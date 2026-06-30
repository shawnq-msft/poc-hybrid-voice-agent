#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>
#include <speechapi_cxx_audio_config.h>
#include <speechapi_cxx_audio_data_stream.h>
#include <speechapi_cxx_audio_stream_format.h>
#include <speechapi_cxx_audio_stream.h>
#include <speechapi_cxx_embedded_speech_config.h>
#include <speechapi_cxx_speech_recognizer.h>
#include <speechapi_cxx_speech_synthesizer.h>

#include "azure_embedded_speech.grpc.pb.h"

namespace pb = voice_agent::azure_embedded::v1;
namespace fs = std::filesystem;
namespace speech = Microsoft::CognitiveServices::Speech;
namespace audio = Microsoft::CognitiveServices::Speech::Audio;

namespace {

std::string env_or(std::string_view name, std::string_view fallback) {
  const char* value = std::getenv(std::string(name).c_str());
  if (value == nullptr || std::string(value).empty()) {
    return std::string(fallback);
  }
  return value;
}

struct ModelEntry {
  std::string id;
  std::string locale;
  std::string path;
};

class AzureEmbeddedSpeechService final : public pb::AzureEmbeddedSpeech::Service {
 public:
  AzureEmbeddedSpeechService() {
    const auto model_root = env_or("VOICE_AGENT_AZURE_EMBEDDED_MODEL_ROOT", "models/azure-embedded");
    asr_models_.push_back({"azure-embedded-zh-CN-35M", "zh-CN", env_or("VOICE_AGENT_AZURE_EMBEDDED_ASR_ZH_CN_MODEL_DIR", model_root + "/asr/zh-CN/encrypted/35M")});
    asr_models_.push_back({"azure-embedded-en-GB-35M", "en-GB", env_or("VOICE_AGENT_AZURE_EMBEDDED_ASR_EN_GB_MODEL_DIR", model_root + "/asr/en-GB/encrypted/v6/35M")});
    tts_models_.push_back({"azure-embedded-zh-CN-XiaoxiaoNeuralHD", "zh-CN", env_or("VOICE_AGENT_AZURE_EMBEDDED_TTS_ZH_CN_MODEL_DIR", model_root + "/tts/zh-CN/XiaoxiaoNeuralHD")});
    tts_models_.push_back({"azure-embedded-en-US-AvaNeuralHD", "en-US", env_or("VOICE_AGENT_AZURE_EMBEDDED_TTS_EN_US_MODEL_DIR", model_root + "/tts/en-US/AvaNeuralHDv2")});
  }

  grpc::Status Health(grpc::ServerContext*, const pb::HealthRequest*, pb::HealthResponse* response) override {
    response->set_status("ok");
    append_status(asr_models_, response->mutable_asr_models());
    append_status(tts_models_, response->mutable_tts_models());
    return grpc::Status::OK;
  }

  grpc::Status Recognize(grpc::ServerContext*, grpc::ServerReaderWriter<pb::AsrEvent, pb::AsrRequest>* stream) override {
    pb::AsrRequest request;
    std::string model = "azure-embedded-zh-CN-35M";
    std::string locale = "zh-CN";
    std::int64_t bytes = 0;
    const auto started = std::chrono::steady_clock::now();
    std::string final_text;
    bool started_recognition = false;
    bool stopped = false;
    std::mutex mutex;
    std::condition_variable stopped_cv;
    std::shared_ptr<audio::PushAudioInputStream> push_stream;
    std::shared_ptr<speech::SpeechRecognizer> recognizer;

    auto start_recognition = [&]() {
      if (started_recognition) return;
      const auto* entry = find_model(asr_models_, model, locale);
      if (entry == nullptr) throw std::runtime_error("Unknown Azure Embedded ASR model or locale: " + model + " / " + locale);
      const auto speech_config = speech::EmbeddedSpeechConfig::FromPath(entry->path);
      const auto reco_model_name = first_recognition_model_name(speech_config, entry->locale);
      speech_config->SetSpeechRecognitionModel(reco_model_name, env_or("PASCO_MODEL_KEY", ""));
      const auto format = audio::AudioStreamFormat::GetWaveFormatPCM(16000, 16, 1);
      push_stream = audio::AudioInputStream::CreatePushStream(format);
      const auto audio_config = audio::AudioConfig::FromStreamInput(push_stream);
      recognizer = speech::SpeechRecognizer::FromConfig(speech_config, audio_config);
      recognizer->Recognizing += [&](const speech::SpeechRecognitionEventArgs& args) {
        const auto text = args.Result->Text;
        if (!text.empty()) {
          pb::AsrEvent event;
          event.set_type("partial");
          event.set_model(entry->id);
          event.set_locale(entry->locale);
          event.set_text(text);
          event.set_bytes(bytes);
          event.set_elapsed_ms(elapsed_ms(started));
          stream->Write(event);
        }
      };
      recognizer->Recognized += [&](const speech::SpeechRecognitionEventArgs& args) {
        const auto text = args.Result->Text;
        if (args.Result->Reason == speech::ResultReason::RecognizedSpeech && !text.empty()) {
          std::lock_guard<std::mutex> lock(mutex);
          if (!final_text.empty()) final_text += " ";
          final_text += text;
        }
      };
      recognizer->Canceled += [&](const speech::SpeechRecognitionCanceledEventArgs& args) {
        std::lock_guard<std::mutex> lock(mutex);
        stopped = true;
        stopped_cv.notify_one();
      };
      recognizer->SessionStopped += [&](const speech::SessionEventArgs&) {
        std::lock_guard<std::mutex> lock(mutex);
        stopped = true;
        stopped_cv.notify_one();
      };
      recognizer->StartContinuousRecognitionAsync().get();
      started_recognition = true;
      pb::AsrEvent event;
      event.set_type("started");
      event.set_model(entry->id);
      event.set_locale(entry->locale);
      event.set_detail(reco_model_name);
      stream->Write(event);
    };

    try {
      while (stream->Read(&request)) {
        if (request.has_config()) {
          const auto& config = request.config();
          if (!config.model().empty()) model = config.model();
          if (!config.locale().empty()) locale = config.locale();
          start_recognition();
        } else if (request.has_pcm16()) {
          start_recognition();
          auto audio = request.pcm16();
          push_stream->Write(reinterpret_cast<uint8_t*>(audio.data()), static_cast<uint32_t>(audio.size()));
          bytes += static_cast<std::int64_t>(audio.size());
        } else if (request.has_end()) {
          break;
        }
      }
      if (push_stream) push_stream->Close();
      if (recognizer) {
        std::unique_lock<std::mutex> lock(mutex);
        stopped_cv.wait_for(lock, std::chrono::seconds(8), [&]() { return stopped; });
        lock.unlock();
        recognizer->StopContinuousRecognitionAsync().get();
      }
    } catch (const std::exception& exc) {
      pb::AsrEvent error_event;
      error_event.set_type("error");
      error_event.set_model(model);
      error_event.set_locale(locale);
      error_event.set_bytes(bytes);
      error_event.set_elapsed_ms(elapsed_ms(started));
      error_event.set_detail(exc.what());
      stream->Write(error_event);
      return grpc::Status::OK;
    }

    pb::AsrEvent final_event;
    final_event.set_type("final");
    final_event.set_model(model);
    final_event.set_locale(locale);
    final_event.set_text(final_text);
    final_event.set_bytes(bytes);
    final_event.set_elapsed_ms(elapsed_ms(started));
    stream->Write(final_event);
    return grpc::Status::OK;
  }

  grpc::Status Synthesize(grpc::ServerContext*, const pb::TtsRequest* request, pb::TtsResponse* response) override {
    const auto started = std::chrono::steady_clock::now();
    const auto voice = request->voice().empty() ? std::string("azure-embedded-zh-CN-XiaoxiaoNeuralHD") : request->voice();
    const auto locale = request->locale().empty() ? std::string("zh-CN") : request->locale();
    const auto* entry = find_model(tts_models_, voice, locale);
    if (entry == nullptr) {
      return grpc::Status(grpc::StatusCode::NOT_FOUND, "Unknown Azure Embedded TTS voice or locale: " + voice + " / " + locale);
    }
    try {
      const auto speech_config = speech::EmbeddedSpeechConfig::FromPath(entry->path);
      speech_config->SetSpeechSynthesisVoice(entry->id, env_or("PASCO_MODEL_KEY", ""));
      speech_config->SetSpeechSynthesisOutputFormat(speech::SpeechSynthesisOutputFormat::Riff24Khz16BitMonoPcm);
      const auto synthesizer = speech::SpeechSynthesizer::FromConfig(speech_config, nullptr);
      const auto result = synthesizer->SpeakText(request->text());
      if (result->Reason != speech::ResultReason::SynthesizingAudioCompleted) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "Azure Embedded TTS did not complete synthesis");
      }
      const auto audio_data = result->GetAudioData();
      response->set_voice(entry->id);
      response->set_locale(entry->locale);
      response->set_media_type("audio/wav");
      response->set_audio(audio_data->data(), audio_data->size());
      response->set_elapsed_ms(elapsed_ms(started));
      return grpc::Status::OK;
    } catch (const std::exception& exc) {
      return grpc::Status(grpc::StatusCode::INTERNAL, exc.what());
    }
  }

 private:
  static const ModelEntry* find_model(const std::vector<ModelEntry>& models, const std::string& id, const std::string& locale) {
    for (const auto& model : models) {
      if (model.id == id || model.locale == locale) return &model;
    }
    return nullptr;
  }

  static std::string first_recognition_model_name(std::shared_ptr<speech::EmbeddedSpeechConfig> config, const std::string& locale) {
    auto models = config->GetSpeechRecognitionModels();
    if (models.empty()) throw std::runtime_error("No Azure Embedded ASR recognition models found");
    for (const auto& model : models) {
      for (const auto& candidate_locale : model->Locales) {
        if (candidate_locale == locale) return model->Name;
      }
    }
    return models.front()->Name;
  }

  static void append_status(const std::vector<ModelEntry>& models, google::protobuf::RepeatedPtrField<pb::ModelStatus>* statuses) {
    for (const auto& model : models) {
      auto* status = statuses->Add();
      status->set_id(model.id);
      status->set_locale(model.locale);
      status->set_path(model.path);
      status->set_loaded(fs::exists(model.path));
      status->set_detail(fs::exists(model.path) ? "present" : "missing");
    }
  }

  static std::int64_t elapsed_ms(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
  }

  std::vector<ModelEntry> asr_models_;
  std::vector<ModelEntry> tts_models_;
};

}  // namespace

int main(int argc, char** argv) {
  const auto address = env_or("AZURE_EMBEDDED_GRPC_URL", "127.0.0.1:8792");
  AzureEmbeddedSpeechService service;
  grpc::ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();
  if (!server) {
    std::cerr << "Failed to start Azure Embedded Speech gRPC server on " << address << '\n';
    return 1;
  }
  std::cout << "Azure Embedded Speech gRPC server listening on " << address << " with Speech SDK target 1.47\n";
  server->Wait();
  return 0;
}
