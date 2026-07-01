#include <chrono>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>
#include <speechapi_cxx_audio_config.h>
#include <speechapi_cxx_audio_data_stream.h>
#include <speechapi_cxx_audio_stream_format.h>
#include <speechapi_cxx_audio_stream.h>
#include <speechapi_cxx_embedded_speech_config.h>
#include <speechapi_cxx_hybrid_speech_config.h>
#include <speechapi_cxx_speech_recognizer.h>
#include <speechapi_cxx_speech_synthesizer.h>

#include "azure_embedded_speech.grpc.pb.h"

#ifndef SPEECHSDK_VERSION
#define SPEECHSDK_VERSION "unknown"
#endif

namespace azurepb = voice_agent::azure_embedded::v1;
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

enum class ModelKind {
  Asr,
  Tts,
};

struct ModelRuntime {
  ModelEntry entry;
  ModelKind kind;
  mutable std::mutex mutex;
  bool preload_attempted = false;
  bool config_loaded = false;
  bool warmup_ok = false;
  std::string recognition_model_name;
  std::string detail = "not loaded";
  std::shared_ptr<speech::EmbeddedSpeechConfig> config;
};

struct PreparedModel {
  std::shared_ptr<speech::EmbeddedSpeechConfig> config;
  std::string recognition_model_name;
};

bool env_flag(std::string_view name, bool fallback = false) {
  const auto value = env_or(name, fallback ? "1" : "0");
  return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "YES" || value == "on" || value == "ON";
}

bool path_looks_decrypted(const std::string& path) {
  return path.find("decrypted") != std::string::npos;
}

std::string asr_model_key_for_path(const std::string& path) {
  const auto explicit_key = env_or("VOICE_AGENT_AZURE_EMBEDDED_ASR_MODEL_KEY", "");
  if (!explicit_key.empty()) return explicit_key;
  return path_looks_decrypted(path) ? std::string() : env_or("PASCO_MODEL_KEY", "");
}

std::shared_ptr<ModelRuntime> make_model(ModelEntry entry, ModelKind kind) {
  auto runtime = std::make_shared<ModelRuntime>();
  runtime->entry = std::move(entry);
  runtime->kind = kind;
  return runtime;
}

class AzureEmbeddedSpeechService final : public azurepb::AzureEmbeddedSpeech::Service {
 public:
  AzureEmbeddedSpeechService() {
    const auto model_root = env_or("VOICE_AGENT_AZURE_EMBEDDED_MODEL_ROOT", "models/azure-embedded");
    preload_enabled_ = env_flag("VOICE_AGENT_AZURE_EMBEDDED_PRELOAD");
    warmup_enabled_ = env_flag("VOICE_AGENT_AZURE_EMBEDDED_WARMUP");
    asr_models_.push_back(make_model({"azure-embedded-zh-CN-35M", "zh-CN", env_or("VOICE_AGENT_AZURE_EMBEDDED_ASR_ZH_CN_MODEL_DIR", model_root + "/asr/zh-CN/decrypted/35M")}, ModelKind::Asr));
    asr_models_.push_back(make_model({"azure-embedded-en-GB-35M", "en-GB", env_or("VOICE_AGENT_AZURE_EMBEDDED_ASR_EN_GB_MODEL_DIR", model_root + "/asr/en-GB/decrypted/v6/35M")}, ModelKind::Asr));

    if (preload_enabled_) {
      preload_all();
    }
  }

  grpc::Status Health(grpc::ServerContext*, const azurepb::HealthRequest*, azurepb::HealthResponse* response) override {
    response->set_status("ok");
    append_status(asr_models_, response->mutable_asr_models());
    return grpc::Status::OK;
  }

  grpc::Status Recognize(grpc::ServerContext*, grpc::ServerReaderWriter<azurepb::AsrEvent, azurepb::AsrRequest>* stream) override {
    azurepb::AsrRequest request;
    std::string model = "azure-embedded-zh-CN-35M";
    std::string locale = "zh-CN";
    std::atomic<std::int64_t> bytes = 0;
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
      const auto entry = find_model(asr_models_, model, locale);
      if (entry == nullptr) throw std::runtime_error("Unknown Azure Embedded ASR model or locale: " + model + " / " + locale);
      const auto prepared = prepare_model(entry, false);
      const auto model_id = entry->entry.id;
      const auto model_locale = entry->entry.locale;
      const auto format = audio::AudioStreamFormat::GetWaveFormatPCM(16000, 16, 1);
      push_stream = audio::AudioInputStream::CreatePushStream(format);
      const auto audio_config = audio::AudioConfig::FromStreamInput(push_stream);
      recognizer = speech::SpeechRecognizer::FromConfig(prepared.config, audio_config);
      recognizer->Recognizing += [&, model_id, model_locale](const speech::SpeechRecognitionEventArgs& args) {
        const auto text = args.Result->Text;
        if (!text.empty()) {
          azurepb::AsrEvent event;
          event.set_type("partial");
          event.set_model(model_id);
          event.set_locale(model_locale);
          event.set_text(text);
          event.set_bytes(bytes.load());
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
      azurepb::AsrEvent event;
      event.set_type("started");
      event.set_model(model_id);
      event.set_locale(model_locale);
      event.set_detail(prepared.recognition_model_name);
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
          bytes.fetch_add(static_cast<std::int64_t>(audio.size()));
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
      azurepb::AsrEvent error_event;
      error_event.set_type("error");
      error_event.set_model(model);
      error_event.set_locale(locale);
      error_event.set_bytes(bytes.load());
      error_event.set_elapsed_ms(elapsed_ms(started));
      error_event.set_detail(exc.what());
      stream->Write(error_event);
      return grpc::Status::OK;
    }

    azurepb::AsrEvent final_event;
    final_event.set_type("final");
    final_event.set_model(model);
    final_event.set_locale(locale);
    final_event.set_text(final_text);
    final_event.set_bytes(bytes.load());
    final_event.set_elapsed_ms(elapsed_ms(started));
    stream->Write(final_event);
    return grpc::Status::OK;
  }

  grpc::Status Synthesize(grpc::ServerContext*, const azurepb::TtsRequest* request, azurepb::TtsResponse* response) override {
    return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "Use azure_embedded_tts_grpc on 127.0.0.1:8793 for Azure Embedded TTS");
  }

 private:
  static std::shared_ptr<ModelRuntime> find_model(const std::vector<std::shared_ptr<ModelRuntime>>& models, const std::string& id, const std::string& locale) {
    for (const auto& model : models) {
      if (model->entry.id == id || model->entry.locale == locale) return model;
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

  PreparedModel prepare_model(const std::shared_ptr<ModelRuntime>& model, bool warmup) {
    std::lock_guard<std::mutex> lock(model->mutex);
    if (model->config_loaded && model->config) {
      return {model->config, model->recognition_model_name};
    }

    model->preload_attempted = true;
    if (!fs::exists(model->entry.path)) {
      model->config_loaded = false;
      model->detail = "missing: " + model->entry.path;
      throw std::runtime_error(model->detail);
    }

    try {
      auto config = speech::EmbeddedSpeechConfig::FromPath(model->entry.path);
      std::string recognition_model_name;
      if (model->kind == ModelKind::Asr) {
        recognition_model_name = first_recognition_model_name(config, model->entry.locale);
        config->SetSpeechRecognitionModel(recognition_model_name, asr_model_key_for_path(model->entry.path));
      } else {
        config->SetSpeechSynthesisVoice(model->entry.id, env_or("PASCO_MODEL_KEY", ""));
        config->SetSpeechSynthesisOutputFormat(speech::SpeechSynthesisOutputFormat::Riff24Khz16BitMonoPcm);
      }

      model->config = std::move(config);
      model->recognition_model_name = recognition_model_name;
      model->config_loaded = true;
      model->warmup_ok = false;
      model->detail = "config_loaded";
      if (!recognition_model_name.empty()) model->detail += "; recognition_model=" + recognition_model_name;
      if (warmup) {
        try {
          warmup_model(*model);
        } catch (const std::exception& exc) {
          model->warmup_ok = false;
          model->detail += std::string("; warmup_failed: ") + exc.what();
        }
      }
      return {model->config, model->recognition_model_name};
    } catch (const std::exception& exc) {
      model->config.reset();
      model->config_loaded = false;
      model->warmup_ok = false;
      model->detail = std::string("preload_failed: ") + exc.what();
      throw;
    }
  }

  void preload_all() {
    for (const auto& model : asr_models_) {
      try {
        prepare_model(model, warmup_enabled_);
      } catch (const std::exception&) {
      }
    }
  }

  static void warmup_model(ModelRuntime& model) {
    if (!model.config) return;
    if (model.kind == ModelKind::Asr) {
      auto format = audio::AudioStreamFormat::GetWaveFormatPCM(16000, 16, 1);
      auto stream = audio::AudioInputStream::CreatePushStream(format);
      auto audio_config = audio::AudioConfig::FromStreamInput(stream);
      auto recognizer = speech::SpeechRecognizer::FromConfig(model.config, audio_config);
      recognizer->StartContinuousRecognitionAsync().get();
      std::vector<std::uint8_t> silence(3200, 0);
      stream->Write(silence.data(), static_cast<std::uint32_t>(silence.size()));
      stream->Close();
      recognizer->StopContinuousRecognitionAsync().get();
    } else {
      auto synthesizer = speech::SpeechSynthesizer::FromConfig(model.config, nullptr);
      const auto result = synthesizer->SpeakText("hello");
      if (result->Reason != speech::ResultReason::SynthesizingAudioCompleted) {
        throw std::runtime_error("warmup synthesis did not complete");
      }
    }
    model.warmup_ok = true;
    model.detail += "; warmup=ok";
  }

  void append_status(const std::vector<std::shared_ptr<ModelRuntime>>& models, google::protobuf::RepeatedPtrField<azurepb::ModelStatus>* statuses) const {
    for (const auto& model : models) {
      auto* status = statuses->Add();
      std::lock_guard<std::mutex> lock(model->mutex);
      const auto present = fs::exists(model->entry.path);
      status->set_id(model->entry.id);
      status->set_locale(model->entry.locale);
      status->set_path(model->entry.path);
      status->set_loaded((model->config_loaded && (!warmup_enabled_ || model->warmup_ok)) || (!preload_enabled_ && present));
      if (!present) {
        status->set_detail("missing: " + model->entry.path);
      } else if (!preload_enabled_ && !model->preload_attempted) {
        status->set_detail("present; preload=disabled");
      } else {
        std::ostringstream detail;
        detail << model->detail << "; warmup=" << (model->warmup_ok ? "ok" : (warmup_enabled_ ? "failed" : "disabled"));
        status->set_detail(detail.str());
      }
    }
  }

  static std::int64_t elapsed_ms(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
  }

  bool preload_enabled_ = false;
  bool warmup_enabled_ = false;
  std::vector<std::shared_ptr<ModelRuntime>> asr_models_;
};

}  // namespace

#ifndef AZURE_EMBEDDED_SPEECH_GRPC_NO_MAIN
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
  std::cout << "Azure Embedded Speech gRPC server listening on " << address << " with Speech SDK target " << SPEECHSDK_VERSION << '\n';
  server->Wait();
  return 0;
}
#endif
