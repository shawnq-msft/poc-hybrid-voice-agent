#include <chrono>
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
#include <speechapi_cxx_embedded_speech_config.h>
#include <speechapi_cxx_enums.h>
#include <speechapi_cxx_eventsignal.h>
#include <speechapi_cxx_hybrid_speech_config.h>
#include <speechapi_cxx_speech_synthesis_eventargs.h>
#include <speechapi_cxx_speech_synthesis_word_boundary_eventargs.h>
#include <speechapi_cxx_speech_config.h>
#include <speechapi_cxx_speech_synthesizer.h>

#include "azure_embedded_speech.grpc.pb.h"

#ifndef SPEECHSDK_VERSION
#define SPEECHSDK_VERSION "unknown"
#endif

namespace azurepb = voice_agent::azure_embedded::v1;
namespace fs = std::filesystem;
namespace speech = Microsoft::CognitiveServices::Speech;

namespace {

std::string env_or(std::string_view name, std::string_view fallback) {
  const char* value = std::getenv(std::string(name).c_str());
  if (value == nullptr || std::string(value).empty()) {
    return std::string(fallback);
  }
  return value;
}

bool env_flag(std::string_view name, bool fallback = false) {
  const auto value = env_or(name, fallback ? "1" : "0");
  return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "YES" || value == "on" || value == "ON";
}

struct TtsModel {
  std::string id;
  std::string locale;
  std::string path;
  std::string offline_voice;
  std::string online_voice;
};

struct TtsRuntime {
  explicit TtsRuntime(TtsModel model) : model(std::move(model)) {}

  TtsModel model;
  std::mutex mutex;
  bool prepared = false;
  bool warmup_ok = false;
  std::int64_t prepare_elapsed_ms = 0;
  std::int64_t warmup_elapsed_ms = 0;
  std::string detail;
  std::shared_ptr<speech::SpeechConfig> speech_config;
  std::shared_ptr<speech::SpeechSynthesizer> synthesizer;
};

class AzureEmbeddedTtsService final : public azurepb::AzureEmbeddedSpeech::Service {
 public:
  AzureEmbeddedTtsService() {
    const auto model_root = env_or("VOICE_AGENT_AZURE_EMBEDDED_MODEL_ROOT", "models/azure-embedded");
    default_voice_ = env_or("VOICE_AGENT_AZURE_EMBEDDED_TTS_VOICE", "azure-embedded-zh-CN-XiaoxiaoNeuralV6");
    preload_enabled_ = env_flag("VOICE_AGENT_AZURE_EMBEDDED_TTS_PRELOAD", true);
    preload_all_models_ = env_flag("VOICE_AGENT_AZURE_EMBEDDED_TTS_PRELOAD_ALL", false);
    warmup_enabled_ = env_flag("VOICE_AGENT_AZURE_EMBEDDED_TTS_WARMUP", true);
    models_.push_back(std::make_shared<TtsRuntime>(TtsModel{
      "azure-embedded-zh-CN-XiaoxiaoNeuralV6",
      "zh-CN",
      env_or("VOICE_AGENT_AZURE_EMBEDDED_TTS_ZH_CN_MODEL_DIR", model_root + "/tts/zh-CN/XiaoxiaoNeuralV6"),
      env_or("VOICE_AGENT_AZURE_EMBEDDED_TTS_ZH_CN_OFFLINE_VOICE", "Microsoft Server Speech Text to Speech Voice (zh-CN, XiaoxiaoNeural)"),
      env_or("VOICE_AGENT_AZURE_EMBEDDED_TTS_ZH_CN_ONLINE_VOICE", "zh-CN-XiaoxiaoNeural"),
    }));
    models_.push_back(std::make_shared<TtsRuntime>(TtsModel{
      "azure-embedded-en-US-AvaNeuralHD",
      "en-US",
      env_or("VOICE_AGENT_AZURE_EMBEDDED_TTS_EN_US_MODEL_DIR", model_root + "/tts/en-US/AvaNeuralHDv2"),
      env_or("VOICE_AGENT_AZURE_EMBEDDED_TTS_EN_US_OFFLINE_VOICE", "Microsoft Server Speech Text to Speech Voice (en-US, AvaHD)"),
      env_or("VOICE_AGENT_AZURE_EMBEDDED_TTS_EN_US_ONLINE_VOICE", "en-US-AvaMultilingualNeural"),
    }));
    if (preload_enabled_) {
      preload_all();
    }
  }

  grpc::Status Health(grpc::ServerContext*, const azurepb::HealthRequest*, azurepb::HealthResponse* response) override {
    response->set_status("ok");
    for (const auto& runtime : models_) {
      std::lock_guard<std::mutex> lock(runtime->mutex);
      auto* status = response->mutable_tts_models()->Add();
      const auto present = fs::exists(runtime->model.path);
      status->set_id(runtime->model.id);
      status->set_locale(runtime->model.locale);
      status->set_path(runtime->model.path);
      status->set_loaded(present && runtime->prepared);
      if (!present) {
        status->set_detail("missing: " + runtime->model.path);
      } else {
        std::ostringstream detail;
        detail << "present; sdk=" << SPEECHSDK_VERSION
               << "; offline_voice=" << runtime->model.offline_voice
               << "; resident=" << (runtime->prepared ? "yes" : "no")
               << "; prepare_ms=" << runtime->prepare_elapsed_ms
               << "; warmup=" << (runtime->warmup_ok ? "ok" : "not_run")
               << "; warmup_ms=" << runtime->warmup_elapsed_ms;
        if (!runtime->detail.empty()) detail << "; " << runtime->detail;
        status->set_detail(detail.str());
      }
    }
    return grpc::Status::OK;
  }

  grpc::Status Synthesize(grpc::ServerContext*, const azurepb::TtsRequest* request, azurepb::TtsResponse* response) override {
    const auto started = std::chrono::steady_clock::now();
    const auto voice = request->voice().empty() ? std::string("azure-embedded-zh-CN-XiaoxiaoNeuralV6") : request->voice();
    const auto locale = request->locale().empty() ? std::string("zh-CN") : request->locale();
    const auto runtime = find_model(voice, locale);
    if (!runtime) {
      return grpc::Status(grpc::StatusCode::NOT_FOUND, "Unknown Azure Embedded TTS voice or locale: " + voice + " / " + locale);
    }
    if (!fs::exists(runtime->model.path)) {
      return grpc::Status(grpc::StatusCode::NOT_FOUND, "Missing Azure Embedded TTS model: " + runtime->model.path);
    }
    if (request->text().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Cannot synthesize empty text");
    }

    try {
      std::lock_guard<std::mutex> lock(runtime->mutex);
      prepare_runtime(*runtime, false);
      const auto result = runtime->synthesizer->SpeakText(request->text());
      if (result->Reason != speech::ResultReason::SynthesizingAudioCompleted) {
        const auto cancellation = speech::SpeechSynthesisCancellationDetails::FromResult(result);
        std::ostringstream detail;
        detail << "Azure Embedded TTS did not complete synthesis; result_reason=" << static_cast<int>(result->Reason)
               << "; cancellation_reason=" << static_cast<int>(cancellation->Reason)
               << "; error_code=" << static_cast<int>(cancellation->ErrorCode)
               << "; error_details=" << cancellation->ErrorDetails;
        return grpc::Status(grpc::StatusCode::INTERNAL, detail.str());
      }
      const auto audio_data = result->GetAudioData();
      response->set_voice(runtime->model.id);
      response->set_locale(runtime->model.locale);
      response->set_media_type("audio/wav");
      response->set_audio(audio_data->data(), audio_data->size());
      response->set_elapsed_ms(elapsed_ms(started));
      return grpc::Status::OK;
    } catch (const std::exception& exc) {
      return grpc::Status(grpc::StatusCode::INTERNAL, exc.what());
    }
  }

 private:
  std::shared_ptr<TtsRuntime> find_model(const std::string& id, const std::string& locale) const {
    for (const auto& runtime : models_) {
      if (runtime->model.id == id || runtime->model.locale == locale) return runtime;
    }
    return {};
  }

  void preload_all() {
    for (const auto& runtime : models_) {
      if (!preload_all_models_ && runtime->model.id != default_voice_ && runtime->model.locale != default_voice_) continue;
      if (!fs::exists(runtime->model.path)) continue;
      std::lock_guard<std::mutex> lock(runtime->mutex);
      try {
        prepare_runtime(*runtime, warmup_enabled_);
      } catch (const std::exception& exc) {
        runtime->detail = std::string("preload_failed: ") + exc.what();
      }
    }
  }

  void prepare_runtime(TtsRuntime& runtime, bool warmup) const {
    if (runtime.prepared) return;
    const auto started = std::chrono::steady_clock::now();
    runtime.speech_config = make_tts_config(runtime.model);
    runtime.synthesizer = speech::SpeechSynthesizer::FromConfig(runtime.speech_config, nullptr);
    runtime.prepared = true;
    runtime.prepare_elapsed_ms = elapsed_ms(started);
    if (warmup) {
      const auto warmup_started = std::chrono::steady_clock::now();
      const auto warmup_text = env_or("VOICE_AGENT_AZURE_EMBEDDED_TTS_WARMUP_TEXT", "Ready.");
      const auto result = runtime.synthesizer->SpeakText(warmup_text);
      runtime.warmup_elapsed_ms = elapsed_ms(warmup_started);
      runtime.warmup_ok = result->Reason == speech::ResultReason::SynthesizingAudioCompleted;
      if (!runtime.warmup_ok) {
        const auto cancellation = speech::SpeechSynthesisCancellationDetails::FromResult(result);
        runtime.detail = "warmup_failed: " + cancellation->ErrorDetails;
      }
    }
  }

  std::shared_ptr<speech::SpeechConfig> make_tts_config(const TtsModel& model) const {
    const auto subscription = env_or("AZURE_SPEECH_KEY", env_or("SPEECH_KEY", "unused"));
    const auto region = env_or("AZURE_SPEECH_REGION", env_or("SPEECH_REGION", "eastus"));
    auto speech_config = speech::SpeechConfig::FromSubscription(subscription, region);
    speech_config->SetSpeechSynthesisVoiceName(model.online_voice);
    speech_config->SetSpeechSynthesisOutputFormat(speech::SpeechSynthesisOutputFormat::Riff24Khz16BitMonoPcm);
    speech_config->SetProperty(speech::PropertyId::SpeechServiceConnection_SynthEnableCompressedAudioTransmission, "true");
    speech_config->SetProperty("SpeechSynthesis_KeepConnectionAfterStopping", "true");
    speech_config->SetProperty(speech::PropertyId::SpeechServiceConnection_SynthBackend, env_or("VOICE_AGENT_AZURE_EMBEDDED_TTS_BACKEND", "hybrid"));
    speech_config->SetProperty(speech::PropertyId::SpeechServiceConnection_SynthOfflineDataPath, model.path);
    speech_config->SetProperty(speech::PropertyId::SpeechServiceConnection_SynthOfflineVoice, model.offline_voice);
    const auto model_key = env_or("VOICE_AGENT_AZURE_EMBEDDED_TTS_MODEL_KEY", env_or("PASCO_MODEL_KEY", ""));
    if (!model_key.empty()) {
      speech_config->SetProperty(speech::PropertyId::SpeechServiceConnection_SynthModelKey, model_key);
    }
    speech_config->SetProperty("SPEECH-SynthBackendSwitchingPolicy", env_or("VOICE_AGENT_AZURE_EMBEDDED_TTS_POLICY", "force_offline"));
    speech_config->SetProperty("SPEECH-SynthBackendFallbackBufferTimeoutMs", env_or("VOICE_AGENT_AZURE_EMBEDDED_TTS_FALLBACK_TIMEOUT_MS", "800"));
    speech_config->SetProperty("SPEECH-SynthBackendFallbackBufferLengthMs", env_or("VOICE_AGENT_AZURE_EMBEDDED_TTS_FALLBACK_BUFFER_MS", "200"));
    if (env_flag("VOICE_AGENT_AZURE_EMBEDDED_TTS_CACHE")) {
      speech_config->SetProperty("SPEECH-SynthesisCachingPath", env_or("VOICE_AGENT_AZURE_EMBEDDED_TTS_CACHE_DIR", fs::temp_directory_path().string()));
      speech_config->SetProperty("SPEECH-SynthesisCachingMaxNumber", env_or("VOICE_AGENT_AZURE_EMBEDDED_TTS_CACHE_MAX_NUMBER", "10000"));
    }

    auto embedded_config = speech::EmbeddedSpeechConfig::FromPath(model.path);
    (void)embedded_config;
    return speech_config;
  }

  static std::int64_t elapsed_ms(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
  }

  bool preload_enabled_ = true;
  bool preload_all_models_ = false;
  bool warmup_enabled_ = true;
  std::string default_voice_;
  std::vector<std::shared_ptr<TtsRuntime>> models_;
};

}  // namespace

int main(int argc, char** argv) {
  const auto address = env_or("AZURE_EMBEDDED_TTS_GRPC_URL", env_or("AZURE_EMBEDDED_GRPC_URL", "127.0.0.1:8793"));
  AzureEmbeddedTtsService service;
  grpc::ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();
  if (!server) {
    std::cerr << "Failed to start Azure Embedded TTS gRPC server on " << address << '\n';
    return 1;
  }
  std::cout << "Azure Embedded TTS gRPC server listening on " << address << " with Speech SDK target " << SPEECHSDK_VERSION << '\n';
  server->Wait();
  return 0;
}