#include <cstdlib>
#include <iostream>
#include <string>

#define AZURE_EMBEDDED_SPEECH_GRPC_NO_MAIN
#include "../src/main.cpp"

namespace {

void set_env(const char* name, const char* value) {
#ifdef _WIN32
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

}  // namespace

int main() {
  set_env("VOICE_AGENT_AZURE_EMBEDDED_MODEL_ROOT", "__missing_azure_embedded_models__");
  set_env("VOICE_AGENT_AZURE_EMBEDDED_PRELOAD", "1");
  set_env("VOICE_AGENT_AZURE_EMBEDDED_WARMUP", "1");

  AzureEmbeddedSpeechService service;
  azurepb::HealthRequest request;
  azurepb::HealthResponse response;
  const auto status = service.Health(nullptr, &request, &response);

  require(status.ok(), "Health RPC should succeed even when preloaded model assets are missing");
  require(response.status() == "ok", "Health status should be ok");
  require(response.asr_models_size() == 2, "Expected two ASR model entries");
  require(response.tts_models_size() == 0, "ASR sidecar should not report TTS model entries");

  for (const auto& model : response.asr_models()) {
    require(!model.loaded(), "Missing ASR model should not report loaded when preload is enabled");
    require(model.detail().find("missing:") != std::string::npos, "Missing ASR model detail should include missing path");
  }
  return 0;
}
