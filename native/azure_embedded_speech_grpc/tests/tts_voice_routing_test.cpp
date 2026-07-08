#include <cstdlib>
#include <iostream>
#include <string>

#define AZURE_EMBEDDED_TTS_GRPC_NO_MAIN
#include "../src/tts_main.cpp"

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
  set_env("VOICE_AGENT_AZURE_EMBEDDED_TTS_PRELOAD", "0");
  set_env("VOICE_AGENT_AZURE_EMBEDDED_TTS_WARMUP", "0");

  AzureEmbeddedTtsService service;
  azurepb::TtsRequest request;
  azurepb::TtsResponse response;
  request.set_voice("azure-embedded-en-US-AvaNeuralHD");
  request.set_locale("zh-CN");
  request.set_text("Hello.");
  request.set_sample_rate_hz(24000);

  const auto status = service.Synthesize(nullptr, &request, &response);

  require(!status.ok(), "Synthesize should fail when the selected model path is missing");
  require(status.error_message().find("tts/en-US/AvaNeuralHDv2") != std::string::npos, "Explicit Ava voice should select the Ava model path");
  require(status.error_message().find("tts/zh-CN/XiaoxiaoNeuralV6") == std::string::npos, "Conflicting zh-CN locale must not override explicit Ava voice");
  return 0;
}