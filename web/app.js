const state = {
  localStream: null,
  mediaRecorder: null,
  recordedChunks: [],
  audioContext: null,
  audioSource: null,
  analyser: null,
  analyserData: null,
  pcmProcessor: null,
  volumeLoop: null,
  muted: false,
  connected: false,
  config: null,
  modelsLoaded: false,
  loadedModelSignature: null,
  llmConfigSyncTimer: null,
  sending: false,
  azureAsr: null,
  turnStream: null,
  responseSocket: null,
  responseTurnId: 0,
  interruptedTurnId: 0,
  currentTurn: {
    vadE2E: null,
    asrE2E: null,
    llmTtft: null,
    llmFirstPunctuation: null,
    ttsStartedAt: null,
    ttsFirstByteMs: null,
    voiceToVoiceFirstByteMs: null,
    streamedAudioChunks: 0,
    streamedAssistantText: false,
    streamingAsr: false,
  },
  typewriterQueue: [],
  typewriterActive: false,
  audioQueue: [],
  playingAudio: false,
  vad: {
    active: false,
    speechStartedAt: 0,
    lastSpeechAt: 0,
    speechEndedAt: 0,
  },
  metrics: {
    values: { vad: [], asr: [], llm: [], tts: [], voiceToVoice: [] },
  },
};

const VAD = {
  speechThreshold: 0.035,
  silenceThreshold: 0.018,
  minSpeechMs: 350,
  minTurnMs: 900,
  silenceEndMs: 500,
  recorderTimesliceMs: 500,
  minBlobBytes: 1400,
};

const connectionStatus = document.querySelector("#connectionStatus");
const providerStatus = document.querySelector("#providerStatus");
const loadModelsButton = document.querySelector("#loadModelsButton");
const connectButton = document.querySelector("#connectButton");
const backendCheckButton = document.querySelector("#backendCheckButton");
const muteButton = document.querySelector("#muteButton");
const disconnectButton = document.querySelector("#disconnectButton");
const eventLog = document.querySelector("#eventLog");
const userTranscript = document.querySelector("#userTranscript");
const assistantTranscript = document.querySelector("#assistantTranscript");
const remoteAudio = document.querySelector("#remoteAudio");
const volumeFill = document.querySelector("#volumeFill");
const volumeValue = document.querySelector("#volumeValue");
const micStatus = document.querySelector("#micStatus");
const vadStatus = document.querySelector("#vadStatus");
const asrStatus = document.querySelector("#asrStatus");
const llmStatus = document.querySelector("#llmStatus");
const ttsStatus = document.querySelector("#ttsStatus");
const vadSelector = document.querySelector("#vadSelector");
const asrSelector = document.querySelector("#asrSelector");
const asrMetricLabel = document.querySelector("#asrMetricLabel");
const llmSelector = document.querySelector("#llmSelector");
const ttsSelector = document.querySelector("#ttsSelector");
const llmPromptInput = document.querySelector("#llmPromptInput");
const llmContextInput = document.querySelector("#llmContextInput");

const configurableControls = [vadSelector, asrSelector, llmSelector, ttsSelector, llmPromptInput, llmContextInput].filter(Boolean);

const metricElements = {
  vad: [document.querySelector("#vadTurn"), document.querySelector("#vadAvg"), document.querySelector("#vadP90")],
  asr: [document.querySelector("#asrTurn"), document.querySelector("#asrAvg"), document.querySelector("#asrP90")],
  llm: [document.querySelector("#llmTurn"), document.querySelector("#llmAvg"), document.querySelector("#llmP90")],
  tts: [document.querySelector("#ttsTurn"), document.querySelector("#ttsAvg"), document.querySelector("#ttsP90")],
  voiceToVoice: [document.querySelector("#v2vTurn"), document.querySelector("#v2vAvg"), document.querySelector("#v2vP90")],
};

function logEvent(message) {
  const item = document.createElement("li");
  item.textContent = `${new Date().toLocaleTimeString()} ${message}`;
  eventLog.appendChild(item);
  item.scrollIntoView({ block: "nearest" });
}

function setStatus(message, muted = false) {
  connectionStatus.textContent = message;
  connectionStatus.classList.toggle("muted", muted);
}

function setVad(status) {
  vadStatus.textContent = status;
}

function setModuleStatus(updates) {
  if (updates.mic) micStatus.textContent = updates.mic;
  if (updates.asr) asrStatus.textContent = updates.asr;
  if (updates.llm) llmStatus.textContent = updates.llm;
  if (updates.tts) ttsStatus.textContent = updates.tts;
}

function setConfigurationLocked(locked) {
  for (const control of configurableControls) {
    control.disabled = locked;
  }
  loadModelsButton.disabled = locked;
}

function syncSelectors(config) {
  setSelectByValue(vadSelector, "silero:500");
  updateVadSelection(false);
  setSelectByValue(asrSelector, "azure-embedded:zh-CN");
  updateAsrMetricLabel();
  setSelectByText(llmSelector, "Foundry qwen2.5 0.5B");
  setSelectByValue(ttsSelector, ["azure-embedded", "edge-tts", "windows-sapi"].includes(config.providers.tts) ? config.providers.tts : "azure-embedded");
  if (llmPromptInput) llmPromptInput.value = config.llmDefaults?.prompt || "";
  if (llmContextInput) llmContextInput.value = config.llmDefaults?.context || "";
}

function setSelectByText(select, text) {
  const option = Array.from(select.options).find((candidate) => candidate.textContent === text);
  if (option) {
    select.value = option.value;
  }
}

function setSelectByValue(select, value) {
  const option = Array.from(select.options).find((candidate) => candidate.value === value && !candidate.disabled);
  if (option) {
    select.value = option.value;
  }
}

function updateVadSelection(announce = true) {
  const [provider, silenceMs] = vadSelector.value.split(":");
  VAD.silenceEndMs = Number(silenceMs);
  if (announce) {
    logEvent(`VAD set to ${provider} with ${VAD.silenceEndMs} ms end silence.`);
  }
}

function selectedTtsProvider() {
  return ttsSelector.value;
}

function selectedLlmOptions() {
  return {
    llmModel: llmSelector.value,
    llmPrompt: llmPromptInput?.value || "",
    llmContext: llmContextInput?.value || "",
  };
}

function selectedRuntimeOptions() {
  return { ttsProvider: selectedTtsProvider(), ...selectedAsrOptions(), ...selectedLlmOptions() };
}

function selectedLlmConfigPayload() {
  return { prompt: llmPromptInput?.value || "", context: llmContextInput?.value || "" };
}

function selectedModelOptions() {
  return { ttsProvider: selectedTtsProvider(), ...selectedAsrOptions(), llmModel: llmSelector.value };
}

function currentModelSignature() {
  return JSON.stringify(selectedModelOptions());
}

function markModelsDirty(reason = "Configuration changed. Load Models again before Start.") {
  if (!state.modelsLoaded) {
    return;
  }
  state.modelsLoaded = false;
  state.loadedModelSignature = null;
  connectButton.disabled = true;
  setStatus("Models need reload", true);
  setModuleStatus({ asr: "Stale", llm: "Stale", tts: "Stale" });
  logEvent(reason);
}

function selectedAsrOptions() {
  const [asrProvider, asrDetail, asrLanguage] = asrSelector.value.split(":");
  if (["foundry-local", "faster-whisper"].includes(asrProvider)) {
    return { asrProvider, asrModel: asrDetail, asrLanguage: asrLanguage || "auto", asrLocale: "auto" };
  }
  return { asrProvider, asrLocale: asrDetail || "auto", asrLanguage: "auto" };
}

function selectedAsrMode() {
  const { asrProvider } = selectedAsrOptions();
  return state.config?.providers?.asrCapabilities?.[asrProvider]?.transportMode || "streaming";
}

function isStreamingAsrSelected() {
  return selectedAsrMode() === "streaming";
}

function isAzureEmbeddedAsrSelected() {
  return selectedAsrOptions().asrProvider === "azure-embedded";
}

function updateAsrMetricLabel() {
  if (asrMetricLabel) {
    asrMetricLabel.textContent = isStreamingAsrSelected() ? "Final" : "E2E";
  }
}

async function loadConfig() {
  try {
    const response = await fetch("/api/config");
    const config = await response.json();
    state.config = config;
    providerStatus.textContent = `${config.providers.asr} ASR / ${config.providers.tts} TTS / ${config.providers.llm} LLM`;
    syncSelectors(config);
    logEvent("Loaded provider configuration.");
  } catch (error) {
    providerStatus.textContent = "Provider config unavailable";
    logEvent(`Config failed: ${error.message}`);
  }
}

function scheduleLlmConfigSync() {
  if (state.llmConfigSyncTimer) {
    clearTimeout(state.llmConfigSyncTimer);
  }
  state.llmConfigSyncTimer = setTimeout(() => {
    state.llmConfigSyncTimer = null;
    syncLlmConfig().catch((error) => logEvent(`LLM config sync failed: ${error.message}`));
  }, 300);
}

async function syncLlmConfig() {
  const response = await fetch("/api/llm-config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(selectedLlmConfigPayload()),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail?.message || payload.detail || "LLM config sync failed");
  }
  const payload = await response.json();
  if (llmPromptInput && document.activeElement !== llmPromptInput) llmPromptInput.value = payload.prompt || "";
  if (llmContextInput && document.activeElement !== llmContextInput) llmContextInput.value = payload.context || "";
}

async function loadModels() {
  const signature = currentModelSignature();
  setStatus("Loading models");
  setModuleStatus({ asr: "Loading", llm: "Loading", tts: "Loading" });
  loadModelsButton.disabled = true;
  const payload = await readModelLoadEvents(selectedRuntimeOptions());
  state.modelsLoaded = true;
  state.loadedModelSignature = signature;
  connectButton.disabled = false;
  loadModelsButton.disabled = false;
  setModuleStatus({ asr: "Idle", llm: "Idle", tts: "Idle" });
  setStatus("Models ready");
  logEvent(`Models loaded: ASR ${formatMs(payload.timingsMs?.asr)}, LLM ${formatMs(payload.timingsMs?.llm)}, TTS ${formatMs(payload.timingsMs?.tts)}.`);
}

async function readModelLoadEvents(requestPayload) {
  const response = await fetch("/api/models/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestPayload),
  });
  if (!response.body) {
    const payload = await response.json().catch(() => ({}));
    const detail = payload.detail?.message || payload.detail || "Model load failed";
    throw new Error(detail);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.event === "model_loaded") applyModelLoadEvent(event);
      if (event.event === "result") result = event;
      if (event.event === "error") throw new Error(event.message || "Model load failed");
    }
  }
  if (!response.ok && !result) throw new Error("Model load failed");
  if (!result) throw new Error("Model load did not return a result");
  return result;
}

function applyModelLoadEvent(event) {
  const stage = event.stage;
  const label = stage.toUpperCase();
  const memory = formatMb(event.memoryRssMb);
  const delta = formatSignedMb(event.memoryDeltaMb);
  const details = event.details || {};
  const model = details.model || details.voice || details.modelStatus?.id || details.provider || "model";
  const memorySource = event.memorySource === "sidecar" ? "sidecar RSS" : "server RSS";
  logEvent(`${label} ${model} loaded in ${formatMs(event.latencyMs)}; ${memorySource} ${memory}; delta ${delta}.`);
  if (stage === "asr") setModuleStatus({ asr: "Loaded" });
  if (stage === "llm") setModuleStatus({ llm: "Loaded" });
  if (stage === "tts") setModuleStatus({ tts: "Loaded" });
}

async function start() {
  if (!state.modelsLoaded) {
    throw new Error("Load models before starting the microphone.");
  }
  if (state.loadedModelSignature !== currentModelSignature()) {
    state.modelsLoaded = false;
    state.loadedModelSignature = null;
    connectButton.disabled = true;
    throw new Error("Configuration changed. Load models again before starting the microphone.");
  }
  setStatus("Requesting microphone");
  setModuleStatus({ mic: "Opening" });
  state.localStream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    video: false,
  });
  setupAnalyser();
  if (state.audioContext?.state === "suspended") {
    await state.audioContext.resume();
  }
  startVolumeLoop();
  state.connected = true;
  setConfigurationLocked(true);
  connectButton.disabled = true;
  muteButton.disabled = false;
  disconnectButton.disabled = false;
  setStatus("Listening");
  setModuleStatus({ mic: "Open", asr: "Idle", llm: "Idle", tts: "Idle" });
  setVad("Listening");
  logEvent("Microphone started. VAD is listening for speech.");
}

function setupAnalyser() {
  state.audioContext = new AudioContext();
  const source = state.audioContext.createMediaStreamSource(state.localStream);
  state.audioSource = source;
  state.analyser = state.audioContext.createAnalyser();
  state.analyser.fftSize = 1024;
  state.analyserData = new Uint8Array(state.analyser.fftSize);
  source.connect(state.analyser);
  setupPcmProcessor(source);
}

function setupPcmProcessor(source) {
  const processor = state.audioContext.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = (event) => {
    event.outputBuffer.getChannelData(0).fill(0);
    if (!state.azureAsr?.active || state.muted) {
      return;
    }
    const input = event.inputBuffer.getChannelData(0);
    sendAzureAsrPcm(encodePcm16(downsampleTo16k(input, state.audioContext.sampleRate)));
  };
  source.connect(processor);
  processor.connect(state.audioContext.destination);
  state.pcmProcessor = processor;
}

function startRecorder() {
  if (!state.localStream || state.localStream.getAudioTracks().length === 0) {
    throw new Error("No microphone audio track is available");
  }
  if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
    state.mediaRecorder.stop();
  }
  state.recordedChunks = [];
  const options = pickRecorderOptions();
  state.mediaRecorder = new MediaRecorder(state.localStream, options);
  state.mediaRecorder.addEventListener("dataavailable", (event) => {
    if (event.data.size > 0) {
      state.recordedChunks.push(event.data);
      sendTurnStreamChunk(event.data);
    }
  });
  state.mediaRecorder.start(VAD.recorderTimesliceMs);
}

function pickRecorderOptions() {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
  for (const mimeType of candidates) {
    if (MediaRecorder.isTypeSupported(mimeType)) {
      return { mimeType };
    }
  }
  return {};
}

function startVolumeLoop() {
  cancelAnimationFrame(state.volumeLoop);
  const tick = () => {
    if (!state.connected || !state.analyser) {
      return;
    }
    const volume = readVolume();
    renderVolume(volume);
    updateVad(volume, performance.now());
    state.volumeLoop = requestAnimationFrame(tick);
  };
  state.volumeLoop = requestAnimationFrame(tick);
}

function readVolume() {
  state.analyser.getByteTimeDomainData(state.analyserData);
  let sum = 0;
  for (const sample of state.analyserData) {
    const normalized = (sample - 128) / 128;
    sum += normalized * normalized;
  }
  return Math.sqrt(sum / state.analyserData.length);
}

function renderVolume(volume) {
  const percent = Math.min(100, Math.round(volume * 450));
  volumeFill.style.width = `${percent}%`;
  volumeValue.textContent = `${percent}%`;
}

function updateVad(volume, now) {
  if (state.sending || state.muted) {
    return;
  }
  if (volume >= VAD.speechThreshold) {
    if (!state.vad.active) {
      interruptAssistantResponse();
      state.vad.active = true;
      state.vad.speechStartedAt = now;
      state.currentTurn.streamingAsr = isStreamingAsrSelected();
      if (isAzureEmbeddedAsrSelected()) {
        startAzureAsrStream();
      } else {
        startRecorder();
        startTurnStream();
      }
      logEvent("VAD speech start.");
    }
    state.vad.lastSpeechAt = now;
    setVad("Speech");
    return;
  }

  if (!state.vad.active) {
    setVad("Listening");
    return;
  }

  const speechDuration = now - state.vad.speechStartedAt;
  const silenceDuration = now - state.vad.lastSpeechAt;
  setVad("Trailing silence");
  if (speechDuration >= VAD.minSpeechMs && speechDuration >= VAD.minTurnMs && silenceDuration >= VAD.silenceEndMs) {
    state.vad.speechEndedAt = now;
    state.currentTurn.vadE2E = silenceDuration;
    updateSingleMetric("vad", silenceDuration);
    sendDetectedTurn();
  }
}

async function sendDetectedTurn() {
  if (isAzureEmbeddedAsrSelected()) {
    await sendAzureEmbeddedTurn();
    return;
  }
  if (state.sending || !state.connected) {
    return;
  }
  state.sending = true;
  setStatus("Processing turn");
  setVad("Triggered");
  setModuleStatus({ asr: "Finalizing", llm: "Queued", tts: "Queued" });
  try {
    const blob = await stopRecorder();
    setVad("Listening");
    if (blob.size < VAD.minBlobBytes) {
      throw new Error("Captured speech was too short. Speak for at least one second.");
    }
    const responseStarted = performance.now();
    const result = state.turnStream ? await finishTurnStream() : await postTurn(blob);
    const speechToSpeechMs = performance.now() - state.vad.speechEndedAt;
    renderRealTurn(result);
    updateMetrics(result.timingsMs || {}, speechToSpeechMs);
    setModuleStatus({ asr: "Idle", llm: "Idle", tts: "Idle" });
    logEvent(`Turn complete. Upload+response ${formatMs(performance.now() - responseStarted)}, speech-to-speech ${formatMs(speechToSpeechMs)}.`);
  } catch (error) {
    logEvent(error.message);
    setModuleStatus({ asr: "Idle", llm: "Idle", tts: "Idle" });
  } finally {
    resetVad();
    if (state.connected) {
      setStatus("Listening");
      setVad("Listening");
      setModuleStatus({ mic: "Open" });
    }
    state.sending = false;
  }
}

async function sendAzureEmbeddedTurn() {
  if (state.sending || !state.connected) {
    return;
  }
  const turnId = state.responseTurnId + 1;
  state.responseTurnId = turnId;
  state.sending = true;
  setStatus("Processing turn");
  setModuleStatus({ asr: "Finalizing", llm: "Queued", tts: "Queued" });
  try {
    const responseStarted = performance.now();
    const finalAsr = await finishAzureAsrStream();
    const speechEndedAt = state.vad.speechEndedAt;
    const vadLatencyMs = state.currentTurn.vadE2E || 0;
    const asrLatencyMs = state.currentTurn.asrE2E || 0;
    state.vad.active = false;
    setVad("Listening");
    setModuleStatus({ asr: "Idle", llm: "Queued", tts: "Queued" });
    state.sending = false;
    const userText = finalAsr.text.trim();
    if (!userText) {
      throw new Error("Azure Embedded ASR returned empty text.");
    }
    userTranscript.textContent = userText;
    const result = await postTextTurnWebSocket(userText, turnId, { vadLatencyMs, asrLatencyMs });
    if (turnId <= state.interruptedTurnId || !result) {
      return;
    }
    const speechToSpeechMs = performance.now() - speechEndedAt;
    renderRealTurn(result);
    updateMetrics(result.timingsMs || {}, speechToSpeechMs);
    setModuleStatus({ asr: "Idle", llm: "Idle", tts: "Idle" });
    logEvent(`Turn complete. Text+response ${formatMs(performance.now() - responseStarted)}, speech-to-speech ${formatMs(speechToSpeechMs)}.`);
  } catch (error) {
    if (turnId > state.interruptedTurnId) logEvent(error.message);
    setModuleStatus({ asr: "Idle", llm: "Idle", tts: "Idle" });
  } finally {
    const wasInterrupted = turnId <= state.interruptedTurnId;
    if (!wasInterrupted) resetVad();
    if (!wasInterrupted && state.connected) {
      setStatus("Listening");
      setVad("Listening");
      setModuleStatus({ mic: "Open" });
    }
    state.sending = false;
  }
}

async function postTurn(blob) {
  try {
    return await postTurnWebSocket(blob);
  } catch (error) {
    logEvent(`WebSocket turn fallback: ${error.message}`);
  }

  const formData = new FormData();
  const extension = blob.type.includes("mp4") ? "m4a" : "webm";
  formData.append("audio", blob, `recording.${extension}`);
  const runtimeOptions = selectedRuntimeOptions();
  formData.append("tts_provider", runtimeOptions.ttsProvider);
  formData.append("llm_model", runtimeOptions.llmModel);
  formData.append("llm_prompt", runtimeOptions.llmPrompt);
  formData.append("llm_context", runtimeOptions.llmContext);
  const asrOptions = runtimeOptions;
  formData.append("asr_provider", asrOptions.asrProvider);
  formData.append("asr_locale", asrOptions.asrLocale);
  if (asrOptions.asrModel) formData.append("asr_model", asrOptions.asrModel);
  if (asrOptions.asrLanguage) formData.append("asr_language", asrOptions.asrLanguage);
  const response = await fetch("/api/session/turn-events", { method: "POST", body: formData });
  if (!response.ok || !response.body) {
    const payload = await response.json().catch(() => ({}));
    const detail = payload.detail?.message || payload.detail || "Real audio turn failed";
    throw new Error(detail);
  }
  return await readTurnEvents(response.body);
}

function startTurnStream() {
  if (state.turnStream?.active) {
    return state.turnStream;
  }
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${protocol}//${location.host}/api/session/turn-ws`);
  socket.binaryType = "arraybuffer";
  let resolveResult;
  let rejectResult;
  const stream = {
    active: true,
    ready: false,
    ending: false,
    socket,
    pendingChunks: [],
    pendingAudio: null,
    result: null,
    resultPromise: new Promise((resolve, reject) => {
      resolveResult = resolve;
      rejectResult = reject;
    }),
    resolveResult,
    rejectResult,
  };
  state.turnStream = stream;
  setModuleStatus({ asr: "Streaming", llm: "Waiting", tts: "Waiting" });

  socket.addEventListener("open", () => {
    const recorderType = state.mediaRecorder?.mimeType || "audio/webm";
    const extension = recorderType.includes("mp4") ? "m4a" : "webm";
    stream.ready = true;
    socket.send(JSON.stringify({
      type: "config",
      ...selectedRuntimeOptions(),
      filename: `recording.${extension}`,
      mediaType: recorderType,
    }));
    flushTurnStream(stream);
    if (stream.ending) {
      socket.send(JSON.stringify({ type: "end" }));
    }
  });

  socket.addEventListener("message", (message) => handleTurnStreamMessage(stream, message));
  socket.addEventListener("error", () => rejectTurnStream(stream, new Error("WebSocket turn stream failed")));
  socket.addEventListener("close", () => {
    if (stream.active) {
      rejectTurnStream(stream, new Error("WebSocket turn stream closed before completion"));
    }
  });
  return stream;
}

async function sendTurnStreamChunk(blob) {
  const stream = state.turnStream;
  if (!stream?.active || stream.ending) {
    return;
  }
  const chunk = await blob.arrayBuffer();
  if (stream.ready && stream.socket.readyState === WebSocket.OPEN) {
    stream.socket.send(chunk);
    return;
  }
  stream.pendingChunks.push(chunk);
}

function flushTurnStream(stream) {
  while (stream.pendingChunks.length > 0 && stream.socket.readyState === WebSocket.OPEN) {
    stream.socket.send(stream.pendingChunks.shift());
  }
}

async function finishTurnStream() {
  const stream = state.turnStream;
  if (!stream) {
    throw new Error("Turn stream was not started.");
  }
  stream.ending = true;
  if (stream.socket.readyState === WebSocket.OPEN) {
    flushTurnStream(stream);
    stream.socket.send(JSON.stringify({ type: "end" }));
  }
  return await stream.resultPromise.finally(() => {
    stream.active = false;
    state.turnStream = null;
  });
}

function handleTurnStreamMessage(stream, message) {
  if (typeof message.data === "string") {
    const event = JSON.parse(message.data);
    if (event.event === "progress") {
      if (event.stage === "tts" && event.status === "audio") {
        stream.pendingAudio = event;
      }
      applyProgressEvent(event);
    }
    if (event.event === "result") stream.result = event.result;
    if (event.event === "error") rejectTurnStream(stream, new Error(event.message || "Real audio turn failed"));
    if (event.event === "done") {
      stream.socket.close();
      stream.active = false;
      stream.result ? stream.resolveResult(stream.result) : stream.rejectResult(new Error("Real audio turn did not return a result"));
    }
    return;
  }

  if (message.data instanceof ArrayBuffer && stream.pendingAudio) {
    enqueueAudioBlob(stream.pendingAudio, message.data);
    stream.pendingAudio = null;
  }
}

function rejectTurnStream(stream, error) {
  stream.active = false;
  stream.rejectResult(error);
  if (state.turnStream === stream) {
    state.turnStream = null;
  }
}

function postTurnWebSocket(blob) {
  return new Promise((resolve, reject) => {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${protocol}//${location.host}/api/session/turn-ws`);
    socket.binaryType = "arraybuffer";
    let result = null;
    let pendingAudio = null;

    socket.addEventListener("open", async () => {
      const extension = blob.type.includes("mp4") ? "m4a" : "webm";
      socket.send(JSON.stringify({
        type: "config",
        ttsProvider: selectedTtsProvider(),
        ...selectedAsrOptions(),
        filename: `recording.${extension}`,
        mediaType: blob.type || "audio/webm",
      }));
      socket.send(await blob.arrayBuffer());
      socket.send(JSON.stringify({ type: "end" }));
    });

    socket.addEventListener("message", (message) => {
      if (typeof message.data === "string") {
        const event = JSON.parse(message.data);
        if (event.event === "progress") {
          if (event.stage === "tts" && event.status === "audio") {
            pendingAudio = event;
          }
          applyProgressEvent(event);
        }
        if (event.event === "result") result = event.result;
        if (event.event === "error") reject(new Error(event.message || "Real audio turn failed"));
        if (event.event === "done") {
          socket.close();
          result ? resolve(result) : reject(new Error("Real audio turn did not return a result"));
        }
        return;
      }

      if (message.data instanceof ArrayBuffer && pendingAudio) {
        enqueueAudioBlob(pendingAudio, message.data);
        pendingAudio = null;
      }
    });

    socket.addEventListener("error", () => reject(new Error("WebSocket turn failed")));
  });
}

function postTextTurnWebSocket(text, turnId = state.responseTurnId, timings = {}) {
  return new Promise((resolve, reject) => {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${protocol}//${location.host}/api/session/text-turn-ws`);
    state.responseSocket = socket;
    socket.binaryType = "arraybuffer";
    let result = null;
    let pendingAudio = null;

    socket.addEventListener("open", () => {
      socket.send(JSON.stringify({
        type: "text_turn",
        text,
        ...selectedRuntimeOptions(),
        vadProvider: "browser-vad",
        vadLatencyMs: timings.vadLatencyMs || 0,
        asrLatencyMs: timings.asrLatencyMs || 0,
      }));
    });

    socket.addEventListener("message", (message) => {
      if (typeof message.data === "string") {
        const event = JSON.parse(message.data);
        if (event.event === "progress") {
          if (turnId <= state.interruptedTurnId) return;
          if (event.stage === "tts" && event.status === "audio") {
            pendingAudio = event;
          }
          applyProgressEvent(event);
        }
        if (event.event === "result" && turnId > state.interruptedTurnId) result = event.result;
        if (event.event === "error" && turnId > state.interruptedTurnId) reject(new Error(event.message || "Text turn failed"));
        if (event.event === "done") {
          socket.close();
          if (state.responseSocket === socket) state.responseSocket = null;
          if (turnId <= state.interruptedTurnId) {
            resolve(null);
            return;
          }
          result ? resolve(result) : reject(new Error("Text turn did not return a result"));
        }
        return;
      }

      if (message.data instanceof ArrayBuffer && pendingAudio && turnId > state.interruptedTurnId) {
        enqueueAudioBlob(pendingAudio, message.data);
        pendingAudio = null;
      }
    });

    socket.addEventListener("error", () => {
      if (state.responseSocket === socket) state.responseSocket = null;
      if (turnId > state.interruptedTurnId) reject(new Error("Text turn WebSocket failed"));
    });
    socket.addEventListener("close", () => {
      if (state.responseSocket === socket) state.responseSocket = null;
      if (turnId <= state.interruptedTurnId) {
        resolve(null);
      }
    });
  });
}

function interruptAssistantResponse() {
  const hasResponseSocket = state.responseSocket && state.responseSocket.readyState !== WebSocket.CLOSED;
  const activeTurnStream = state.turnStream?.active ? state.turnStream : null;
  if (!hasResponseSocket && !activeTurnStream && !state.playingAudio && state.audioQueue.length === 0) {
    return;
  }
  state.interruptedTurnId = state.responseTurnId;
  if (activeTurnStream) {
    try {
      activeTurnStream.socket.close();
    } catch {
      // Ignore close races; the next turn owns UI state from here.
    }
    rejectTurnStream(activeTurnStream, new Error("Assistant response interrupted by speech."));
  }
  if (hasResponseSocket) {
    try {
      state.responseSocket.close();
    } catch {
      // Ignore close races; the next turn owns UI state from here.
    }
  }
  state.responseSocket = null;
  state.audioQueue = [];
  state.playingAudio = false;
  const activeAudioUrl = remoteAudio.currentSrc || remoteAudio.src;
  remoteAudio.pause();
  remoteAudio.removeAttribute("src");
  remoteAudio.load();
  if (activeAudioUrl.startsWith("blob:")) URL.revokeObjectURL(activeAudioUrl);
  setModuleStatus({ llm: "Interrupted", tts: "Interrupted" });
  logEvent("Assistant response interrupted by speech.");
}

function startAzureAsrStream() {
  if (!isAzureEmbeddedAsrSelected() || state.azureAsr?.active) {
    return state.azureAsr;
  }
  const sidecarUrl = state.config?.audio?.azureEmbeddedAsr?.sidecarUrl || "/api/azure-embedded/asr-ws";
  const socket = new WebSocket(resolveLocalWebSocketUrl(sidecarUrl));
  socket.binaryType = "arraybuffer";
  const asrOptions = selectedAsrOptions();
  let resolveFinal;
  let rejectFinal;
  const stream = {
    active: true,
    ready: false,
    ending: false,
    finalReceived: false,
    socket,
    pendingChunks: [],
    lastPartial: "",
    finalPromise: new Promise((resolve, reject) => {
      resolveFinal = resolve;
      rejectFinal = reject;
    }),
    resolveFinal,
    rejectFinal,
  };
  state.azureAsr = stream;
  setModuleStatus({ asr: "Streaming", llm: "Waiting", tts: "Waiting" });
  userTranscript.textContent = "";

  socket.addEventListener("open", () => {
    stream.ready = true;
    socket.send(JSON.stringify({ type: "start", locale: asrOptions.asrLocale, model: `azure-embedded-${asrOptions.asrLocale}-35M` }));
    flushAzureAsrStream(stream);
    if (stream.ending) {
      socket.send(JSON.stringify({ type: "end" }));
    }
  });

  socket.addEventListener("message", (message) => {
    if (typeof message.data !== "string") {
      return;
    }
    handleAzureAsrEvent(stream, JSON.parse(message.data));
  });
  socket.addEventListener("error", () => rejectAzureAsrStream(stream, new Error("Azure Embedded ASR WebSocket failed")));
  socket.addEventListener("close", () => {
    if (!stream.finalReceived && stream.active) {
      rejectAzureAsrStream(stream, new Error("Azure Embedded ASR closed before final text."));
    }
  });
  return stream;
}

function sendAzureAsrPcm(pcm16) {
  const stream = state.azureAsr;
  if (!stream?.active || stream.ending || stream.finalReceived) {
    return;
  }
  if (stream.ready && stream.socket.readyState === WebSocket.OPEN) {
    stream.socket.send(pcm16);
    return;
  }
  stream.pendingChunks.push(pcm16.slice(0));
}

function flushAzureAsrStream(stream) {
  while (stream.pendingChunks.length > 0 && stream.socket.readyState === WebSocket.OPEN) {
    stream.socket.send(stream.pendingChunks.shift());
  }
}

async function finishAzureAsrStream() {
  const stream = state.azureAsr || startAzureAsrStream();
  if (!stream) {
    throw new Error("Azure Embedded ASR stream was not started.");
  }
  stream.ending = true;
  if (stream.socket.readyState === WebSocket.OPEN) {
    flushAzureAsrStream(stream);
    stream.socket.send(JSON.stringify({ type: "end" }));
  }
  return await stream.finalPromise.finally(() => {
    stream.active = false;
    state.azureAsr = null;
  });
}

function handleAzureAsrEvent(stream, event) {
  if (event.type === "started") {
    setModuleStatus({ asr: "Streaming" });
    return;
  }
  if (event.type === "partial" && event.text) {
    stream.lastPartial = event.text;
    userTranscript.textContent = event.text;
    return;
  }
  if (event.type === "final") {
    const text = String(event.text || stream.lastPartial || "").trim();
    stream.finalReceived = true;
    stream.active = false;
    if (state.vad.speechEndedAt) {
      state.currentTurn.asrE2E = performance.now() - state.vad.speechEndedAt;
      updateSingleMetric("asr", state.currentTurn.asrE2E, false);
    }
    stream.resolveFinal({ text, event });
    if (stream.socket.readyState === WebSocket.OPEN) {
      stream.socket.close();
    }
    return;
  }
  if (event.type === "error" || event.type === "canceled") {
    rejectAzureAsrStream(stream, new Error(event.message || event.details || "Azure Embedded ASR failed"));
  }
}

function rejectAzureAsrStream(stream, error) {
  stream.active = false;
  stream.rejectFinal(error);
  if (state.azureAsr === stream) {
    state.azureAsr = null;
  }
}

function closeAzureAsrStream() {
  const stream = state.azureAsr;
  if (!stream) return;
  stream.active = false;
  if (stream.socket.readyState === WebSocket.OPEN || stream.socket.readyState === WebSocket.CONNECTING) {
    stream.socket.close();
  }
  state.azureAsr = null;
}

function resolveLocalWebSocketUrl(url) {
  if (location.protocol !== "https:" || !url.startsWith("ws://")) {
    return url;
  }
  return `wss://${url.slice("ws://".length)}`;
}

function downsampleTo16k(input, inputSampleRate) {
  const outputSampleRate = 16000;
  if (inputSampleRate === outputSampleRate) {
    return input;
  }
  const ratio = inputSampleRate / outputSampleRate;
  const outputLength = Math.max(1, Math.round(input.length / ratio));
  const output = new Float32Array(outputLength);
  for (let index = 0; index < outputLength; index += 1) {
    const start = Math.floor(index * ratio);
    const end = Math.min(input.length, Math.floor((index + 1) * ratio));
    let sum = 0;
    let count = 0;
    for (let inputIndex = start; inputIndex < end; inputIndex += 1) {
      sum += input[inputIndex];
      count += 1;
    }
    output[index] = count > 0 ? sum / count : input[Math.min(start, input.length - 1)];
  }
  return output;
}

function encodePcm16(samples) {
  const buffer = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buffer);
  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(index * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return buffer;
}

async function readTurnEvents(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.event === "progress") applyProgressEvent(event);
      if (event.event === "result") result = event.result;
      if (event.event === "error") throw new Error(event.message || "Real audio turn failed");
    }
  }
  if (!result) throw new Error("Real audio turn did not return a result");
  return result;
}

function applyProgressEvent(event) {
  if (event.stage === "vad") {
    if (event.status === "running") setVad("Running");
    if (event.status === "idle") setVad("Listening");
    if (Number.isFinite(event.latencyMs)) updateSingleMetric("vad", event.latencyMs, false);
  }
  if (event.stage === "asr") {
    if (event.status === "running") setModuleStatus({ asr: "Running", llm: "Waiting", tts: "Waiting" });
    if (event.status === "idle") {
      setModuleStatus({ asr: "Idle", llm: "Running", tts: "Waiting" });
      if (state.vad.speechEndedAt) {
        state.currentTurn.asrE2E = performance.now() - state.vad.speechEndedAt;
        updateSingleMetric("asr", state.currentTurn.asrE2E, false);
      } else if (Number.isFinite(event.latencyMs)) {
        updateSingleMetric("asr", event.latencyMs, false);
      }
      if (event.text) userTranscript.textContent = event.text;
    }
  }
  if (event.stage === "llm") {
    if (event.status === "running") setModuleStatus({ asr: "Idle", llm: "Running", tts: "Waiting" });
    if (event.status === "ttft" && Number.isFinite(event.latencyMs)) {
      state.currentTurn.llmTtft = event.latencyMs;
      updateSingleMetric("llm", event.latencyMs, false);
    }
    if (event.status === "first_sentence" && Number.isFinite(event.latencyMs)) {
      state.currentTurn.llmFirstPunctuation = Math.max(0, event.latencyMs - (state.currentTurn.llmTtft || 0));
    }
    if (event.status === "token" && event.text) {
      appendAssistantToken(event.text);
    }
    if (event.status === "idle") setModuleStatus({ asr: "Idle", llm: "Idle", tts: "Running" });
    if (event.status !== "token" && event.text && !state.currentTurn.streamedAssistantText) assistantTranscript.textContent = event.text;
  }
  if (event.stage === "tts") {
    if (event.status === "running") {
      if (state.currentTurn.ttsStartedAt === null) state.currentTurn.ttsStartedAt = performance.now();
      setModuleStatus({ tts: "Running" });
    }
    if (event.status === "audio") {
      enqueueAudioChunk(event);
    }
    if (event.status === "idle") setModuleStatus({ tts: "Idle" });
  }
}

function enqueueAudioChunk(event) {
  if (!event.audioBase64 || !event.audioMediaType) {
    return;
  }
  recordFirstPlayableAudio(event);
  state.currentTurn.streamedAudioChunks += 1;
  state.audioQueue.push(`data:${event.audioMediaType};base64,${event.audioBase64}`);
  logEvent(`TTS audio chunk ${state.currentTurn.streamedAudioChunks} queued.`);
  playNextAudioChunk();
}

function enqueueAudioBlob(event, arrayBuffer) {
  recordFirstPlayableAudio(event);
  state.currentTurn.streamedAudioChunks += 1;
  const blob = new Blob([arrayBuffer], { type: event.audioMediaType || "audio/wav" });
  state.audioQueue.push(URL.createObjectURL(blob));
  logEvent(`TTS binary audio chunk ${state.currentTurn.streamedAudioChunks} queued.`);
  playNextAudioChunk();
}

function recordFirstPlayableAudio(event) {
  if (state.currentTurn.ttsFirstByteMs !== null) {
    return;
  }
  const now = performance.now();
  const ttsFirstByteMs = state.currentTurn.ttsStartedAt
    ? now - state.currentTurn.ttsStartedAt
    : event.latencyMs;
  state.currentTurn.ttsFirstByteMs = ttsFirstByteMs;
  if (Number.isFinite(ttsFirstByteMs)) {
    updateSingleMetric("tts", ttsFirstByteMs, false);
  }
  if (state.vad.speechEndedAt) {
    state.currentTurn.voiceToVoiceFirstByteMs = now - state.vad.speechEndedAt;
    updateSingleMetric("voiceToVoice", state.currentTurn.voiceToVoiceFirstByteMs, false);
  }
}

function appendAssistantToken(text) {
  if (!state.currentTurn.streamedAssistantText) {
    assistantTranscript.textContent = "";
    state.currentTurn.streamedAssistantText = true;
  }
  state.typewriterQueue.push(...Array.from(text));
  runTypewriter();
}

function runTypewriter() {
  if (state.typewriterActive) {
    return;
  }
  state.typewriterActive = true;
  const tick = () => {
    const batch = state.typewriterQueue.splice(0, 3).join("");
    if (batch) assistantTranscript.textContent += batch;
    if (state.typewriterQueue.length > 0) {
      setTimeout(tick, 20);
    } else {
      state.typewriterActive = false;
    }
  };
  tick();
}

function playNextAudioChunk() {
  if (state.playingAudio || state.audioQueue.length === 0) {
    return;
  }
  state.playingAudio = true;
  remoteAudio.src = state.audioQueue.shift();
  remoteAudio.onended = () => {
    if (remoteAudio.src.startsWith("blob:")) URL.revokeObjectURL(remoteAudio.src);
    state.playingAudio = false;
    playNextAudioChunk();
  };
  remoteAudio.onerror = () => {
    if (remoteAudio.src.startsWith("blob:")) URL.revokeObjectURL(remoteAudio.src);
    state.playingAudio = false;
    logEvent("Audio chunk playback failed.");
    playNextAudioChunk();
  };
  remoteAudio.play().catch((error) => {
    if (remoteAudio.src.startsWith("blob:")) URL.revokeObjectURL(remoteAudio.src);
    state.playingAudio = false;
    logEvent(`Audio playback blocked: ${error.message}`);
  });
}

async function runBackendCheck() {
  setStatus("Running backend check");
  setModuleStatus({ asr: "Running", llm: "Running", tts: "Running" });
  backendCheckButton.disabled = true;
  const responseStarted = performance.now();
  const response = await fetch("/api/session/backend-check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(selectedRuntimeOptions()),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail?.message || payload.detail || "Backend real chain check failed";
    throw new Error(detail);
  }
  const fallbackTiming = performance.now() - responseStarted;
  renderRealTurn(payload.result);
  updateMetrics(payload.result.timingsMs || {}, payload.result.timingsMs?.backendTotal || fallbackTiming);
  setStatus("Backend check passed");
  setModuleStatus({ asr: "Idle", llm: "Idle", tts: "Idle" });
  logEvent(`${payload.mode} passed with ${payload.result.asr.provider} ASR.`);
}

function stopRecorder() {
  return new Promise((resolve, reject) => {
    const recorder = state.mediaRecorder;
    if (!recorder || recorder.state === "inactive") {
      reject(new Error("Recorder is not active"));
      return;
    }
    recorder.addEventListener(
      "stop",
      () => {
        const type = recorder.mimeType || "audio/webm";
        resolve(new Blob(state.recordedChunks, { type }));
      },
      { once: true },
    );
    if (recorder.state === "recording") {
      recorder.requestData();
    }
    recorder.stop();
    state.mediaRecorder = null;
  });
}

function resetVad() {
  closeAzureAsrStream();
  state.vad.active = false;
  state.vad.speechStartedAt = 0;
  state.vad.lastSpeechAt = 0;
  state.vad.speechEndedAt = 0;
  state.currentTurn.vadE2E = null;
  state.currentTurn.asrE2E = null;
  state.currentTurn.llmTtft = null;
  state.currentTurn.llmFirstPunctuation = null;
  state.currentTurn.ttsStartedAt = null;
  state.currentTurn.ttsFirstByteMs = null;
  state.currentTurn.voiceToVoiceFirstByteMs = null;
  state.currentTurn.streamedAudioChunks = 0;
  state.currentTurn.streamedAssistantText = false;
  state.currentTurn.streamingAsr = false;
  state.typewriterQueue = [];
  state.typewriterActive = false;
  state.audioQueue = [];
}

function renderRealTurn(result) {
  userTranscript.textContent = result.userText;
  if (!state.currentTurn.streamedAssistantText) {
    assistantTranscript.textContent = result.assistantText;
  }
  logEvent(`real-audio-turn passed with ${result.vad?.provider || "unknown"} VAD, ${result.asr.provider} ASR and ${result.tts.provider} TTS.`);
  if (state.currentTurn.streamedAudioChunks > 0) {
    return;
  }
  if (result.tts.audioBase64) {
    recordFirstPlayableAudio({ latencyMs: result.timingsMs?.tts });
    const audioUrl = `data:${result.tts.audioMediaType};base64,${result.tts.audioBase64}`;
    remoteAudio.src = audioUrl;
    remoteAudio.play().catch((error) => logEvent(`Audio playback blocked: ${error.message}`));
  } else if (result.tts.browserFallback && "speechSynthesis" in window) {
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance(result.assistantText));
    logEvent("Using browser speechSynthesis fallback for TTS playback.");
  }
}

function updateMetrics(timings, speechToSpeechMs, includeAverage = true) {
  const llmTtft = timings.llm;
  const llmFirstPunctuation = Number.isFinite(state.currentTurn.llmFirstPunctuation)
    ? state.currentTurn.llmFirstPunctuation
    : Math.max(0, (timings.llmFirstSentence ?? timings.llm ?? 0) - (timings.llm ?? 0));
  const ttsFirstByte = state.currentTurn.ttsFirstByteMs ?? timings.tts;
  const voiceToVoice = state.currentTurn.voiceToVoiceFirstByteMs ?? (state.currentTurn.streamingAsr
    ? sumFinite(state.currentTurn.asrE2E ?? timings.asr, llmTtft, llmFirstPunctuation, ttsFirstByte)
    : sumFinite(
        state.currentTurn.vadE2E ?? timings.vad,
        state.currentTurn.asrE2E ?? timings.asr,
        llmTtft,
        llmFirstPunctuation,
        ttsFirstByte,
      ));
  const normalized = {
    vad: state.currentTurn.vadE2E ?? timings.vad,
    asr: state.currentTurn.asrE2E ?? timings.asr,
    llm: llmTtft,
    tts: ttsFirstByte,
    voiceToVoice: Number.isFinite(voiceToVoice) ? voiceToVoice : undefined,
  };
  for (const [key, value] of Object.entries(normalized)) {
    const [currentElement, averageElement] = metricElements[key];
    if (!Number.isFinite(value)) {
      continue;
    }
    currentElement.textContent = formatMs(value);
    if (includeAverage) {
      recordMetric(key, value);
    }
  }
}

function sumFinite(...values) {
  if (values.some((value) => !Number.isFinite(value))) {
    return NaN;
  }
  return values.reduce((sum, value) => sum + value, 0);
}

function updateSingleMetric(key, value, includeAverage = true) {
  const [currentElement, averageElement] = metricElements[key];
  if (!Number.isFinite(value)) {
    return;
  }
  currentElement.textContent = formatMs(value);
  if (includeAverage) {
    recordMetric(key, value);
  }
}

function recordMetric(key, value) {
  const [, averageElement, p90Element] = metricElements[key];
  state.metrics.values[key].push(value);
  averageElement.textContent = formatMs(average(state.metrics.values[key]));
  p90Element.textContent = formatMs(percentile(state.metrics.values[key], 0.9));
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function percentile(values, quantile) {
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(sorted.length - 1, Math.ceil(sorted.length * quantile) - 1);
  return sorted[index];
}

function formatMs(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return `${Math.round(value)} ms`;
}

function formatMb(value) {
  if (!Number.isFinite(value)) {
    return "n/a";
  }
  return `${value.toFixed(1)} MB`;
}

function formatSignedMb(value) {
  if (!Number.isFinite(value)) {
    return "n/a";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(1)} MB`;
}

function disconnect() {
  cancelAnimationFrame(state.volumeLoop);
  closeAzureAsrStream();
  if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
    state.mediaRecorder.stop();
  }
  state.mediaRecorder = null;
  state.recordedChunks = [];
  if (state.localStream) {
    for (const track of state.localStream.getTracks()) {
      track.stop();
    }
    state.localStream = null;
  }
  if (state.audioContext) {
    if (state.pcmProcessor) {
      state.pcmProcessor.onaudioprocess = null;
      state.pcmProcessor.disconnect();
      state.pcmProcessor = null;
    }
    state.audioContext.close();
    state.audioContext = null;
  }
  state.audioSource = null;
  state.analyser = null;
  state.connected = false;
  state.sending = false;
  resetVad();
  setConfigurationLocked(false);
  connectButton.disabled = !state.modelsLoaded;
  muteButton.disabled = true;
  disconnectButton.disabled = true;
  setStatus("Idle", true);
  setVad("Idle");
  setModuleStatus({ mic: "Idle", asr: "Idle", llm: "Idle", tts: "Idle" });
  renderVolume(0);
  logEvent("Session stopped.");
}

loadModelsButton.addEventListener("click", async () => {
  try {
    await loadModels();
  } catch (error) {
    state.modelsLoaded = false;
    state.loadedModelSignature = null;
    connectButton.disabled = true;
    setStatus("Model load failed", true);
    setModuleStatus({ asr: "Idle", llm: "Idle", tts: "Idle" });
    loadModelsButton.disabled = false;
    logEvent(error.message);
  }
});

connectButton.addEventListener("click", async () => {
  try {
    await start();
  } catch (error) {
    setStatus("Start failed", true);
    logEvent(`${error.message}. Allow microphone permission for live turns, or use Backend Check.`);
    disconnect();
  }
});

backendCheckButton.addEventListener("click", async () => {
  try {
    await runBackendCheck();
  } catch (error) {
    setStatus("Backend check failed", true);
    logEvent(error.message);
  } finally {
    backendCheckButton.disabled = false;
  }
});

muteButton.addEventListener("click", () => {
  state.muted = !state.muted;
  if (state.localStream) {
    for (const track of state.localStream.getAudioTracks()) {
      track.enabled = !state.muted;
    }
  }
  muteButton.textContent = state.muted ? "Unmute" : "Mute";
  setVad(state.muted ? "Muted" : "Listening");
  logEvent(state.muted ? "Microphone muted." : "Microphone unmuted.");
});

disconnectButton.addEventListener("click", disconnect);
vadSelector.addEventListener("change", () => {
  updateVadSelection();
  markModelsDirty("VAD selection changed. Load Models again before Start.");
});
asrSelector.addEventListener("change", () => {
  updateAsrMetricLabel();
  markModelsDirty("ASR selection changed. Load Models again before Start.");
});
llmSelector.addEventListener("change", () => markModelsDirty("LLM model changed. Load Models again before Start."));
ttsSelector.addEventListener("change", () => markModelsDirty("TTS model changed. Load Models again before Start."));
llmPromptInput?.addEventListener("input", scheduleLlmConfigSync);
llmContextInput?.addEventListener("input", scheduleLlmConfigSync);

loadConfig();