import React from "react";

export function VoiceAgentApp() {
  return (
    <main className="app-shell">
      <section className="control-panel" aria-label="Voice session controls">
        <div>
          <p className="eyebrow">Local-first voice agent</p>
          <h1>Hybrid Voice Agent</h1>
        </div>
        <div className="status-row">
          <span id="connectionStatus" className="status-pill">Idle</span>
          <span id="providerStatus" className="status-pill muted">Loading providers</span>
        </div>
        <div className="button-row">
          <button id="loadModelsButton" type="button">Load Models</button>
          <button id="connectButton" type="button" disabled>Start</button>
          <button id="backendCheckButton" type="button">Backend Check</button>
          <button id="muteButton" type="button" disabled>Mute</button>
          <button id="disconnectButton" type="button" disabled>Stop</button>
        </div>
        <details className="llm-config" id="llmConfigPanel">
          <summary><span>LLM Prompt &amp; Context</span><button id="resetContextButton" type="button">Reset Context</button></summary>
          <div className="llm-config-grid">
            <label>Prompt<textarea id="llmPromptInput" rows="4" /></label>
            <label>Context<textarea id="llmContextInput" rows="4" readOnly placeholder="Recent conversation is assembled automatically" /></label>
          </div>
        </details>
        <div className="module-strip" aria-label="Speech pipeline modules">
          <section className="module-card" aria-label="Microphone module">
            <div className="module-heading"><span>MIC</span><strong id="micStatus">Idle</strong></div>
            <label>Input/API<select id="micSelector"><option>Browser getUserMedia</option></select></label>
            <div className="volume-meter" aria-hidden="true"><span id="volumeFill" /></div>
            <div className="module-metric"><span id="volumeValue">0%</span><small>volume</small></div>
          </section>
          <section className="module-card" aria-label="VAD module">
            <div className="module-heading"><span>VAD</span><strong id="vadStatus">Idle</strong></div>
            <label>Model/API<select id="vadSelector" defaultValue="silero:500"><option value="silero:300">Silero VAD · 300 ms</option><option value="silero:500">Silero VAD · 500 ms</option><option value="silero:800">Silero VAD · 800 ms</option></select></label>
            <dl className="latency-list">
              <div><dt>End</dt><dd id="vadTurn">-</dd></div>
              <div><dt>AVG</dt><dd id="vadAvg">-</dd></div>
              <div><dt>P90</dt><dd id="vadP90">-</dd></div>
            </dl>
          </section>
          <section className="module-card" aria-label="ASR module">
            <div className="module-heading"><span>ASR</span><strong id="asrStatus">Idle</strong></div>
            <label>Model/API<select id="asrSelector" defaultValue="azure-embedded:zh-CN"><option value="azure-embedded:zh-CN">Azure Embedded ASR zh-CN 35M</option><option value="azure-embedded:en-GB">Azure Embedded ASR en-GB 35M</option><option value="lfm2-audio:LiquidAI/LFM2.5-Audio-1.5B:en">LiquidAI LFM2.5 Audio 1.5B</option><option value="foundry-local:nemotron-3.5-asr-streaming-0.6b:auto">Foundry Nemotron ASR multilingual 0.6B</option><option value="foundry-local:nemotron-speech-streaming-en-0.6b:en">Foundry Nemotron Speech English 0.6B</option><option value="faster-whisper:tiny:auto">faster-whisper tiny CPU fallback</option><option disabled>whisper.cpp</option><option disabled>Vosk</option></select></label>
            <dl className="latency-list">
              <div><dt id="asrMetricLabel">Final</dt><dd id="asrTurn">-</dd></div>
              <div><dt>AVG</dt><dd id="asrAvg">-</dd></div>
              <div><dt>P90</dt><dd id="asrP90">-</dd></div>
            </dl>
          </section>
          <section className="module-card" aria-label="LLM module">
            <div className="module-heading"><span>LLM</span><strong id="llmStatus">Idle</strong></div>
            <label>Model/API<select id="llmSelector" defaultValue="foundry-local:qwen2.5-0.5b-instruct-cuda-gpu:4"><option value="foundry-local:qwen2.5-0.5b-instruct-cuda-gpu:4">Foundry qwen2.5 0.5B</option><option value="lfm2-audio:LiquidAI/LFM2.5-Audio-1.5B">LiquidAI LFM2.5 Audio 1.5B</option><option value="foundry-local:gemma-4-e2b">Foundry Gemma 4 E2B (requires catalog model)</option><option value="llama-cpp:gemma-3n-e2b-it">llama.cpp Gemma 3n E2B IT Q4_K_M</option><option disabled>Cloud fallback</option></select></label>
            <dl className="latency-list">
              <div><dt>TTFT</dt><dd id="llmTurn">-</dd></div>
              <div><dt>AVG</dt><dd id="llmAvg">-</dd></div>
              <div><dt>P90</dt><dd id="llmP90">-</dd></div>
            </dl>
          </section>
          <section className="module-card" aria-label="TTS module">
            <div className="module-heading"><span>TTS</span><strong id="ttsStatus">Idle</strong></div>
            <label>Model/API<select id="ttsSelector" defaultValue="azure-embedded:azure-embedded-zh-CN-XiaoxiaoNeuralV6"><option value="azure-embedded:azure-embedded-zh-CN-XiaoxiaoNeuralV6">Azure Embedded Xiaoxiao V6</option><option value="lfm2-audio:LiquidAI/LFM2.5-Audio-1.5B">LiquidAI LFM2.5 Audio 1.5B</option><option value="azure-embedded:azure-embedded-en-US-AvaNeuralHD">Azure Embedded Ava HD en-US</option><option value="edge-tts">Edge Xiaoxiao Neural</option><option value="windows-sapi">Windows SAPI</option><option value="browser-speech" disabled>Browser speechSynthesis</option><option value="azure-speech" disabled>Azure Speech</option></select></label>
            <dl className="latency-list">
              <div><dt>TTFB</dt><dd id="ttsTurn">-</dd></div>
              <div><dt>AVG</dt><dd id="ttsAvg">-</dd></div>
              <div><dt>P90</dt><dd id="ttsP90">-</dd></div>
            </dl>
          </section>
        </div>
        <section className="voice-latency" aria-label="Voice to voice total latency">
          <strong>Voice2Voice</strong>
          <span>Turn <b id="v2vTurn">-</b></span>
          <span>AVG <b id="v2vAvg">-</b></span>
          <span>P90 <b id="v2vP90">-</b></span>
        </section>
        <audio id="remoteAudio" autoPlay playsInline />
      </section>

      <section className="transcript-grid" aria-label="Conversation transcript">
        <article>
          <h2>You</h2>
          <div id="userTranscript" className="transcript-box">Waiting for speech...</div>
        </article>
        <article>
          <h2>Assistant</h2>
          <div id="assistantTranscript" className="transcript-box">No response yet.</div>
        </article>
      </section>

      <section className="event-log" aria-label="Session events">
        <h2>Events</h2>
        <ol id="eventLog" />
      </section>
    </main>
  );
}