import React from "react";
import { flushSync } from "react-dom";
import { createRoot } from "react-dom/client";
import { VoiceAgentApp } from "./VoiceAgentApp.jsx";

const rootElement = document.querySelector("#root");

if (!rootElement) {
  throw new Error("React root element #root was not found.");
}

flushSync(() => {
  createRoot(rootElement).render(<VoiceAgentApp />);
});