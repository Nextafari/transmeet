const micStatusEl = document.getElementById("micStatus");
const driveStatusEl = document.getElementById("driveStatus");
const backendStatusEl = document.getElementById("backendStatus");
const liveOverlayStatusEl = document.getElementById("liveOverlayStatus");
const liveOverlayGroupEl = document.getElementById("liveCaptionOverlayGroup");
const systemAudioStatusEl = document.getElementById("systemAudioStatus");
const systemAudioGroupEl = document.getElementById("systemAudioGroup");
const systemAudioLabelEl = document.getElementById("systemAudioLabel");
const microphoneGroupEl = document.getElementById("microphoneGroup");
const microphoneStatusEl = document.getElementById("microphoneStatus");
const recoverySectionEl = document.getElementById("recoverySection");
const recoveryStatusEl = document.getElementById("recoveryStatus");
const recoverRecordingEl = document.getElementById("recoverRecording");
const discardRecoveryEl = document.getElementById("discardRecovery");

const SETTINGS_KEY = "recordingSettings";
const DEFAULT_SETTINGS = {
  captureType: "tab",
  includeSystemAudio: true,
  includeMicrophone: false,
  enableSubtitles: false,
  enableTranslation: false,
  sourceLanguage: "en",
  targetLanguage: "en",
  subtitleFormat: "srt",
  burnSubtitles: true,
  liveCaptionOverlay: false,
  compressionPreset: "balanced",
  enableDriveUpload: false
};

function sendRuntimeMessage(message) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(message, (response) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
        return;
      }
      resolve(response);
    });
  });
}

async function getSettings() {
  const result = await chrome.storage.local.get(SETTINGS_KEY);
  const stored = result[SETTINGS_KEY] || {};
  const merged = normalizeRecordingSettings({ ...DEFAULT_SETTINGS, ...stored });
  if (stored.enableSubtitles === undefined && stored.enableTranslation) {
    merged.enableSubtitles = true;
  }
  return merged;
}

async function setSettings(nextSettings) {
  await chrome.storage.local.set({ [SETTINGS_KEY]: nextSettings });
}

function isLiveOverlaySupported(settings) {
  return settings.captureType === "tab" && settings.includeSystemAudio;
}

function hasCapturableTabAudio(settings) {
  return settings.captureType === "tab" && settings.includeSystemAudio;
}

function getLiveOverlayUnavailableReason(settings) {
  if (settings.captureType !== "tab") {
    return "Live overlay beta is available for Current Tab recordings only.";
  }
  if (!settings.includeSystemAudio) {
    return "Turn on Tab/System Audio to use live overlay beta.";
  }
  return "";
}

function normalizeRecordingSettings(settings) {
  const normalized = { ...DEFAULT_SETTINGS, ...settings };
  normalized.sourceLanguage = normalized.sourceLanguage || DEFAULT_SETTINGS.sourceLanguage;
  if (normalized.enableTranslation) {
    normalized.enableSubtitles = true;
  }
  if (!isLiveOverlaySupported(normalized)) {
    normalized.liveCaptionOverlay = false;
  }
  return normalized;
}

function settingsDiffer(left, right) {
  return JSON.stringify(left) !== JSON.stringify(right);
}

async function updateSettings(patch) {
  const current = await getSettings();
  const next = normalizeRecordingSettings({ ...current, ...patch });
  await setSettings(next);
  applySettingsToUI(next);
}

function setDriveStatus(text, state) {
  if (!driveStatusEl) return;
  driveStatusEl.textContent = text;
  driveStatusEl.classList.remove("drive-status-connected", "drive-status-unavailable", "drive-status-neutral");
  if (state === "connected") {
    driveStatusEl.classList.add("drive-status-connected");
    return;
  }
  if (state === "unavailable") {
    driveStatusEl.classList.add("drive-status-unavailable");
    return;
  }
  driveStatusEl.classList.add("drive-status-neutral");
}

function formatBytes(bytes) {
  const size = Number(bytes) || 0;
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  if (size < 1024 * 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  return `${(size / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

function setRecoveryStatus(text, state) {
  if (!recoveryStatusEl) return;
  recoveryStatusEl.textContent = text;
  recoveryStatusEl.classList.remove(
    "recovery-status-available",
    "recovery-status-unavailable",
    "recovery-status-neutral"
  );
  if (state === "available") {
    recoveryStatusEl.classList.add("recovery-status-available");
    return;
  }
  if (state === "unavailable") {
    recoveryStatusEl.classList.add("recovery-status-unavailable");
    return;
  }
  recoveryStatusEl.classList.add("recovery-status-neutral");
}

async function refreshRecoveryStatus() {
  if (!recoverySectionEl) return;

  try {
    const response = await sendRuntimeMessage({ type: "AUTOSAVE_STATUS_REQUEST" });
    if (!response?.success || !response.hasRecoverable || !response.recording) {
      recoverySectionEl.hidden = true;
      if (recoverRecordingEl) recoverRecordingEl.disabled = false;
      if (discardRecoveryEl) discardRecoveryEl.disabled = false;
      return;
    }

    const recording = response.recording;
    const recordingActive = !!response.recordingActive;
    recoverySectionEl.hidden = false;
    if (recoverRecordingEl) recoverRecordingEl.disabled = recordingActive;
    if (discardRecoveryEl) discardRecoveryEl.disabled = recordingActive;
    const savedAt = recording.updatedAt
      ? new Date(recording.updatedAt).toLocaleString()
      : "recently";
    const extraCount = response.count > 1 ? ` (${response.count} autosaves)` : "";
    setRecoveryStatus(
      recordingActive
        ? `Autosaving current recording: ${formatBytes(recording.size)} saved`
        : `Autosave found${extraCount}: ${formatBytes(recording.size)} saved ${savedAt}`,
      recordingActive ? "neutral" : "available"
    );
  } catch (error) {
    recoverySectionEl.hidden = false;
    setRecoveryStatus(`Autosave check failed: ${error.message}`, "unavailable");
  }
}

async function refreshDriveStatus() {
  try {
    const response = await sendRuntimeMessage({ type: "DRIVE_STATUS" });
    if (!response?.success || !response.connected) {
      setDriveStatus("Drive: Not connected", "neutral");
      return;
    }

    const suffix = response.email ? ` (${response.email})` : "";
    setDriveStatus(`Drive: Connected${suffix}`, "connected");
  } catch (error) {
    setDriveStatus("Drive: Status unavailable", "unavailable");
  }
}

function bindSettingElement(elementId, key, parser = (value) => value) {
  const element = document.getElementById(elementId);
  if (!element) return;
  element.addEventListener("change", async () => {
    const rawValue = element.type === "checkbox" ? element.checked : element.value;
    await updateSettings({ [key]: parser(rawValue) });
  });
}

async function initializeSettingsUI() {
  const result = await chrome.storage.local.get(SETTINGS_KEY);
  const stored = result[SETTINGS_KEY] || {};
  const settings = normalizeRecordingSettings({ ...DEFAULT_SETTINGS, ...stored });
  if (settingsDiffer(settings, { ...DEFAULT_SETTINGS, ...stored })) {
    await setSettings(settings);
  }

  applySettingsToUI(settings);

  bindSettingElement("captureType", "captureType");
  bindSettingElement("includeSystemAudio", "includeSystemAudio");
  bindSettingElement("includeMicrophone", "includeMicrophone");
  bindSettingElement("enableSubtitles", "enableSubtitles");
  bindSettingElement("enableTranslation", "enableTranslation");
  bindSettingElement("sourceLanguage", "sourceLanguage");
  bindSettingElement("targetLanguage", "targetLanguage");
  bindSettingElement("subtitleFormat", "subtitleFormat");
  bindSettingElement("burnSubtitles", "burnSubtitles");
  bindSettingElement("liveCaptionOverlay", "liveCaptionOverlay");
  bindSettingElement("compressionPreset", "compressionPreset");
  bindSettingElement("enableDriveUpload", "enableDriveUpload");
}

function applySettingsToUI(settings) {
  document.getElementById("captureType").value = settings.captureType;
  document.getElementById("includeMicrophone").checked = !!settings.includeMicrophone;
  document.getElementById("enableSubtitles").checked = settings.enableSubtitles;
  document.getElementById("enableTranslation").checked = settings.enableTranslation;
  document.getElementById("sourceLanguage").value = settings.sourceLanguage || "auto";
  document.getElementById("targetLanguage").value = settings.targetLanguage;
  document.getElementById("subtitleFormat").value = settings.subtitleFormat;
  document.getElementById("burnSubtitles").checked = settings.burnSubtitles;
  document.getElementById("liveCaptionOverlay").checked = settings.liveCaptionOverlay;
  document.getElementById("compressionPreset").value = settings.compressionPreset;
  document.getElementById("enableDriveUpload").checked = settings.enableDriveUpload;

  syncAudioControls(settings);
  syncSubtitleControls(settings);
  syncLiveOverlayControls(settings);
}

function syncAudioControls(settings) {
  const includeSystemAudioEl = document.getElementById("includeSystemAudio");
  if (!includeSystemAudioEl) return;

  const tabCapture = settings.captureType === "tab";
  const includeMicrophone = !!settings.includeMicrophone;
  includeSystemAudioEl.disabled = !tabCapture;
  includeSystemAudioEl.checked = tabCapture && settings.includeSystemAudio;

  if (systemAudioGroupEl) {
    systemAudioGroupEl.classList.toggle("is-disabled", !tabCapture);
  }

  if (systemAudioLabelEl) {
    systemAudioLabelEl.textContent = tabCapture
      ? "Include current tab audio"
      : "Current tab audio unavailable";
  }

  if (systemAudioStatusEl) {
    if (tabCapture && settings.includeSystemAudio) {
      systemAudioStatusEl.textContent = "Meeting/tab audio is recorded without using the microphone.";
    } else if (tabCapture) {
      systemAudioStatusEl.textContent = "Off: meeting/tab audio will not be recorded.";
    } else {
      systemAudioStatusEl.textContent = "Window and screen recordings cannot capture meeting audio here.";
    }
    systemAudioStatusEl.classList.toggle(
      "warning",
      !tabCapture || !settings.includeSystemAudio
    );
  }

  if (microphoneGroupEl) {
    microphoneGroupEl.classList.toggle("is-enabled", includeMicrophone);
  }

  if (microphoneStatusEl) {
    microphoneStatusEl.textContent = includeMicrophone
      ? "On: room and microphone audio will be mixed into the recording."
      : "Off: nearby voices and room noise are not recorded.";
    microphoneStatusEl.classList.toggle("warning", includeMicrophone);
  }
}

function syncSubtitleControls(settings) {
  const enableSubtitles = !!settings.enableSubtitles;
  const subtitleFormatEl = document.getElementById("subtitleFormat");
  const burnSubtitlesEl = document.getElementById("burnSubtitles");
  const targetLanguageEl = document.getElementById("targetLanguage");
  const sourceLanguageEl = document.getElementById("sourceLanguage");
  if (sourceLanguageEl) {
    sourceLanguageEl.disabled = !enableSubtitles && !settings.liveCaptionOverlay;
  }
  if (subtitleFormatEl) {
    subtitleFormatEl.disabled = !enableSubtitles;
  }
  if (burnSubtitlesEl) {
    burnSubtitlesEl.disabled = !enableSubtitles;
  }
  if (targetLanguageEl) {
    targetLanguageEl.disabled = !settings.enableTranslation;
  }
}

function syncLiveOverlayControls(settings) {
  const liveOverlayEl = document.getElementById("liveCaptionOverlay");
  if (!liveOverlayEl) return;

  const reason = getLiveOverlayUnavailableReason(settings);
  liveOverlayEl.disabled = !!reason;
  if (reason) {
    liveOverlayEl.checked = false;
  }

  if (liveOverlayGroupEl) {
    liveOverlayGroupEl.classList.toggle("is-disabled", !!reason);
  }

  if (liveOverlayStatusEl) {
    liveOverlayStatusEl.textContent = reason || "Beta: Current Tab audio only.";
    liveOverlayStatusEl.classList.toggle("warning", !!reason);
  }
}

async function getMicPermissionState() {
  try {
    if (!("permissions" in navigator)) return "prompt";
    const status = await navigator.permissions.query({ name: "microphone" });
    return status.state;
  } catch {
    return "prompt";
  }
}

function setMicStatus(text, state) {
  if (!micStatusEl) return;
  micStatusEl.textContent = text;
  micStatusEl.classList.remove("mic-status-enabled", "mic-status-unavailable", "mic-status-neutral");
  if (state === "enabled") {
    micStatusEl.classList.add("mic-status-enabled");
    return;
  }
  if (state === "unavailable") {
    micStatusEl.classList.add("mic-status-unavailable");
    return;
  }
  micStatusEl.classList.add("mic-status-neutral");
}

async function refreshMicStatus() {
  const state = await getMicPermissionState();
  if (state === "granted") {
    setMicStatus("Microphone permission: Granted", "enabled");
    return;
  }

  if (state === "denied") {
    setMicStatus("Microphone permission: Blocked", "unavailable");
    return;
  }

  setMicStatus("Microphone permission: Not granted", "neutral");
}

document.getElementById("start").addEventListener("click", async () => {
  const settings = normalizeRecordingSettings(await getSettings());
  await setSettings(settings);
  applySettingsToUI(settings);

  const captureType = settings.captureType;
  const includeSystemAudio = hasCapturableTabAudio(settings);
  let enableSubtitles = settings.enableSubtitles;
  let enableTranslation = settings.enableTranslation;
  const sourceLanguage = settings.sourceLanguage || "auto";
  const targetLanguage = settings.targetLanguage;
  const subtitleFormat = settings.subtitleFormat;
  let burnSubtitles = settings.burnSubtitles && enableSubtitles;
  const liveCaptionOverlay = settings.liveCaptionOverlay && isLiveOverlaySupported(settings);
  const compressionPreset = settings.compressionPreset;
  const enableDriveUpload = settings.enableDriveUpload;
  let includeMicrophone = !!settings.includeMicrophone;
  let permissionProbe = null;
  let preferredMicrophoneDeviceId = null;
  const cleanupPermissionProbe = () => {
    if (permissionProbe) {
      permissionProbe.getTracks().forEach((track) => track.stop());
      permissionProbe = null;
    }
  };

  if (includeMicrophone) {
    const micPermissionState = await getMicPermissionState();
    if (micPermissionState !== "granted") {
      const doSetup = confirm(
        "Microphone recording is turned on, but microphone permission has not been granted.\n\n" +
        "Click OK to open the setup page. This recording will continue without microphone audio."
      );
      if (doSetup) {
        chrome.tabs.create({ url: "setup.html" });
      }
      includeMicrophone = false;
      setMicStatus("Microphone permission: Not granted", "neutral");
    }
  }

  if (includeMicrophone) {
    try {
      permissionProbe = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: false
      });

      const [probeTrack] = permissionProbe.getAudioTracks();
      preferredMicrophoneDeviceId = probeTrack?.getSettings?.().deviceId || null;
      console.log("Microphone permission granted from popup");
      setMicStatus("Microphone permission: Granted", "enabled");
      
      cleanupPermissionProbe();

    } catch (error) {
      const errorMessage = error?.message || String(error);
      console.error("Microphone permission denied:", errorMessage);
      
      // Check if it is a permission denied error
      if (errorMessage.includes("Permission denied") || errorMessage.includes("NotAllowedError") || errorMessage.includes("permission was not granted")) {
        const doSetup = confirm(
          "Microphone access is blocked. Chrome prevents microphone prompts in the popup menu.\n\n" +
            "Click OK to open a setup page where you can grant permission once.\n" +
            "This recording will continue without microphone audio."
        );
        if (doSetup) {
          chrome.tabs.create({ url: "setup.html" });
        }
      } else {
        alert("Microphone error: " + errorMessage + "\nRecording will continue without microphone.");
      }
      setMicStatus("Microphone permission: Blocked", "unavailable");
      includeMicrophone = false;
      preferredMicrophoneDeviceId = null;
    }
  }

  if (enableSubtitles && !includeSystemAudio && !includeMicrophone) {
    alert(
      "Subtitles need captured audio. Use Current Tab with tab audio, or enable the microphone for narrated recordings.\n\n" +
      "Recording will continue without subtitle generation."
    );
    enableSubtitles = false;
    enableTranslation = false;
    burnSubtitles = false;
  }

  if (enableSubtitles && !includeSystemAudio && includeMicrophone) {
    const continueWithMicOnly = confirm(
      "Subtitles will be generated from microphone audio only.\n\n" +
      "If you are trying to caption a video playing in Chrome, choose Current Tab and keep tab audio enabled so the actual video audio is recorded.\n\n" +
      "Continue with microphone-only subtitles?"
    );

    if (!continueWithMicOnly) {
      cleanupPermissionProbe();
      return;
    }
  }

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.runtime.sendMessage({
    type: "START_RECORDING",
    tabId: tab.id,
    captureType,
    includeSystemAudio,
    includeMicrophone,
    preferredMicrophoneDeviceId,
    processingOptions: {
      enableTranslation,
      enableSubtitles,
      sourceLanguage,
      targetLanguage,
      subtitleFormat,
      burnSubtitles,
      liveCaptionOverlay,
      compressionPreset,
      enableDriveUpload
    }
  }, (response) => {
    cleanupPermissionProbe();

    if (response?.success) {
      console.log("Recording started");
    } else {
      console.error("Failed to start recording:", response?.error);
    }
    // Close popup AFTER receiving response
    window.close();
  });
});

document.getElementById("stop").addEventListener("click", async () => {
  chrome.runtime.sendMessage({
    type: "STOP_RECORDING"
  }, (response) => {
    if (response?.success) {
      console.log("Recording stopped");
    }
    // Close popup AFTER receiving response
    window.close();
  });
});

document.getElementById("enableMic")?.addEventListener("click", async () => {
  chrome.tabs.create({ url: "setup.html" });
});

recoverRecordingEl?.addEventListener("click", async () => {
  recoverRecordingEl.disabled = true;
  if (discardRecoveryEl) discardRecoveryEl.disabled = true;
  setRecoveryStatus("Recovering autosaved recording…", "neutral");

  try {
    const response = await sendRuntimeMessage({ type: "RECOVER_AUTOSAVE_REQUEST" });
    if (!response?.success) {
      throw new Error(response?.error || "Recovery failed");
    }

    setRecoveryStatus(
      `Recovered ${response.filename || "recording"} (${formatBytes(response.size)})`,
      "available"
    );
    setTimeout(refreshRecoveryStatus, 1200);
  } catch (error) {
    setRecoveryStatus(`Recovery failed: ${error.message}`, "unavailable");
  } finally {
    recoverRecordingEl.disabled = false;
    if (discardRecoveryEl) discardRecoveryEl.disabled = false;
  }
});

discardRecoveryEl?.addEventListener("click", async () => {
  if (!confirm("Discard the autosaved recording chunks? This cannot be undone.")) {
    return;
  }

  if (recoverRecordingEl) recoverRecordingEl.disabled = true;
  discardRecoveryEl.disabled = true;
  setRecoveryStatus("Discarding autosave…", "neutral");

  try {
    const response = await sendRuntimeMessage({ type: "DISCARD_AUTOSAVE_REQUEST" });
    if (!response?.success) {
      throw new Error(response?.error || "Discard failed");
    }
    recoverySectionEl.hidden = true;
  } catch (error) {
    setRecoveryStatus(`Discard failed: ${error.message}`, "unavailable");
  } finally {
    if (recoverRecordingEl) recoverRecordingEl.disabled = false;
    discardRecoveryEl.disabled = false;
  }
});

document.getElementById("connectDrive")?.addEventListener("click", async () => {
  setDriveStatus("Drive: Connecting...", "neutral");
  try {
    const response = await sendRuntimeMessage({ type: "DRIVE_CONNECT" });
    if (response?.success) {
      const suffix = response.email ? ` (${response.email})` : "";
      setDriveStatus(`Drive: Connected${suffix}`, "connected");
      return;
    }
    setDriveStatus(`Drive: ${response?.error || "Connect failed"}`, "unavailable");
  } catch (error) {
    setDriveStatus(`Drive: ${error.message}`, "unavailable");
  }
});

document.getElementById("disconnectDrive")?.addEventListener("click", async () => {
  try {
    const response = await sendRuntimeMessage({ type: "DRIVE_DISCONNECT" });
    if (response?.success) {
      setDriveStatus("Drive: Not connected", "neutral");
      return;
    }
    setDriveStatus(`Drive: ${response?.error || "Disconnect failed"}`, "unavailable");
  } catch (error) {
    setDriveStatus(`Drive: ${error.message}`, "unavailable");
  }
});

function setBackendStatus(text, state) {
  if (!backendStatusEl) return;
  backendStatusEl.textContent = text;
  backendStatusEl.classList.remove(
    "backend-status-connected",
    "backend-status-unavailable",
    "backend-status-neutral"
  );
  if (state === "connected") {
    backendStatusEl.classList.add("backend-status-connected");
    return;
  }
  if (state === "unavailable") {
    backendStatusEl.classList.add("backend-status-unavailable");
    return;
  }
  backendStatusEl.classList.add("backend-status-neutral");
}

async function refreshBackendStatus() {
  try {
    const response = await sendRuntimeMessage({ type: "BACKEND_HEALTH" });
    if (response?.healthy) {
      setBackendStatus("Backend: Connected", "connected");
      return;
    }
    setBackendStatus(
      `Backend: ${response?.error || "Unavailable"}`,
      "unavailable"
    );
  } catch (error) {
    setBackendStatus("Backend: Not configured", "neutral");
  }
}

document.getElementById("openBackendSettings")?.addEventListener("click", () => {
  chrome.tabs.create({ url: "setup.html" });
});

window.addEventListener("focus", () => {
  refreshMicStatus();
  refreshDriveStatus();
  refreshBackendStatus();
  refreshRecoveryStatus();
});

refreshMicStatus();
initializeSettingsUI();
refreshDriveStatus();
refreshBackendStatus();
refreshRecoveryStatus();
