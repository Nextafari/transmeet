async function setupOffscreenDocument() {
  // Check if offscreen document already exists
  const existingContexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT']
  });

  if (existingContexts.length > 0) {
    console.log("[BG] Offscreen document already exists");
    return;
  }

  // Create offscreen document
  await chrome.offscreen.createDocument({
    url: "offscreen.html",
    reasons: ["USER_MEDIA", "AUDIO_PLAYBACK", "BLOBS"],
    justification: "Recording media and preparing recording files for download"
  });
  
  console.log("[BG] Offscreen document created");
  // Give offscreen document time to load
  await new Promise(resolve => setTimeout(resolve, 100));
}

function waitForDownloadCompletion(downloadId, timeoutMs = 10 * 60 * 1000) {
  return new Promise((resolve, reject) => {
    let settled = false;

    const finish = (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      chrome.downloads.onChanged.removeListener(handleChange);
      if (error) {
        reject(error);
      } else {
        resolve();
      }
    };

    const inspectState = (state, error) => {
      if (state === "complete") {
        finish();
      } else if (state === "interrupted") {
        finish(new Error(`Download interrupted${error ? `: ${error}` : ""}`));
      }
    };

    const handleChange = (delta) => {
      if (delta.id !== downloadId) return;
      inspectState(delta.state?.current, delta.error?.current);
    };

    const timer = setTimeout(() => {
      finish(new Error("Download did not complete within 10 minutes"));
    }, timeoutMs);

    chrome.downloads.onChanged.addListener(handleChange);
    chrome.downloads.search({ id: downloadId }, (items) => {
      if (chrome.runtime.lastError) {
        finish(new Error(chrome.runtime.lastError.message));
        return;
      }

      const item = items?.[0];
      if (!item) {
        finish(new Error("Download could not be found after it started"));
        return;
      }
      inspectState(item.state, item.error);
    });
  });
}

async function downloadUrl({ url, filename }) {
  if (!url || !String(url).startsWith("blob:")) {
    throw new Error("Invalid recording download URL");
  }

  let lastError = null;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const downloadId = await new Promise((resolve, reject) => {
        chrome.downloads.download(
          {
            url,
            filename,
            saveAs: false,
          },
          (id) => {
            if (chrome.runtime.lastError || typeof id !== "number") {
              reject(new Error(chrome.runtime.lastError?.message || "Download failed"));
              return;
            }
            resolve(id);
          }
        );
      });

      await waitForDownloadCompletion(downloadId);
      return downloadId;
    } catch (error) {
      lastError = error;
      if (attempt === 0) {
        console.info("[BG] Retrying interrupted download:", error.message);
      }
    }
  }

  throw lastError || new Error("Download failed");
}

const DRIVE_AUTH_KEY = "driveAuthState";

function getAuthToken(interactive) {
  return new Promise((resolve, reject) => {
    chrome.identity.getAuthToken({ interactive }, (token) => {
      if (chrome.runtime.lastError || !token) {
        reject(new Error(chrome.runtime.lastError?.message || "No auth token available"));
        return;
      }
      resolve(token);
    });
  });
}

function removeCachedAuthToken(token) {
  return new Promise((resolve) => {
    if (!token) {
      resolve();
      return;
    }
    chrome.identity.removeCachedAuthToken({ token }, () => resolve());
  });
}

async function fetchUserEmail(token) {
  const response = await fetch("https://www.googleapis.com/oauth2/v2/userinfo", {
    headers: {
      Authorization: `Bearer ${token}`
    }
  });

  if (!response.ok) {
    throw new Error(`Unable to load account profile (${response.status})`);
  }

  const data = await response.json();
  return data.email || null;
}

async function updateDriveState(nextState) {
  await chrome.storage.local.set({ [DRIVE_AUTH_KEY]: nextState });
}

async function readDriveState() {
  const stored = await chrome.storage.local.get(DRIVE_AUTH_KEY);
  return stored[DRIVE_AUTH_KEY] || { connected: false, email: null };
}

async function connectDriveAccount() {
  const token = await getAuthToken(true);
  const email = await fetchUserEmail(token);
  await updateDriveState({ connected: true, email });
  return { connected: true, email };
}

async function disconnectDriveAccount() {
  try {
    const token = await getAuthToken(false);
    await removeCachedAuthToken(token);
  } catch {
    // No active token cached.
  }

  await updateDriveState({ connected: false, email: null });
}

async function uploadToDrive({ blob, filename, mimeType }) {
  const driveState = await readDriveState();
  if (!driveState.connected) {
    throw new Error("Google Drive is not connected. Connect Drive from the popup first.");
  }

  let token;
  try {
    token = await getAuthToken(false);
  } catch {
    throw new Error("Drive session expired. Reconnect Drive from the popup.");
  }

  const metadata = {
    name: filename,
    mimeType: mimeType || "video/webm"
  };

  async function attemptUpload(uploadToken) {
    const formData = new FormData();
    formData.append("metadata", new Blob([JSON.stringify(metadata)], { type: "application/json" }));
    formData.append("file", blob, filename);

    const response = await fetch("https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id,name,webViewLink", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${uploadToken}`
      },
      body: formData
    });

    return response;
  }

  let response = await attemptUpload(token);
  if (response.status === 401) {
    await removeCachedAuthToken(token);
    token = await getAuthToken(false);
    response = await attemptUpload(token);
  }

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Drive upload failed (${response.status}): ${body.slice(0, 160)}`);
  }

  return response.json();
}

async function sendToOffscreen(message) {
  // Send message to offscreen document with error handling
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(message, (response) => {
      if (chrome.runtime.lastError) {
        console.warn("[BG] Message send error:", chrome.runtime.lastError.message);
        reject(chrome.runtime.lastError);
      } else {
        resolve(response);
      }
    });
  });
}

const BACKEND_URL_KEY = "backendUrl";
const OVERLAY_ID = "screenrec-live-caption-overlay";
const ACTIVE_TAB_KEY = "activeRecordingTabId";
const LIVE_OVERLAY_KEY = "liveOverlayEnabled";

let activeRecordingTabId = null;
let liveOverlayEnabled = false;
let overlayClearTimer = null;
let stoppingRecording = false;
let lastOverlayTabId = null;

async function fetchWithTimeout(url, options = {}, timeoutMs = 5000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, {
      ...options,
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timer);
  }
}

async function updateLiveCaptionOverlay(tabId, captionText) {
  if (!tabId) return;
  cancelOverlayClear();

  await chrome.scripting.executeScript({
    target: { tabId },
    func: (overlayId, text) => {
      let overlay = document.getElementById(overlayId);
      if (!overlay) {
        overlay = document.createElement("div");
        overlay.id = overlayId;
        overlay.style.position = "fixed";
        overlay.style.left = "50%";
        overlay.style.bottom = "28px";
        overlay.style.transform = "translateX(-50%)";
        overlay.style.zIndex = "2147483647";
        overlay.style.maxWidth = "80vw";
        overlay.style.maxHeight = "132px";
        overlay.style.overflow = "hidden";
        overlay.style.padding = "10px 14px";
        overlay.style.borderRadius = "10px";
        overlay.style.background = "rgba(15, 23, 42, 0.82)";
        overlay.style.color = "#f8fafc";
        overlay.style.fontSize = "15px";
        overlay.style.fontFamily = "system-ui, -apple-system, Segoe UI, sans-serif";
        overlay.style.fontWeight = "600";
        overlay.style.lineHeight = "1.3";
        overlay.style.textAlign = "center";
        overlay.style.boxShadow = "0 8px 28px rgba(0,0,0,0.35)";
        overlay.style.pointerEvents = "none";
        document.documentElement.appendChild(overlay);
      }

      overlay.textContent = text;
      overlay.style.display = text ? "block" : "none";
    },
    args: [OVERLAY_ID, captionText || ""],
  });
}

async function clearLiveCaptionOverlay(tabId) {
  if (!tabId) return;

  await chrome.scripting.executeScript({
    target: { tabId },
    func: (overlayId) => {
      const overlay = document.getElementById(overlayId);
      if (overlay) {
        overlay.remove();
      }
    },
    args: [OVERLAY_ID],
  });
}

function cancelOverlayClear() {
  if (!overlayClearTimer) return;
  clearTimeout(overlayClearTimer);
  overlayClearTimer = null;
}

function scheduleOverlayClear(tabId, delayMs = 5000) {
  cancelOverlayClear();
  overlayClearTimer = setTimeout(() => {
    clearLiveCaptionOverlay(tabId).catch(() => {});
    overlayClearTimer = null;
  }, delayMs);
}

async function getBackendUrl() {
  const stored = await chrome.storage.local.get(BACKEND_URL_KEY);
  const value = String(stored[BACKEND_URL_KEY] || "").trim();
  if (!value) return null;

  const parsed = new URL(value);
  if (!["http:", "https:"].includes(parsed.protocol)) {
    throw new Error("Backend URL must start with http:// or https://");
  }

  return parsed.href.replace(/\/+$/, "");
}

function buildBackendUrl(baseUrl, path) {
  const cleanBase = String(baseUrl || "").replace(/\/+$/, "");
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  return `${cleanBase}${cleanPath}`;
}

function backendRequestHeaders(url, initialHeaders) {
  const headers = new Headers(initialHeaders || {});
  const hostname = new URL(url).hostname.toLowerCase();
  if (hostname.endsWith(".ngrok-free.dev") ||
      hostname.endsWith(".ngrok-free.app") ||
      hostname.endsWith(".ngrok.io")) {
    headers.set("ngrok-skip-browser-warning", "1");
  }
  return headers;
}

async function readBackendJson(response, label) {
  const contentType = response.headers.get("content-type") || "";
  const body = await response.text();

  if (!response.ok) {
    throw new Error(`${label} returned ${response.status}: ${body.slice(0, 120)}`);
  }

  if (!contentType.toLowerCase().includes("application/json")) {
    throw new Error(
      `${label} returned ${contentType || "unknown content"} instead of JSON`
    );
  }

  return JSON.parse(body);
}

async function checkBackendHealth() {
  try {
    const backendUrl = await getBackendUrl();
    if (!backendUrl) {
      return { healthy: false, error: "Backend URL not configured" };
    }

    const healthUrl = buildBackendUrl(backendUrl, "/api/v1/health/");
    const response = await fetchWithTimeout(
      healthUrl,
      {
        method: "GET",
        headers: backendRequestHeaders(healthUrl),
      },
      5000
    );

    const payload = await readBackendJson(response, "Backend health");
    if (payload?.status !== "ok") {
      return { healthy: false, error: "Backend health response was not ok" };
    }

    return { healthy: true };
  } catch (error) {
    return { healthy: false, error: error.message };
  }
}

async function translateText(text, targetLanguage) {
  const backendUrl = await getBackendUrl();
  if (!backendUrl) {
    throw new Error("Backend URL not configured");
  }

  const translationUrl = buildBackendUrl(backendUrl, "/api/v1/translation/text");
  const response = await fetch(translationUrl, {
    method: "POST",
    headers: backendRequestHeaders(translationUrl, {
      "Content-Type": "application/json",
    }),
    body: JSON.stringify({
      text,
      target_language: targetLanguage,
    }),
  });

  return readBackendJson(response, "Translation API");
}

function supportsLiveCaptionOverlay(message) {
  return message.captureType === "tab" && !!message.includeSystemAudio;
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "RECORDING_STATUS") {
    (async () => {
      try {
        let tabId = activeRecordingTabId || message.tabId || lastOverlayTabId;
        if (!tabId) {
          const stored = await chrome.storage.local.get(ACTIVE_TAB_KEY);
          tabId = stored[ACTIVE_TAB_KEY] || null;
          if (tabId) {
            lastOverlayTabId = tabId;
          }
        }

        if (!liveOverlayEnabled) {
          const stored = await chrome.storage.local.get(LIVE_OVERLAY_KEY);
          liveOverlayEnabled = !!stored[LIVE_OVERLAY_KEY];
        }

        if (message.stage === "overlay-clear") {
          if (tabId) {
            await clearLiveCaptionOverlay(tabId);
          }
          sendResponse({ success: true });
          return;
        }

        if (!liveOverlayEnabled || !tabId || stoppingRecording) {
          sendResponse({ success: true });
          return;
        }

        if (message.stage === "translation-started") {
          const statusText = message.subtitleOnly
            ? "Generating subtitles..."
            : "Generating translated subtitles...";
          await updateLiveCaptionOverlay(
            tabId,
            statusText
          );
        }

        if (message.stage === "translation-completed") {
          const preview = message.translation
            ? String(message.translation).slice(0, 180)
            : "Subtitles generated";
          await updateLiveCaptionOverlay(tabId, preview);
        }

        if (message.stage === "translation-error") {
          const errorText = message.error
            ? `Subtitles failed: ${message.error}`
            : "Subtitles failed";
          await updateLiveCaptionOverlay(tabId, errorText);
          scheduleOverlayClear(tabId, 7000);
        }

        if (message.stage === "burn-error") {
          const errorText = message.error
            ? `Subtitle burn failed: ${message.error}`
            : "Subtitle burn failed";
          await updateLiveCaptionOverlay(tabId, errorText);
          scheduleOverlayClear(tabId, 7000);
        }

        if (message.stage === "live-caption") {
          await updateLiveCaptionOverlay(
            tabId,
            message.caption || ""
          );
        }

        if (message.stage === "live-caption-empty") {
          scheduleOverlayClear(tabId, 1800);
        }

        if (message.stage === "live-caption-error") {
          const errorText = message.error
            ? `Live captions failed: ${message.error}`
            : "Live captions failed";
          await updateLiveCaptionOverlay(tabId, errorText);
          scheduleOverlayClear(tabId, 6000);
        }

        sendResponse({ success: true });
      } catch (error) {
        sendResponse({ success: false, error: error.message });
      }
    })();

    return true;
  }

  if (message.type === "BACKEND_HEALTH") {
    (async () => {
      const health = await checkBackendHealth();
      sendResponse(health);
    })();
    return true;
  }

  if (message.type === "BACKEND_URL_GET") {
    (async () => {
      try {
        const backendUrl = await getBackendUrl();
        sendResponse({ success: true, backendUrl: backendUrl || null });
      } catch (error) {
        sendResponse({ success: false, error: error.message });
      }
    })();
    return true;
  }

  if (message.type === "BACKEND_TRANSLATE") {
    (async () => {
      try {
        const result = await translateText(
          message.text,
          message.targetLanguage
        );
        sendResponse({ success: true, ...result });
      } catch (error) {
        sendResponse({ success: false, error: error.message });
      }
    })();
    return true;
  }

  if (message.type === "DRIVE_STATUS") {
    (async () => {
      try {
        const state = await readDriveState();
        sendResponse({ success: true, ...state });
      } catch (error) {
        sendResponse({ success: false, error: error.message });
      }
    })();

    return true;
  }

  if (message.type === "DRIVE_CONNECT") {
    (async () => {
      try {
        const state = await connectDriveAccount();
        sendResponse({ success: true, ...state });
      } catch (error) {
        sendResponse({ success: false, error: error.message });
      }
    })();

    return true;
  }

  if (message.type === "DRIVE_DISCONNECT") {
    (async () => {
      try {
        await disconnectDriveAccount();
        sendResponse({ success: true });
      } catch (error) {
        sendResponse({ success: false, error: error.message });
      }
    })();

    return true;
  }

  if (message.type === "UPLOAD_TO_DRIVE") {
    (async () => {
      try {
        const uploaded = await uploadToDrive({
          blob: message.blob,
          filename: message.filename,
          mimeType: message.mimeType
        });
        sendResponse({ success: true, file: uploaded });
      } catch (error) {
        console.error("[BG] Drive upload error:", error);
        sendResponse({ success: false, error: error.message });
      }
    })();

    return true;
  }

  if (message.type === "AUTOSAVE_STATUS_REQUEST" ||
      message.type === "RECOVER_AUTOSAVE_REQUEST" ||
      message.type === "DISCARD_AUTOSAVE_REQUEST") {
    (async () => {
      try {
        await setupOffscreenDocument();
        const response = await sendToOffscreen({
          ...message,
          type: message.type.replace("_REQUEST", ""),
        });
        sendResponse(response || { success: false, error: "No autosave response" });
      } catch (error) {
        sendResponse({ success: false, error: error.message });
      }
    })();

    return true;
  }

  if (message.type === "START_RECORDING") {
    console.log("[BG] Start recording requested");

    (async () => {
      try {
        stoppingRecording = false;
        await setupOffscreenDocument();
        activeRecordingTabId = message.tabId ?? sender?.tab?.id ?? null;
        lastOverlayTabId = activeRecordingTabId;
        if (activeRecordingTabId) {
          await chrome.storage.local.set({
            [ACTIVE_TAB_KEY]: activeRecordingTabId,
          });
        }
        liveOverlayEnabled =
          !!message.processingOptions?.liveCaptionOverlay &&
          supportsLiveCaptionOverlay(message);
        await chrome.storage.local.set({
          [LIVE_OVERLAY_KEY]: liveOverlayEnabled,
        });

        if (liveOverlayEnabled && activeRecordingTabId) {
          await updateLiveCaptionOverlay(
            activeRecordingTabId,
            "Live caption overlay enabled"
          );
        }

        let streamId;

        if (message.captureType === "tab") {
          // Use tabCapture for current tab
          streamId = await chrome.tabCapture.getMediaStreamId({
            targetTabId: message.tabId
          });
        } else {
          // Use desktopCapture for window/screen
          // This will show Chrome's native picker
          const sources = message.captureType === "screen" 
            ? ["screen", "window"] 
            : ["window"];

          const tabId = message.tabId ?? sender?.tab?.id;
          if (!tabId) {
            throw new Error("Missing active tab for screen selection");
          }

          const tab = await chrome.tabs.get(tabId);

          streamId = await new Promise((resolve, reject) => {
            chrome.desktopCapture.chooseDesktopMedia(
              sources,
              tab,
              (id) => {
                if (id) {
                  resolve(id);
                } else {
                  reject(new Error("User cancelled screen selection"));
                }
              }
            );
          });
        }

        await sendToOffscreen({
          type: "BEGIN_CAPTURE",
          tabId: activeRecordingTabId,
          streamId: streamId,
          captureType: message.captureType,
          includeSystemAudio: message.includeSystemAudio,
          includeMicrophone: message.includeMicrophone,
          preferredMicrophoneDeviceId: message.preferredMicrophoneDeviceId,
          processingOptions: {
            ...(message.processingOptions || {}),
            liveCaptionOverlay: liveOverlayEnabled,
          }
        });

        sendResponse({ success: true });
      } catch (error) {
        console.error("[BG] Error starting recording:", error);
        sendResponse({ success: false, error: error.message });
      }
    })();

    return true; // Keep message channel open for async response
  }

  if (message.type === "STOP_RECORDING") {
    console.log("[BG] Stop recording requested");
    
    (async () => {
      try {
        stoppingRecording = true;

        const tabId = activeRecordingTabId || lastOverlayTabId;
        if (tabId) {
          await clearLiveCaptionOverlay(tabId);
        }

        liveOverlayEnabled = false;
        activeRecordingTabId = null;
        await chrome.storage.local.remove([ACTIVE_TAB_KEY, LIVE_OVERLAY_KEY]);

        await sendToOffscreen({
          type: "STOP_RECORDING"
        });

        sendResponse({ success: true });
      } catch (error) {
        console.error("[BG] Error stopping recording:", error);
        liveOverlayEnabled = false;
        activeRecordingTabId = null;
        sendResponse({ success: false, error: error.message });
      }
    })();
    
    return true;
  }

  if (message.type === "DOWNLOAD_URL") {
    (async () => {
      try {
        const filename = message.filename || `recording-${Date.now()}.webm`;
        const downloadId = await downloadUrl({
          url: String(message.url || ""),
          filename,
        });

        sendResponse({ success: true, downloadId });
      } catch (error) {
        console.error("[BG] Download error:", error);
        sendResponse({ success: false, error: error.message });
      }
    })();

    return true;
  }
});
