(function() {
  // Private state - encapsulated within this closure
  let stream;
  let mediaRecorder;
  let recordedChunks = [];
  let sourceVideoStream = null;
  let sourceMicStream = null;
  let mixAudioContext = null;
  let monitorAudio = null;
  let currentProcessingOptions = {
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
  let liveCaptionSeq = 0;
  let liveCaptionNextDisplaySeq = 1;
  let liveCaptionPending = [];
  let liveCaptionInflight = new Map();
  let liveCaptionCompleted = new Map();
  let liveCaptionSession = 0;
  let liveCaptionPlaceholderShown = false;
  let lastLiveCaptionErrorAt = 0;
  let liveCaptionRecorder = null;
  let liveCaptionTimer = null;
  let liveCaptionChunks = [];
  let lastLiveCaptionRequestAt = 0;
  let lastLiveCaptionFingerprint = "";
  let liveCaptionTimeoutStreak = 0;
  let liveCaptionEverUpdated = false;
  let activeTabId = null;
  let activeAutosave = null;
  let autosaveWriteChain = Promise.resolve();
  let autosaveDbPromise = null;
  const DEFAULT_LIVE_CAPTION_CHUNK_MS = 1000;
  const DEFAULT_LIVE_CAPTION_MIN_INTERVAL_MS = 0;
  const DEFAULT_LIVE_CAPTION_MIN_BLOB_BYTES = 1200;
  const LIVE_CAPTION_MAX_QUEUE = 4;
  const LIVE_CAPTION_MAX_INFLIGHT = 2;
  const LIVE_CAPTION_SKIP_AHEAD_MS = 2500;
  const LIVE_CAPTION_ERROR_THROTTLE_MS = 5000;
  const RECORDING_CHUNK_MS = 5000;
  const LIVE_CAPTION_TIMEOUT_MS = 15000;
  const LIVE_CAPTION_SLOW_AFTER_TIMEOUTS = 3;
  const AUTOSAVE_DB_NAME = "screenrec-autosave";
  const AUTOSAVE_DB_VERSION = 1;
  const AUTOSAVE_RECORDINGS_STORE = "recordings";
  const AUTOSAVE_CHUNKS_STORE = "chunks";
  let liveCaptionChunkMs = DEFAULT_LIVE_CAPTION_CHUNK_MS;
  let liveCaptionMinIntervalMs = DEFAULT_LIVE_CAPTION_MIN_INTERVAL_MS;
  let liveCaptionMinBlobBytes = DEFAULT_LIVE_CAPTION_MIN_BLOB_BYTES;
  let liveStreamSession = null;
  let liveStreamAudioContext = null;
  let liveStreamWorkletNode = null;
  let liveStreamActive = false;
  let liveStreamDowngraded = false;
  let streamingConfig = {
    enabled: false,
    wsPath: "/api/v1/media/live-stream",
    frameMs: 80,
  };

  /**
   * Formats error for logging (handles DOMException properly)
   */
  function formatError(error) {
    if (error instanceof DOMException) {
      return `${error.name}: ${error.message}`;
    }
    if (error instanceof Error) {
      return `${error.name}: ${error.message}`;
    }
    return String(error);
  }

  function isAbortError(error) {
    return error?.name === "AbortError" || String(error?.message || "").includes("aborted");
  }

  function requestToPromise(request) {
    return new Promise((resolve, reject) => {
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error("IndexedDB request failed"));
    });
  }

  function transactionDone(transaction) {
    return new Promise((resolve, reject) => {
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error || new Error("IndexedDB transaction failed"));
      transaction.onabort = () => reject(transaction.error || new Error("IndexedDB transaction aborted"));
    });
  }

  function openAutosaveDb() {
    if (autosaveDbPromise) {
      return autosaveDbPromise;
    }

    autosaveDbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(AUTOSAVE_DB_NAME, AUTOSAVE_DB_VERSION);

      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(AUTOSAVE_RECORDINGS_STORE)) {
          db.createObjectStore(AUTOSAVE_RECORDINGS_STORE, { keyPath: "id" });
        }

        if (!db.objectStoreNames.contains(AUTOSAVE_CHUNKS_STORE)) {
          const chunks = db.createObjectStore(AUTOSAVE_CHUNKS_STORE, {
            keyPath: ["recordingId", "index"],
          });
          chunks.createIndex("recordingId", "recordingId", { unique: false });
        }
      };

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error("Unable to open autosave database"));
    });

    return autosaveDbPromise;
  }

  function queueAutosaveWrite(operation) {
    const run = autosaveWriteChain.catch(() => {}).then(operation);
    autosaveWriteChain = run.catch((error) => {
      console.warn("[OFFSCREEN] Recording autosave write failed:", formatError(error));
    });
    return autosaveWriteChain;
  }

  async function waitForAutosaveWrites() {
    await autosaveWriteChain.catch(() => {});
  }

  async function createAutosaveSession(filename, mimeType) {
    const now = Date.now();
    const session = {
      id: `recording-${now}-${Math.random().toString(36).slice(2, 8)}`,
      filename,
      mimeType: mimeType || "video/webm",
      startedAt: now,
      updatedAt: now,
      chunkCount: 0,
      size: 0,
    };

    const db = await openAutosaveDb();
    const transaction = db.transaction(AUTOSAVE_RECORDINGS_STORE, "readwrite");
    const done = transactionDone(transaction);
    transaction.objectStore(AUTOSAVE_RECORDINGS_STORE).put(session);
    await done;
    return session;
  }

  function saveAutosaveChunk(chunk) {
    if (!activeAutosave || !chunk?.size) {
      return;
    }

    const recordingId = activeAutosave.id;
    const index = activeAutosave.chunkCount;
    const now = Date.now();
    activeAutosave = {
      ...activeAutosave,
      updatedAt: now,
      chunkCount: index + 1,
      size: activeAutosave.size + chunk.size,
    };
    const metadata = { ...activeAutosave };
    const chunkRecord = {
      recordingId,
      index,
      blob: chunk,
      size: chunk.size,
      createdAt: now,
    };

    queueAutosaveWrite(async () => {
      const db = await openAutosaveDb();
      const transaction = db.transaction(
        [AUTOSAVE_RECORDINGS_STORE, AUTOSAVE_CHUNKS_STORE],
        "readwrite"
      );
      const done = transactionDone(transaction);
      transaction.objectStore(AUTOSAVE_CHUNKS_STORE).put(chunkRecord);
      transaction.objectStore(AUTOSAVE_RECORDINGS_STORE).put(metadata);
      await done;
    });
  }

  async function listAutosaveRecordings() {
    const db = await openAutosaveDb();
    const transaction = db.transaction(AUTOSAVE_RECORDINGS_STORE, "readonly");
    const done = transactionDone(transaction);
    const recordings = await requestToPromise(
      transaction.objectStore(AUTOSAVE_RECORDINGS_STORE).getAll()
    );
    await done;
    return recordings
      .filter((recording) => recording.chunkCount > 0)
      .sort((left, right) => right.updatedAt - left.updatedAt);
  }

  async function getAutosaveChunks(recordingId) {
    const db = await openAutosaveDb();
    const transaction = db.transaction(AUTOSAVE_CHUNKS_STORE, "readonly");
    const done = transactionDone(transaction);
    const index = transaction.objectStore(AUTOSAVE_CHUNKS_STORE).index("recordingId");

    const chunks = await new Promise((resolve, reject) => {
      const results = [];
      const request = index.openCursor(IDBKeyRange.only(recordingId));
      request.onsuccess = () => {
        const cursor = request.result;
        if (!cursor) {
          resolve(results);
          return;
        }
        results.push(cursor.value);
        cursor.continue();
      };
      request.onerror = () => reject(request.error || new Error("Unable to read autosave chunks"));
    });

    await done;
    return chunks.sort((left, right) => left.index - right.index);
  }

  async function deleteAutosaveRecording(recordingId) {
    const db = await openAutosaveDb();
    const transaction = db.transaction(
      [AUTOSAVE_RECORDINGS_STORE, AUTOSAVE_CHUNKS_STORE],
      "readwrite"
    );
    const done = transactionDone(transaction);
    transaction.objectStore(AUTOSAVE_RECORDINGS_STORE).delete(recordingId);

    const index = transaction.objectStore(AUTOSAVE_CHUNKS_STORE).index("recordingId");
    const request = index.openCursor(IDBKeyRange.only(recordingId));
    request.onsuccess = () => {
      const cursor = request.result;
      if (!cursor) return;
      cursor.delete();
      cursor.continue();
    };

    await done;
  }

  async function discardAutosaveRecordings() {
    const db = await openAutosaveDb();
    const transaction = db.transaction(
      [AUTOSAVE_RECORDINGS_STORE, AUTOSAVE_CHUNKS_STORE],
      "readwrite"
    );
    const done = transactionDone(transaction);
    transaction.objectStore(AUTOSAVE_RECORDINGS_STORE).clear();
    transaction.objectStore(AUTOSAVE_CHUNKS_STORE).clear();
    await done;
    activeAutosave = null;
  }

  async function recoverAutosaveRecording(recordingId) {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      throw new Error("Stop the active recording before recovering an autosave.");
    }

    await waitForAutosaveWrites();
    const recordings = await listAutosaveRecordings();
    const recording = recordingId
      ? recordings.find((candidate) => candidate.id === recordingId)
      : recordings[0];
    if (!recording) {
      throw new Error("No autosaved recording was found.");
    }

    const chunks = await getAutosaveChunks(recording.id);
    if (!chunks.length) {
      throw new Error("The autosaved recording has no saved chunks.");
    }

    const recoveredBlob = new Blob(
      chunks.map((chunk) => chunk.blob),
      { type: recording.mimeType || "video/webm" }
    );
    const baseName = String(recording.filename || "recording.webm").replace(/\.[^/.]+$/, "");
    const recoveredFilename = `${baseName}-recovered.webm`;
    await downloadBlob(recoveredBlob, recoveredFilename);
    await deleteAutosaveRecording(recording.id);

    if (activeAutosave?.id === recording.id) {
      activeAutosave = null;
    }

    return {
      filename: recoveredFilename,
      size: recoveredBlob.size,
      chunkCount: chunks.length,
    };
  }

  /**
   * Captures video stream based on the capture type (tab, window, or screen)
   */
  async function captureVideoStream(message) {
    const isTab = message.captureType === "tab";

    // For tab capture, we need a streamId
    if (isTab) {
      if (!message.streamId) {
        throw new Error("Missing streamId for tab capture");
      }

      return await navigator.mediaDevices.getUserMedia({
        audio: message.includeSystemAudio ? {
          mandatory: {
            chromeMediaSource: "tab",
            chromeMediaSourceId: message.streamId
          }
        } : false,
        video: {
          mandatory: {
            chromeMediaSource: "tab",
            chromeMediaSourceId: message.streamId
          }
        }
      });
    }

    // For screen/window capture, try getUserMedia with streamId first
    if (message.streamId) {
      try {
        console.log("[OFFSCREEN] Trying getUserMedia with desktop streamId");
        return await navigator.mediaDevices.getUserMedia({
          audio: false, // System audio not supported on macOS for desktop
          video: {
            mandatory: {
              chromeMediaSource: "desktop",
              chromeMediaSourceId: message.streamId,
              maxWidth: 1920,
              maxHeight: 1080
            }
          }
        });
      } catch (error) {
        console.warn("[OFFSCREEN] getUserMedia with streamId failed:", formatError(error));
        console.log("[OFFSCREEN] Falling back to getDisplayMedia");
      }
    }

    // Fallback: use getDisplayMedia for screen/window capture
    // This will show Chrome's native picker
    console.log("[OFFSCREEN] Using getDisplayMedia for screen capture");
    return await navigator.mediaDevices.getDisplayMedia({
      video: {
        displaySurface: message.captureType === "window" ? "window" : "monitor",
        width: { max: 1920 },
        height: { max: 1080 }
      },
      audio: false // System audio via getDisplayMedia is also limited on macOS
    });
  }

  /**
   * Captures microphone audio stream if requested
   */
  async function captureMicrophoneStream(includeMicrophone, preferredMicrophoneDeviceId) {
    if (!includeMicrophone) {
      console.log("[OFFSCREEN] Microphone not requested, skipping");
      return null;
    }

    // Check permission state first
    try {
      const permissionStatus = await navigator.permissions.query({ name: "microphone" });
      console.log("[OFFSCREEN] Microphone permission state:", permissionStatus.state);
    } catch (e) {
      console.log("[OFFSCREEN] Could not query microphone permission:", formatError(e));
    }

    try {
      const baseAudioConstraints = {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      };

      if (preferredMicrophoneDeviceId) {
        baseAudioConstraints.deviceId = { exact: preferredMicrophoneDeviceId };
      }

      const audioStream = await navigator.mediaDevices.getUserMedia({
        audio: baseAudioConstraints,
        video: false
      });
      console.log("[OFFSCREEN] Microphone stream acquired, tracks:", audioStream.getAudioTracks().length);
      
      const track = audioStream.getAudioTracks()[0];
      if (track) {
        console.log("[OFFSCREEN] Mic track state:", track.readyState, "enabled:", track.enabled, "muted:", track.muted);
      }
      
      return audioStream;
    } catch (error) {
      console.warn("[OFFSCREEN] Microphone capture with enhanced constraints failed:", formatError(error));
      try {
        const fallbackStream = await navigator.mediaDevices.getUserMedia({
          audio: preferredMicrophoneDeviceId
            ? { deviceId: { ideal: preferredMicrophoneDeviceId } }
            : true,
          video: false
        });
        console.log("[OFFSCREEN] Microphone stream acquired with fallback constraints");
        return fallbackStream;
      } catch (fallbackError) {
        console.error("[OFFSCREEN] Could not get microphone:", formatError(fallbackError));
        return null;
      }
    }
  }

  /**
   * Combines video and audio streams into a single MediaStream
   */
  function combineStreams(videoStream, audioStream) {
    // Both system audio and microphone - mix them
    if (audioStream && videoStream.getAudioTracks().length > 0) {
      mixAudioContext = new AudioContext();
      const systemAudioSource = mixAudioContext.createMediaStreamSource(videoStream);
      const micSource = mixAudioContext.createMediaStreamSource(audioStream);
      const destination = mixAudioContext.createMediaStreamDestination();

      systemAudioSource.connect(destination);
      micSource.connect(destination);

      const videoTracks = videoStream.getVideoTracks();
      const mixedAudioTracks = destination.stream.getAudioTracks();
      return new MediaStream([...videoTracks, ...mixedAudioTracks]);
    } 
    
    // Only microphone, combine with video
    if (audioStream) {
      return new MediaStream([
        ...videoStream.getVideoTracks(),
        ...audioStream.getAudioTracks()
      ]);
    } 
    
    // No additional audio, use original stream
    return videoStream;
  }

  function setupTabAudioMonitoring(videoStream, includeSystemAudio) {
    if (!includeSystemAudio) return;

    const tabAudioTracks = videoStream.getAudioTracks();
    if (!tabAudioTracks.length) return;

    monitorAudio = new Audio();
    monitorAudio.srcObject = new MediaStream(tabAudioTracks);
    monitorAudio.volume = 1;
    monitorAudio.play().catch((error) => {
      console.warn("[OFFSCREEN] Tab audio monitoring failed:", formatError(error));
    });
  }

  function cleanupResources() {
    stopLiveCaptionRecorder();
    stopLiveStream();
    try {
      if (monitorAudio) {
        monitorAudio.pause();
        monitorAudio.srcObject = null;
        monitorAudio = null;
      }
    } catch {}

    try {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    } catch {}

    try {
      if (sourceVideoStream) {
        sourceVideoStream.getTracks().forEach((track) => track.stop());
      }
    } catch {}

    try {
      if (sourceMicStream) {
        sourceMicStream.getTracks().forEach((track) => track.stop());
      }
    } catch {}

    try {
      if (mixAudioContext) {
        mixAudioContext.close();
      }
    } catch {}

    currentProcessingOptions.liveCaptionOverlay = false;
    stream = null;
    sourceVideoStream = null;
    sourceMicStream = null;
    mixAudioContext = null;
  }

  function getRecorderConfig(compressionPreset) {
    if (compressionPreset === "fast") {
      return {
        mimeType: "video/webm;codecs=vp9",
        videoBitsPerSecond: 3200000
      };
    }

    if (compressionPreset === "max") {
      return {
        mimeType: "video/webm;codecs=vp8",
        videoBitsPerSecond: 1300000
      };
    }

    return {
      mimeType: "video/webm;codecs=vp9",
      videoBitsPerSecond: 2000000
    };
  }

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

  async function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    try {
      const response = await sendRuntimeMessage({
        type: "DOWNLOAD_URL",
        url,
        filename,
      });
      if (!response?.success) {
        throw new Error(response?.error || "Download failed");
      }
      return response.downloadId;
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  async function downloadTextFile(content, filename, mimeType) {
    return await downloadBlob(new Blob([content], { type: mimeType }), filename);
  }

  async function downloadRecordingArtifacts(outputBlob, outputFilename, results) {
    const errors = [];
    let videoDownloadId = null;
    let subtitleDownloadId = null;

    try {
      videoDownloadId = await downloadBlob(outputBlob, outputFilename);
      console.log("[OFFSCREEN] Recording saved:", outputFilename);
    } catch (error) {
      const message = formatError(error);
      errors.push(`Video download failed: ${message}`);
      console.warn("[OFFSCREEN] Recording download failed:", message);
    }

    if (results.subtitleContent && results.subtitleFilename) {
      try {
        subtitleDownloadId = await downloadTextFile(
          results.subtitleContent,
          results.subtitleFilename,
          results.subtitleFormat === "vtt"
            ? "text/vtt;charset=utf-8"
            : "application/x-subrip;charset=utf-8"
        );
        console.log("[OFFSCREEN] Subtitle file saved:", results.subtitleFilename);
      } catch (error) {
        const message = formatError(error);
        errors.push(`Subtitle download failed: ${message}`);
        console.warn("[OFFSCREEN] Subtitle download failed:", message);
      }
    } else if (currentProcessingOptions.enableSubtitles) {
      console.warn("[OFFSCREEN] Subtitle generation produced no downloadable file");
    }

    if (errors.length) {
      chrome.runtime.sendMessage({
        type: "RECORDING_STATUS",
        stage: "download-error",
        tabId: activeTabId,
        error: errors.join(" | ").slice(0, 180),
      });
    }

    if (!videoDownloadId && !subtitleDownloadId) {
      throw new Error(errors.join(" | ") || "No recording artifacts were downloaded");
    }

    return { videoDownloadId, subtitleDownloadId };
  }

  async function getBackendUrl() {
    const response = await sendRuntimeMessage({
      type: "BACKEND_URL_GET",
    });
    const value = String(response?.backendUrl || "").trim();
    if (!value) {
      return null;
    }

    try {
      const parsed = new URL(value);
      if (!["http:", "https:"].includes(parsed.protocol)) {
        throw new Error("Backend URL must start with http:// or https://");
      }
      return parsed.href.replace(/\/+$/, "");
    } catch (error) {
      throw new Error(`Invalid backend URL: ${formatError(error)}`);
    }
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

  async function backendFetch(path, options = {}, timeoutMs = 180000) {
    const backendUrl = await getBackendUrl();
    if (!backendUrl) {
      throw new Error("Backend URL not configured");
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const requestUrl = buildBackendUrl(backendUrl, path);
    const { externalSignal, ...fetchOptions } = options;
    const onExternalAbort = () => controller.abort();
    if (externalSignal) {
      if (externalSignal.aborted) {
        controller.abort();
      } else {
        externalSignal.addEventListener("abort", onExternalAbort, { once: true });
      }
    }

    try {
      return await fetch(requestUrl, {
        ...fetchOptions,
        headers: backendRequestHeaders(requestUrl, options.headers),
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timer);
      if (externalSignal) {
        externalSignal.removeEventListener("abort", onExternalAbort);
      }
    }
  }

  function numericConfigValue(value, fallback, minValue = 0) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      return fallback;
    }
    return Math.max(minValue, Math.floor(parsed));
  }

  function supportsLiveCaptionOverlay(message) {
    return message.captureType === "tab" && !!message.includeSystemAudio;
  }

  async function readJsonResponse(response, label) {
    const contentType = response.headers.get("content-type") || "";
    const body = await response.text();

    if (!response.ok) {
      throw new Error(`${label} API error (${response.status}): ${body.slice(0, 180)}`);
    }

    if (!contentType.toLowerCase().includes("application/json")) {
      throw new Error(
        `${label} API returned ${contentType || "unknown content"} instead of JSON: ${body.slice(0, 80)}`
      );
    }

    try {
      return JSON.parse(body);
    } catch (error) {
      throw new Error(`${label} API returned invalid JSON: ${formatError(error)}`);
    }
  }

  function backendErrorMessage(response, body, label) {
    const contentType = (response.headers.get("content-type") || "").toLowerCase();
    const responseUrl = response.url || "the configured backend URL";
    const looksLikeHtml = contentType.includes("text/html") || /^\s*</.test(body);

    if (looksLikeHtml) {
      return (
        `${label} API unavailable (${response.status}) at ${responseUrl}. ` +
        "The request returned an HTML gateway page before it reached the API."
      );
    }

    return `${label} API error (${response.status}): ${body.slice(0, 180)}`;
  }

  function wait(delayMs) {
    return new Promise((resolve) => setTimeout(resolve, delayMs));
  }

  async function backendFetchWithRetry(
    path,
    options,
    timeoutMs,
    label,
    retryDelays = [1000, 2500]
  ) {
    for (let attempt = 0; ; attempt += 1) {
      const response = await backendFetch(path, options, timeoutMs);
      const retryable = [502, 503, 504].includes(response.status);

      if (!retryable || attempt >= retryDelays.length) {
        return response;
      }

      const body = await response.text();
      console.info(
        `[OFFSCREEN] ${label} API temporarily unavailable (${response.status}); ` +
        `retrying in ${retryDelays[attempt]}ms: ${body.slice(0, 80)}`
      );
      await wait(retryDelays[attempt]);
    }
  }

  async function loadLiveCaptionConfig() {
    liveCaptionChunkMs = DEFAULT_LIVE_CAPTION_CHUNK_MS;
    liveCaptionMinIntervalMs = DEFAULT_LIVE_CAPTION_MIN_INTERVAL_MS;
    liveCaptionMinBlobBytes = DEFAULT_LIVE_CAPTION_MIN_BLOB_BYTES;

    try {
      const response = await backendFetch(
        "/api/v1/media/live-caption/config",
        { method: "GET" },
        5000
      );

      const config = await readJsonResponse(response, "Config");
      liveCaptionChunkMs = numericConfigValue(
        config?.chunk_ms,
        DEFAULT_LIVE_CAPTION_CHUNK_MS,
        500
      );
      liveCaptionMinIntervalMs = numericConfigValue(
        config?.min_interval_ms,
        DEFAULT_LIVE_CAPTION_MIN_INTERVAL_MS,
        0
      );
      liveCaptionMinBlobBytes = numericConfigValue(
        config?.min_audio_bytes,
        DEFAULT_LIVE_CAPTION_MIN_BLOB_BYTES,
        0
      );

      const streaming = config?.streaming || {};
      streamingConfig = {
        enabled: streaming.enabled === true,
        wsPath: typeof streaming.ws_path === "string" && streaming.ws_path
          ? streaming.ws_path
          : "/api/v1/media/live-stream",
        frameMs: numericConfigValue(streaming.frame_ms, 80, 20),
      };

      console.log("[OFFSCREEN] Live caption config:", {
        backend: config?.backend,
        backends: config?.backends,
        chunkMs: liveCaptionChunkMs,
        minIntervalMs: liveCaptionMinIntervalMs,
        minBlobBytes: liveCaptionMinBlobBytes,
        streaming: streamingConfig
      });
    } catch (error) {
      console.info("[OFFSCREEN] Using default live caption config:", formatError(error));
    }
  }

  async function requestCaptionPackage(blob, filename) {
    const formData = new FormData();
    formData.append("file", blob, filename);
    formData.append("source_language", currentProcessingOptions.sourceLanguage || "en");
    formData.append("target_language", currentProcessingOptions.targetLanguage);
    formData.append(
      "subtitle_format",
      currentProcessingOptions.subtitleFormat || "srt"
    );
    formData.append(
      "translate",
      currentProcessingOptions.enableTranslation ? "true" : "false"
    );

    const response = await backendFetchWithRetry(
      "/api/v1/media/captions",
      {
        method: "POST",
        body: formData,
      },
      300000,
      "Caption"
    );

    if (!response.ok) {
      const body = await response.text();
      throw new Error(backendErrorMessage(response, body, "Caption"));
    }

    return readJsonResponse(response, "Caption");
  }

  async function requestBurnedVideo(blob, filename, subtitleContent, subtitleFormat) {
    const formData = new FormData();
    formData.append("file", blob, filename);
    formData.append("subtitle_content", subtitleContent);
    formData.append("subtitle_format", subtitleFormat || "srt");
    formData.append("output_ext", "mp4");

    const response = await backendFetchWithRetry(
      "/api/v1/media/burn-subtitles",
      {
        method: "POST",
        body: formData,
      },
      300000,
      "Burn subtitles"
    );

    if (!response.ok) {
      const body = await response.text();
      throw new Error(backendErrorMessage(response, body, "Burn subtitles"));
    }

    return response.blob();
  }

  function emitLiveCaption(caption) {
    if (!currentProcessingOptions.liveCaptionOverlay) {
      return;
    }

    liveCaptionEverUpdated = true;
    if (!caption) {
      chrome.runtime.sendMessage({
        type: "RECORDING_STATUS",
        stage: "live-caption-empty",
        tabId: activeTabId,
      });
      return;
    }

    chrome.runtime.sendMessage({
      type: "RECORDING_STATUS",
      stage: "live-caption",
      tabId: activeTabId,
      caption,
    });
  }

  async function requestLiveCaption(chunkBlob, signal) {
    const formData = new FormData();
    formData.append("file", chunkBlob, `chunk-${Date.now()}.webm`);
    formData.append("source_language", currentProcessingOptions.sourceLanguage || "en");
    formData.append("target_language", currentProcessingOptions.targetLanguage);
    formData.append(
      "translate",
      currentProcessingOptions.enableTranslation ? "true" : "false"
    );

    const response = await backendFetch(
      "/api/v1/media/live-caption",
      {
        method: "POST",
        body: formData,
        externalSignal: signal,
      },
      LIVE_CAPTION_TIMEOUT_MS
    );

    if (!response.ok) {
      const body = await response.text();
      throw new Error(`Live caption API error (${response.status}): ${body.slice(0, 180)}`);
    }

    const payload = await readJsonResponse(response, "Live caption");
    return String(payload?.caption_text || "").trim();
  }

  function resetLiveCaptionPipeline() {
    for (const entry of liveCaptionInflight.values()) {
      try {
        entry.controller.abort();
      } catch (error) {}
    }
    liveCaptionSeq = 0;
    liveCaptionNextDisplaySeq = 1;
    liveCaptionPending = [];
    liveCaptionInflight = new Map();
    liveCaptionCompleted = new Map();
    liveCaptionSession += 1;
    liveCaptionPlaceholderShown = false;
  }

  function enqueueLiveCaptionChunk(chunkBlob) {
    void gateAndQueueLiveCaptionChunk(chunkBlob);
  }

  async function gateAndQueueLiveCaptionChunk(chunkBlob) {
    if (!currentProcessingOptions.liveCaptionOverlay) {
      return;
    }

    if (chunkBlob.size < liveCaptionMinBlobBytes) {
      return;
    }

    const now = Date.now();
    if (now - lastLiveCaptionRequestAt < liveCaptionMinIntervalMs) {
      return;
    }

    const chunkBuffer = await chunkBlob.arrayBuffer();
    const bytes = new Uint8Array(chunkBuffer);
    let hash = 2166136261;
    const stride = Math.max(1, Math.floor(bytes.length / 2048));
    for (let i = 0; i < bytes.length; i += stride) {
      hash ^= bytes[i];
      hash = Math.imul(hash, 16777619);
    }
    const fingerprint = `${chunkBlob.size}:${hash >>> 0}`;
    if (fingerprint === lastLiveCaptionFingerprint) {
      return;
    }

    lastLiveCaptionFingerprint = fingerprint;
    lastLiveCaptionRequestAt = now;

    if (!liveCaptionEverUpdated) {
      liveCaptionPlaceholderShown = true;
      emitLiveCaption("Transcribing audio…");
    }

    liveCaptionSeq += 1;
    liveCaptionPending.push({
      seq: liveCaptionSeq,
      blob: chunkBlob,
      enqueuedAt: Date.now(),
    });
    if (liveCaptionPending.length > LIVE_CAPTION_MAX_QUEUE) {
      const dropped = liveCaptionPending.shift();
      // Fill the hole so ordered display never waits on a dropped chunk.
      if (dropped.seq >= liveCaptionNextDisplaySeq) {
        liveCaptionCompleted.set(dropped.seq, "");
      }
      console.info("[OFFSCREEN] Live caption queue full; dropped oldest chunk");
    }
    pumpLiveCaptionQueue();
  }

  function pumpLiveCaptionQueue() {
    if (!currentProcessingOptions.liveCaptionOverlay) {
      return;
    }
    while (
      liveCaptionInflight.size < LIVE_CAPTION_MAX_INFLIGHT &&
      liveCaptionPending.length > 0
    ) {
      startLiveCaptionRequest(liveCaptionPending.shift());
    }
  }

  function startLiveCaptionRequest(item) {
    const session = liveCaptionSession;
    const controller = new AbortController();
    liveCaptionInflight.set(item.seq, { controller, startedAt: Date.now() });
    const requestStarted = performance.now();
    requestLiveCaption(item.blob, controller.signal).then(
      (caption) => {
        if (session !== liveCaptionSession) {
          return;
        }
        const entry = liveCaptionInflight.get(item.seq);
        liveCaptionInflight.delete(item.seq);
        if ((entry && entry.skipped) || item.seq < liveCaptionNextDisplaySeq) {
          return;
        }
        if (!currentProcessingOptions.liveCaptionOverlay) {
          return;
        }

        liveCaptionTimeoutStreak = 0;
        const elapsedMs = Math.round(performance.now() - requestStarted);
        console.log("[OFFSCREEN] Live caption response:", {
          elapsedMs,
          blobBytes: item.blob.size,
          captionChars: caption.length
        });

        liveCaptionCompleted.set(item.seq, caption);
        drainOrderedLiveCaptions();
        pumpLiveCaptionQueue();
      },
      (error) => {
        if (session !== liveCaptionSession) {
          return;
        }
        const entry = liveCaptionInflight.get(item.seq);
        liveCaptionInflight.delete(item.seq);
        if ((entry && entry.skipped) || item.seq < liveCaptionNextDisplaySeq) {
          return;
        }
        // A failed chunk renders as silence: skip it so ordering keeps moving.
        liveCaptionCompleted.set(item.seq, "");
        if (isAbortError(error)) {
          liveCaptionTimeoutStreak += 1;
          console.info("[OFFSCREEN] Live caption chunk timed out; recording continues");
          if (
            currentProcessingOptions.liveCaptionOverlay &&
            liveCaptionTimeoutStreak >= LIVE_CAPTION_SLOW_AFTER_TIMEOUTS
          ) {
            liveCaptionEverUpdated = true;
            chrome.runtime.sendMessage({
              type: "RECORDING_STATUS",
              stage: "live-caption-error",
              tabId: activeTabId,
              error: "backend is slow or warming up (requests timing out)",
            });
          }
        } else if (currentProcessingOptions.liveCaptionOverlay) {
          const formatted = formatError(error);
          console.warn("[OFFSCREEN] Live caption chunk failed:", formatted);
          const errorAt = Date.now();
          if (errorAt - lastLiveCaptionErrorAt >= LIVE_CAPTION_ERROR_THROTTLE_MS) {
            lastLiveCaptionErrorAt = errorAt;
            liveCaptionEverUpdated = true;
            chrome.runtime.sendMessage({
              type: "RECORDING_STATUS",
              stage: "live-caption-error",
              tabId: activeTabId,
              error: formatted.slice(0, 180),
            });
          }
        }
        drainOrderedLiveCaptions();
        pumpLiveCaptionQueue();
      }
    );
  }

  function abortLiveCaptionRequest(seq) {
    const entry = liveCaptionInflight.get(seq);
    if (!entry) {
      return;
    }
    entry.skipped = true;
    try {
      entry.controller.abort();
    } catch (error) {}
  }

  function clearLiveCaptionPlaceholder() {
    if (!liveCaptionPlaceholderShown) {
      return;
    }
    liveCaptionPlaceholderShown = false;
    emitLiveCaption("");
  }

  function drainOrderedLiveCaptions() {
    for (;;) {
      const head = liveCaptionNextDisplaySeq;
      if (liveCaptionCompleted.has(head)) {
        const caption = liveCaptionCompleted.get(head);
        liveCaptionCompleted.delete(head);
        liveCaptionNextDisplaySeq = head + 1;
        if (caption) {
          liveCaptionPlaceholderShown = false;
          emitLiveCaption(caption);
        } else {
          // Empty result (silence, VAD, backend hiccup): keep the last
          // caption on screen instead of blinking the overlay.
          clearLiveCaptionPlaceholder();
        }
        continue;
      }
      if (liveCaptionPending.some((item) => item.seq === head)) {
        pumpLiveCaptionQueue();
        return;
      }
      const inflight = liveCaptionInflight.get(head);
      if (!inflight) {
        // Head was dropped or never started; skip past it.
        liveCaptionNextDisplaySeq = head + 1;
        continue;
      }
      let newerWaiting = false;
      for (const seq of liveCaptionCompleted.keys()) {
        if (seq > head) {
          newerWaiting = true;
          break;
        }
      }
      if (
        newerWaiting &&
        Date.now() - inflight.startedAt >= LIVE_CAPTION_SKIP_AHEAD_MS
      ) {
        console.info("[OFFSCREEN] Live caption skipping stalled chunk");
        abortLiveCaptionRequest(head);
        liveCaptionNextDisplaySeq = head + 1;
        continue;
      }
      return;
    }
  }

  async function waitForLiveCaptionIdle(timeoutMs = 1500) {
    const startedAt = Date.now();
    while (liveCaptionPending.length > 0 || liveCaptionInflight.size > 0) {
      if (Date.now() - startedAt >= timeoutMs) {
        console.info("[OFFSCREEN] Continuing finalization with live caption request still pending");
        return;
      }
      await wait(50);
    }
  }

  async function postProcessRecording(blob, filename) {
    const results = {
      translation: null,
      subtitleContent: null,
      subtitleFilename: null,
      burnedVideoBlob: null,
      burnedVideoFilename: null,
      subtitleFormat: null,
      driveFile: null,
    };

    if (currentProcessingOptions.enableSubtitles) {
      try {
        const subtitleOnly = !currentProcessingOptions.enableTranslation;
        const subtitleNote = subtitleOnly ? "" : " with translation";
        console.log(
          `[OFFSCREEN] Subtitles requested${subtitleNote} for ${currentProcessingOptions.targetLanguage}`
        );

        chrome.runtime.sendMessage({
          type: "RECORDING_STATUS",
          stage: "translation-started",
          subtitleOnly,
          tabId: activeTabId,
          filename,
        });

        const captionPayload = await requestCaptionPackage(blob, filename);
        results.translation = captionPayload.translated_text || "";
        results.subtitleContent = captionPayload.subtitle_content || "";
        if (!results.subtitleContent.trim()) {
          throw new Error("Caption API returned no subtitle content");
        }
        const subtitleFormat = (captionPayload.subtitle_format || "srt").toLowerCase();
        const baseName = filename.replace(/\.[^/.]+$/, "");
        results.subtitleFormat = subtitleFormat;
        results.subtitleFilename = `${baseName}.${subtitleFormat}`;

        chrome.runtime.sendMessage({
          type: "RECORDING_STATUS",
          stage: "translation-completed",
          subtitleOnly,
          tabId: activeTabId,
          filename,
          translation: (results.translation || "").slice(0, 220),
        });

        if (currentProcessingOptions.burnSubtitles && results.subtitleContent) {
          try {
            const burned = await requestBurnedVideo(
              blob,
              filename,
              results.subtitleContent,
              subtitleFormat
            );
            results.burnedVideoBlob = burned;
            results.burnedVideoFilename = `${baseName}-captioned.mp4`;
          } catch (error) {
            console.warn("[OFFSCREEN] Subtitle burn failed:", formatError(error));
            chrome.runtime.sendMessage({
              type: "RECORDING_STATUS",
              stage: "burn-error",
              tabId: activeTabId,
              error: formatError(error).slice(0, 180),
            });
          }
        }
      } catch (error) {
        console.warn("[OFFSCREEN] Translation error:", formatError(error));
        const subtitleOnly = !currentProcessingOptions.enableTranslation;
        chrome.runtime.sendMessage({
          type: "RECORDING_STATUS",
          stage: "translation-error",
          subtitleOnly,
          tabId: activeTabId,
          error: formatError(error).slice(0, 180),
        });
      }
    }

    return results;
  }

  /**
   * Initializes and starts the MediaRecorder
   */
  async function setupRecorder(mediaStream) {
    recordedChunks = [];
    resetLiveCaptionPipeline();
    const recorderConfig = getRecorderConfig(currentProcessingOptions.compressionPreset);
    mediaRecorder = new MediaRecorder(mediaStream, recorderConfig);
    const filename = `recording-${Date.now()}.webm`;

    try {
      activeAutosave = await createAutosaveSession(
        filename,
        mediaRecorder.mimeType || recorderConfig.mimeType || "video/webm"
      );
      console.log("[OFFSCREEN] Recording autosave started:", activeAutosave.id);
    } catch (error) {
      activeAutosave = null;
      console.warn("[OFFSCREEN] Recording autosave unavailable:", formatError(error));
    }

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        recordedChunks.push(event.data);
        saveAutosaveChunk(event.data);
      }
    };

    mediaRecorder.onerror = (event) => {
      const error = event.error || event;
      console.error("[OFFSCREEN] MediaRecorder error:", formatError(error));
      try {
        if (mediaRecorder && mediaRecorder.state !== "inactive") {
          mediaRecorder.requestData();
          mediaRecorder.stop();
        }
      } catch (stopError) {
        console.warn("[OFFSCREEN] MediaRecorder emergency stop failed:", formatError(stopError));
      }
    };

    mediaRecorder.onstop = async () => {
      await waitForAutosaveWrites();
      const blob = new Blob(recordedChunks, {
        type: mediaRecorder?.mimeType || activeAutosave?.mimeType || "video/webm",
      });
      let outputBlob = blob;
      let outputFilename = filename;

      chrome.runtime.sendMessage({
        type: "RECORDING_STATUS",
        stage: "overlay-clear",
        tabId: activeTabId,
      });

      try {
        try {
          await waitForLiveCaptionIdle();
        } catch (error) {
          console.warn("[OFFSCREEN] Live caption queue drain failed:", formatError(error));
        }

        const results = await postProcessRecording(blob, filename);
        if (results.burnedVideoBlob) {
          outputBlob = results.burnedVideoBlob;
          outputFilename = results.burnedVideoFilename || outputFilename;
        }

        const downloads = await downloadRecordingArtifacts(outputBlob, outputFilename, results);
        if (downloads.videoDownloadId && activeAutosave) {
          await deleteAutosaveRecording(activeAutosave.id);
          console.log("[OFFSCREEN] Recording autosave cleared:", activeAutosave.id);
          activeAutosave = null;
        }

        if (currentProcessingOptions.enableDriveUpload) {
          try {
            chrome.runtime.sendMessage({
              type: "RECORDING_STATUS",
              stage: "upload-started",
              tabId: activeTabId,
              filename: outputFilename,
            });

            const response = await sendRuntimeMessage({
              type: "UPLOAD_TO_DRIVE",
              blob: outputBlob,
              filename: outputFilename,
              mimeType: outputBlob.type || "video/webm",
            });

            if (!response?.success) {
              throw new Error(response?.error || "Drive upload failed");
            }

            chrome.runtime.sendMessage({
              type: "RECORDING_STATUS",
              stage: "upload-completed",
              filename: outputFilename,
              tabId: activeTabId,
              file: response.file,
            });
          } catch (error) {
            console.warn("[OFFSCREEN] Drive upload failed:", formatError(error));
            chrome.runtime.sendMessage({
              type: "RECORDING_STATUS",
              stage: "upload-error",
              tabId: activeTabId,
              error: formatError(error).slice(0, 180),
            });
          }
        }
      } catch (error) {
        console.error("[OFFSCREEN] Recording finalization failed:", formatError(error));
        chrome.runtime.sendMessage({
          type: "RECORDING_STATUS",
          stage: "download-error",
          tabId: activeTabId,
          error: formatError(error).slice(0, 180),
        });
      } finally {
        cleanupResources();
        mediaRecorder = null;
        recordedChunks = [];
      }
    };

    mediaStream.getTracks().forEach((track) => {
      track.addEventListener("ended", () => {
        if (mediaRecorder && mediaRecorder.state === "recording") {
          console.info("[OFFSCREEN] Capture track ended; finalizing recording");
          try {
            currentProcessingOptions.liveCaptionOverlay = false;
            stopLiveCaptionRecorder();
            stopLiveStream();
            mediaRecorder.requestData();
            mediaRecorder.stop();
          } catch (error) {
            console.warn("[OFFSCREEN] Track-ended stop failed:", formatError(error));
          }
        }
      });
    });

    mediaRecorder.start(RECORDING_CHUNK_MS);
    console.log("[OFFSCREEN] MediaRecorder started");
  }

  function stopLiveCaptionRecorder() {
    if (liveCaptionTimer) {
      clearTimeout(liveCaptionTimer);
      liveCaptionTimer = null;
    }

    if (liveCaptionRecorder && liveCaptionRecorder.state !== "inactive") {
      const recorder = liveCaptionRecorder;
      liveCaptionRecorder = null;
      try {
        recorder.stop();
      } catch (error) {
        console.warn("[OFFSCREEN] Live caption recorder stop failed:", formatError(error));
      }
    } else {
      liveCaptionRecorder = null;
    }

    liveCaptionChunks = [];
    lastLiveCaptionRequestAt = 0;
    lastLiveCaptionFingerprint = "";
    lastLiveCaptionErrorAt = 0;
    liveCaptionTimeoutStreak = 0;
    liveCaptionEverUpdated = false;
    resetLiveCaptionPipeline();
  }

  function liveAudioOnlyStream() {
    try {
      if (typeof stream !== "undefined" && stream && stream.getAudioTracks().length > 0) {
        return new MediaStream(stream.getAudioTracks());
      }
    } catch (error) {
      console.warn("[OFFSCREEN] Audio-only stream failed:", formatError(error));
    }
    return null;
  }

  function buildBackendWsBase(httpBase) {
    const url = new URL(httpBase);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    url.search = "";
    url.hash = "";
    return url.toString().replace(/\/+$/, "");
  }

  function stopLiveStream() {
    liveStreamActive = false;

    if (liveStreamSession) {
      const session = liveStreamSession;
      liveStreamSession = null;
      try {
        session.end();
        session.close();
      } catch (error) {
        console.warn("[OFFSCREEN] Live stream session close failed:", formatError(error));
      }
    }

    if (liveStreamWorkletNode) {
      try {
        liveStreamWorkletNode.port.postMessage({ type: "stop" });
        liveStreamWorkletNode.disconnect();
      } catch (error) {}
      liveStreamWorkletNode = null;
    }

    if (liveStreamAudioContext) {
      const context = liveStreamAudioContext;
      liveStreamAudioContext = null;
      context.close().catch(() => {});
    }
  }

  function describeLiveCaptionChunk() {
    const seconds = liveCaptionChunkMs / 1000;
    const label = Number.isInteger(seconds) ? String(seconds) : seconds.toFixed(1);
    return `${label}-second chunks`;
  }

  function downgradeToChunked(reason) {
    if (liveStreamDowngraded) {
      return;
    }
    liveStreamDowngraded = true;
    console.info("[OFFSCREEN] Live stream downgrading to chunked captions:", reason || "unavailable");
    stopLiveStream();

    if (!currentProcessingOptions.liveCaptionOverlay) {
      return;
    }
    emitLiveCaption(
      `Realtime captions unavailable; using ${describeLiveCaptionChunk()} (${
        String(reason || "unavailable").slice(0, 80)
      })`
    );
    const audioStream = liveAudioOnlyStream();
    if (audioStream) {
      startLiveCaptionRecorder(audioStream);
    }
  }

  async function startLiveStream() {
    if (liveStreamActive) {
      return;
    }
    liveStreamDowngraded = false;

    const audioStream = liveAudioOnlyStream();
    if (!audioStream) {
      console.info("[OFFSCREEN] Live stream needs audio input; using chunked recorder");
      startLiveCaptionRecorder(new MediaStream());
      return;
    }

    if (typeof TransmeetLiveStream === "undefined") {
      console.warn("[OFFSCREEN] Live stream client missing; using chunked recorder");
      downgradeToChunked("client missing");
      return;
    }

    let backendUrl;
    try {
      backendUrl = await getBackendUrl();
    } catch (error) {
      console.warn("[OFFSCREEN] Live stream backend URL failed:", formatError(error));
      downgradeToChunked("backend url");
      return;
    }
    if (!backendUrl) {
      downgradeToChunked("backend url");
      return;
    }

    liveStreamActive = true;
    try {
      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(audioStream);
      await audioContext.audioWorklet.addModule(
        chrome.runtime.getURL("audio-worklet/pcm-tap.js")
      );
      const workletNode = new AudioWorkletNode(audioContext, "pcm-tap");
      const silence = audioContext.createGain();
      silence.gain.value = 0;
      source.connect(workletNode);
      workletNode.connect(silence);
      silence.connect(audioContext.destination);

      const session = TransmeetLiveStream.createLiveStreamSession({
        wsBase: buildBackendWsBase(backendUrl),
        wsPath: streamingConfig.wsPath,
        sourceLanguage: currentProcessingOptions.sourceLanguage || "auto",
        targetLanguage: currentProcessingOptions.targetLanguage || "en",
        translate: !!currentProcessingOptions.enableTranslation,
        onCaption: (text) => {
          if (!liveStreamActive) {
            return;
          }
          if (!text) {
            return;
          }
          emitLiveCaption(text);
        },
        onDowngrade: (reason) => {
          if (!liveStreamActive) {
            return;
          }
          downgradeToChunked(reason);
        },
        onError: (error) => {
          if (!liveStreamActive) {
            return;
          }
          console.warn("[OFFSCREEN] Live stream error:", formatError(error));
          downgradeToChunked(formatError(error));
        },
      });

      workletNode.port.onmessage = (event) => {
        const data = event.data || {};
        if (data.type === "pcm" && data.buffer) {
          session.sendPcm(data.buffer);
        }
      };
      workletNode.port.postMessage({
        type: "config",
        targetRate: 16000,
        frameSamples: Math.max(320, 16 * streamingConfig.frameMs),
      });

      liveStreamAudioContext = audioContext;
      liveStreamWorkletNode = workletNode;
      liveStreamSession = session;

      await session.ready;
      console.log("[OFFSCREEN] Live stream session open");
    } catch (error) {
      console.warn("[OFFSCREEN] Live stream start failed:", formatError(error));
      downgradeToChunked(formatError(error));
    }
  }

  function startLiveCaptionRecorder(audioStream) {
    if (!currentProcessingOptions.liveCaptionOverlay) {
      return;
    }

    if (!audioStream || audioStream.getAudioTracks().length === 0) {
      chrome.runtime.sendMessage({
        type: "RECORDING_STATUS",
        stage: "live-caption",
        tabId: activeTabId,
        caption: "Live captions need audio input",
      });
      return;
    }

    stopLiveCaptionRecorder();

    let recorder;
    try {
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "";
      recorder = new MediaRecorder(
        audioStream,
        mimeType ? { mimeType } : undefined
      );
    } catch (error) {
      console.warn("[OFFSCREEN] Live caption recorder init failed:", formatError(error));
      return;
    }

    liveCaptionRecorder = recorder;
    liveCaptionChunks = [];

    function scheduleLiveCaptionStop() {
      if (liveCaptionTimer) {
        clearTimeout(liveCaptionTimer);
      }
      liveCaptionTimer = setTimeout(() => {
        if (recorder.state === "recording") {
          try {
            recorder.stop();
          } catch (error) {
            console.warn("[OFFSCREEN] Live caption recorder stop error:", formatError(error));
          }
        }
      }, liveCaptionChunkMs);
    }

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        liveCaptionChunks.push(event.data);
      }
    };

    recorder.onstop = () => {
      if (recorder !== liveCaptionRecorder) {
        return;
      }

      const mimeType = recorder.mimeType || "audio/webm";
      const blob = new Blob(liveCaptionChunks, { type: mimeType });
      liveCaptionChunks = [];

      if (blob.size > 0) {
        enqueueLiveCaptionChunk(blob);
      }

      if (liveCaptionRecorder && currentProcessingOptions.liveCaptionOverlay) {
        try {
          recorder.start();
          scheduleLiveCaptionStop();
        } catch (error) {
          console.warn("[OFFSCREEN] Live caption recorder restart failed:", formatError(error));
        }
      }
    };

    try {
      recorder.start();
      scheduleLiveCaptionStop();
    } catch (error) {
      console.warn("[OFFSCREEN] Live caption recorder start failed:", formatError(error));
    }
  }

  /**
   * Handles the complete recording start process
   */
  async function beginCapture(message) {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      stopRecording();
    }

    cleanupResources();
    lastLiveCaptionRequestAt = 0;
    lastLiveCaptionFingerprint = "";
    liveCaptionTimeoutStreak = 0;
    liveCaptionEverUpdated = false;
    activeTabId = message.tabId ?? null;
    const liveCaptionOverlay =
      !!message.processingOptions?.liveCaptionOverlay &&
      supportsLiveCaptionOverlay(message);

    if (message.processingOptions?.liveCaptionOverlay && !liveCaptionOverlay) {
      console.info(
        "[OFFSCREEN] Live caption overlay disabled for unsupported capture mode"
      );
    }

    currentProcessingOptions = {
      enableSubtitles: !!message.processingOptions?.enableSubtitles,
      enableTranslation: !!message.processingOptions?.enableTranslation,
      sourceLanguage: message.processingOptions?.sourceLanguage || "en",
      targetLanguage: message.processingOptions?.targetLanguage || "en",
      subtitleFormat: message.processingOptions?.subtitleFormat || "srt",
      burnSubtitles: message.processingOptions?.burnSubtitles !== false,
      liveCaptionOverlay,
      compressionPreset: message.processingOptions?.compressionPreset || "balanced",
      enableDriveUpload: !!message.processingOptions?.enableDriveUpload
    };

    console.log("[OFFSCREEN] beginCapture called with:", {
      captureType: message.captureType,
      includeSystemAudio: message.includeSystemAudio,
      includeMicrophone: message.includeMicrophone,
      hasStreamId: !!message.streamId,
      processingOptions: currentProcessingOptions
    });

    if (currentProcessingOptions.liveCaptionOverlay) {
      await loadLiveCaptionConfig();
    }

    const videoStream = await captureVideoStream(message);
    sourceVideoStream = videoStream;
    console.log("[OFFSCREEN] Video stream acquired - audio tracks:", videoStream.getAudioTracks().length, "video tracks:", videoStream.getVideoTracks().length);

    const audioStream = await captureMicrophoneStream(
      message.includeMicrophone,
      message.preferredMicrophoneDeviceId
    );
    sourceMicStream = audioStream;
    console.log("[OFFSCREEN] Microphone stream:", audioStream ? "acquired (" + audioStream.getAudioTracks().length + " tracks)" : "not requested or failed");
    
    stream = combineStreams(videoStream, audioStream);
    console.log("[OFFSCREEN] Final stream - audio tracks:", stream.getAudioTracks().length, "video tracks:", stream.getVideoTracks().length);

    if (currentProcessingOptions.enableSubtitles && stream.getAudioTracks().length === 0) {
      console.warn("[OFFSCREEN] Subtitles disabled because the recording stream has no audio tracks");
      currentProcessingOptions.enableSubtitles = false;
      currentProcessingOptions.enableTranslation = false;
      currentProcessingOptions.burnSubtitles = false;
      chrome.runtime.sendMessage({
        type: "RECORDING_STATUS",
        stage: "translation-error",
        subtitleOnly: true,
        tabId: activeTabId,
        error: "Subtitles need captured audio",
      });
    }

    setupTabAudioMonitoring(videoStream, message.includeSystemAudio);

    await setupRecorder(stream);
    if (currentProcessingOptions.liveCaptionOverlay) {
      if (streamingConfig.enabled) {
        startLiveStream();
      } else {
        startLiveCaptionRecorder(new MediaStream(stream.getAudioTracks()));
      }
    }
  }

  /**
   * Stops the recording and cleans up resources
   */
  function stopRecording() {
    currentProcessingOptions.liveCaptionOverlay = false;
    stopLiveCaptionRecorder();
    stopLiveStream();

    chrome.runtime.sendMessage({
      type: "RECORDING_STATUS",
      stage: "overlay-clear",
      tabId: activeTabId,
    });

    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      try {
        mediaRecorder.requestData();
      } catch (error) {
        console.warn("[OFFSCREEN] Final recording chunk request failed:", formatError(error));
      }
      mediaRecorder.stop();
      console.log("[OFFSCREEN] Recording stopped");
      return;
    }

    cleanupResources();
    mediaRecorder = null;
  }

  // Message listener - the public interface
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "BEGIN_CAPTURE") {
      console.log("[OFFSCREEN] Starting capture");

      (async () => {
        try {
          await beginCapture(message);
          sendResponse({ success: true });
        } catch (error) {
          console.error("[OFFSCREEN] Error:", formatError(error));
          sendResponse({ success: false, error: error.message || String(error) });
        }
      })();

      return true; // Keep message channel open for async response
    }

    if (message.type === "STOP_RECORDING") {
      stopRecording();
      sendResponse({ success: true });
      return true;
    }

    if (message.type === "AUTOSAVE_STATUS") {
      (async () => {
        try {
          await waitForAutosaveWrites();
          const recordings = await listAutosaveRecordings();
          sendResponse({
            success: true,
            hasRecoverable: recordings.length > 0,
            recording: recordings[0] || null,
            count: recordings.length,
            recordingActive: !!(mediaRecorder && mediaRecorder.state !== "inactive"),
          });
        } catch (error) {
          sendResponse({ success: false, error: error.message || String(error) });
        }
      })();

      return true;
    }

    if (message.type === "RECOVER_AUTOSAVE") {
      (async () => {
        try {
          const recovered = await recoverAutosaveRecording(message.recordingId || null);
          sendResponse({ success: true, ...recovered });
        } catch (error) {
          sendResponse({ success: false, error: error.message || String(error) });
        }
      })();

      return true;
    }

    if (message.type === "DISCARD_AUTOSAVE") {
      (async () => {
        try {
          await discardAutosaveRecordings();
          sendResponse({ success: true });
        } catch (error) {
          sendResponse({ success: false, error: error.message || String(error) });
        }
      })();

      return true;
    }
  });
})();
