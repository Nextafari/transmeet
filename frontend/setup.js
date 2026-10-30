const BACKEND_URL_KEY = "backendUrl";

function normalizeBackendUrl(rawUrl) {
  const value = String(rawUrl || "").trim();
  if (!value) {
    throw new Error("Please enter a backend URL");
  }

  const parsed = new URL(value);
  if (!["http:", "https:"].includes(parsed.protocol)) {
    throw new Error("Backend URL must start with http:// or https://");
  }

  return parsed.href.replace(/\/+$/, "");
}

function backendRequestHeaders(url) {
  const headers = new Headers();
  const hostname = new URL(url).hostname.toLowerCase();
  if (hostname.endsWith(".ngrok-free.dev") ||
      hostname.endsWith(".ngrok-free.app") ||
      hostname.endsWith(".ngrok.io")) {
    headers.set("ngrok-skip-browser-warning", "1");
  }
  return headers;
}

async function readHealthResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  const body = await response.text();

  if (!response.ok) {
    throw new Error(`Server returned ${response.status}: ${body.slice(0, 120)}`);
  }

  if (!contentType.toLowerCase().includes("application/json")) {
    throw new Error(
      `Backend health returned ${contentType || "unknown content"} instead of JSON`
    );
  }

  const payload = JSON.parse(body);
  if (payload?.status !== "ok") {
    throw new Error("Backend health response did not report status ok");
  }
}

async function verifyBackendUrl(url) {
  const healthUrl = `${url}/api/v1/health/`;
  const response = await fetch(healthUrl, {
    method: 'GET',
    headers: backendRequestHeaders(healthUrl),
  });
  await readHealthResponse(response);
}

function showMessage(elementId, show = true) {
  const el = document.getElementById(elementId);
  if (el) el.style.display = show ? 'block' : 'none';
}

function setMessageText(elementId, text) {
  const el = document.getElementById(elementId);
  if (el) el.textContent = text;
}

async function loadBackendUrl() {
  const stored = await chrome.storage.local.get(BACKEND_URL_KEY);
  if (stored[BACKEND_URL_KEY]) {
    document.getElementById('backendUrl').value = stored[BACKEND_URL_KEY];
  }
}

async function testBackendConnection() {
  try {
    const url = normalizeBackendUrl(document.getElementById('backendUrl').value);
    await verifyBackendUrl(url);
    document.getElementById('backendUrl').value = url;
    setMessageText('backendSuccessMsg', 'Backend connection successful!');
    showMessage('backendSuccessMsg', true);
    showMessage('backendErrorMsg', false);
  } catch (error) {
    setMessageText('backendErrorMsg', `Connection failed: ${error.message}`);
    showMessage('backendErrorMsg', true);
    showMessage('backendSuccessMsg', false);
  }
}

async function saveBackendUrl() {
  try {
    const url = normalizeBackendUrl(document.getElementById('backendUrl').value);
    await verifyBackendUrl(url);
    await chrome.storage.local.set({ [BACKEND_URL_KEY]: url });
    document.getElementById('backendUrl').value = url;
    setMessageText('backendSuccessMsg', 'Backend URL saved successfully!');
    showMessage('backendSuccessMsg', true);
    showMessage('backendErrorMsg', false);
  } catch (error) {
    setMessageText('backendErrorMsg', `Save failed: ${error.message}`);
    showMessage('backendErrorMsg', true);
  }
}

document.getElementById('requestBtn').addEventListener('click', async () => {
    const btn = document.getElementById('requestBtn');
  
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(track => track.stop());
      
      btn.style.display = 'none';
      showMessage('micSuccessMsg', true);
      showMessage('micErrorMsg', false);

    } catch (err) {
      console.error(err);
      showMessage('micErrorMsg', true);
      setMessageText('micErrorMsg', "Error: " + err.message + ". Please ensure you click 'Allow' when prompted.");
    }
});

document.getElementById('testBackendBtn').addEventListener('click', testBackendConnection);
document.getElementById('saveBackendBtn').addEventListener('click', saveBackendUrl);

loadBackendUrl();
