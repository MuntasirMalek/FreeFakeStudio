/**
 * FreeFakeStudio Web — Connector & Studio Controller
 */

(function () {
  // Elements
  const launcherView = document.getElementById('launcher-view');
  const studioView = document.getElementById('studio-view');
  const connectionStatus = document.getElementById('connection-status');
  const statusText = document.getElementById('status-text');
  const disconnectBtn = document.getElementById('disconnect-btn');
  const backendDisplay = document.getElementById('backend-display');
  const studioIframe = document.getElementById('studio-iframe');
  const frameLoader = document.getElementById('frame-loading');
  const loaderStatus = document.getElementById('loader-status');
  const openNewtabBtn = document.getElementById('open-newtab-btn');
  const switchEngineBtn = document.getElementById('switch-engine-btn');

  const modelSelector = document.getElementById('model-selector');
  const modelHint = document.getElementById('model-hint');
  const colabLaunchBtn = document.getElementById('colab-launch-btn');
  const manualUrlInput = document.getElementById('manual-url');
  const manualConnectBtn = document.getElementById('manual-connect-btn');

  const STORAGE_KEY = 'freefake_backend_url';

  // Hints per model
  const MODEL_HINTS = {
    z_image: "⚡ Recommended for first-time users. Takes only ~1.5 minutes to boot on Colab (8 steps).",
    flux_4b: "🌊 Photorealistic all-rounder. Great for text-to-image and inpainting (~11 GB download).",
    flux_9b: "🔮 Ultra detailed high-resolution FP8 model (~13 GB download).",
    qwen: "🎨 Conversational AI image editing. Describe changes in natural language (~15 GB download).",
    ernie: "🖌️ Baidu's turbo text-to-image model. Strong text rendering (~9 GB download)."
  };

  // Base Colab URL
  const COLAB_NOTEBOOK_URL = "https://colab.research.google.com/github/MuntasirMalek/FreeFakeStudio/blob/main/FreeFakeStudio_1Click.ipynb";

  // Initialize
  function init() {
    setupModelSelector();
    setupListeners();
    checkConnection();
  }

  function setupModelSelector() {
    if (!modelSelector) return;
    modelSelector.addEventListener('change', (e) => {
      const selected = e.target.value;
      if (MODEL_HINTS[selected]) {
        modelHint.textContent = MODEL_HINTS[selected];
      }
      // Update Colab URL with parameter hash
      colabLaunchBtn.href = `${COLAB_NOTEBOOK_URL}#model=${encodeURIComponent(selected)}`;
    });
  }

  function setupListeners() {
    manualConnectBtn.addEventListener('click', () => {
      const url = manualUrlInput.value.trim();
      if (!url) {
        alert("Please enter a valid gradio.live URL!");
        return;
      }
      connectToBackend(url);
    });

    manualUrlInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        manualConnectBtn.click();
      }
    });

    disconnectBtn.addEventListener('click', disconnect);
    switchEngineBtn.addEventListener('click', disconnect);

    openNewtabBtn.addEventListener('click', () => {
      const currentUrl = localStorage.getItem(STORAGE_KEY);
      if (currentUrl) {
        window.open(currentUrl, '_blank');
      }
    });

    // Iframe load handler
    studioIframe.addEventListener('load', () => {
      if (studioIframe.src && studioIframe.src !== 'about:blank') {
        frameLoader.classList.add('fade-out');
      }
    });
  }

  function checkConnection() {
    // 1. Check URL parameters
    const params = new URLSearchParams(window.location.search);
    const backendFromUrl = params.get('backend') || params.get('api');

    if (backendFromUrl) {
      connectToBackend(backendFromUrl);
      return;
    }

    // 2. Check local storage
    const savedBackend = localStorage.getItem(STORAGE_KEY);
    if (savedBackend) {
      connectToBackend(savedBackend);
    } else {
      showLauncher();
    }
  }

  function connectToBackend(rawUrl) {
    let cleanUrl = rawUrl.trim();
    if (!cleanUrl.startsWith('http://') && !cleanUrl.startsWith('https://')) {
      cleanUrl = 'https://' + cleanUrl;
    }
    // Remove trailing slash
    cleanUrl = cleanUrl.replace(/\/+$/, '');

    // Save
    localStorage.setItem(STORAGE_KEY, cleanUrl);

    // Update query string cleanly without reload
    const currentUrl = new URL(window.location);
    currentUrl.searchParams.set('backend', cleanUrl);
    window.history.replaceState({}, '', currentUrl);

    // Switch UI
    showStudio(cleanUrl);
  }

  function showStudio(url) {
    launcherView.classList.remove('active');
    studioView.classList.add('active');

    // Update status indicator
    connectionStatus.className = 'status-pill status-connected';
    statusText.textContent = 'GPU Connected';
    disconnectBtn.classList.remove('hidden');

    backendDisplay.textContent = url.replace(/^https?:\/\//, '');

    // Show spinner while iframe loads
    frameLoader.classList.remove('fade-out');
    loaderStatus.textContent = 'Loading your FreeFakeStudio interface...';

    // Set iframe source
    if (studioIframe.src !== url) {
      studioIframe.src = url;
    } else {
      frameLoader.classList.add('fade-out');
    }
  }

  function showLauncher() {
    studioView.classList.remove('active');
    launcherView.classList.add('active');

    connectionStatus.className = 'status-pill status-disconnected';
    statusText.textContent = 'GPU Disconnected';
    disconnectBtn.classList.add('hidden');

    studioIframe.src = 'about:blank';
  }

  function disconnect() {
    localStorage.removeItem(STORAGE_KEY);
    
    // Clear URL search params
    const currentUrl = new URL(window.location);
    currentUrl.searchParams.delete('backend');
    currentUrl.searchParams.delete('api');
    window.history.replaceState({}, '', currentUrl.pathname);

    showLauncher();
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
