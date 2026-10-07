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

  // Metrics & Presets
  const metricsCount = document.getElementById('metrics-model-count');
  const metricsSize = document.getElementById('metrics-download-size');
  const metricsTime = document.getElementById('metrics-boot-time');

  const presetFast = document.getElementById('preset-fast');
  const presetFlux = document.getElementById('preset-flux');
  const presetEdit = document.getElementById('preset-edit');
  const presetAll = document.getElementById('preset-all');
  const presetChips = [presetFast, presetFlux, presetEdit, presetAll].filter(Boolean);

  const modelCheckboxes = document.querySelectorAll('input[name="model"]');

  const STORAGE_KEY = 'freefake_backend_url';
  const MODELS_STORAGE_KEY = 'freefake_selected_models';

  // Initialize
  function init() {
    setupModelCheckboxes();
    setupPresetButtons();
    setupListeners();
    checkConnection();
    updateMetrics();
  }

  function setupModelCheckboxes() {
    // Restore saved checkbox preferences if available
    try {
      const saved = JSON.parse(localStorage.getItem(MODELS_STORAGE_KEY));
      if (Array.isArray(saved) && saved.length > 0) {
        modelCheckboxes.forEach((cb) => {
          cb.checked = saved.includes(cb.value);
        });
      }
    } catch (e) {}

    modelCheckboxes.forEach((cb) => {
      updateCardStyle(cb);
      cb.addEventListener('change', () => {
        // Guarantee at least one model is selected
        const anyChecked = Array.from(modelCheckboxes).some(c => c.checked);
        if (!anyChecked) {
          cb.checked = true;
        }
        updateCardStyle(cb);
        clearActivePreset();
        saveSelectedModels();
        updateMetrics();
      });
    });
  }

  function setupPresetButtons() {
    if (presetFast) {
      presetFast.addEventListener('click', () => {
        setActivePreset(presetFast);
        modelCheckboxes.forEach((cb) => {
          cb.checked = (cb.value === '⚡ Z-Image Turbo');
          updateCardStyle(cb);
        });
        saveSelectedModels();
        updateMetrics();
      });
    }

    if (presetFlux) {
      presetFlux.addEventListener('click', () => {
        setActivePreset(presetFlux);
        modelCheckboxes.forEach((cb) => {
          cb.checked = (cb.value === '🌊 FLUX.2-klein 4B' || cb.value === '🔮 FLUX.2-klein 9B');
          updateCardStyle(cb);
        });
        saveSelectedModels();
        updateMetrics();
      });
    }

    if (presetEdit) {
      presetEdit.addEventListener('click', () => {
        setActivePreset(presetEdit);
        modelCheckboxes.forEach((cb) => {
          cb.checked = (cb.value === '🎨 Qwen-Image-Edit' || cb.value === '⚡ Z-Image Turbo');
          updateCardStyle(cb);
        });
        saveSelectedModels();
        updateMetrics();
      });
    }

    if (presetAll) {
      presetAll.addEventListener('click', () => {
        setActivePreset(presetAll);
        modelCheckboxes.forEach((cb) => {
          cb.checked = true;
          updateCardStyle(cb);
        });
        saveSelectedModels();
        updateMetrics();
      });
    }
  }

  function setActivePreset(activeChip) {
    presetChips.forEach(chip => chip.classList.remove('active'));
    if (activeChip) activeChip.classList.add('active');
  }

  function clearActivePreset() {
    presetChips.forEach(chip => chip.classList.remove('active'));
  }

  function updateCardStyle(cb) {
    const parent = cb.closest('.model-card');
    if (parent) {
      if (cb.checked) {
        parent.classList.add('checked');
      } else {
        parent.classList.remove('checked');
      }
    }
  }

  function updateMetrics() {
    let count = 0;
    let totalSize = 0;
    let totalTime = 0;

    modelCheckboxes.forEach((cb) => {
      if (cb.checked) {
        count++;
        const card = cb.closest('.model-card');
        if (card) {
          const s = parseFloat(card.getAttribute('data-size')) || 0;
          const t = parseFloat(card.getAttribute('data-time')) || 0;
          totalSize += s;
          totalTime = Math.max(totalTime, t); // parallel downloads overlap, so time scales gracefully
        }
      }
    });

    if (metricsCount) metricsCount.textContent = `${count} / ${modelCheckboxes.length}`;
    if (metricsSize) metricsSize.textContent = `~${totalSize.toFixed(1)} GB`;
    
    // Estimate boot time based on total volume with aria2 1Gbps (~100MB/s) + 40s environment setup
    const estMinutes = (totalSize / 5.5).toFixed(1);
    if (metricsTime) metricsTime.textContent = `~${Math.max(1.2, parseFloat(estMinutes))} min`;
  }

  function saveSelectedModels() {
    const selected = Array.from(modelCheckboxes)
      .filter(c => c.checked)
      .map(c => c.value);
    localStorage.setItem(MODELS_STORAGE_KEY, JSON.stringify(selected));
  }

  function setupListeners() {
    if (disconnectBtn) disconnectBtn.addEventListener('click', disconnect);
    if (switchEngineBtn) switchEngineBtn.addEventListener('click', disconnect);

    if (openNewtabBtn) {
      openNewtabBtn.addEventListener('click', () => {
        const currentUrl = localStorage.getItem(STORAGE_KEY);
        if (currentUrl) {
          window.open(currentUrl, '_blank');
        }
      });
    }

    // Iframe load handler
    if (studioIframe) {
      studioIframe.addEventListener('load', () => {
        if (studioIframe.src && studioIframe.src !== 'about:blank') {
          if (frameLoader) frameLoader.classList.add('fade-out');
        }
      });
    }
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
    cleanUrl = cleanUrl.replace(/\/+$/, '');

    localStorage.setItem(STORAGE_KEY, cleanUrl);

    const currentUrl = new URL(window.location);
    currentUrl.searchParams.set('backend', cleanUrl);
    window.history.replaceState({}, '', currentUrl);

    showStudio(cleanUrl);
  }

  function showStudio(url) {
    if (launcherView) launcherView.classList.remove('active');
    if (studioView) studioView.classList.add('active');

    if (connectionStatus) connectionStatus.className = 'status-pill status-connected';
    if (statusText) statusText.textContent = 'GPU Active';
    if (disconnectBtn) disconnectBtn.classList.remove('hidden');

    if (backendDisplay) backendDisplay.textContent = url.replace(/^https?:\/\//, '');

    if (frameLoader) {
      frameLoader.classList.remove('fade-out');
      if (loaderStatus) loaderStatus.textContent = 'Connecting to FreeFakeStudio GPU Engine...';
    }

    if (studioIframe) {
      if (studioIframe.src !== url) {
        studioIframe.src = url;
      } else {
        if (frameLoader) frameLoader.classList.add('fade-out');
      }
    }
  }

  function showLauncher() {
    if (studioView) studioView.classList.remove('active');
    if (launcherView) launcherView.classList.add('active');

    if (connectionStatus) connectionStatus.className = 'status-pill status-disconnected';
    if (statusText) statusText.textContent = 'Disconnected';
    if (disconnectBtn) disconnectBtn.classList.add('hidden');

    if (studioIframe) studioIframe.src = 'about:blank';
  }

  function disconnect() {
    localStorage.removeItem(STORAGE_KEY);
    
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
