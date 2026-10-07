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

  const selectFastBtn = document.getElementById('select-fast-btn');
  const selectAllBtn = document.getElementById('select-all-btn');
  const modelCheckboxes = document.querySelectorAll('input[name="model"]');

  const STORAGE_KEY = 'freefake_backend_url';
  const MODELS_STORAGE_KEY = 'freefake_selected_models';

  // Initialize
  function init() {
    setupModelCheckboxes();
    setupListeners();
    checkConnection();
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
        if (selectFastBtn) selectFastBtn.classList.remove('active');
        if (selectAllBtn) selectAllBtn.classList.remove('active');
        saveSelectedModels();
      });
    });

    if (selectFastBtn) {
      selectFastBtn.addEventListener('click', () => {
        modelCheckboxes.forEach((cb) => {
          cb.checked = (cb.value === '⚡ Z-Image Turbo');
          updateCardStyle(cb);
        });
        selectFastBtn.classList.add('active');
        if (selectAllBtn) selectAllBtn.classList.remove('active');
        saveSelectedModels();
      });
    }

    if (selectAllBtn) {
      selectAllBtn.addEventListener('click', () => {
        modelCheckboxes.forEach((cb) => {
          cb.checked = true;
          updateCardStyle(cb);
        });
        selectAllBtn.classList.add('active');
        if (selectFastBtn) selectFastBtn.classList.remove('active');
        saveSelectedModels();
      });
    }
  }

  function updateCardStyle(cb) {
    const parent = cb.closest('.model-checkbox-item');
    if (parent) {
      if (cb.checked) {
        parent.classList.add('checked');
      } else {
        parent.classList.remove('checked');
      }
    }
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
      if (loaderStatus) loaderStatus.textContent = 'Loading your FreeFakeStudio interface...';
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
