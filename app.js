/* ==========================================================================
   PlantDiag — Webapp version 1.2.2
   Client-side plant disease classification (TF.js) + AI reasoning (Mistral via proxy).
   Online: requires internet for Mistral reasoning.
   Mistral API key: pre-wired below + saved in browser localStorage via Settings.
   ========================================================================== */

'use strict';

/* ------------------------------------------------------------------ *
 * Constants
 * ------------------------------------------------------------------ */
const INPUT_SIZE = 224;
const MODEL_URL = 'model/model.json';
const LABELS_URL = 'model/labels.txt';
const DATASETS_URL = 'datasets.json';
const PROXY_URL = window.location.origin;

const FALLBACK_LABELS = [
  'Apple___Apple_scab',
  'Apple___Black_rot',
  'Apple___Cedar_apple_rust',
  'Apple___healthy',
  'Blueberry___healthy',
  'Cherry_(including_sour)___Powdery_mildew',
  'Cherry_(including_sour)___healthy',
  'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
  'Corn_(maize)___Common_rust_',
  'Corn_(maize)___Northern_Leaf_Blight',
  'Corn_(maize)___healthy',
  'Grape___Black_rot',
  'Grape___Esca_(Black_Measles)',
  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
  'Grape___healthy',
  'Orange___Haunglongbing_(Citrus_greening)',
  'Peach___Bacterial_spot',
  'Peach___healthy',
  'Pepper,_bell___Bacterial_spot',
  'Pepper,_bell___healthy',
  'Potato___Early_blight',
  'Potato___Late_blight',
  'Potato___healthy',
  'Raspberry___healthy',
  'Soybean___healthy',
  'Squash___Powdery_mildew',
  'Strawberry___Leaf_scorch',
  'Strawberry___healthy',
  'Tomato___Bacterial_spot',
  'Tomato___Early_blight',
  'Tomato___Late_blight',
  'Tomato___Leaf_Mold',
  'Tomato___Septoria_leaf_spot',
  'Tomato___Spider_mites Two-spotted_spider_mite',
  'Tomato___Target_Spot',
  'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
  'Tomato___Tomato_mosaic_virus',
  'Tomato___healthy',
];

const CONVERT_HELP =
  'Run <code>python convert_model.py</code> in this folder, then serve the app over HTTP. ' +
  'This creates <code>model/model.json</code> + weight files from your Colab .tflite. ' +
  'See README.md for the full setup.';

/* ------------------------------------------------------------------ *
 * App state
 * ------------------------------------------------------------------ */
const state = {
  model: null,
  labels: [],
  labelsFromFile: false,
  currentImageUrl: null,
  busy: false,
  engine: null,
  datasets: null,
  lastPrediction: null,
  mistralKey: localStorage.getItem('plantdiag_mistral_key') || 'mQah8aLILXzDBkNcur9Q8NiWZ85escYm',
  mistralModel: localStorage.getItem('plantdiag_mistral_model') || 'mistral-small-latest',
};

/* ------------------------------------------------------------------ *
 * DOM references
 * ------------------------------------------------------------------ */
const els = {
  statusTime: document.getElementById('statusTime'),
  statusBanner: document.getElementById('statusBanner'),
  uploadZone: document.getElementById('uploadZone'),
  galleryBtn: document.getElementById('galleryBtn'),
  cameraBtn: document.getElementById('cameraBtn'),
  galleryInput: document.getElementById('galleryInput'),
  cameraInput: document.getElementById('cameraInput'),
  previewCard: document.getElementById('previewCard'),
  previewImg: document.getElementById('previewImg'),
  analyzeBtn: document.getElementById('analyzeBtn'),
  loadingCard: document.getElementById('loadingCard'),
  errorCard: document.getElementById('errorCard'),
  errorMessage: document.getElementById('errorMessage'),
  errorDetailsWrap: document.getElementById('errorDetailsWrap'),
  errorDetails: document.getElementById('errorDetails'),
  errorDismissBtn: document.getElementById('errorDismissBtn'),
  resultsCard: document.getElementById('resultsCard'),
  resultHero: document.getElementById('resultHero'),
  resultStatus: document.getElementById('resultStatus'),
  resultLabel: document.getElementById('resultLabel'),
  resultPlant: document.getElementById('resultPlant'),
  confidenceValue: document.getElementById('confidenceValue'),
  predictionList: document.getElementById('predictionList'),
  infoDesc: document.getElementById('infoDesc'),
  infoTips: document.getElementById('infoTips'),
  againBtn: document.getElementById('againBtn'),
  appBody: document.getElementById('appBody'),
  engineBadge: document.getElementById('engineBadge'),
  datasetMarkers: document.getElementById('datasetMarkers'),
  // Settings modal
  settingsModal: document.getElementById('settingsModal'),
  settingsBtn: document.getElementById('settingsBtn'),
  settingsCloseBtn: document.getElementById('settingsCloseBtn'),
  settingsCancelBtn: document.getElementById('settingsCancelBtn'),
  settingsSaveBtn: document.getElementById('settingsSaveBtn'),
  mistralKeyInput: document.getElementById('mistralKey'),
  mistralModelInput: document.getElementById('mistralModel'),
  // Reasoning
  reasoningCard: document.getElementById('reasoningCard'),
  reasoningBtn: document.getElementById('reasoningBtn'),
  reasoningText: document.getElementById('reasoningText'),
  reasoningLoading: document.getElementById('reasoningLoading'),
  reasoningStatusBadge: document.getElementById('reasoningStatusBadge'),
  userNote: document.getElementById('userNote'),
};

/* ------------------------------------------------------------------ *
 * Boot
 * ------------------------------------------------------------------ */

initClock();
initSettings();
initUploadUI();
initTabs();
initModelAsync();

/* ------------------------------------------------------------------ *
 * Clock (status bar)
 * ------------------------------------------------------------------ */

function initClock() {
  const tick = () => {
    const now = new Date();
    let h = now.getHours();
    const m = String(now.getMinutes()).padStart(2, '0');
    h = h % 12 || 12;
    els.statusTime.textContent = `${h}:${m}`;
  };
  tick();
  setInterval(tick, 30_000);
}

/* ------------------------------------------------------------------ *
 * Settings (Mistral API key)
 * ------------------------------------------------------------------ */

function initSettings() {
  if (state.mistralKey) els.mistralKeyInput.value = state.mistralKey;
  if (state.mistralModel) els.mistralModelInput.value = state.mistralModel;

  els.settingsBtn.addEventListener('click', () => { els.settingsModal.hidden = false; });
  els.settingsCloseBtn.addEventListener('click', () => { els.settingsModal.hidden = true; });
  els.settingsCancelBtn.addEventListener('click', () => { els.settingsModal.hidden = true; });

  els.settingsSaveBtn.addEventListener('click', () => {
    const key = els.mistralKeyInput.value.trim();
    const model = els.mistralModelInput.value;
    if (!key.startsWith('sk-')) {
      showError('Mistral API key should start with "sk-".');
      return;
    }
    state.mistralKey = key;
    state.mistralModel = model;
    localStorage.setItem('plantdiag_mistral_key', key);
    localStorage.setItem('plantdiag_mistral_model', model);
    updateReasoningStatus();
    els.settingsModal.hidden = true;
    showBanner('success', '✅ Mistral API key saved. Ready for AI explanation.');
  });

  updateReasoningStatus();
}

function updateReasoningStatus() {
  const configured = !!state.mistralKey && state.mistralKey.startsWith('sk-');
  els.reasoningStatusBadge.textContent = configured ? '✔ Mistral configured' : '⚪ Not configured';
  els.reasoningStatusBadge.className = `reasoning-status-badge ${configured ? 'configured' : ''}`;
}

/* ------------------------------------------------------------------ *
 * Datasets manifest (PlantVillage + NaijaStaple)
 * ------------------------------------------------------------------ */

async function loadDatasets() {
  try {
    const res = await fetch(DATASETS_URL, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.datasets = await res.json();
  } catch (err) {
    console.warn('[PlantDiag] Could not load datasets.json — using inline defaults.', err);
    state.datasets = {
      engine: {
        name: 'MobileNetV3Small (lightweight)',
        version: '1.2.2-lite',
        input: [224, 224, 3],
        classes: 38,
        backend: 'TensorFlow.js (WebGL, on-device)',
        notes: 'Low-weight engine for browser deployment.',
      },
      data_sources: [
        {
          id: 'plantvillage',
          name: 'PlantVillage',
          description: '54,000+ leaf images across 38 plant disease classes.',
          classes: 38,
          status: 'connected',
          check: true,
        },
        {
          id: 'naijastaple',
          name: 'NaijaStaple',
          description: 'Nigerian staple crops (cassava, maize, yam, cocoyam, plantain, sweet potato).',
          classes: 12,
          status: 'connected',
          check: true,
        },
      ],
    };
  }
  renderEngineBadge();
  renderDatasetMarkers();
}

function renderEngineBadge() {
  const eng = state.datasets?.engine;
  if (!eng) return;
  els.engineBadge.textContent = `⚡ ${eng.name} · ${eng.version} · ${eng.backend}`;
}

function renderDatasetMarkers() {
  const sources = state.datasets?.data_sources ?? [];
  els.datasetMarkers.innerHTML = '';
  sources.forEach((ds) => {
    const connected = ds.status === 'connected' && ds.check !== false;
    const marker = document.createElement('div');
    marker.className = `dataset-marker ${connected ? 'connected' : ''}`;
    marker.innerHTML = `
      <span class="check">${connected ? '✔' : '✖'}</span>
      <span class="name">${escapeHtml(ds.name)}</span>
      <span class="desc">(${ds.classes} classes)</span>
    `;
    els.datasetMarkers.appendChild(marker);
  });
}

/* ------------------------------------------------------------------ *
 * Model + labels loading
 * ------------------------------------------------------------------ */

async function initModelAsync() {
  showBanner('info', '⏳ Loading model… make sure the <code>model/</code> folder exists (run <code>convert_model.py</code> first).');

  await loadDatasets();

  try {
    const res = await fetch(LABELS_URL, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${LABELS_URL}`);
    const text = await res.text();
    const labels = text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0 && !line.startsWith('#'));
    if (labels.length === 0) throw new Error('labels.txt is empty');
    state.labels = labels;
    state.labelsFromFile = true;
  } catch (err) {
    console.warn('[PlantDiag] Could not load labels.txt — using embedded fallback labels.', err);
    state.labels = FALLBACK_LABELS.slice();
    state.labelsFromFile = false;
  }

  try {
    if (typeof tf === 'undefined') {
      throw new Error('TensorFlow.js failed to load (no internet for the CDN?)');
    }
    await tf.ready();
    console.info('[PlantDiag] TF.js backend:', tf.getBackend());

    state.model = await tf.loadGraphModel(MODEL_URL);

    const out = state.model.outputs[0];
    const shape = out.shape && out.shape.length ? out.shape.join('×') : 'unknown';
    console.info('[PlantDiag] Model loaded. Output shape:', shape);

    showBanner(
      'success',
      `✅ ${state.datasets?.engine?.name ?? 'Model'} ready (${state.labels.length} classes). Snap a leaf to diagnose!`
    );
  } catch (err) {
    console.error('[PlantDiag] Model load failed:', err);
    state.model = null;
    showBanner(
      'error',
      `🚫 Model not loaded — convert your .tflite first. ${CONVERT_HELP}<br><small>Details: ${escapeHtml(String(err && err.message || err))}</small>`
    );
  }
}

/* ------------------------------------------------------------------ *
 * Upload / camera / drag-and-drop UI
 * ------------------------------------------------------------------ */

function initUploadUI() {
  els.uploadZone.addEventListener('click', () => els.galleryInput.click());
  els.uploadZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      els.galleryInput.click();
    }
  });

  els.galleryBtn.addEventListener('click', () => els.galleryInput.click());
  els.cameraBtn.addEventListener('click', () => {
    els.cameraInput.click();
  });

  els.galleryInput.addEventListener('change', onFileChosen);
  els.cameraInput.addEventListener('change', onFileChosen);

  const zone = els.uploadZone;
  ['dragenter', 'dragover'].forEach((evtName) => {
    zone.addEventListener(evtName, (e) => {
      e.preventDefault();
      zone.classList.add('dragover');
    });
  });
  ['dragleave', 'drop'].forEach((evtName) => {
    zone.addEventListener(evtName, (e) => {
      e.preventDefault();
      zone.classList.remove('dragover');
    });
  });
  zone.addEventListener('drop', (e) => {
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleImageFile(file);
  });

  els.analyzeBtn.addEventListener('click', runDiagnosis);
  els.againBtn.addEventListener('click', resetToUpload);
  els.errorDismissBtn.addEventListener('click', hideError);
  els.reasoningBtn.addEventListener('click', requestReasoning);
}

function onFileChosen(e) {
  const file = e.target.files && e.target.files[0];
  if (file) handleImageFile(file);
  e.target.value = '';
}

function handleImageFile(file) {
  hideError();

  if (!file.type.startsWith('image/')) {
    showError('That file is not an image. Please choose a JPG or PNG photo of a leaf.');
    return;
  }

  if (state.currentImageUrl) {
    URL.revokeObjectURL(state.currentImageUrl);
  }
  state.currentImageUrl = URL.createObjectURL(file);

  els.previewImg.onload = () => {
    els.previewCard.hidden = false;
    els.resultsCard.hidden = true;
    els.reasoningCard.hidden = true;
    els.previewCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  };
  els.previewImg.onerror = () => {
    showError('Could not read that image file. It may be corrupt — try another photo.');
    els.previewCard.hidden = true;
  };
  els.previewImg.src = state.currentImageUrl;
}

/* ------------------------------------------------------------------ *
 * Inference pipeline
 * ------------------------------------------------------------------ */

async function runDiagnosis() {
  if (state.busy) return;
  hideError();

  if (!state.model) {
    showError(`Model not loaded — convert your .tflite first. ${CONVERT_HELP}`);
    return;
  }
  if (!els.previewImg.src) {
    showError('Please choose a leaf photo first.');
    return;
  }

  state.busy = true;
  els.analyzeBtn.disabled = true;
  els.resultsCard.hidden = true;
  els.reasoningCard.hidden = true;
  els.loadingCard.hidden = false;
  els.loadingCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  let inputTensor = null;

  try {
    inputTensor = await imageToTensor(els.previewImg);
    const output = state.model.predict(inputTensor);
    const scores = await toProbabilities(output);
    const top3 = topK(scores, 3);
    renderResults(top3, scores);
  } catch (err) {
    console.error('[PlantDiag] Inference failed:', err);
    showError(`Inference failed: ${escapeHtml(String(err && err.message || err))}.`);
  } finally {
    if (inputTensor) inputTensor.dispose();
    state.busy = false;
    els.analyzeBtn.disabled = false;
    els.loadingCard.hidden = true;
  }
}

async function imageToTensor(img) {
  const canvas = document.createElement('canvas');
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });

  const side = Math.min(img.naturalWidth, img.naturalHeight);
  const sx = (img.naturalWidth - side) / 2;
  const sy = (img.naturalHeight - side) / 2;
  ctx.drawImage(img, sx, sy, side, side, 0, 0, INPUT_SIZE, INPUT_SIZE);

  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);

  const floats = new Float32Array(INPUT_SIZE * INPUT_SIZE * 3);
  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
    floats[j] = data[i] / 255.0;
    floats[j + 1] = data[i + 1] / 255.0;
    floats[j + 2] = data[i + 2] / 255.0;
  }

  return tf.tensor4d(floats, [1, INPUT_SIZE, INPUT_SIZE, 3]);
}

async function toProbabilities(output) {
  const tensor = Array.isArray(output) ? output[0] : output;

  let values = await tensor.data();
  values = Array.from(values);

  if (Array.isArray(output)) {
    output.forEach((t) => t.dispose && t.dispose());
  } else if (tensor.dispose) {
    tensor.dispose();
  }

  const sum = values.reduce((a, b) => a + b, 0);
  const hasNegatives = values.some((v) => v < 0);
  const looksNormalized = !hasNegatives && Math.abs(sum - 1) < 0.01;

  if (looksNormalized) {
    return Float32Array.from(values);
  }

  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const expSum = exps.reduce((a, b) => a + b, 0);
  return Float32Array.from(exps.map((e) => e / expSum));
}

function topK(scores, k) {
  return Array.from(scores)
    .map((score, index) => ({ index, score }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

/* ------------------------------------------------------------------ *
 * Rendering
 * ------------------------------------------------------------------ */

function prettyLabel(raw) {
  if (!raw) return 'Unknown class';
  return raw
    .split('___')
    .map((part, i) =>
      part
        .split(/[\s,]+/)
        .filter(Boolean)
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(' ')
    )
    .join(' — ');
}

function plantOf(raw) {
  return (raw || '').split('___')[0].replace(/_/g, ' ') || 'Unknown plant';
}

function diseaseOf(raw) {
  const parts = (raw || '').split('___');
  return parts.length > 1 ? parts.slice(1).join(' — ').replace(/_/g, ' ') : '';
}

function renderResults(top3, _scores) {
  const best = top3[0];
  const bestLabel = state.labels[best.index] || `class_${best.index}`;
  const isHealthy = /healthy/i.test(bestLabel);

  els.resultHero.classList.toggle('healthy', isHealthy);
  els.resultHero.classList.toggle('diseased', !isHealthy);
  els.resultStatus.textContent = isHealthy ? '🟢' : '🟠';
  els.resultLabel.textContent = prettyLabel(bestLabel);
  els.resultPlant.textContent = plantOf(bestLabel);
  els.confidenceValue.textContent = `${(best.score * 100).toFixed(1)}%`;

  els.predictionList.innerHTML = '';
  top3.forEach((item, rank) => {
    const label = state.labels[item.index] || `class_${item.index}`;
    const pct = (item.score * 100).toFixed(1);

    const li = document.createElement('li');
    li.className = 'prediction-row' + (rank === 0 ? ' top' : '');
    li.innerHTML = `
      <span class="prediction-rank">${rank + 1}</span>
      <span class="prediction-name" title="${escapeAttr(label)}">${escapeHtml(prettyLabel(label))}</span>
      <span class="prediction-pct">${pct}%</span>
      <span class="prediction-bar"><span class="prediction-fill"></span></span>
    `;
    els.predictionList.appendChild(li);

    const fill = li.querySelector('.prediction-fill');
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        fill.style.width = `${item.score * 100}%`;
      });
    });
  });

  state.lastPrediction = {
    prediction: bestLabel,
    confidence: best.score,
    plant: plantOf(bestLabel),
    disease: diseaseOf(bestLabel),
  };

  renderDiseaseInfo(bestLabel, best.score);

  els.resultsCard.hidden = false;
  els.resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

  els.reasoningCard.hidden = false;
  els.reasoningText.textContent = '';
}

function renderDiseaseInfo(label, score) {
  const plant = plantOf(label);
  const disease = diseaseOf(label);
  const isHealthy = /healthy/i.test(label);

  let desc;
  let tips;

  if (isHealthy) {
    desc =
      `The model sees no signs of disease on this ${plant} leaf ` +
      `(confidence ${(score * 100).toFixed(1)}%). The leaf appearance matches the healthy class it was trained on.`;
    tips = [
      'Keep monitoring leaves weekly for early spots, wilting, or discoloration.',
      'Maintain consistent watering and avoid overhead irrigation that wets foliage late in the day.',
      'Remove fallen debris around the plant to reduce pest and disease pressure.',
    ];
  } else {
    desc =
      `Detected: ${disease} on ${plant} ` +
      `(confidence ${(score * 100).toFixed(1)}%). ` +
      'This is an AI-generated suggestion from a leaf photo — confirm with local agronomy guidance before acting.';
    tips = [
      `Isolate or remove severely affected ${plant} leaves and dispose of them (do not compost).`,
      'Improve airflow around plants; avoid wetting leaves when watering.',
      'Consider an appropriate fungicide or bactericide for "' + disease + '" — ask your local extension office for a product registered in your area.',
      'If spread continues, rotate crops and remove plant residues at season end.',
    ];
  }

  els.infoDesc.textContent = desc;
  els.infoTips.innerHTML = '';
  tips.forEach((t) => {
    const li = document.createElement('li');
    li.textContent = t;
    els.infoTips.appendChild(li);
  });
}

/* ------------------------------------------------------------------ *
 * Mistral reasoning (via proxy)
 * ------------------------------------------------------------------ */

async function requestReasoning() {
  if (!state.lastPrediction) {
    showError('Run a diagnosis first before asking for AI reasoning.');
    return;
  }
  if (!state.mistralKey || !state.mistralKey.startsWith('sk-')) {
    showError('No Mistral API key saved. Open Settings (⚙️) and add your key.');
    return;
  }
  if (!navigator.onLine) {
    showError('You appear to be offline. AI reasoning requires internet.');
    return;
  }

  els.reasoningBtn.disabled = true;
  els.reasoningLoading.hidden = false;
  els.reasoningText.textContent = 'Thinking…';
  els.reasoningCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

  try {
    const resp = await fetch(`${PROXY_URL}/reason`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        apiKey: state.mistralKey,
        model: state.mistralModel,
        ...state.lastPrediction,
        user_note: els.userNote.value.trim(),
      }),
    });
    if (!resp.ok) throw new Error(`Proxy error ${resp.status}`);
    const data = await resp.json();
    els.reasoningText.textContent = data.reasoning || 'No response from AI.';
  } catch (err) {
    console.error('[PlantDiag] Reasoning failed:', err);
    els.reasoningText.textContent =
      `Could not get AI reasoning: ${escapeHtml(String(err && err.message || err))}. ` +
      'Make sure proxy.py is running on port 5000.';
  } finally {
    els.reasoningBtn.disabled = false;
    els.reasoningLoading.hidden = true;
  }
}

/* ------------------------------------------------------------------ *
 * UI helpers
 * ------------------------------------------------------------------ */

function showBanner(type, html) {
  els.statusBanner.className = `status-banner ${type}`;
  els.statusBanner.innerHTML = html;
  els.statusBanner.hidden = false;
}

function showError(message) {
  els.errorMessage.innerHTML = message;
  els.errorCard.hidden = false;
  els.errorCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() {
  els.errorCard.hidden = true;
  els.errorDetailsWrap.hidden = true;
  els.errorDetails.textContent = '';
}

function resetToUpload() {
  hideError();
  els.previewCard.hidden = true;
  els.resultsCard.hidden = true;
  els.reasoningCard.hidden = true;
  els.previewImg.removeAttribute('src');
  if (state.currentImageUrl) {
    URL.revokeObjectURL(state.currentImageUrl);
    state.currentImageUrl = null;
  }
  state.lastPrediction = null;
  els.uploadZone.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function initTabs() {
  const tabs = document.querySelectorAll('.tab-item');
  const scrollTo = {
    home: () => els.appBody.scrollTo({ top: 0, behavior: 'smooth' }),
    scan: () => els.uploadZone.scrollIntoView({ behavior: 'smooth', block: 'start' }),
    help: () => document.querySelector('.help-card').scrollIntoView({ behavior: 'smooth', block: 'start' }),
  };
  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      tabs.forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      const key = tab.dataset.tab;
      if (scrollTo[key]) scrollTo[key]();
    });
  });
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function escapeAttr(str) {
  return escapeHtml(str).replace(/'/g, '&#39;');
}
