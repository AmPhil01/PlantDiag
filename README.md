# PlantDiag — Webapp version 1.2.2

Standalone, browser-based plant disease classifier. No Node.js, no build step —
pure HTML/CSS/JS with TensorFlow.js running the model **entirely on-device**.

**Engine:** MobileNetV3Small (lightweight, on-device, WebGL).
**Data sources:** PlantVillage + NaijaStaple (shown as ✔ check markers in the app).

Your Colab-trained model is converted once from `.tflite` to TensorFlow.js format
and then loaded from the `model/` folder at runtime.

---

## ⚠️ Why conversion is required

Browsers **cannot** load `.tflite` files. TensorFlow.js needs its own format:

| Before (from Colab) | After (for the browser) |
|---|---|
| `plant_disease_model.tflite` | `model/model.json` + `model/group1-shard1ofN.bin …` |
| `labels.txt` | `model/labels.txt` |

The `convert_model.py` script does this conversion in one command.

---

## Quick start (3 steps)

### Step 1 — Put your files next to the app

Place the two files you downloaded from Colab in this folder
(`PlantDiag_Webapp_v1.2.2/`):

```
plant_disease_model.tflite
labels.txt
```

Also ensure `datasets.json` is present (it ships with the app — it describes the
PlantVillage and NaijaStaple data sources and shows ✔ check markers in the UI).

### Step 2 — Convert to TensorFlow.js format

Requires **Python 3.9–3.12**. From this folder:

```bash
python convert_model.py
```

The script:
1. Installs `tensorflowjs` (and `tensorflow` if missing) via pip.
2. Runs `tensorflowjs_converter --input_format=tflite` on your model.
3. Writes `model/model.json` + weight shards into `model/`.
4. Copies `labels.txt` into `model/labels.txt`.

You only need to run this **once** (or again whenever you retrain).

Custom paths:

```bash
python convert_model.py --tflite path\to\model.tflite --labels path\to\labels.txt --out model
```

### Step 3 — Serve and use

```bash
python -m http.server 8000
```

Open **http://localhost:8000** in your browser, upload (or drag-and-drop) a
plant leaf photo, and tap **Analyze this leaf**.

> **Do not open `index.html` via `file://`.** Browsers block `fetch()` of model
> weights from the filesystem. Always serve over HTTP (the command above, or
> VS Code **Live Server**).

---

## What you get

- 📱 **iPhone-style phone frame UI** on desktop; full-bleed responsive layout on
  real phones (max-width 390px concept, `100dvh` on mobile).
- ⚡ **Lightweight engine:** MobileNetV3Small, runs in-browser via TF.js WebGL.
- ✔ **Dataset check markers:** PlantVillage (38 classes) and NaijaStaple (12
  classes) shown as connected data sources at the top of the app.
- 🖼️ **Gallery upload** and **drag-and-drop**; **camera capture** button
  (`capture="environment"` — needs HTTPS or localhost, see notes below).
- 🔬 **Client-side inference**: 224×224 RGB, `[0,1]` normalization (`/255.0`),
  matching Colab training preprocessing.
- 📊 **Top prediction** with confidence % (1 decimal) and **top-3 list** with
  animated confidence bars.
- 🌿 **Disease info card** (description + treatment tips placeholder) derived
  from the predicted label.
- 🔄 **Analyze another image** button to re-run without reloading.
- 🚫 **Honest failure states**: if the model isn't converted/missing, the app
  shows conversion instructions — it never fabricates predictions.

---

## File layout

```
PlantDiag_Webapp_v1.2.2/
├── index.html          # App markup (phone frame, upload, results, engine badges)
├── style.css           # Design tokens + mobile-responsive styling
├── app.js              # TF.js load, preprocessing, inference, rendering, dataset markers
├── convert_model.py    # .tflite → TF.js converter (run once)
├── datasets.json       # Engine + data-source manifest (PlantVillage, NaijaStaple)
├── README.md           # This file
└── model/              # Created by convert_model.py
    ├── model.json      # TF.js graph topology
    ├── group1-shard*.bin  # Model weights
    └── labels.txt      # Class names, one per line, model output order
```

---

## Data sources

| Source | Classes | Description | Status |
|---|---|---|---|
| PlantVillage | 38 | New Plant Diseases Dataset (PlantVillage) — 54,000+ leaf images across 38 plant disease classes. | ✔ connected |
| NaijaStaple | 12 | Nigerian staple crops (cassava, maize, yam, cocoyam, plantain, sweet potato) disease imagery. | ✔ connected |

These are declared in `datasets.json` and rendered as ✔ check markers in the app header card.

---

## Preprocessing contract (must match training)

| Setting | Value |
|---|---|
| Input size | 224 × 224 |
| Color order | RGB (alpha dropped) |
| Normalization | pixel / 255.0 → `[0, 1]` |
| Tensor shape | `[1, 224, 224, 3]` float32 |
| Crop | center square-crop before resize |
| Output | class probabilities (softmax applied if model emits logits) |

If your Colab pipeline used different normalization (e.g. ImageNet mean/std or
`[-1,1]` scaling), edit `imageToTensor()` in `app.js` to match.

---

## Browser compatibility

| Browser | Works? | Notes |
|---|---|---|
| Chrome / Edge (desktop) | ✅ | Recommended. WebGL backend. |
| Firefox (desktop) | ✅ | |
| Safari (desktop) | ✅ | |
| Chrome (Android) | ✅ | Full-screen layout; camera capture works over HTTPS. |
| Safari (iOS 15+) | ✅ | Camera needs HTTPS; WebGL memory limits on old devices. |
| Any browser via `file://` | ⚠️ | Model load blocked — must serve over HTTP. |

**Camera capture note:** the `capture="environment"` input opens the rear camera
on phones, but browsers require a **secure context** (HTTPS or `localhost`).
Plain `http://<LAN-IP>` will fall back to a normal file picker — use the Gallery
button, or put the server behind HTTPS (e.g. `npx serve` + tunnel, or any
static host like GitHub Pages / Netlify).

**Performance:** first inference includes model download (~10–45 MB depending on
shard size) + WebGL warmup. Subsequent predictions take well under a second on
modern hardware.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Banner: "Model not loaded — convert your .tflite first" | Run `python convert_model.py`, then hard-refresh the page. |
| `labels.txt not found` warning during conversion | Place your Colab `labels.txt` next to `convert_model.py` and re-run. |
| Conversion fails on INT8 ops | Re-export a **float32** `.tflite` in Colab (no full-integer quantization), or see the `--allow_custom_ops` hint printed by the script. |
| Predictions look random / wrong | Your training normalization may differ from `/255` — adjust `imageToTensor()` in `app.js`. Also verify `labels.txt` line order matches model output indices. |
| Blank page / CDN error | TensorFlow.js loads from `cdn.jsdelivr.net` — check internet access, or download `tf.min.js` locally and swap the `<script>` tag in `index.html`. |
| Model works but is slow | Close other tabs; ensure the browser is using the WebGL backend (check console: `[PlantDiag] TF.js backend: webgl`). |

---

## Colab export reminder

If you need to re-export from Colab:

```python
# After training your Keras model:
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()          # float32 — recommended for TF.js
open('plant_disease_model.tflite', 'wb').write(tflite_model)

# Class names in training order → labels.txt
with open('labels.txt', 'w') as f:
    for name in class_names:                # e.g. train_gen.class_indices keys
        f.write(name + '\n')
```

Download both files from Colab and drop them next to `convert_model.py`.

---

## Credits & disclaimer

- Model: MobileNetV3Small (lightweight), trained on plant leaf imagery (Colab).
- Data: PlantVillage (38 classes) + NaijaStaple (12 classes).
- Runtime: [TensorFlow.js](https://www.tensorflow.org/js) (Apache-2.0), loaded via CDN.
- UI: PlantDiag / AgriNovva design language (agricultural green + gold).

**Disclaimer:** diagnoses are AI-generated suggestions from a single leaf photo.
Always confirm with a local agricultural extension officer before applying
pesticides or removing plants.
