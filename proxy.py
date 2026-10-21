#!/usr/bin/env python3
"""
proxy.py — PlantDiag Webapp v1.2.2 Mistral reasoning proxy
==========================================================
Secure backend that forwards Mistral API requests from the browser.
The API key is never embedded in the frontend — it is sent per-request
from the settings modal and lives only in browser localStorage.

REQUIRED:
    pip install flask flask-cors requests python-dotenv

RUN:
    python proxy.py
    # listens on http://localhost:5000

ENDPOINTS:
    GET  /health        — proxy health check
    POST /reason        — forwards to Mistral chat/completions
                          Body: { apiKey, model, prediction, confidence,
                                  plant, disease, user_note }
"""

import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
DEFAULT_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "proxy": "plantdiag-mistral-proxy"})


@app.route("/reason", methods=["POST"])
def reason():
    """
    Forward a reasoning request to Mistral.

    Expected JSON body:
    {
      "apiKey": "sk-...",           // from settings modal (localStorage)
      "model": "mistral-small-latest", // optional
      "prediction": "Tomato___Late_blight",
      "confidence": 0.87,
      "plant": "Tomato",
      "disease": "Late blight",
      "user_note": "optional farmer note"
    }
    """
    data = request.get_json(silent=True) or {}

    api_key = data.get("apiKey", "").strip()
    model = data.get("model") or DEFAULT_MODEL
    prediction = data.get("prediction", "unknown")
    confidence = data.get("confidence", 0.0)
    plant = data.get("plant", "unknown plant")
    disease = data.get("disease", "unknown disease")
    user_note = data.get("user_note", "")

    if not api_key:
        return jsonify({"error": "No Mistral API key provided. Set it in Settings."}), 401

    prompt = (
        f"You are an expert agricultural assistant helping a farmer diagnose a plant disease.\n"
        f"A leaf image was classified by an on-device AI model as: {prediction} "
        f"(confidence: {confidence:.1%}).\n"
        f"Plant: {plant}. Detected disease: {disease}.\n"
    )
    if user_note:
        prompt += f"Farmer's additional note: {user_note}\n"

    prompt += (
        "\nPlease provide:\n"
        "1. A plain-language explanation of what this disease is.\n"
        "2. Likely symptoms to confirm visually.\n"
        "3. Immediate treatment or management steps.\n"
        "4. Prevention tips for next season.\n"
        "5. A disclaimer that this is AI-generated and they should confirm with a local extension officer.\n"
        "Keep it concise, practical, and written for a farmer, not a scientist."
    )

    try:
        resp = requests.post(
            MISTRAL_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful agricultural extension assistant."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.4,
                "max_tokens": 700,
            },
            timeout=45,
        )

        if not resp.ok:
            # Pass through Mistral error messages helpfully
            try:
                err_detail = resp.json()
            except Exception:
                err_detail = resp.text
            return jsonify({
                "error": f"Mistral API error {resp.status_code}",
                "detail": err_detail,
            }), 502

        result = resp.json()
        text = result["choices"][0]["message"]["content"]
        return jsonify({"reasoning": text, "model": model})

    except requests.RequestException as e:
        return jsonify({"error": f"Failed to reach Mistral: {str(e)}"}), 502


if __name__ == "__main__":
    print("=" * 55)
    print(" PlantDiag Mistral Proxy — http://localhost:5000")
    print(" " * 55)
    print(" Endpoints:")
    print("   GET  /health")
    print("   POST /reason   (send apiKey from frontend settings)")
    print("=" * 55)
    app.run(host="127.0.0.1", port=5000, debug=False)
