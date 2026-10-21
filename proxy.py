#!/usr/bin/env python3
"""
proxy.py — PlantDiag Webapp v1.2.2
Flask backend serving the static frontend + Mistral reasoning proxy.
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)


@app.route('/reason', methods=['POST'])
def reason():
    data = request.get_json(silent=True) or {}
    api_key = data.get('apiKey', '').strip()
    model = data.get('model') or 'mistral-small-latest'
    prediction = data.get('prediction', 'unknown')
    confidence = data.get('confidence', 0.0)
    plant = data.get('plant', 'unknown plant')
    disease = data.get('disease', 'unknown disease')
    user_note = data.get('user_note', '')

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
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful agricultural extension assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.4,
                "max_tokens": 700
            },
            timeout=45
        )
        if not resp.ok:
            return jsonify({"error": f"Mistral API error {resp.status_code}"}), 502
        result = resp.json()
        return jsonify({"reasoning": result["choices"][0]["message"]["content"], "model": model})
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to reach Mistral: {str(e)}"}), 502


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
