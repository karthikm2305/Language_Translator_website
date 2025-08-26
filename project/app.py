import os
import difflib
import pandas as pd
from flask import Flask, render_template, request, jsonify, url_for
from gtts import gTTS

app = Flask(__name__, static_folder="static", template_folder="templates")

CSV_PATH = "thenglish_translations.csv"
AUDIO_DIR = os.path.join("static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# Load dictionary
try:
    thenglish_df = pd.read_csv(CSV_PATH)
    thenglish_df.columns = [c.strip() for c in thenglish_df.columns]
except Exception as e:
    print(f"⚠️ Could not load {CSV_PATH}: {e}")
    thenglish_df = pd.DataFrame(columns=["Thenglish", "English", "Hindi"])

def dict_translate(text: str, target: str):
    """Translate text from Thenglish -> English/Hindi (with fuzzy matching)."""
    if not isinstance(text, str) or not text.strip():
        return None

    text = text.strip().lower()
    df = thenglish_df.copy()
    df["Thenglish"] = df["Thenglish"].astype(str).str.strip().str.lower()

    # Exact match
    row = df[df["Thenglish"] == text]
    if not row.empty:
        return row["English"].values[0] if target == "en" else row["Hindi"].values[0]

    # Fuzzy match
    suggestions = difflib.get_close_matches(text, df["Thenglish"].tolist(), n=1, cutoff=0.6)
    if suggestions:
        row = df[df["Thenglish"] == suggestions[0]]
        if not row.empty:
            return row["English"].values[0] if target == "en" else row["Hindi"].values[0]
    return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    lang = data.get("language", "en")
    if lang not in ["en", "hi"]:
        return jsonify({"error": "Language must be 'en' or 'hi'"}), 400

    result = dict_translate(text, lang)
    if result:
        return jsonify({"translation": result, "source": "dictionary"})
    else:
        return jsonify({"translation": None, "source": "none", "message": "Word not found in dictionary"})

@app.route("/api/pronounce", methods=["POST"])
def api_pronounce():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    lang = data.get("tts_lang", "en")
    if not text:
        return jsonify({"error": "Empty text"}), 400

    filename = f"pron_{abs(hash(text+lang))}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(filepath)
        return jsonify({"audio_url": url_for("static", filename=f"audio/{filename}")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/words", methods=["GET"])
def api_words():
    try:
        words = thenglish_df["Thenglish"].dropna().astype(str).tolist()
        return jsonify({"words": words})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
