# Main Flask application
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from ocr_utils import extract_text
from firebase_utils import get_user_profile
from ml_model import predict_food, analyze_ingredient_risk

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def home():
    return {"status": "NutriScan Backend Running"}

@app.route("/analyze", methods=["POST"])
def analyze():
    print("Analyze API hit")

    file = request.files.get("file")
    uid = request.form.get("uid")

    if not file or not uid:
        return jsonify({"error": "Missing file or UID"}), 400

    # Save uploaded image
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Extract ingredients using OCR
    ingredients_text = extract_text(filepath)
    print("OCR TEXT:", ingredients_text)

    # Get user profile
    user_profile = get_user_profile(uid)

    # Predict food safety
    prediction = predict_food(user_profile, ingredients_text)

    is_safe = prediction == 1

    if is_safe:
        result = "✅ Safe to Consume"
        alert = "Food matches your health profile."
        confidence = 85
    else:
        result = "❌ Not Recommended"
        alert = "Food may be harmful for you."
        confidence = 25

    # Analyze ingredient risk
    risk_data = analyze_ingredient_risk(ingredients_text)

    high_count = len(risk_data.get("high", []))
    medium_count = len(risk_data.get("medium", []))
    low_count = len(risk_data.get("low", []))

    total = high_count + medium_count + low_count

    # Prevent division by zero
    if total == 0:
        total = 1

    high_percent = round((high_count / total) * 100, 2)
    medium_percent = round((medium_count / total) * 100, 2)
    low_percent = round((low_count / total) * 100, 2)

    return jsonify({
        "ingredients": ingredients_text[:500],
        "health_result": result,
        "alert": alert,
        "status": "SAFE" if is_safe else "NOT SAFE",
        "confidence": confidence,
        "total_ingredients": total,
        "risk_chart": {
            "high": high_percent,
            "medium": medium_percent,
            "low": low_percent
        },
        "risk_details": {
            "high": risk_data.get("high", []),
            "medium": risk_data.get("medium", []),
            "low": risk_data.get("low", [])
        }
    })


if __name__ == "__main__":
    app.run(debug=True)
