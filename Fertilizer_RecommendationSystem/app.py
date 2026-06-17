from flask import Flask, render_template, request, flash
import sqlite3
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.secret_key = "fertilizer123"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "fertilizer.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "tuned_model.pkl")
DATASET_PATH = os.path.join(BASE_DIR,"Fertilizer Prediction.csv")

# Prepare label encoders for categorical features so inference matches training
soil_le = None
crop_le = None
try:
    _df_for_enc = pd.read_csv(DATASET_PATH)
    # training CSV uses these exact column names (including spaces/typos),
    # so fit encoders on the same values used during training.
    if "Soil Type" in _df_for_enc.columns:
        soil_le = LabelEncoder()
        soil_le.fit(_df_for_enc["Soil Type"].astype(str))
    if "Crop Type" in _df_for_enc.columns:
        crop_le = LabelEncoder()
        crop_le.fit(_df_for_enc["Crop Type"].astype(str))
except Exception as _e:
    print("Could not prepare label encoders from dataset:", _e)

# ==========================
# DATABASE
# ==========================
def create_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    temperature REAL,
    humidity REAL,
    moisture REAL,
    soil_type TEXT,
    crop_type TEXT,
    nitrogen REAL,
    potassium REAL,
    phosphorous REAL,
    fertilizer TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
    conn.commit()
    conn.close()

create_database()

# ==========================
# MODEL LOADING (optional ML path)
# ==========================
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model Loaded Successfully!")

        if hasattr(model, "classes_"):
            print("Classes:", model.classes_)

    except Exception as e:
        print("❌ Model load failed:", e)
else:
    print("❌ Model file not found at", MODEL_PATH)

# Fertilizer Name Mapping (for model predictions)
fertilizer_map = {
    0: "10-26-26",
    1: "14-35-14",
    2: "17-17-17",
    3: "20-20",
    4: "28-28",
    5: "DAP",
    6: "Urea"
}

# Simple rule-based fallback
def get_fertilizer_recommendation(temp, hum, moist, n, k, p):
    # High Nitrogen → Urea
    if n >= 45:
        return "Urea"
    
    # High Phosphorus → DAP
    elif p >= 40:
        return "DAP"
    
    # Balanced NPK → 17-17-17 or 20-20
    elif n >= 20 and p >= 20 and k >= 15:
        if temp > 28:
            return "20-20"
        else:
            return "17-17-17"
    
    # Low nutrients + high moisture → 28-28
    elif n < 25 and p < 25 and moist > 50:
        return "28-28"
    
    # High temperature + low moisture → 10-26-26
    elif temp > 30 or moist < 30:
        return "10-26-26"
    
    # High humidity + moderate nutrients → 14-35-14
    elif hum > 65:
        return "14-35-14"
    
    # Default
    else:
        return "28-28"
# ==========================
# ROUTES
# ==========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/records")
def records():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    data = cursor.fetchall()
    conn.close()
    return render_template("records.html", records=data)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        try:
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            moisture = float(request.form["moisture"])
            soil_type = request.form.get("soil_type", "Loamy")
            crop_type = request.form.get("crop_type", "Rice")
            nitrogen = float(request.form["nitrogen"])
            potassium = float(request.form["potassium"])
            phosphorous = float(request.form["phosphorous"])

            # Try ML model first (if available), otherwise fallback to rule-based
            if model is not None:
                # Build input features matching the training data order and names.
                # Training CSV columns (in order) are: 
                # ['Temparature','Humidity ','Moisture','Soil Type','Crop Type',
                #  'Nitrogen','Potassium','Phosphorous']
                try:
                    # encode categorical features the same way they were encoded during training
                    if soil_le is not None:
                        soil_encoded = int(soil_le.transform([soil_type])[0])
                    else:
                        soil_encoded = 0

                    if crop_le is not None:
                        crop_encoded = int(crop_le.transform([crop_type])[0])
                    else:
                        crop_encoded = 0

                    # Use the model's expected feature names to avoid mismatches
                    if hasattr(model, "feature_names_in_"):
                        cols = list(model.feature_names_in_)
                    else:
                        cols = [
                            "Temparature",
                            "Humidity",
                            "Moisture",
                            "Soil Type",
                            "Crop Type",
                            "Nitrogen",
                            "Potassium",
                            "Phosphorous",
                        ]

                    input_data = pd.DataFrame([
                        [
                            temperature,
                            humidity,
                            moisture,
                            soil_encoded,
                            crop_encoded,
                            nitrogen,
                            potassium,
                            phosphorous,
                        ]
                    ], columns=cols)

                    pred_num = model.predict(input_data)[0]
                    prediction = fertilizer_map.get(int(pred_num), get_fertilizer_recommendation(
                        temperature, humidity, moisture, nitrogen, potassium, phosphorous))

                except Exception as e:
                    print("Model prediction failed:", e)
                    prediction = get_fertilizer_recommendation(
                        temperature, humidity, moisture, nitrogen, potassium, phosphorous)
            else:
                prediction = get_fertilizer_recommendation(
                    temperature, humidity, moisture, nitrogen, potassium, phosphorous)

            # Save to DB
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
INSERT INTO predictions
(temperature, humidity, moisture, soil_type, crop_type,
 nitrogen, potassium, phosphorous, fertilizer, created_at)
VALUES (?,?,?,?,?,?,?,?,?, datetime('now','localtime'))
""", (
                temperature, humidity, moisture, soil_type, crop_type,
                nitrogen, potassium, phosphorous, prediction
            ))
            conn.commit()
            conn.close()

            flash("✅ Prediction Generated Successfully!")

        except Exception as e:
            prediction = f"Error: {str(e)}"
            print("Error:", e)

    return render_template("predict.html", prediction=prediction)
@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(temperature) FROM predictions")
    avg_temp = round(cursor.fetchone()[0] or 0, 1)

    cursor.execute("SELECT AVG(humidity) FROM predictions")
    avg_humidity = round(cursor.fetchone()[0] or 0, 1)

    cursor.execute("SELECT AVG(moisture) FROM predictions")
    avg_moisture = round(cursor.fetchone()[0] or 0, 1)

    cursor.execute("""
        SELECT fertilizer, COUNT(*)
        FROM predictions
        GROUP BY fertilizer
        ORDER BY COUNT(*) DESC
        LIMIT 1
    """)
    result = cursor.fetchone()
    top_fertilizer = result[0] if result else "N/A"

    conn.close()

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        avg_temp=round(avg_temp, 1),
        avg_humidity=round(avg_humidity, 1),
        avg_moisture=round(avg_moisture, 1),
        top_fertilizer=top_fertilizer
    )


@app.route("/analytics")
def analytics():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT fertilizer, COUNT(*)
        FROM predictions
        GROUP BY fertilizer
        ORDER BY COUNT(*) DESC
    """)
    fertilizer_data = cursor.fetchall()

    cursor.execute("SELECT AVG(nitrogen) FROM predictions")
    avg_n = round(cursor.fetchone()[0] or 0, 1)

    cursor.execute("SELECT AVG(phosphorous) FROM predictions")
    avg_p = round(cursor.fetchone()[0] or 0, 1)

    cursor.execute("SELECT AVG(potassium) FROM predictions")
    avg_k = round(cursor.fetchone()[0] or 0, 1)

    conn.close()

    labels = [row[0] for row in fertilizer_data]
    counts = [row[1] for row in fertilizer_data]

    return render_template(
        "analytics.html",
        fertilizer_data=fertilizer_data,
        labels=labels,
        counts=counts,
        avg_n=avg_n,
        avg_p=avg_p,
        avg_k=avg_k,
    ) 


@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)