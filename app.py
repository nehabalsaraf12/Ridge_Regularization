"""
Diabetes Risk Predictor - Flask app for Vercel
Model : overfitting_ridge_model.pkl (sklearn Ridge, 8 Pima-diabetes features)
Run   : python app.py            (local)
Deploy: vercel --prod            (Vercel auto-detects the `app` object)
"""
import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# ---------------------------------------------------------------- model ----
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "overfitting_ridge_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# (name, label, min, max, step, default, hint)
FEATURES = [
    ("Pregnancies", "Pregnancies", 0, 20, 1, 1, "Number of times pregnant"),
    ("Glucose", "Glucose", 0, 300, 1, 110, "Plasma glucose (mg/dL)"),
    ("BloodPressure", "Blood Pressure", 0, 200, 1, 72, "Diastolic BP (mm Hg)"),
    ("SkinThickness", "Skin Thickness", 0, 100, 1, 23, "Triceps skinfold (mm)"),
    ("Insulin", "Insulin", 0, 900, 1, 80, "2-hour serum insulin (mu U/ml)"),
    ("BMI", "BMI", 0, 70, 0.1, 28.0, "Body mass index (kg/m²)"),
    ("DiabetesPedigreeFunction", "Pedigree Function", 0, 3, 0.001, 0.47,
     "Diabetes family-history score"),
    ("Age", "Age", 1, 120, 1, 33, "Age in years"),
]


def classify(score: float):
    """Turn the regression score into a friendly label + colour."""
    if score < 0.30:
        return "Low risk", "low"
    if score < 0.55:
        return "Moderate risk", "mid"
    return "High risk", "high"


def predict_from(values: dict) -> float:
    row = pd.DataFrame([[float(values[f[0]]) for f in FEATURES]],
                       columns=[f[0] for f in FEATURES])
    return float(model.predict(row)[0])


# ----------------------------------------------------------------- page ----
PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diabetes Risk Predictor</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🩺</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg1:#0b1026; --bg2:#10325a; --bg3:#0f766e;
    --card:rgba(255,255,255,.08); --border:rgba(255,255,255,.18);
    --text:#eef4ff; --muted:#a9b8d6;
    --accent:#22d3ee; --accent2:#818cf8;
    --low:#34d399; --mid:#fbbf24; --high:#fb7185;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body {
    font-family:'Poppins',system-ui,sans-serif; color:var(--text);
    min-height:100vh; padding:32px 16px 48px;
    background:
      radial-gradient(circle at 15% 10%, rgba(34,211,238,.25), transparent 40%),
      radial-gradient(circle at 85% 90%, rgba(129,140,248,.28), transparent 45%),
      linear-gradient(135deg, var(--bg1), var(--bg2) 55%, var(--bg3));
    background-attachment:fixed;
  }
  .blob { position:fixed; border-radius:50%; filter:blur(70px); opacity:.35; z-index:-1; }
  .b1 { width:320px; height:320px; background:#22d3ee; top:-80px; right:-60px; }
  .b2 { width:380px; height:380px; background:#6366f1; bottom:-120px; left:-100px; }

  .wrap { max-width:1020px; margin:0 auto; }
  header { text-align:center; margin-bottom:28px; }
  header svg { width:96px; height:96px; filter:drop-shadow(0 8px 24px rgba(34,211,238,.5)); }
  h1 {
    font-size:clamp(1.8rem,4vw,2.8rem); font-weight:700; margin-top:8px;
    background:linear-gradient(90deg,#fff,var(--accent),var(--accent2));
    -webkit-background-clip:text; background-clip:text; color:transparent;
  }
  header p { color:var(--muted); margin-top:8px; font-weight:300; }

  .grid { display:grid; grid-template-columns:1.6fr 1fr; gap:22px; }
  @media (max-width:860px){ .grid{ grid-template-columns:1fr; } }

  .card {
    background:var(--card); border:1px solid var(--border);
    border-radius:22px; padding:26px;
    backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);
    box-shadow:0 20px 50px rgba(0,0,0,.35);
  }
  .card h2 { font-size:1.1rem; font-weight:600; margin-bottom:18px; }

  .fields { display:grid; grid-template-columns:repeat(2,1fr); gap:16px; }
  @media (max-width:560px){ .fields{ grid-template-columns:1fr; } }
  label { display:block; font-size:.85rem; font-weight:500; margin-bottom:6px; }
  label small { display:block; color:var(--muted); font-weight:300; font-size:.72rem; }
  input {
    width:100%; padding:12px 14px; border-radius:12px;
    border:1px solid var(--border); background:rgba(255,255,255,.07);
    color:var(--text); font:inherit; font-size:.95rem; outline:none;
    transition:.2s;
  }
  input:focus { border-color:var(--accent); box-shadow:0 0 0 3px rgba(34,211,238,.25); }

  button {
    margin-top:22px; width:100%; padding:14px; border:0; border-radius:14px;
    font:inherit; font-weight:600; font-size:1rem; color:#04202b; cursor:pointer;
    background:linear-gradient(90deg,var(--accent),var(--accent2));
    box-shadow:0 10px 26px rgba(34,211,238,.35); transition:.2s;
  }
  button:hover { transform:translateY(-2px); box-shadow:0 14px 32px rgba(129,140,248,.45); }

  .result { text-align:center; }
  .gauge {
    --p:0; --c:var(--muted);
    width:190px; height:190px; border-radius:50%; margin:6px auto 18px;
    display:grid; place-items:center;
    background:conic-gradient(var(--c) calc(var(--p)*1%), rgba(255,255,255,.1) 0);
  }
  .gauge div {
    width:150px; height:150px; border-radius:50%; background:#0e1b3a;
    display:grid; place-items:center; font-size:2.2rem; font-weight:700;
  }
  .low  { --c:var(--low);  } .mid { --c:var(--mid); } .high { --c:var(--high); }
  .badge {
    display:inline-block; padding:8px 20px; border-radius:999px; font-weight:600;
    background:rgba(255,255,255,.1); border:1px solid var(--c); color:var(--c);
  }
  .placeholder { color:var(--muted); font-weight:300; padding:40px 8px; }
  .raw { color:var(--muted); font-size:.8rem; margin-top:14px; }
  .error { color:var(--high); margin-bottom:14px; font-size:.9rem; }

  footer { text-align:center; color:var(--muted); font-size:.75rem; margin-top:26px; font-weight:300; }
</style>
</head>
<body>
<div class="blob b1"></div><div class="blob b2"></div>
<div class="wrap">
  <header>
    <!-- inline SVG hero image -->
    <svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#818cf8"/>
        </linearGradient>
      </defs>
      <circle cx="60" cy="60" r="56" fill="rgba(255,255,255,.08)" stroke="url(#g)" stroke-width="3"/>
      <path d="M60 94 C28 70 26 46 40 38 C50 33 58 38 60 44 C62 38 70 33 80 38 C94 46 92 70 60 94Z" fill="url(#g)"/>
      <polyline points="26,62 46,62 54,46 64,78 72,58 94,58" fill="none" stroke="#fff" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <h1>Diabetes Risk Predictor</h1>
    <p>Ridge regression model · enter patient measurements to get an instant risk score</p>
  </header>

  <div class="grid">
    <form class="card" method="post" action="/">
      <h2>🧪 Patient details</h2>
      {% if error %}<div class="error">⚠️ {{ error }}</div>{% endif %}
      <div class="fields">
        {% for name,label,lo,hi,step,default,hint in features %}
        <div>
          <label for="{{ name }}">{{ label }}<small>{{ hint }}</small></label>
          <input type="number" id="{{ name }}" name="{{ name }}" min="{{ lo }}" max="{{ hi }}"
                 step="{{ step }}" value="{{ values.get(name, default) }}" required>
        </div>
        {% endfor %}
      </div>
      <button type="submit">Predict risk →</button>
    </form>

    <div class="card result">
      <h2>📊 Result</h2>
      {% if score is not none %}
        <div class="gauge {{ level }}" style="--p: {{ pct }}"><div>{{ pct }}%</div></div>
        <span class="badge {{ level }}">{{ label }}</span>
        <p class="raw">Raw model output: {{ "%.4f"|format(score) }}</p>
      {% else %}
        <div class="placeholder">Fill in the form and press <b>Predict risk</b> to see the score here.</div>
      {% endif %}
    </div>
  </div>

  <footer>For educational purposes only — not a medical diagnosis. Consult a healthcare professional.</footer>
</div>
</body>
</html>
"""


# --------------------------------------------------------------- routes ----
@app.route("/", methods=["GET", "POST"])
def home():
    ctx = dict(features=FEATURES, values={}, score=None, pct=0,
               label="", level="", error=None)
    if request.method == "POST":
        ctx["values"] = request.form.to_dict()
        try:
            score = predict_from(request.form)
            label, level = classify(score)
            ctx.update(score=score, label=label, level=level,
                       pct=round(max(0.0, min(1.0, score)) * 100))
        except (ValueError, KeyError):
            ctx["error"] = "Please enter valid numbers in every field."
    return render_template_string(PAGE, **ctx)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API: POST {"Pregnancies":2, "Glucose":120, ...}"""
    data = request.get_json(silent=True) or {}
    try:
        score = predict_from(data)
    except (ValueError, KeyError, TypeError) as e:
        return jsonify(error=f"Invalid input: {e}"), 400
    label, _ = classify(score)
    return jsonify(score=score, risk=label)


@app.route("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(debug=True)
