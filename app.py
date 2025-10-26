from flask import Flask, request, jsonify, render_template_string, redirect
import pandas as pd
import joblib
import os
MODEL_PATH = os.path.join("artifacts", "obs_value_model.joblib")
META_PATH  = os.path.join("artifacts", "metadata.joblib")

model = joblib.load(MODEL_PATH)          
meta  = joblib.load(META_PATH)
INPUT_COLS = meta["columns"]             
app = Flask(__name__)
FREQ_OPTIONS = [
    ("Annual", "A"),
]
SIZE_OPTIONS = [
    ("Enterprises with 10+ employees", "GE10"),
]
NACE_OPTIONS = [
    ("All market sectors (B–N, S95)", "B-N_S95_XK"),  # optional, if present
    ("Manufacturing", "C"),
    ("Wholesale & retail trade", "G"),
    ("Information & communication", "J"),
]
INDIC_OPTIONS = [
    ("Enterprises selling online", "E_SELL"),
    ("Enterprises buying online", "PC_BUY"),
    ("Enterprises using computers / ICT", "PC_ENT"),
]
UNIT_OPTIONS = [
    ("% of enterprises", "PC_ENT"),
    ("% of persons employed", "PC_PERS"),
]
# Common EU geos + datalist fallback
GEO_SUGGESTIONS = [
    ("Germany", "DE"), ("France", "FR"), ("Austria", "AT"), ("Belgium", "BE"),
    ("Netherlands", "NL"), ("Italy", "IT"), ("Spain", "ES"), ("Poland", "PL"),
    ("Czechia", "CZ"), ("Denmark", "DK"), ("Sweden", "SE"), ("Finland", "FI"),
    ("Ireland", "IE"), ("Portugal", "PT"), ("Greece", "EL"), ("Romania", "RO"),
    ("Bulgaria", "BG"), ("Hungary", "HU"), ("Croatia", "HR"), ("Slovakia", "SK"),
    ("Slovenia", "SI"), ("Estonia", "EE"), ("Latvia", "LV"), ("Lithuania", "LT"),
]

COUNTRY_NAMES = {val: label for (label, val) in GEO_SUGGESTIONS}

INDIC_TEXT = {
    "E_SELL": "enterprises selling online",
    "PC_BUY": "enterprises buying online",
    "PC_ENT": "enterprises using ICT or computers"
}
INDUSTRY_TEXT = {
    "C": "the manufacturing sector",
    "G": "the wholesale & retail trade sector",
    "J": "the information & communication sector",
    "B-N_S95_XK": "all market sectors"
}
HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>E-commerce OBS_VALUE Predictor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root { --bg:#f7f8fb; --card:#fff; --text:#0f172a; --muted:#64748b; --border:#e2e8f0; --brand:#2563eb; --ok:#16a34a; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Arial; }
    .wrap { max-width:980px; margin:32px auto; padding:0 16px; }
    h1 { margin:0 0 6px; font-size:32px; }
    p.muted { color:var(--muted); margin:0 0 18px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:18px; }
    .card { background:var(--card); border:1px solid var(--border); border-radius:16px; padding:20px; box-shadow:0 2px 12px rgba(0,0,0,.04); }
    label { display:block; font-weight:600; margin-bottom:6px; }
    .hint { display:block; color:var(--muted); font-size:12px; margin-top:6px; }
    select, input { width:100%; padding:12px 12px; border:1px solid var(--border); border-radius:12px; background:#fff; }
    button { padding:12px 16px; border-radius:12px; border:0; background:var(--brand); color:#fff; font-weight:700; cursor:pointer; }
    button.secondary { background:#0ea5e9; }
    .row { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
    .result { font-size:18px; }
    .result-card { border-left:6px solid var(--ok); background:#f0fdf4; }
    code { background:#eef2ff; padding:2px 6px; border-radius:6px; }
    .footer { color:var(--muted); font-size:13px; margin-top:16px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>E-commerce OBS_VALUE Predictor</h1>
    <p class="muted">Pick from the lists; the app converts your choices to Eurostat codes for the model. Result is shown as an easy sentence (percentage).</p>

    <div class="card">
      <form method="POST" action="/predict_form">
        <div class="grid">
          <div>
            <label>Frequency</label>
            <select name="freq" required>
              {% for label, val in FREQ_OPTIONS %}
                <option value="{{val}}">{{label}}</option>
              {% endfor %}
            </select>
            <span class="hint">Most datasets use Annual.</span>
          </div>

          <div>
            <label>Indicator</label>
            <select name="indic_is" required>
              {% for label, val in INDIC_OPTIONS %}
                <option value="{{val}}">{{label}}</option>
              {% endfor %}
            </select>
            <span class="hint">What aspect of e-commerce?</span>
          </div>

          <div>
            <label>Company size</label>
            <select name="size_emp" required>
              {% for label, val in SIZE_OPTIONS %}
                <option value="{{val}}">{{label}}</option>
              {% endfor %}
            </select>
            <span class="hint">e.g., 10+ employees</span>
          </div>

          <div>
            <label>Unit</label>
            <select name="unit" required>
              {% for label, val in UNIT_OPTIONS %}
                <option value="{{val}}">{{label}}</option>
              {% endfor %}
            </select>
            <span class="hint">How the value is measured</span>
          </div>

          <div>
            <label>Industry (NACE Rev.2)</label>
            <select name="nace_r2" required>
              {% for label, val in NACE_OPTIONS %}
                <option value="{{val}}">{{label}}</option>
              {% endfor %}
            </select>
            <span class="hint">Choose the industry section</span>
          </div>

          <div>
            <label>Country / Region</label>
            <input name="geo" list="geo_list" placeholder="Start typing a country code…" required>
            <datalist id="geo_list">
              {% for label, val in GEO_SUGGESTIONS %}
                <option value="{{val}}">{{label}}</option>
              {% endfor %}
            </datalist>
            <span class="hint">Type a country code (e.g., DE, FR) or pick a suggestion</span>
          </div>

          <div>
            <label>Year</label>
            <input name="TIME_PERIOD" type="number" min="2000" max="2100" placeholder="e.g., 2021" required>
            <span class="hint">Calendar year</span>
          </div>
        </div>

        <div class="row" style="margin-top:16px;">
          <button type="submit">Predict</button>
          <form method="POST" action="/quick_example_1">
            <button class="secondary" formaction="/quick_example_1">Try example 1 (DE, Manufacturing, E_SELL, 2021)</button>
          </form>
          <form method="POST" action="/quick_example_2">
            <button class="secondary" formaction="/quick_example_2">Try example 2 (FR, ICT, PC_ENT, 2022)</button>
          </form>
        </div>
      </form>
    </div>

    {% if prediction is defined %}
    <div class="card result-card" style="margin-top:14px;">
      <div class="result">{{ prediction|safe }}</div>
      <p class="muted">Model: {{ model_name }} &nbsp;·&nbsp; Inputs: <code>{{ inputs }}</code></p>
    </div>
    {% endif %}

    <div class="footer">Tip: Use the example buttons to see a demo prediction, then adjust values.</div>
  </div>
</body>
</html>
"""
def _make_dataframe(payload: dict) -> pd.DataFrame:
    """Match training columns and recreate 'year_bucket' feature."""
    row = {col: payload.get(col, None) for col in INPUT_COLS}
    df = pd.DataFrame([row])
    df["TIME_PERIOD"] = pd.to_numeric(df["TIME_PERIOD"], errors="coerce")
    df["year_bucket"] = (df["TIME_PERIOD"] // 5) * 5  # same as training
    return df

def _friendly_message(payload, y_pred_float) -> str:
    """Return a plain-English explanation line for common users."""
    rounded = round(y_pred_float, 1)
    country = payload.get("geo", "")
    year    = payload.get("TIME_PERIOD", "")
    industry_code  = payload.get("nace_r2", "")
    indicator_code = payload.get("indic_is", "")

    country_text  = COUNTRY_NAMES.get(country, f"country code {country}")
    indicator_txt = INDIC_TEXT.get(indicator_code, "enterprises engaged in digital activity")
    industry_txt  = INDUSTRY_TEXT.get(industry_code, "the selected industry")

    # Compose the message
    return (
        f"Estimated percentage of <b>{indicator_txt}</b> "
        f"in <b>{industry_txt}</b> of <b>{country_text}</b> "
        f"in <b>{year}</b>: <b>{rounded}%</b>"
    )

# ======== Flask app ========
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template_string(
        HTML,
        FREQ_OPTIONS=FREQ_OPTIONS,
        SIZE_OPTIONS=SIZE_OPTIONS,
        NACE_OPTIONS=NACE_OPTIONS,
        INDIC_OPTIONS=INDIC_OPTIONS,
        UNIT_OPTIONS=UNIT_OPTIONS,
        GEO_SUGGESTIONS=GEO_SUGGESTIONS
    )

@app.route("/predict_form", methods=["POST"])
def predict_form():
    payload = {k: request.form.get(k) for k in INPUT_COLS}
    df = _make_dataframe(payload)
    pred = float(model.predict(df)[0])
    message = _friendly_message(payload, pred)
    return render_template_string(
        HTML,
        prediction=message,
        model_name=type(model.named_steps["model"]).__name__,
        inputs=payload,
        FREQ_OPTIONS=FREQ_OPTIONS,
        SIZE_OPTIONS=SIZE_OPTIONS,
        NACE_OPTIONS=NACE_OPTIONS,
        INDIC_OPTIONS=INDIC_OPTIONS,
        UNIT_OPTIONS=UNIT_OPTIONS,
        GEO_SUGGESTIONS=GEO_SUGGESTIONS
    )

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        payload = request.get_json(force=True)
        df = _make_dataframe(payload)
        pred = float(model.predict(df)[0])
        return jsonify({
            "prediction": pred,
            "message": _friendly_message(payload, pred),
            "model": type(model.named_steps["model"]).__name__,
            "inputs": payload
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- Quick demo buttons (fill the form and show a prediction) ---
@app.route("/quick_example_1", methods=["POST"])
def quick_example_1():
    payload = {
        "freq": "A", "size_emp": "GE10", "nace_r2": "C",
        "indic_is": "E_SELL", "unit": "PC_ENT", "geo": "DE", "TIME_PERIOD": "2021"
    }
    df = _make_dataframe(payload)
    pred = float(model.predict(df)[0])
    message = _friendly_message(payload, pred)
    return render_template_string(
        HTML,
        prediction=message,
        model_name=type(model.named_steps["model"]).__name__,
        inputs=payload,
        FREQ_OPTIONS=FREQ_OPTIONS,
        SIZE_OPTIONS=SIZE_OPTIONS,
        NACE_OPTIONS=NACE_OPTIONS,
        INDIC_OPTIONS=INDIC_OPTIONS,
        UNIT_OPTIONS=UNIT_OPTIONS,
        GEO_SUGGESTIONS=GEO_SUGGESTIONS
    )

@app.route("/quick_example_2", methods=["POST"])
def quick_example_2():
    payload = {
        "freq": "A", "size_emp": "GE10", "nace_r2": "J",
        "indic_is": "PC_ENT", "unit": "PC_ENT", "geo": "FR", "TIME_PERIOD": "2022"
    }
    df = _make_dataframe(payload)
    pred = float(model.predict(df)[0])
    message = _friendly_message(payload, pred)
    return render_template_string(
        HTML,
        prediction=message,
        model_name=type(model.named_steps["model"]).__name__,
        inputs=payload,
        FREQ_OPTIONS=FREQ_OPTIONS,
        SIZE_OPTIONS=SIZE_OPTIONS,
        NACE_OPTIONS=NACE_OPTIONS,
        INDIC_OPTIONS=INDIC_OPTIONS,
        UNIT_OPTIONS=UNIT_OPTIONS,
        GEO_SUGGESTIONS=GEO_SUGGESTIONS
    )

if __name__ == "__main__":
    # Use a different port than JupyterLab’s default; disable reloader for notebooks
    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False, threaded=True)
