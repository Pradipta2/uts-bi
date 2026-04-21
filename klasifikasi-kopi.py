import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Kopi ☕",
    page_icon="☕",
    layout="centered",
)

# ── CSS styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp { background-color: #fdf6ec; }

    /* Title */
    h1 { color: #4a2c0a !important; font-family: 'Georgia', serif; }
    h2, h3 { color: #6b3d14 !important; }

    /* Cards */
    .result-card {
        background: linear-gradient(135deg, #6b3d14, #a0522d);
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        color: white;
        margin-top: 20px;
        box-shadow: 0 6px 20px rgba(106,61,20,0.35);
    }
    .result-label {
        font-size: 1.1rem;
        opacity: 0.85;
        margin-bottom: 6px;
    }
    .result-class {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: 2px;
    }
    .result-desc {
        font-size: 1.05rem;
        margin-top: 10px;
        opacity: 0.9;
    }
    .prob-bar-label { font-size: 0.85rem; color: #4a2c0a; font-weight: 600; }
    .info-box {
        background: #fff8f0;
        border-left: 5px solid #a0522d;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 16px;
        font-size: 0.93rem;
        color: #4a2c0a;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6b3d14, #a0522d);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 40px;
        font-size: 1.05rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton>button:hover { opacity: 0.88; }

    /* Divider */
    hr { border-color: #d4a96a44; }
</style>
""", unsafe_allow_html=True)

# ── Load model & features ─────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "model_klasifikasi_kopi_rf.pkl")
FEATURE_PATH = os.path.join(os.path.dirname(__file__), "daftar_fitur_kopi.pkl")

@st.cache_resource(show_spinner="Memuat model…")
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource(show_spinner="Memuat fitur…")
def load_features():
    import pickle
    with open(FEATURE_PATH, "rb") as f:
        return pickle.load(f)

try:
    model    = load_model()
    features = load_features()           # full column list (incl. caffeine_concentration)
    model_features = list(model.feature_names_in_)   # features actually used by model
except Exception as e:
    st.error(f"❌ Gagal memuat model atau fitur: {e}")
    st.stop()

# ── Label mapping for classes ─────────────────────────────────────────────────
CLASS_LABELS = {
    0: ("Kelas 0", "☕", "#795548"),
    1: ("Kelas 1", "🟤", "#8D6E63"),
    2: ("Kelas 2", "🟠", "#A1887F"),
    3: ("Kelas 3", "🟡", "#BCAAA4"),
    4: ("Kelas 4", "⭐", "#D7CCC8"),
    5: ("Kelas 5", "🌟", "#EFEBE9"),
}

# If you know your actual drink-category names, replace the dict above, e.g.:
# CLASS_LABELS = {0: ("Espresso","☕","#3e1a00"), 1: ("Latte","🥛","#b07a4f"), ...}

# ── Drink encoding (for the 'drink' input feature) ────────────────────────────
DRINK_OPTIONS = {
    "Brewed Coffee"    : 0,
    "Espresso"         : 1,
    "Latte"            : 2,
    "Cappuccino"       : 3,
    "Americano"        : 4,
    "Macchiato"        : 5,
    "Mocha"            : 6,
    "Cold Brew"        : 7,
    "Flat White"       : 8,
    "Iced Coffee"      : 9,
    "Lainnya (manual)" : -1,
}

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# ☕ Klasifikasi Kopi")
st.markdown("Masukkan parameter kopi Anda, lalu klik **Prediksi** untuk mengetahui kategori kopi.")

st.markdown('<div class="info-box">🔬 Model: <b>Random Forest Classifier</b> &nbsp;|&nbsp; Fitur: 6 &nbsp;|&nbsp; Kelas: 6</div>', unsafe_allow_html=True)

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.subheader("📋 Parameter Input")

    col1, col2 = st.columns(2)

    with col1:
        # --- drink (encoded) ---
        if "drink" in model_features:
            st.markdown("**Jenis Minuman (`drink`)**")
            drink_choice = st.selectbox(
                "Pilih jenis minuman",
                options=list(DRINK_OPTIONS.keys()),
                index=0,
                label_visibility="collapsed",
            )
            if DRINK_OPTIONS[drink_choice] == -1:
                drink_val = st.number_input("Nilai encode manual", min_value=0, max_value=50, value=0, step=1)
            else:
                drink_val = DRINK_OPTIONS[drink_choice]
            st.caption(f"Nilai encode: `{drink_val}`")

        # --- volume ---
        volume = st.number_input(
            "Volume (ml) 💧",
            min_value=0.0, max_value=1000.0, value=240.0, step=10.0,
            help="Volume minuman dalam mililiter",
        )

        # --- calories ---
        calories = st.number_input(
            "Kalori (kcal) 🔥",
            min_value=0.0, max_value=1000.0, value=5.0, step=1.0,
            help="Kandungan kalori dalam kkal",
        )

    with col2:
        # --- caffeine ---
        caffeine = st.number_input(
            "Kafein (mg) ⚡",
            min_value=0.0, max_value=500.0, value=95.0, step=1.0,
            help="Kandungan kafein dalam miligram",
        )

        # --- caffeine_per_ml ---
        caffeine_per_ml = st.number_input(
            "Kafein per ml (mg/ml) 📊",
            min_value=0.0, max_value=10.0, value=round(95/240, 4), step=0.001,
            format="%.4f",
            help="Konsentrasi kafein per mililiter",
        )

        # --- calories_per_ml ---
        calories_per_ml = st.number_input(
            "Kalori per ml (kcal/ml) 📉",
            min_value=0.0, max_value=10.0, value=round(5/240, 4), step=0.001,
            format="%.4f",
            help="Kalori per mililiter",
        )

    st.divider()

    # Auto-compute helper
    st.markdown("**💡 Hitung Otomatis**")
    auto_compute = st.checkbox(
        "Hitung `caffeine_per_ml` & `calories_per_ml` otomatis dari Volume, Kalori, dan Kafein",
        value=False,
    )
    if auto_compute and volume > 0:
        caffeine_per_ml = round(caffeine / volume, 6)
        calories_per_ml = round(calories / volume, 6)
        st.info(f"Auto-calc → kafein/ml: `{caffeine_per_ml}` | kalori/ml: `{calories_per_ml}`")

    st.divider()
    submitted = st.form_submit_button("🔍 Prediksi Kategori Kopi", use_container_width=True)

# ── Prediction ─────────────────────────────────────────────────────────────────
if submitted:
    # Build input row aligned to model feature order
    input_map = {
        "drink"          : drink_val if "drink" in model_features else 0,
        "volume"         : volume,
        "calories"       : calories,
        "caffeine"       : caffeine,
        "caffeine_per_ml": caffeine_per_ml,
        "calories_per_ml": calories_per_ml,
    }
    input_row = [[input_map[f] for f in model_features]]
    input_df  = pd.DataFrame(input_row, columns=model_features)

    prediction  = model.predict(input_df)[0]
    proba       = model.predict_proba(input_df)[0]
    classes     = model.classes_

    label, icon, color = CLASS_LABELS.get(int(prediction), (f"Kelas {prediction}", "❓", "#795548"))

    # Result card
    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Hasil Prediksi</div>
        <div class="result-class">{icon} {label}</div>
        <div class="result-desc">Keyakinan model: <b>{proba[list(classes).index(prediction)]:.1%}</b></div>
    </div>
    """, unsafe_allow_html=True)

    # Probability breakdown
    st.markdown("### 📈 Distribusi Probabilitas")
    prob_df = pd.DataFrame({
        "Kelas"      : [CLASS_LABELS.get(int(c), (f"Kelas {c}", "", ""))[0] for c in classes],
        "Probabilitas": proba,
    }).sort_values("Probabilitas", ascending=False)

    st.dataframe(
        prob_df.style.format({"Probabilitas": "{:.2%}"})
               .bar(subset=["Probabilitas"], color="#a0522d", vmin=0, vmax=1),
        use_container_width=True,
        hide_index=True,
    )

    # Input summary
    with st.expander("🔎 Ringkasan Input", expanded=False):
        st.dataframe(input_df.T.rename(columns={0: "Nilai"}), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#a0522d;font-size:0.85rem;'>"
    "☕ Aplikasi Klasifikasi Kopi · Random Forest · Dibuat dengan Streamlit"
    "</p>",
    unsafe_allow_html=True,
)