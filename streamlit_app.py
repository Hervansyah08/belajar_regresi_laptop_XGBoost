import streamlit as st
import pandas as pd
import joblib
import requests

st.set_page_config(
    page_title="Prediksi Harga Laptop",
    page_icon="💻",
    layout="wide",
)

# Load data & model
df = pd.read_csv("laptop_price.csv", encoding="latin1")
model = joblib.load("model_xgb_laptop.pkl")

# ======================
# HEADER
# ======================
st.markdown(
    """
    <h1 style='text-align: center;'>💻 Prediksi Harga Laptop</h1>
    <p style='text-align: center; font-size:18px;'>
    Gunakan Machine Learning (XGBoost) untuk memprediksi harga laptop secara otomatis
    </p>
""",
    unsafe_allow_html=True,
)

st.divider()

# ======================
# LAYOUT 2 KOLOM
# ======================
col1, col2 = st.columns([1, 1])

# ======================
# KOLOM INPUT
# ======================
with col1:
    st.subheader("🔧 Spesifikasi Laptop")

    company = st.selectbox("Brand", sorted(df["Company"].unique()))
    typename = st.selectbox("Tipe", sorted(df["TypeName"].unique()))
    inches = st.slider("Ukuran Layar (inch)", 10.0, 20.0, 15.6)

    cpu = st.selectbox("Processor", sorted(df["Cpu"].unique()))

    ram_options = sorted(df["Ram"].str.replace("GB", "").astype(int).unique())
    ram = st.selectbox("RAM", ram_options, format_func=lambda x: f"{x} GB")

    memory = st.selectbox("Storage", sorted(df["Memory"].unique()))
    gpu = st.selectbox("GPU", sorted(df["Gpu"].unique()))
    opsys = st.selectbox("Operating System", sorted(df["OpSys"].unique()))

    weight = st.slider("Berat (kg)", 0.5, 5.0, 1.5)

    predict_btn = st.button("🚀 Prediksi Sekarang", use_container_width=True)

# ======================
# KOLOM HASIL
# ======================
with col2:
    st.subheader("📊 Hasil Prediksi")

    if predict_btn:
        input_data = pd.DataFrame(
            {
                "Company": [company],
                "TypeName": [typename],
                "Inches": [inches],
                "Cpu": [cpu],
                "Ram": [ram],
                "Memory": [memory],
                "Gpu": [gpu],
                "OpSys": [opsys],
                "Weight": [weight],
            }
        )

        hasil = model.predict(input_data)
        harga_euro = hasil[0]

        # Kurs API
        try:
            response = requests.get(
                "https://api.frankfurter.app/latest?from=EUR&to=IDR"
            )
            kurs = response.json()["rates"]["IDR"]
        except:
            kurs = 17000
            st.warning("⚠️ Menggunakan kurs default")

        harga_rupiah = harga_euro * kurs

        # ======================
        # CARD HASIL
        # ======================
        st.markdown(
            f"""
        <div style="
            background-color:#1e293b;
            padding:25px;
            border-radius:15px;
            text-align:center;
        ">
            <h2 style='color:#38bdf8;'>💰 Estimasi Harga</h2>
            <h1 style='color:white;'>Rp {harga_rupiah:,.0f}</h1>
            <p style='color:gray;'>≈ €{harga_euro:,.2f}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.snow()

        # Detail tambahan
        with st.expander("🔍 Detail Input"):
            st.dataframe(input_data)

    else:
        st.info("👈 Masukkan spesifikasi lalu klik prediksi")

# ======================
# FOOTER
# ======================
st.divider()
st.markdown(
    "<center>Dibuat oleh Akbar Hervansyah dengan menggunakan Streamlit & XGBoost</center>",
    unsafe_allow_html=True,
)
