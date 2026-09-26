import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi halaman utama
st.set_page_config(
    page_title="Prediksi Segmen Pelanggan",
    page_icon="🛒",
    layout="centered"
)

# Fungsi untuk memuat model
@st.cache_resource
def load_assets():
    model = joblib.load('kmeans_customer.pkl')
    scaler = joblib.load('scaler_customer.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error("Gagal memuat model. Pastikan file .pkl berada di direktori yang sama.")
    st.stop()

st.title("🛒 Customer Profiling & Segmentation")
st.markdown("""
Aplikasi ini menggunakan algoritma **Machine Learning (K-Means)** untuk menganalisis metrik finansial dan data pengeluaran pelanggan dari nota pembelian, lalu mengkategorikannya ke dalam target segmen yang tepat guna optimalisasi margin bisnis.
""")

st.divider()

col1, col2 = st.columns(2)

with st.form("input_form"):
    st.subheader("Data Finansial Pelanggan")
    
    with col1:
        income = st.number_input("Pendapatan Tahunan ($)", min_value=0.0, value=55000.0, step=1000.0)
        wines = st.number_input("Total Belanja Anggur ($)", min_value=0.0, value=250.0, step=10.0)
        fruits = st.number_input("Total Belanja Buah ($)", min_value=0.0, value=30.0, step=5.0)
        
    with col2:
        meat = st.number_input("Total Belanja Daging ($)", min_value=0.0, value=150.0, step=10.0)
        fish = st.number_input("Total Belanja Ikan ($)", min_value=0.0, value=45.0, step=5.0)
        sweets = st.number_input("Total Belanja Manisan ($)", min_value=0.0, value=20.0, step=5.0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Analisis Profil Pelanggan", use_container_width=True)

if submitted:
    input_df = pd.DataFrame(
        [[income, wines, fruits, meat, fish, sweets]], 
        columns=['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']
    )
    
    input_scaled = scaler.transform(input_df)
    cluster_id = model.predict(input_scaled)[0]
    
    st.divider()
    st.subheader("📊 Hasil Analisis")
    
    if cluster_id == 0:
        st.success(f"**Segmen Terdeteksi: Cluster {cluster_id} (Konsumen Ekonomis)**")
        st.info("Karakteristik: Pendapatan menengah ke bawah. Frekuensi belanja minim dan berfokus pada barang kebutuhan pokok murah.")
        st.write("**Rekomendasi Aksi:** Tawarkan diskon kuantitas (bundle pricing) dan promosi kebutuhan harian untuk memicu konversi transaksi rutin.")
        
    elif cluster_id == 1:
        st.success(f"**Segmen Terdeteksi: Cluster {cluster_id} (Konsumen Menengah-Aktif)**")
        st.info("Karakteristik: Pendapatan stabil. Distribusi pengeluaran yang merata di berbagai kategori, sangat responsif terhadap kampanye musiman.")
        st.write("**Rekomendasi Aksi:** Tawarkan program poin loyalitas (loyalty rewards) atau cashback untuk mempertahankan retensi pembelian.")
        
    elif cluster_id == 2:
        st.success(f"**Segmen Terdeteksi: Cluster {cluster_id} (Konsumen Premium)**")
        st.info("Karakteristik: Pendapatan tinggi. Menghasilkan margin laba paling besar, didominasi dari pembelian produk daging berkualitas dan anggur.")
        st.write("**Rekomendasi Aksi:** Jangan tawarkan diskon murah. Berikan akses eksklusif untuk produk baru, pelayanan prioritas, atau sistem membership VIP.")
