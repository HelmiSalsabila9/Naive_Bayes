from sklearn.metrics import accuracy_score
import streamlit as st
import pandas as pd
import numpy as np
import pickle

from PIL import Image
from sklearn.naive_bayes import GaussianNB

# Preprocessing

st.caption('Universitas Logistik dan Bisnis Internasional')
st.title('TUGAS BESAR DATA MINING')
st.subheader("""
        Penerapan Metode Klasifikasi Naive Bayes Dalam Memprediksi Siswa Unggulan oleh:
""")
st.subheader('1194018 - Helmi Salsabila - D4TI3A')

img1 = Image.open('img1.jpg')
img1 = img1.resize((700, 418))
st.image(img1, use_column_width=False)

# Sidebar
st.sidebar.header('Masukan Data')
upload = st.file_uploader('Masukan file CSV', type=['CSV'])
if upload is not None:
    inputan = pd.read_csv(upload)
else:
    def input_user():
        kelas = st.sidebar.number_input('Kelas', 7, 9)
        indo = st.sidebar.number_input('Nilai B.Indo', 0,100)
        mtk = st.sidebar.number_input('Nilai MTK', 0, 100)
        ipa = st.sidebar.number_input('Nilai IPA', 0, 100)
        ing = st.sidebar.number_input('Nilai B.Ing', 0, 100)
        data = {
            'Kelas': kelas,
            'Nilai B.Indo': indo,
            'Nilai MTK': mtk,
            'Nilai IPA': ipa,
            'Nilai B.Ing': ing
            }
        fitur = pd.DataFrame(data, index=[0])
        return fitur
    inputan = input_user()

ds = pd.read_csv('data_fiks.csv')
siswa = ds.drop(columns=['Grade'])
df = pd.concat([inputan, siswa], axis=0)

# Encode
encode = ['Kelas']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1] # Ambil baris pertama data input user

# Menampilkan hasil parameter inputan user
st.subheader('Hasil Inputan User')
if upload is not None:
    st.write(df)
else:
    st.caption('Menunggu file CSV diupload')
    st.write(df)
    
# Load modelnya
load_model = pickle.load(open('modelNBC_smp.pkl', 'rb'))

# Terapkan model prediksi
predik = load_model.predict(df)
proba = load_model.predict_proba(df)

# Tampilkan
st.subheader('Keterangan Label Kelas')
grade_siswa = np.array(['A', 'B', 'C'])
st.write(grade_siswa)

# Hasil prediksi
# A = 7,85,84,84,89
# Default
# C = 7,70,70,70,70 
st.subheader('Hasil Prediksi [Klasifikasi Siswa]')
if grade_siswa[predik] == 'A':
    st.success('Grade A')
    st.caption('Pertahankan!')
elif grade_siswa[predik] == 'B':
    st.warning('Grade B')
    st.caption('Tingkatkan!')
else:
    st.error('Grade C')
    st.caption('Belajar lagi!')
    
# Hasil Probabilitas
st.subheader('Hasil Probabilitas [Label Kelas]')
st.write(proba)

