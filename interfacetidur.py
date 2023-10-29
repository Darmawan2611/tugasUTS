import pickle
import numpy as np
import streamlit as st

model = pickle.load(open('gangguantidur.sav', 'rb'))
st.title('Prediksi Gangguan Tidur')

with st.form(key='sleep_form'):
    Age = st.number_input('Masukan Usia Anda', min_value=0)
    Sleep_Duration = st.number_input('Masukan Durasi Tidur Anda (jam)')
    Quality_of_Sleep = st.number_input('Masukan Jumlah Jam Tidur yang Berkualitas', min_value=0)
    Physical_Activity_Level = st.number_input('Masukan Tingkatan Aktivitas Secara Fisik', min_value=0)
    Stress_Level = st.number_input('Tingkatan Stress', min_value=0)
    Heart_Rate = st.number_input('Detak Jantung Rata - Rata', min_value=0)
    Daily_Steps = st.number_input('Jumlah Langkah Kaki Harian', min_value=0)

    submitted = st.form_submit_button('Lakukan Prediksi Gangguan Tidur')

if submitted:
    input_data = [[Age, Sleep_Duration, Quality_of_Sleep, Physical_Activity_Level, Stress_Level, Heart_Rate, Daily_Steps]]
    sleep_prediction = model.predict(input_data)

    if sleep_prediction[0] == 1:
        sleep_diagnosis = 'Pasien Terkena Gangguan Tidur'
    else:
        sleep_diagnosis = 'Pasien Tidak Terkena Gangguan Tidur'
    st.success(sleep_diagnosis)
