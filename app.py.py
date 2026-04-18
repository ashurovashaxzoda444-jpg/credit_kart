import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ===================== SETTINGS =====================
st.set_page_config(page_title="Fraud Detection", layout="wide")

# ===================== HEADER =====================
st.markdown("<h1 style='text-align:center;'>💳 Kredit karta firibgarligini aniqlash</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Muallif: Ashurova Shahzoda</h4>", unsafe_allow_html=True)

st.write("")

# ===================== MODEL =====================
model_data = pickle.load(open('model.pkl', 'rb'))

# model yoki (model, scaler) ekanini aniqlash
if isinstance(model_data, tuple):
    model, scaler = model_data
else:
    model = model_data
    scaler = None

# model nechta feature kutyapti
n_features = model.n_features_in_

st.info(f"Model {n_features} ta o‘zgaruvchi (feature) qabul qiladi")

# ===================== INPUT =====================
st.markdown("### 📥 Ma'lumotlarni kiriting")

cols = st.columns(3)
inputs = []

for i in range(n_features):
    if i % 3 == 0:
        val = cols[0].number_input(f'Feature {i+1}', value=0.0)
    elif i % 3 == 1:
        val = cols[1].number_input(f'Feature {i+1}', value=0.0)
    else:
        val = cols[2].number_input(f'Feature {i+1}', value=0.0)
    
    inputs.append(val)

# ===================== PREDICTION =====================
if st.button("🔍 Bashorat qilish"):
    data = np.array(inputs).reshape(1, -1)

    # scaler bo‘lsa ishlatadi, bo‘lmasa o‘tkazib yuboradi
    if scaler is not None:
        data = scaler.transform(data)

    prediction = model.predict(data)[0]

    # ehtimollik bo‘lsa chiqaradi
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(data)[0][1]
    else:
        prob = None

    st.markdown("## 📊 Natija")

    if prediction == 1:
        st.error(f"⚠️ Firibgarlik aniqlandi!")
    else:
        st.success(f"✅ Tranzaksiya xavfsiz")

    if prob is not None:
        st.write(f"📈 Ehtimollik: {prob:.2f}")

    # ===================== GRAPH 1 =====================
    st.markdown("### 📈 Feature grafigi")
    fig, ax = plt.subplots()
    ax.plot(inputs, marker='o')
    ax.set_title("Kiritilgan qiymatlar")
    st.pyplot(fig)

    # ===================== GRAPH 2 =====================
    if prob is not None:
        st.markdown("### 📊 Ehtimollik diagrammasi")
        fig2, ax2 = plt.subplots()
        ax2.bar(["Normal", "Fraud"], [1-prob, prob])
        st.pyplot(fig2)

    # ===================== GRAPH 3 =====================
    st.markdown("### 📉 Trend (Regressionga o‘xshash)")
    fig3, ax3 = plt.subplots()
    ax3.plot(sorted(inputs))
    st.pyplot(fig3)

# ===================== FOOTER =====================
st.markdown("---")
st.markdown("<center>© 2026 Ashurova Shahzoda</center>", unsafe_allow_html=True)
