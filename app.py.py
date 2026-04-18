import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ===================== STYLE =====================
st.set_page_config(page_title="Fraud Detection", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: white;
}
.author {
    text-align: center;
    font-size: 16px;
    color: #e0e0e0;
}
.card {
    background-color: rgba(255,255,255,0.1);
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
st.markdown('<div class="title">💳 Kredit karta firibgarligini aniqlash</div>', unsafe_allow_html=True)
st.markdown('<div class="author">Muallif: Ashurova Shahzoda</div>', unsafe_allow_html=True)

st.write("")

# ===================== MODEL =====================
model, scaler = pickle.load(open('model.pkl', 'rb'))

# ===================== INPUT =====================
st.markdown("### 📥 Ma'lumotlarni kiriting")

cols = st.columns(3)
inputs = []

for i in range(1, 29):
    if i <= 10:
        val = cols[0].number_input(f'V{i}', value=0.0)
    elif i <= 20:
        val = cols[1].number_input(f'V{i}', value=0.0)
    else:
        val = cols[2].number_input(f'V{i}', value=0.0)
    inputs.append(val)

amount = st.number_input("💰 Transaction Amount", value=0.0)
inputs.append(amount)

# ===================== PREDICTION =====================
if st.button("🔍 Bashorat qilish"):
    data = np.array(inputs).reshape(1, -1)
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0][1]

    st.markdown("## 📊 Natija")

    if prediction == 1:
        st.error(f"⚠️ Firibgarlik aniqlandi! (Ehtimollik: {prob:.2f})")
    else:
        st.success(f"✅ Tranzaksiya xavfsiz (Ehtimollik: {prob:.2f})")

    # ===================== GRAPH 1 =====================
    st.markdown("### 📈 Feature grafigi")
    fig, ax = plt.subplots()
    ax.plot(inputs, marker='o')
    ax.set_title("Kiritilgan qiymatlar")
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Qiymat")
    st.pyplot(fig)

    # ===================== GRAPH 2 =====================
    st.markdown("### 📊 Ehtimollik diagrammasi")
    fig2, ax2 = plt.subplots()
    ax2.bar(["Normal", "Fraud"], [1-prob, prob])
    ax2.set_title("Model ehtimolligi")
    st.pyplot(fig2)

    # ===================== GRAPH 3 =====================
    st.markdown("### 📉 Regressionga o‘xshash trend")
    fig3, ax3 = plt.subplots()
    ax3.plot(sorted(inputs))
    ax3.set_title("Trend (Regression ko‘rinishi)")
    st.pyplot(fig3)

# ===================== FOOTER =====================
st.markdown("---")
st.markdown('<div class="author">© 2026 Ashurova Shahzoda | AI Fraud Detection System</div>', unsafe_allow_html=True)
