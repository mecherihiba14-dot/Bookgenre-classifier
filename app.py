import streamlit as st
import joblib
from langdetect import detect

st.set_page_config(page_title="Book Genre Classifier", page_icon="📚", layout="centered")

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

st.title("📚 Book Genre Classifier")
st.caption("Predict a book's genre from its title and/or description — powered by a Support Vector Classifier trained on 52,000+ Goodreads books.")

st.divider()

title = st.text_input("Book Title", placeholder="e.g. The Great Gatsby")
description = st.text_area("Description", placeholder="Paste the book's synopsis or blurb here...", height=150)

try:
    lang = detect(st.text_input)
    if lang != 'en':
        st.warning("⚠️ This app only supports English text. Please enter the title or description in English for accurate results.")
        st.stop()
except:
    pass

if title and description:
    st.caption("✅ Using title + description — best accuracy")
elif title:
    st.caption("ℹ️ Using title only")
elif description:
    st.caption("ℹ️ Using description only")
    
def clear_inputs():
    st.session_state.title = ""
    st.session_state.description = ""

st.divider()

col1, col2 = st.columns([3, 1])

with col1:
    classify = st.button("🔍 Classify Genre", use_container_width=True, key="classify_btn")
with col2:
    st.button("🗑️ Clear", use_container_width=True, on_click=clear_inputs)

if st.button("🔍 Classify Genre", use_container_width=True):
    if not title and not description:
        st.warning("Please enter a title or description before classifying.")
    else:
        if title and description:
            text_input = title + ' ' + description
        elif description:
            text_input = description
        else:
            text_input = title

        with st.spinner("Classifying..."):
            vectorized = vectorizer.transform([text_input])
            prediction = label_encoder.inverse_transform(model.predict(vectorized))[0]

            try:
                proba = model.predict_proba(vectorized)[0]
                confidence = round(max(proba) * 100)
            except:
                confidence = None

        st.success(f"**Predicted Genre: {prediction}**")

        if confidence:
            st.metric(label="Model Confidence", value=f"{confidence}%")
            st.progress(confidence / 100)

        st.divider()
        with st.expander("ℹ️ How this prediction was made"):
            if title and description:
                st.write(f"**Input used:** Title + Description")
            elif description:
                st.write(f"**Input used:** Description only")
            else:
                st.write(f"**Input used:** Title only")
            st.write(f"**Model:** Support Vector Classifier (linear kernel)")
            st.write(f"**Vectorizer:** TF-IDF (15,000 features, 1-2 ngrams)")

st.divider()
st.caption("Built by Hiba · GOMYCODE Certificate Project")
