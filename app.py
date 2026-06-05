import streamlit as st
import joblib
from langdetect import detect

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Book Genre Classifier",
    page_icon="📚",
    layout="centered"
)

# --------------------------------------------------
# Load Trained Model Components
# --------------------------------------------------
# model.pkl           -> Trained SVC model
# vectorizer.pkl      -> TF-IDF vectorizer
# label_encoder.pkl   -> LabelEncoder used during training
# --------------------------------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# --------------------------------------------------
# App Header
# --------------------------------------------------
st.title("📚 Book Genre Classifier")
st.caption(
    "Predict a book's genre from its title and/or description — "
    "powered by a Support Vector Classifier trained on 52,000+ Goodreads books."
)

st.divider()

# --------------------------------------------------
# Function to Clear Inputs
# --------------------------------------------------
def clear_inputs():
    """Reset title and description fields."""
    st.session_state.title = ""
    st.session_state.description = ""

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
title = st.text_input(
    "Book Title",
    placeholder="e.g. The Great Gatsby",
    key="title"
)

description = st.text_area(
    "Description",
    placeholder="Paste the book's synopsis or blurb here...",
    height=150,
    key="description"
)

# --------------------------------------------------
# Display Input Status
# --------------------------------------------------
if title and description:
    st.caption("✅ Using title + description — best accuracy")
elif title:
    st.caption("ℹ️ Using title only")
elif description:
    st.caption("ℹ️ Using description only")

st.divider()

# --------------------------------------------------
# Action Buttons
# --------------------------------------------------
col1, col2 = st.columns([3, 1])

with col1:
    classify = st.button(
        "🔍 Classify Genre",
        use_container_width=True
    )

with col2:
    st.button(
        "🗑️ Clear",
        use_container_width=True,
        on_click=clear_inputs
    )

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------
if classify:

    # Ensure at least one field is filled
    if not title and not description:
        st.warning(
            "Please enter a title or description before classifying."
        )

    else:

        # ------------------------------------------
        # Build Input Text
        # ------------------------------------------
        # Use both title and description when available
        if title and description:
            text_input = f"{title} {description}"
        elif description:
            text_input = description
        else:
            text_input = title

        # ------------------------------------------
        # Language Detection
        # ------------------------------------------
        # Only English text is supported
        try:
            lang = detect(text_input)

            if lang != "en":
                st.warning(
                    "⚠️ This app only supports English text."
                )
                st.stop()

        except Exception:
            # Continue if language detection fails
            pass

        # ------------------------------------------
        # Run Prediction
        # ------------------------------------------
        with st.spinner("Classifying..."):

            # Convert text into TF-IDF features
            vectorized = vectorizer.transform([text_input])

            # Get encoded prediction from model
            pred_encoded = model.predict(vectorized)[0]

            # Decode numeric prediction into genre name
            prediction = label_encoder.inverse_transform(
                [pred_encoded]
            )[0]

            # --------------------------------------
            # Calculate Confidence Score
            # --------------------------------------
            try:
                probabilities = model.predict_proba(vectorized)[0]
                confidence = round(max(probabilities) * 100)

            except Exception:
                # Some SVC models don't support predict_proba
                confidence = None

        # ------------------------------------------
        # Display Prediction
        # ------------------------------------------
        st.success(
            f"**Predicted Genre: {prediction}**"
        )

        # Display confidence if available
        if confidence is not None:
            st.metric(
                label="Model Confidence",
                value=f"{confidence}%"
            )

            st.progress(confidence / 100)

        st.divider()

        # ------------------------------------------
        # Prediction Details
        # ------------------------------------------
        with st.expander("ℹ️ How this prediction was made"):

            if title and description:
                st.write("**Input used:** Title + Description")
            elif description:
                st.write("**Input used:** Description only")
            else:
                st.write("**Input used:** Title only")

            st.write(
                "**Model:** Support Vector Classifier (Linear Kernel)"
            )

            st.write(
                "**Vectorizer:** TF-IDF (15,000 features, 1-2 ngrams)"
            )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption("Built by Hiba · GOMYCODE Certificate Project")
