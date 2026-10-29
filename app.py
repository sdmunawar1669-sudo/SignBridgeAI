import os
import io
import time
import joblib
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(APP_DIR, "gesture_samples.csv")
MODEL_PATH = os.path.join(APP_DIR, "signbridge_model.joblib")
LABELS = ["HELLO", "THANK YOU", "HELP", "PLEASE", "YES", "NO", "WATER", "FOOD", "HOSPITAL", "EMERGENCY"]

st.set_page_config(page_title="SignBridge AI", page_icon="🤟", layout="wide")

st.markdown("""
<style>
.main {background: #f4f7fb;}
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.hero {padding: 1.2rem 1.4rem; border-radius: 18px; color: white;
       background: linear-gradient(120deg,#101d35,#2363d9);}
.hero h1 {margin: 0; font-size: 2.2rem;}
.hero p {margin: .45rem 0 0 0; opacity: .9;}
.small-note {color:#60708a;font-size:.9rem;}
</style>
<div class="hero">
  <h1>🤟 SignBridge AI</h1>
  <p>Hand-gesture recognition prototype · Camera snapshots · Text and speech output</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Educational prototype: this app recognizes only gestures you collect and train. "
    "It is not a complete sign-language translator and must not be relied on for emergencies."
)

mp_hands = mp.solutions.hands

def image_to_rgb(upload):
    raw = upload.getvalue()
    arr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def extract_features(rgb):
    """Return 63 normalized landmark values for the first detected hand, or None."""
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5
    ) as hands:
        result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None, None
    pts = np.array([[lm.x, lm.y, lm.z] for lm in result.multi_hand_landmarks[0].landmark],
                   dtype=np.float32)
    # Translation and scale normalization relative to wrist.
    pts -= pts[0]
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts /= scale
    return pts.flatten(), result.multi_hand_landmarks[0]

def read_dataset():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(DATA_PATH)
    except Exception:
        return pd.DataFrame()

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

def save_sample(label, features):
    row = {"label": label}
    row.update({f"f{i}": float(v) for i, v in enumerate(features)})
    df = read_dataset()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

def speak_text(text):
    # Browser speech synthesis is supported by most modern browsers.
    safe_text = text.replace("\\", "\\\\").replace("'", "\\'")
    st.components.v1.html(
        f"""<script>
        const utterance = new SpeechSynthesisUtterance('{safe_text}');
        utterance.rate = 0.9;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utterance);
        </script>""",
        height=0
    )

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Open section", ["Live Recognition", "Collect Training Data", "Train Model", "Project Information"])
    st.divider()
    st.caption("PBL stack: Python · MediaPipe · OpenCV · scikit-learn · Streamlit")

if page == "Live Recognition":
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Camera / image input")
        source = st.radio("Choose input", ["Take a camera snapshot", "Upload an image"], horizontal=True)
        uploaded = None
        if source == "Take a camera snapshot":
            uploaded = st.camera_input("Position one hand clearly in the frame")
        else:
            uploaded = st.file_uploader("Upload a hand-sign image", type=["png", "jpg", "jpeg"])

        if uploaded is not None:
            rgb = image_to_rgb(uploaded)
            if rgb is None:
                st.error("Could not read this image. Try another JPG or PNG.")
            else:
                st.image(rgb, caption="Input image", use_container_width=True)
                features, landmarks = extract_features(rgb)
                if features is None:
                    st.error("No hand detected. Improve lighting and keep your hand fully visible.")
                else:
                    model = load_model()
                    if model is None:
                        st.info("No trained model yet. Open **Collect Training Data** and collect examples for at least two signs, then train the model.")
                    else:
                        probs = model.predict_proba([features])[0]
                        idx = int(np.argmax(probs))
                        prediction = str(model.classes_[idx])
                        confidence = float(probs[idx])
                        st.session_state["last_prediction"] = prediction
                        st.session_state["last_confidence"] = confidence
                        st.success(f"Recognized sign: **{prediction}**")
                        st.metric("Model confidence", f"{confidence:.1%}")
                        if confidence < 0.60:
                            st.warning("Low confidence. Try another image or collect more varied training examples.")
    with right:
        st.subheader("Recognition result")
        prediction = st.session_state.get("last_prediction", "Waiting for input")
        confidence = st.session_state.get("last_confidence", 0.0)
        st.markdown(f"### {prediction}")
        st.progress(min(max(float(confidence), 0.0), 1.0))
        st.caption(f"Confidence: {confidence:.1%}")
        if prediction != "Waiting for input":
            if st.button("🔊 Speak recognized sign", use_container_width=True):
                speak_text(prediction.replace("_", " ").title())
            if st.button("➕ Add to sentence", use_container_width=True):
                current = st.session_state.get("sentence", [])
                current.append(prediction.replace("_", " ").title())
                st.session_state["sentence"] = current
        st.divider()
        st.subheader("Build a sentence")
        sentence = st.session_state.get("sentence", [])
        st.text_area("Recognized words", value=" ".join(sentence), height=100, key="sentence_display")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Speak sentence", use_container_width=True, disabled=not sentence):
                speak_text(" ".join(sentence))
        with c2:
            if st.button("Clear sentence", use_container_width=True):
                st.session_state["sentence"] = []
                st.rerun()
        st.caption("For the first demo, collect clear examples of each sign under similar camera conditions.")

elif page == "Collect Training Data":
    st.subheader("Collect examples for your own sign classes")
    st.write("For each label, show the gesture to the camera and save multiple samples. Aim for 20–40 examples per label, with small changes in hand position and distance.")
    label = st.selectbox("Gesture label", LABELS)
    source = st.radio("Image source", ["Camera snapshot", "Upload image"], horizontal=True)
    uploaded = st.camera_input("Capture a sample") if source == "Camera snapshot" else st.file_uploader("Upload sample", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        rgb = image_to_rgb(uploaded)
        if rgb is not None:
            st.image(rgb, use_container_width=True)
            features, _ = extract_features(rgb)
            if features is None:
                st.error("No hand detected. Try again with a clear, well-lit image.")
            else:
                if st.button(f"Save sample as {label}", type="primary"):
                    save_sample(label, features)
                    st.success(f"Saved one training example for {label}.")
    df = read_dataset()
    if not df.empty:
        st.subheader("Collected samples")
        counts = df["label"].value_counts().rename_axis("Gesture").reset_index(name="Samples")
        st.dataframe(counts, use_container_width=True, hide_index=True)
        st.download_button("Download training dataset CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="gesture_samples.csv", mime="text/csv")
        if st.button("Delete all collected samples"):
            os.remove(DATA_PATH)
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            st.rerun()
    else:
        st.info("No samples collected yet.")

elif page == "Train Model":
    st.subheader("Train the gesture classifier")
    df = read_dataset()
    if df.empty:
        st.warning("Collect training samples first.")
    else:
        counts = df["label"].value_counts()
        st.dataframe(counts.rename_axis("Gesture").reset_index(name="Samples"),
                     use_container_width=True, hide_index=True)
        usable = counts[counts >= 2].index.tolist()
        st.write(f"Labels with at least two examples: {len(usable)}")
        if len(usable) < 2:
            st.warning("Collect examples for at least two different gestures, preferably 20 or more per gesture.")
        if st.button("Train Random Forest model", type="primary", disabled=len(usable) < 2):
            train_df = df[df["label"].isin(usable)].copy()
            feature_cols = [c for c in train_df.columns if c.startswith("f")]
            X = train_df[feature_cols].astype(float).values
            y = train_df["label"].astype(str).values
            model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
            # Small datasets may not support a useful test split; train on all data in that case.
            if len(train_df) >= 10 and train_df["label"].value_counts().min() >= 2:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    model.fit(X_train, y_train)
                    score = accuracy_score(y_test, model.predict(X_test))
                    st.metric("Holdout accuracy", f"{score:.1%}")
                    st.caption("This score is only a small-sample estimate; test with new images too.")
                except ValueError:
                    model.fit(X, y)
                    st.info("Dataset too small for a reliable split; trained using all collected examples.")
            else:
                model.fit(X, y)
                st.info("Small dataset: trained on all samples. Collect more varied examples for a meaningful evaluation.")
            joblib.dump(model, MODEL_PATH)
            st.success("Model trained and saved. Open Live Recognition to test it.")
            st.write("Classes:", ", ".join(model.classes_))

elif page == "Project Information":
    st.subheader("Project overview")
    st.markdown("""
**Title:** SignBridge AI — Sign Language Gesture Recognition and Communication Assistant

**Problem statement:** Communication can be difficult when a signer and a non-signer do not share a language. This prototype recognizes a limited set of hand gestures and presents them as text, with optional speech output.

**AI approach:**
1. MediaPipe detects 21 hand landmarks.
2. Landmark coordinates are normalized to reduce the effect of hand position and scale.
3. A Random Forest classifier learns the gesture labels from examples collected by the user.
4. The model predicts a label for a new image.

**Important limitation:** This prototype recognizes isolated, trained gestures, not full sign language. Real sign languages use grammar, movement, facial expressions, and body posture. Do not use it for emergency or medical communication.
""")
    st.subheader("Suggested viva questions")
    with st.expander("Why use MediaPipe?"):
        st.write("It provides hand landmarks, which are compact features compared with classifying every image pixel.")
    with st.expander("Why use Random Forest?"):
        st.write("It works well on tabular landmark features, is quick to train, and is practical for a small prototype dataset.")
    with st.expander("How can it be improved?"):
        st.write("Collect a larger and more diverse ISL dataset, add temporal sequence modeling for moving signs, and evaluate with unseen users.")
    with st.expander("Does it translate complete Indian Sign Language?"):
        st.write("No. This prototype classifies only the gesture labels you train it on.")

st.divider()
st.caption("SignBridge AI · Artificial Intelligence Lab PBL · Educational prototype")
