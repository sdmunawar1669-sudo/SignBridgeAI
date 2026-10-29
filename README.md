# SignBridge AI — AI Lab PBL

A practical Python prototype for recognizing a limited set of hand gestures from a webcam snapshot or uploaded image, converting predictions into text, and optionally speaking the result.

## Features
- Webcam snapshot and image upload
- MediaPipe hand landmark extraction
- Landmark normalization
- User-collected training examples
- Random Forest gesture classifier
- Model confidence display
- Text-to-speech in supported browsers
- Sentence builder
- Dataset export and project/viva notes

## Important limitation
This is an educational isolated-gesture recognizer, not a complete sign-language translator. Indian Sign Language (ISL) has its own vocabulary and grammar. Only train labels that match the gestures and dataset you are using. Do not rely on the app for emergency, medical, or other high-stakes communication.

## Setup (Windows)
1. Install Python 3.10 or 3.11 for the best compatibility with MediaPipe.
2. Extract this ZIP to a folder.
3. Open Command Prompt in that folder.
4. Create a virtual environment:

   ```bat
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Run the app:

   ```bat
   streamlit run app.py
   ```

6. The app opens in your browser. Allow camera access if prompted.

## How to use
1. Open **Collect Training Data**.
2. Select a label and capture/upload a clear hand image.
3. Click **Save sample**. Repeat at least 20–40 times per label.
4. Collect examples for at least two different labels.
5. Open **Train Model** and click **Train Random Forest model**.
6. Open **Live Recognition** and test with new images.
7. Use **Speak recognized sign** or add recognized labels to the sentence builder.

## Suggested first demo
Start with 3–5 gestures that are easy for your team to reproduce consistently. Do not try to support all signs in the first week. Use the same hand orientation and lighting while collecting initial samples, then test with new images and different hand positions.

## Project algorithm
1. Read an image from the webcam or upload.
2. Detect the hand using MediaPipe.
3. Extract the 21 landmarks (x, y, z) = 63 values.
4. Translate coordinates relative to the wrist and scale-normalize.
5. Pass the features to the trained Random Forest.
6. Display the predicted label and model confidence.

## Suggested extensions
- Add temporal recognition for moving gestures.
- Evaluate on an independently collected dataset.
- Add a properly sourced ISL dataset and document its license.
- Add typed replies and speech output.
- Add a report with confusion matrix, precision, recall, and F1 score.
