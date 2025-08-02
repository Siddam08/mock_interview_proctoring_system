Mock Interview Proctoring System
A real-time proctoring application built using Python, OpenCV, and machine learning. It is designed to monitor candidates during mock interviews by detecting multiple violations such as:

No face detected

Multiple faces

Looking away

Phone detection (YOLOv5)

Voice emotion (e.g., stress, nervousness)

All suspicious activities are logged with:

Timestamped snapshots

Audio recordings

Emotion classification

📂 Project Structure
css
Copy
Edit
mock_interview_proctoring_system/
├── emotion_predictor_full.py
├── extract_features.py
├── multiple_face_eye_tracking.py
├── predict_emotion.py
├── record_audio.py
├── requirements.txt
├── run_all_proctoring.py
├── train_model.py
├── violations/
│   └── [Violation folders with snapshots and audio logs]
└── predictions/
    └── [Predicted emotion audio logs]
✅ Features
✅ Real-time face detection

✅ Multiple face detection

✅ Head pose estimation (to detect looking away)

✅ YOLOv5 phone detection

✅ Audio recording on violations

✅ Voice emotion recognition from 5-second audio clips

✅ Snapshot & audio logging with timestamp and labels

✅ Webcam window overlay for live violation status

⚙️ Requirements
Install required libraries:

bash
Copy
Edit
pip install -r requirements.txt
Dependencies include:

OpenCV

dlib

numpy

librosa

scikit-learn

torch, torchvision (for YOLO)

sounddevice

scipy

🚀 How to Run
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/Siddam08/mock_interview_proctoring_system.git
cd mock_interview_proctoring_system
2. (Optional) Set up virtual environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Train the Emotion Detection Model (One-time)
bash
Copy
Edit
python train_model.py
This will create emotion_model.pkl using 180 MFCC features extracted from recorded audio samples.

5. Start Proctoring System
bash
Copy
Edit
python run_all_proctoring.py
Real-time webcam feed opens with live alerts.

On any violation:

Snapshot is saved in: violations/<violation_type>/

Audio clip saved in: violations/<violation_type>/

Predicted emotion audio saved in: predictions/

📌 Notes
Make sure your webcam and microphone are accessible.

Violations are stored automatically in timestamped folders.

If model is missing: ensure emotion_model.pkl is generated from train_model.py.

📧 Author
Siddam SriTeja
GitHub: Siddam08
