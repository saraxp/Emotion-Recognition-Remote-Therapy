"""
Flask Application with Authentication
Emotion Recognition for Remote Therapy Sessions
"""

from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for
from flask_cors import CORS
import cv2
import numpy as np
import keras
from collections import deque
import datetime
import json
import os
from threading import Lock
from functools import wraps
from huggingface_hub import hf_hub_download

# Import configuration
import config

# Import Firebase database module
import database_firebase as db
from ai_summary import generate_summary_with_gemini, check_gemini_available

# ==================== APP INITIALIZATION ====================

app = Flask(__name__)

# 🔒 Load secret key from config
app.secret_key = config.FLASK_SECRET_KEY

# Load other Flask settings
app.config['DEBUG'] = config.DEBUG

CORS(app)

# Initialize Firebase
db.init_database()
db.create_default_users()

# Load emotion model
try:
    model_path = hf_hub_download(
        repo_id=config.HF_REPO_ID,
        filename=config.HF_MODEL_FILENAME,
        repo_type="model",      # important for model repos
        token=None              # public = no token
    )
    model = keras.models.load_model(model_path)
    print("✅ Model loaded from Hugging Face")
except Exception as e:
    print(f"⚠️  Failed to load model from HF: {e}")
    model = None

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # type: ignore

# Globals
camera = None
emotion_window = deque(maxlen=10)
data_lock = Lock()

current_session = {
    "session_id": None,
    "patient_id": None,
    "patient_name": None,
    "start_time": None,
    "emotions": [],
    "is_recording": False,
}

os.makedirs("session_data", exist_ok=True)

# ==================== AUTH DECORATORS ====================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


def doctor_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session or session.get("role") != "doctor":
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


def patient_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session or session.get("role") != "patient":
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# ==================== AUTH ROUTES ====================

@app.route("/")
def home():
    if "user_id" in session:
        if session.get("role") == "doctor":
            return redirect(url_for("doctor_dashboard"))
        else:
            return redirect(url_for("patient_session"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.json
        username = data.get("username") # type: ignore
        password = data.get("password") # type: ignore

        user = db.authenticate_user(username, password)
        if user:
            session["user_id"] = user["id"] # type: ignore
            session["username"] = user["username"] # type: ignore
            session["full_name"] = user["full_name"] # type: ignore
            session["role"] = user["role"] # type: ignore

            redirect_url = (
                "/doctor_dashboard" if user["role"] == "doctor" else "/patient_session" # type: ignore
            )
            return jsonify(
                {"success": True, "role": user["role"], "redirect": redirect_url} # type: ignore
            )
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ==================== PATIENT ROUTES ====================

@app.route("/patient_session")
@patient_required
def patient_session():
    return render_template("patient_session.html", user=session)


@app.route("/my_sessions")
@patient_required
def my_sessions():
    sessions = db.get_patient_sessions(session["user_id"])
    stats = db.get_session_statistics(session["user_id"])
    return render_template(
        "patient_history.html", sessions=sessions, stats=stats, user=session
    )

# ==================== DOCTOR ROUTES ====================

@app.route("/doctor_dashboard")
@doctor_required
def doctor_dashboard():
    """Doctor dashboard view"""
    import traceback

    def safe_to_str(obj):
        import datetime, numpy as np
        from google.cloud.firestore_v1 import DocumentReference
        if isinstance(obj, dict):
            return {str(k): safe_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [safe_to_str(i) for i in obj]
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, DocumentReference):
            return obj.id
        else:
            return str(obj)

    try:
        patients = db.get_all_patients()
        sessions = db.get_all_sessions(limit=50)
        stats = db.get_session_statistics()
        gemini_status = check_gemini_available()

        # 🔍 Print all data types for debugging
        print("\n================= DEBUG INFO =================")
        print("PATIENTS TYPE:", type(patients))
        if isinstance(patients, list) and patients:
            print("PATIENT[0] TYPE:", type(patients[0]))
            print("PATIENT[0] DATA:", patients[0])

        print("\nSESSIONS TYPE:", type(sessions))
        if isinstance(sessions, list) and sessions:
            print("SESSION[0] TYPE:", type(sessions[0]))
            print("SESSION[0] DATA:", sessions[0])

        print("\nSTATS TYPE:", type(stats))
        print("STATS DATA:", stats)
        print("GEMINI STATUS:", gemini_status)
        print("==============================================\n")

        # ✅ Clean everything before rendering
        patients = safe_to_str(patients)
        sessions = safe_to_str(sessions)
        stats = safe_to_str(stats)

        user_data = {k: str(v) for k, v in session.items()}

        return render_template(
            "doctor_dashboard.html",
            patients=patients,
            sessions=sessions,
            stats=stats,
            gemini_available=str(gemini_status),
            user=user_data,
        )

    except Exception as e:
        print("⚠️ ERROR in doctor_dashboard:", e)
        traceback.print_exc()
        return f"<pre>Error loading dashboard:\n{traceback.format_exc()}</pre>", 500



# ==================== VIDEO & EMOTION ====================

def preprocess_face(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)
    return face_img


def get_smoothed_emotion(predictions):
    emotion_window.append(predictions)
    return np.mean(emotion_window, axis=0)


def generate_frames():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)

    if not model:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "Model not loaded!",
            (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        ret, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        return

    frame_count = 0
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))

        frame_count += 1
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            processed_face = preprocess_face(face_roi)
            preds = model.predict(processed_face, verbose=0)[0]
            smoothed = get_smoothed_emotion(preds)

            idx = np.argmax(smoothed)
            emotion = emotion_labels[idx]
            confidence = smoothed[idx]

            if current_session["is_recording"] and frame_count % 30 == 0:
                with data_lock:
                    current_session["emotions"].append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "emotion": emotion,
                        "confidence": float(confidence)
                    })

            color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{emotion} ({confidence*100:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if current_session["is_recording"]:
            cv2.circle(frame, (20, 20), 10, (0, 0, 255), -1)
            cv2.putText(frame, f"Recording - {current_session['patient_name']}",
                        (40, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

@app.route("/video_feed")
@login_required
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ==================== SESSION MANAGEMENT ====================

@app.route("/start_session", methods=["POST"])
@patient_required
def start_session():
    with data_lock:
        session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{session['user_id']}"
        current_session.update({
            "session_id": session_id,
            "patient_id": session["user_id"],
            "patient_name": session["full_name"],
            "start_time": datetime.datetime.now().isoformat(),
            "emotions": [],
            "is_recording": True,
        })
        emotion_window.clear()
        db.create_session(session_id, session["user_id"])
    return jsonify({"status": "success", "session_id": session_id})

@app.route("/stop_session", methods=["POST"])
@patient_required
def stop_session():
    with data_lock:
        try:
            current_session["is_recording"] = False
            current_session["end_time"] = datetime.datetime.now().isoformat()

            print("🧠 Generating AI summary with Gemini...")
            ai_summary = generate_summary_with_gemini(
                current_session, current_session.get("patient_name", "Unknown")
            ) or "Summary not generated"

            db.end_session(current_session["session_id"], current_session, ai_summary)

            filename = f"session_data/{current_session['session_id']}.json"
            with open(filename, "w") as f:
                json.dump({**current_session, "ai_summary": ai_summary}, f, indent=4)

            return jsonify({
                "status": "success",
                "session_id": current_session["session_id"],
                "ai_summary": ai_summary
            })
        except Exception as e:
            print(f"⚠️ Error stopping session: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

# ==================== API ROUTES ====================

@app.route("/session/<session_id>")
@login_required
def get_session_details(session_id):
    session_data = db.get_session(session_id)
    if not session_data:
        return jsonify({"error": "Session not found"}), 404
    if session["role"] == "patient" and session_data["patient_id"] != session["user_id"]: # type: ignore
        return jsonify({"error": "Unauthorized"}), 403
    emotion_dist = db.get_emotion_distribution(session_id=session_id)
    session_data["emotion_distribution"] = emotion_dist # type: ignore
    return jsonify(session_data)

# ==================== RUN APP ====================

if __name__ == "__main__":
    print("=" * 70)
    print(" EMOTION RECOGNITION THERAPY SYSTEM ")
    print("=" * 70)
    print("✅ Flask initialized\n✅ Firebase connected\n✅ Model ready (if available)\n")
    
    # Check Gemini API status
    if check_gemini_available():
        print("✅ Gemini API configured - AI summaries enabled\n")
    else:
        print("⚠️  Gemini API not configured - using rule-based summaries\n")
        print("To enable AI summaries, set GEMINI_API_KEY environment variable")
        print("Get your free API key at: https://makersuite.google.com/app/apikey\n")
    
    port = int(os.environ.get('PORT', 5005))
    app.run(debug=True, host="0.0.0.0", port=port, threaded=True)

@app.get("/api/health")
def health():
    return {"ok": True}