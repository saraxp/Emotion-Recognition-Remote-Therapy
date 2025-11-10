"""
Flask Application with Authentication
Emotion Recognition for Remote Therapy Sessions
FIXED: Complete emotion detection logic for deployment
"""

from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for
from flask_cors import CORS
import cv2
import numpy as np
import keras
from collections import deque, Counter
import datetime
import json
import os
from threading import Lock
import base64
import re
from functools import wraps
from huggingface_hub import hf_hub_download

# Import configuration
import config

# Import Firebase database module
import database_firebase as db
from ai_summary import generate_summary_with_gemini, check_gemini_available

# ==================== APP INITIALIZATION ====================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB upload cap
app.secret_key = config.FLASK_SECRET_KEY
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
        repo_type="model",
        token=None
    )
    model = keras.models.load_model(model_path)
    print("✅ Model loaded from Hugging Face")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ Failed to load model from HF: {e}")
    print("   Trying local file...")
    
    # Try local file
    if os.path.exists('emotion_model_best.h5'):
        model = keras.models.load_model('emotion_model_best.h5')
        print("✅ Model loaded from local file")
    else:
        model = None
        print("❌ No model found!")

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Load face cascade
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
except:
    cascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)

# Globals
camera = None
emotion_window = deque(maxlen=10)
data_lock = Lock()

# ⭐ FIXED: Per-session frame counter
current_session = {
    "session_id": None,
    "patient_id": None,
    "patient_name": None,
    "start_time": None,
    "emotions": [],
    "is_recording": False,
    "frame_counter": 0,  # NEW: Track frames properly
    "last_save_frame": 0,  # NEW: Track last emotion save
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
        username = data.get("username")
        password = data.get("password")

        user = db.authenticate_user(username, password)
        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["full_name"] = user["full_name"]
            session["role"] = user["role"]

            redirect_url = (
                "/doctor_dashboard" if user["role"] == "doctor" else "/patient_session"
            )
            return jsonify(
                {"success": True, "role": user["role"], "redirect": redirect_url}
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

# ==================== EMOTION PROCESSING ====================

def preprocess_face(face_img):
    """Preprocess face image for model prediction - SAME AS WORKING SCRIPT"""
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)
    return face_img

def get_smoothed_emotion(predictions):
    """Smooth emotion predictions - SAME AS WORKING SCRIPT"""
    emotion_window.append(predictions)
    avg_predictions = np.mean(emotion_window, axis=0)
    return avg_predictions

# ==================== VIDEO FEED (FALLBACK) ====================

def generate_frames():
    """Fallback video feed using OpenCV (for local testing only)"""
    global camera
    
    if camera is None:
        camera = cv2.VideoCapture(0)

    if not model:
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "MODEL NOT LOADED!", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            import time
            time.sleep(0.1)

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            processed_face = preprocess_face(face_roi)
            predictions = model.predict(processed_face, verbose=0)[0]
            smoothed_predictions = get_smoothed_emotion(predictions)
            emotion_idx = np.argmax(smoothed_predictions)
            emotion = emotion_labels[emotion_idx]
            confidence = smoothed_predictions[emotion_idx]

            color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{emotion} ({confidence*100:.1f}%)"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if current_session["is_recording"]:
            cv2.circle(frame, (20, 20), 10, (0, 0, 255), -1)

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
            "frame_counter": 0,  # Reset frame counter
            "last_save_frame": 0,  # Reset save tracker
        })
        emotion_window.clear()
        db.create_session(session_id, session["user_id"])
        
        print(f"\n🎬 SESSION STARTED")
        print(f"   Session ID: {session_id}")
        print(f"   Patient: {session['full_name']}")
    
    return jsonify({"status": "success", "session_id": session_id})

@app.route("/stop_session", methods=["POST"])
@patient_required
def stop_session():
    with data_lock:
        try:
            current_session["is_recording"] = False
            current_session["end_time"] = datetime.datetime.now().isoformat()
            
            emotion_count = len(current_session["emotions"])
            
            print(f"\n🛑 SESSION STOPPED")
            print(f"   Session ID: {current_session['session_id']}")
            print(f"   Emotions captured: {emotion_count}")
            
            if emotion_count == 0:
                print(f"   ⚠️ WARNING: No emotions detected!")

            print("🧠 Generating AI summary...")
            ai_summary = generate_summary_with_gemini(
                current_session, current_session.get("patient_name", "Unknown")
            ) or "Summary not generated"

            db.end_session(current_session["session_id"], current_session, ai_summary)

            filename = f"session_data/{current_session['session_id']}.json"
            with open(filename, "w") as f:
                json.dump({**current_session, "ai_summary": ai_summary}, f, indent=4)
            
            print(f"✅ Session saved: {filename}\n")

            return jsonify({
                "status": "success",
                "session_id": current_session["session_id"],
                "ai_summary": ai_summary,
                "total_emotions": emotion_count  
            })
        except Exception as e:
            print("Error:", e)
            return jsonify({"error": str(e)}), 500

# ==================== FRAME PROCESSING (DEPLOYMENT) ====================

@app.route("/process_frame", methods=["POST"])
@patient_required
def process_frame_api():
    """⭐ FIXED: Accept base64 frame from browser and detect emotions properly"""
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 503

    data = request.get_json(silent=True) or {}
    img_b64 = data.get("image")
    if not img_b64:
        return jsonify({"success": False, "error": "No image provided"}), 400

    # Remove data URI prefix if present
    m = re.match(r"^data:image/\w+;base64,(.*)$", img_b64)
    if m:
        img_b64 = m.group(1)

    try:
        # Decode base64 to image
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"success": False, "error": "Failed to decode image"}), 400

        # Increment frame counter
        with data_lock:
            if current_session.get("is_recording"):
                current_session["frame_counter"] += 1

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces - SAME PARAMETERS AS WORKING SCRIPT
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(48, 48)
        )

        faces_out = []
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess face
            processed_face = preprocess_face(face_roi)
            
            # Predict emotion
            preds = model.predict(processed_face, verbose=0)[0]
            
            # Smooth predictions
            smoothed = get_smoothed_emotion(preds)
            
            # Get dominant emotion
            idx = int(np.argmax(smoothed))
            emotion = emotion_labels[idx]
            confidence = float(smoothed[idx])

            faces_out.append({
                "x": int(x), 
                "y": int(y), 
                "w": int(w), 
                "h": int(h),
                "emotion": emotion, 
                "confidence": confidence
            })

            # Save emotion every 10 frames (roughly every 20 seconds at 2s interval)
            with data_lock:
                if current_session.get("is_recording"):
                    frames_since_save = current_session["frame_counter"] - current_session["last_save_frame"]
                    
                    if frames_since_save >= 10:  # Save every 10 processed frames
                        timestamp = datetime.datetime.now().isoformat()
                        emotion_data = {
                            "timestamp": timestamp,
                            "emotion": emotion,
                            "confidence": confidence,
                            "all_probabilities": {
                                label: float(smoothed[i]) 
                                for i, label in enumerate(emotion_labels)
                            }
                        }
                        current_session["emotions"].append(emotion_data)
                        current_session["last_save_frame"] = current_session["frame_counter"]
                        
                        print(f"💾 Saved emotion #{len(current_session['emotions'])}: {emotion} ({confidence*100:.1f}%) [Frame {current_session['frame_counter']}]")

        # Return first face as primary emotion
        top_emotion = faces_out[0] if faces_out else None
        
        return jsonify({
            "success": True, 
            "faces": faces_out, 
            "emotion": top_emotion
        })

    except Exception as e:
        print(f"❌ Error in process_frame: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# ==================== SESSION STATS (MISSING ENDPOINT) ====================

@app.route("/session_stats")
@patient_required
def session_stats():
    """⭐ NEW: Endpoint for real-time session statistics"""
    with data_lock:
        if not current_session.get("is_recording"):
            return jsonify({"error": "No active session"}), 400
        
        emotions = current_session.get("emotions", [])
        total_detections = len(emotions)
        
        # Calculate emotion distribution
        emotion_counts = Counter([e["emotion"] for e in emotions])
        emotion_percentages = {}
        
        if total_detections > 0:
            for emotion in emotion_labels:
                count = emotion_counts.get(emotion, 0)
                emotion_percentages[emotion] = round((count / total_detections) * 100, 1)
        else:
            emotion_percentages = {emotion: 0 for emotion in emotion_labels}
        
        return jsonify({
            "total_detections": total_detections,
            "emotion_percentages": emotion_percentages,
            "session_id": current_session.get("session_id"),
            "is_recording": current_session.get("is_recording", False)
        })

# ==================== API ROUTES ====================

@app.route("/api/health")
def health():
    return jsonify({
        "ok": True,
        "model_loaded": model is not None,
        "camera_active": camera is not None,
    })

@app.route("/session/<session_id>")
@login_required
def get_session_details(session_id):
    session_data = db.get_session(session_id)
    if not session_data:
        return jsonify({"error": "Session not found"}), 404
    if session["role"] == "patient" and session_data["patient_id"] != session["user_id"]:
        return jsonify({"error": "Unauthorized"}), 403
    emotion_dist = db.get_emotion_distribution(session_id=session_id)
    session_data["emotion_distribution"] = emotion_dist
    return jsonify(session_data)

# ==================== RUN APP ====================

if __name__ == "__main__":
    print("=" * 70)
    print(" EMOTION RECOGNITION THERAPY SYSTEM ")
    print("=" * 70)
    print("✅ Flask initialized")
    print("✅ Firebase connected")
    
    if model:
        print("✅ Emotion model loaded and ready")
    else:
        print("❌ Emotion model NOT loaded - detection will not work!")
    
    print()
    
    if check_gemini_available():
        print("✅ Gemini API configured\n")
    else:
        print("⚠️ Gemini API not configured\n")
    
    port = int(os.environ.get('PORT', 5005))
    app.run(debug=True, host="0.0.0.0", port=port, threaded=True)