"""
Firebase Database Module for Emotion Recognition System
Fully Safe Version with Automatic Type Cleaning
Updated to support environment variables for Render deployment
"""

import json
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
import os

# Firebase imports
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    print("⚠ firebase-admin not installed. Run: pip install firebase-admin")
    FIREBASE_AVAILABLE = False

db = None

# ==================== INITIALIZATION ====================

def init_firebase():
    """Initialize Firebase with service account from environment or file"""
    global db

    if not FIREBASE_AVAILABLE:
        print("✗ Firebase SDK not installed")
        return False

    try:
        # Check if already initialized
        try:
            firebase_admin.get_app()
            print("✓ Firebase already initialized")
            db = firestore.client()
            return True
        except ValueError:
            # Not initialized yet, proceed
            pass

        # Try to load from environment variable first (for Render/production)
        firebase_creds_json = os.environ.get('FIREBASE_CREDENTIALS')
        
        if firebase_creds_json:
            print("📦 Loading Firebase credentials from environment variable...")
            try:
                cred_dict = json.loads(firebase_creds_json)
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
                print("✅ Firebase initialized successfully from environment")
            except json.JSONDecodeError as e:
                print(f"✗ Invalid JSON in FIREBASE_CREDENTIALS: {e}")
                return False
            except Exception as e:
                print(f"✗ Error initializing from environment: {e}")
                return False
                
        elif os.path.exists("serviceAccountKey.json"):
            # Fallback to file for local development
            print("📦 Loading Firebase credentials from local file...")
            cred = credentials.Certificate("serviceAccountKey.json")
            firebase_admin.initialize_app(cred)
            print("✅ Firebase initialized successfully from file")
            
        else:
            print("✗ Firebase credentials not found!")
            print("\nFor production (Render):")
            print("  Set FIREBASE_CREDENTIALS environment variable with your Firebase JSON")
            print("\nFor local development:")
            print("  1. Go to Firebase Console → Project Settings → Service Accounts")
            print("  2. Generate a new private key")
            print("  3. Save as serviceAccountKey.json in the project root")
            return False

        db = firestore.client()
        return True
        
    except Exception as e:
        print(f"✗ Firebase initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False


def init_database():
    """Initialize database connection"""
    if not init_firebase():
        print("✗ Firebase not initialized")
        return False
    print("✅ Firebase Firestore ready")
    print("  Collections will be created automatically when data is added")
    return True


# ==================== DATA SANITIZATION ====================

def clean_for_template(data):
    """Recursively clean data for safe rendering and JSON serialization."""
    import datetime
    import numpy as np
    from google.cloud.firestore_v1 import DocumentReference

    if isinstance(data, dict):
        return {str(k): clean_for_template(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_for_template(i) for i in data]
    elif isinstance(data, (int, float, str, bool)) or data is None:
        return data
    elif isinstance(data, datetime.datetime):
        return data.isoformat()
    elif isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, DocumentReference):
        return data.id
    else:
        try:
            return str(data)
        except Exception:
            return "unreadable"


# ==================== USER MANAGEMENT ====================

def create_default_users():
    """Create default demo users in Firebase"""
    if not db:
        print("✗ Firebase not initialized")
        return

    users_ref = db.collection("users")

    doctor_query = users_ref.where("username", "==", "doctor").limit(1).get()
    if len(list(doctor_query)) > 0:
        print("✓ Demo users already exist")
        return

    doctor_data = {
        "username": "doctor",
        "password_hash": generate_password_hash("doctor123"),
        "full_name": "Dr. Sarah Johnson",
        "email": "doctor@therapy.com",
        "role": "doctor",
        "created_at": firestore.SERVER_TIMESTAMP,
    }
    users_ref.add(doctor_data)

    patients = [
        ("patient1", "patient123", "Parth", "john@email.com"),
        ("patient2", "patient123", "Sara", "jane@email.com"),
        ("patient3", "patient123", "vaibhavi", "mike@email.com"),
    ]

    for username, password, name, email in patients:
        users_ref.add({
            "username": username,
            "password_hash": generate_password_hash(password),
            "full_name": name,
            "email": email,
            "role": "patient",
            "created_at": firestore.SERVER_TIMESTAMP,
        })

    print("✓ Demo users created in Firebase")


def authenticate_user(username, password):
    """Authenticate user and return user data"""
    if not db:
        return None

    query = db.collection("users").where("username", "==", username).limit(1).get()
    for doc in query:
        user_data = doc.to_dict()
        if check_password_hash(user_data["password_hash"], password):
            doc.reference.update({"last_login": firestore.SERVER_TIMESTAMP})
            user_data["id"] = doc.id
            return clean_for_template(user_data)
    return None


def get_user_by_id(user_id):
    """Get user details by document ID"""
    if not db:
        return None

    # Convert ID to string just in case
    user_id = str(user_id)

    doc = db.collection("users").document(user_id).get()

    if doc.exists:
        user_data = doc.to_dict()
        user_data["id"] = doc.id
        return clean_for_template(user_data)

    return None


def get_all_patients():
    """Get all patients"""
    if not db:
        return []
    patients = []
    query = db.collection("users").where("role", "==", "patient").stream()
    for doc in query:
        data = doc.to_dict()
        data["id"] = doc.id
        patients.append(clean_for_template(data))
    return patients


# ==================== SESSION MANAGEMENT ====================

def create_session(session_id, patient_id, doctor_id=None):
    """Create a new therapy session in Firebase"""
    if not db:
        return

    # Ensure IDs are stored as strings
    session_data = {
        "session_id": str(session_id),
        "patient_id": str(patient_id),
        "doctor_id": str(doctor_id) if doctor_id else None,
        "start_time": datetime.now(),
        "created_at": firestore.SERVER_TIMESTAMP,
    }

    db.collection("sessions").document(str(session_id)).set(clean_for_template(session_data))


def end_session(session_id, emotion_data, ai_summary=None):
    """End a therapy session and save data properly with duration"""
    if not db:
        return

    try:
        session_ref = db.collection("sessions").document(str(session_id))
        session_doc = session_ref.get()

        if session_doc.exists:
            session_info = session_doc.to_dict()

            # Handle both datetime and Firestore timestamp formats
            start_time = session_info.get("start_time")
            if hasattr(start_time, "to_datetime"):
                start_time = start_time.to_datetime()
            elif isinstance(start_time, str):
                try:
                    start_time = datetime.fromisoformat(start_time)
                except Exception:
                    start_time = datetime.now()

            end_time = datetime.now()
            duration_seconds = int((end_time - start_time).total_seconds())

            # Update session
            session_ref.update({
                "end_time": end_time,
                "duration_seconds": duration_seconds,
                "total_emotions_detected": len(emotion_data.get("emotions", [])),
                "emotion_data": emotion_data,
                "ai_summary": ai_summary,
            })

            # Store emotion records in subcollection
            emotions_ref = session_ref.collection("emotion_records")
            for record in emotion_data.get("emotions", []):
                emotions_ref.add({
                    "timestamp": datetime.fromisoformat(record["timestamp"]),
                    "emotion": record["emotion"],
                    "confidence": record["confidence"],
                    "modality": "facial",
                })

            print(f"✓ Session {session_id} ended, duration = {duration_seconds}s")

        else:
            print(f"⚠ Session {session_id} not found")

    except Exception as e:
        print(f"✗ Error in end_session: {e}")


def get_session(session_id):
    if not db:
        return None
    doc = db.collection("sessions").document(session_id).get()
    if not doc.exists:
        return None
    data = doc.to_dict()
    data["id"] = doc.id

    if "patient_id" in data:
        patient = get_user_by_id(data["patient_id"])
        if patient:
            data["patient_name"] = patient["full_name"]
            data["patient_username"] = patient["username"]

    return clean_for_template(data)


def get_all_sessions(limit=100):
    """Get all sessions (for doctor dashboard)"""
    if not db:
        return []

    sessions_ref = db.collection("sessions")
    query = sessions_ref.order_by("start_time", direction=firestore.Query.DESCENDING).limit(limit)

    sessions = []
    for doc in query.stream():
        data = doc.to_dict()
        data["id"] = doc.id

        # 🔧 Ensure IDs are strings before querying users
        patient_id = str(data.get("patient_id", ""))
        if patient_id:
            try:
                patient = get_user_by_id(patient_id)
                if patient:
                    data["patient_name"] = patient.get("full_name", "")
                    data["patient_username"] = patient.get("username", "")
            except Exception as e:
                print(f"⚠️ Skipped patient lookup for {patient_id}: {e}")

        sessions.append(clean_for_template(data))

    return sessions


def get_patient_sessions(patient_id, limit=50):
    if not db:
        return []
    query = db.collection("sessions").where("patient_id", "==", patient_id).order_by("start_time", direction=firestore.Query.DESCENDING).limit(limit)
    sessions = []
    for doc in query.stream():
        data = doc.to_dict()
        data["id"] = doc.id
        sessions.append(clean_for_template(data))
    return sessions


def get_session_statistics(patient_id=None):
    """Compute safe statistics"""
    if not db:
        return {}

    query = db.collection("sessions")
    if patient_id:
        query = query.where("patient_id", "==", patient_id)

    total_sessions = 0
    total_duration = 0
    total_emotions = 0

    for doc in query.stream():
        data = doc.to_dict()
        if "end_time" not in data:
            continue
        try:
            total_sessions += 1
            total_duration += int(float(data.get("duration_seconds", 0) or 0))
            total_emotions += int(float(data.get("total_emotions_detected", 0) or 0))
        except Exception:
            continue

    avg_duration = float(total_duration / total_sessions) if total_sessions > 0 else 0.0
    stats = {
        "total_sessions": total_sessions,
        "total_duration": total_duration,
        "avg_duration": avg_duration,
        "total_emotions": total_emotions,
    }
    return clean_for_template(stats)


def get_emotion_distribution(session_id=None, patient_id=None):
    if not db:
        return []
    emotion_counts = {}
    if session_id:
        ref = db.collection("sessions").document(session_id).collection("emotion_records")
        for doc in ref.stream():
            e = doc.to_dict().get("emotion", "Unknown")
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
    elif patient_id:
        sessions = get_patient_sessions(patient_id)
        for s in sessions:
            if "emotion_data" in s and s["emotion_data"]:
                for e in s["emotion_data"].get("emotions", []):
                    emotion_counts[e["emotion"]] = emotion_counts.get(e["emotion"], 0) + 1
    dist = [{"emotion": k, "count": v} for k, v in emotion_counts.items()]
    return clean_for_template(sorted(dist, key=lambda x: x["count"], reverse=True))