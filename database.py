"""
Database module for Emotion Recognition System
Uses SQLite (free, no setup required)
"""

import sqlite3
import json
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
import os

DATABASE_PATH = 'therapy_sessions.db'

def get_db_connection():
    """Create database connection"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn

def init_database():
    """Initialize database with required tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Users table (both patients and doctors)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT,
            role TEXT NOT NULL CHECK(role IN ('patient', 'doctor')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    # Sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            patient_id INTEGER NOT NULL,
            doctor_id INTEGER,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            duration_seconds INTEGER,
            total_emotions_detected INTEGER DEFAULT 0,
            ai_summary TEXT,
            emotion_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES users(id),
            FOREIGN KEY (doctor_id) REFERENCES users(id)
        )
    ''')
    
    # Emotion records table (detailed emotion tracking)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emotion_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            modality TEXT CHECK(modality IN ('facial', 'voice')),
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    ''')
    
    # Create indexes for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_patient ON sessions(patient_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_doctor ON sessions(doctor_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_emotion_records_session ON emotion_records(session_id)')
    
    conn.commit()
    conn.close()
    
    print("✓ Database initialized successfully")

def create_default_users():
    """Create default demo users"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if users already exist
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] > 0:
        conn.close()
        return
    
    # Create demo doctor
    cursor.execute('''
        INSERT INTO users (username, password_hash, full_name, email, role)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        'doctor',
        generate_password_hash('doctor123', method='bcrypt'),
        'Dr. Sarah Johnson',
        'doctor@therapy.com',
        'doctor'
    ))
    
    # Create demo patients
    patients = [
        ('patient1', 'patient123', 'John Doe', 'john@email.com'),
        ('patient2', 'patient123', 'Jane Smith', 'jane@email.com'),
        ('patient3', 'patient123', 'Mike Wilson', 'mike@email.com'),
    ]
    
    for username, password, name, email in patients:
        cursor.execute('''
            INSERT INTO users (username, password_hash, full_name, email, role)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            username,
            generate_password_hash(password, method='bcrypt'),
            name,
            email,
            'patient'
        ))
    
    conn.commit()
    conn.close()
    
    print("✓ Demo users created:")
    print("  Doctor: username='doctor', password='doctor123'")
    print("  Patients: username='patient1/2/3', password='patient123'")

# ==================== USER MANAGEMENT ====================

def authenticate_user(username, password):
    """Authenticate user and return user data"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    
    if user and check_password_hash(user['password_hash'], password):
        # Update last login
        cursor.execute('UPDATE users SET last_login = ? WHERE id = ?',
                      (datetime.now(), user['id']))
        conn.commit()
        conn.close()
        
        return {
            'id': user['id'],
            'username': user['username'],
            'full_name': user['full_name'],
            'email': user['email'],
            'role': user['role']
        }
    
    conn.close()
    return None

def create_user(username, password, full_name, email, role):
    """Create a new user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO users (username, password_hash, full_name, email, role)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            username,
            generate_password_hash(password, method='bcrypt'),
            full_name,
            email,
            role
        ))
        
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        
        return user_id
        
    except sqlite3.IntegrityError:
        conn.close()
        return None

def get_user_by_id(user_id):
    """Get user details by ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return dict(user)
    return None

def get_all_patients():
    """Get all patients (for doctor dashboard)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, full_name, email, created_at, last_login
        FROM users WHERE role = 'patient'
        ORDER BY full_name
    ''')
    
    patients = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return patients

# ==================== SESSION MANAGEMENT ====================

def create_session(session_id, patient_id, doctor_id=None):
    """Create a new therapy session"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO sessions (session_id, patient_id, doctor_id, start_time)
        VALUES (?, ?, ?, ?)
    ''', (session_id, patient_id, doctor_id, datetime.now()))
    
    conn.commit()
    conn.close()

def end_session(session_id, emotion_data, ai_summary=None):
    """End a therapy session and save data"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get session start time
    cursor.execute('SELECT start_time FROM sessions WHERE session_id = ?', (session_id,))
    session = cursor.fetchone()
    
    if session:
        start_time = datetime.fromisoformat(session['start_time'])
        end_time = datetime.now()
        duration = int((end_time - start_time).total_seconds())
        
        # Update session
        cursor.execute('''
            UPDATE sessions
            SET end_time = ?, duration_seconds = ?, 
                total_emotions_detected = ?, emotion_data = ?, ai_summary = ?
            WHERE session_id = ?
        ''', (
            end_time,
            duration,
            len(emotion_data.get('emotions', [])),
            json.dumps(emotion_data),
            ai_summary,
            session_id
        ))
        
        # Save individual emotion records
        for emotion_record in emotion_data.get('emotions', []):
            cursor.execute('''
                INSERT INTO emotion_records (session_id, timestamp, emotion, confidence, modality)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                session_id,
                emotion_record['timestamp'],
                emotion_record['emotion'],
                emotion_record['confidence'],
                'facial'
            ))
        
        conn.commit()
    
    conn.close()

def get_session(session_id):
    """Get session details"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT s.*, u.full_name as patient_name, u.username as patient_username
        FROM sessions s
        JOIN users u ON s.patient_id = u.id
        WHERE s.session_id = ?
    ''', (session_id,))
    
    session = cursor.fetchone()
    conn.close()
    
    if session:
        session_dict = dict(session)
        if session_dict['emotion_data']:
            session_dict['emotion_data'] = json.loads(session_dict['emotion_data'])
        return session_dict
    
    return None

def get_patient_sessions(patient_id, limit=50):
    """Get all sessions for a specific patient"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT s.*, u.full_name as patient_name
        FROM sessions s
        JOIN users u ON s.patient_id = u.id
        WHERE s.patient_id = ?
        ORDER BY s.start_time DESC
        LIMIT ?
    ''', (patient_id, limit))
    
    sessions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return sessions

def get_all_sessions(limit=100):
    """Get all sessions (for doctor dashboard)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT s.*, u.full_name as patient_name, u.username as patient_username
        FROM sessions s
        JOIN users u ON s.patient_id = u.id
        ORDER BY s.start_time DESC
        LIMIT ?
    ''', (limit,))
    
    sessions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return sessions

def get_session_statistics(patient_id=None):
    """Get session statistics"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if patient_id:
        cursor.execute('''
            SELECT 
                COUNT(*) as total_sessions,
                SUM(duration_seconds) as total_duration,
                AVG(duration_seconds) as avg_duration,
                SUM(total_emotions_detected) as total_emotions
            FROM sessions
            WHERE patient_id = ? AND end_time IS NOT NULL
        ''', (patient_id,))
    else:
        cursor.execute('''
            SELECT 
                COUNT(*) as total_sessions,
                SUM(duration_seconds) as total_duration,
                AVG(duration_seconds) as avg_duration,
                SUM(total_emotions_detected) as total_emotions
            FROM sessions
            WHERE end_time IS NOT NULL
        ''')
    
    stats = dict(cursor.fetchone())
    conn.close()
    
    return stats

def get_emotion_distribution(session_id=None, patient_id=None):
    """Get emotion distribution for session or patient"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if session_id:
        cursor.execute('''
            SELECT emotion, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM emotion_records
            WHERE session_id = ?
            GROUP BY emotion
            ORDER BY count DESC
        ''', (session_id,))
    elif patient_id:
        cursor.execute('''
            SELECT er.emotion, COUNT(*) as count, AVG(er.confidence) as avg_confidence
            FROM emotion_records er
            JOIN sessions s ON er.session_id = s.session_id
            WHERE s.patient_id = ?
            GROUP BY er.emotion
            ORDER BY count DESC
        ''', (patient_id,))
    else:
        cursor.execute('''
            SELECT emotion, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM emotion_records
            GROUP BY emotion
            ORDER BY count DESC
        ''')
    
    distribution = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return distribution

# ==================== CLEANUP ====================

def cleanup_old_sessions(days=7):
    """Delete sessions older than specified days"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cutoff_date = datetime.now() - timedelta(days=days)
    
    # Get sessions to delete
    cursor.execute('SELECT session_id FROM sessions WHERE start_time < ?', (cutoff_date,))
    old_sessions = [row['session_id'] for row in cursor.fetchall()]
    
    # Delete emotion records
    cursor.execute('DELETE FROM emotion_records WHERE session_id IN (SELECT session_id FROM sessions WHERE start_time < ?)', (cutoff_date,))
    
    # Delete sessions
    cursor.execute('DELETE FROM sessions WHERE start_time < ?', (cutoff_date,))
    
    conn.commit()
    deleted_count = len(old_sessions)
    conn.close()
    
    return deleted_count

# ==================== INITIALIZATION ====================

if __name__ == "__main__":
    print("Initializing database...")
    init_database()
    create_default_users()
    print("\nDatabase ready!")
    print("\nDefault credentials:")
    print("  Doctor: doctor / doctor123")
    print("  Patient: patient1 / patient123")