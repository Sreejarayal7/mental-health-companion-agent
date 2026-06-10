import sqlite3
import bcrypt
import jwt
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv(override=True)

DB_PATH    = "data/journal.db"
SECRET_KEY = os.getenv("JWT_SECRET", "mental_health_secret_key_2026")
TOKEN_EXPIRY_HOURS = 24

def init_users_table():
    """Create users table if it doesn't exist."""
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email         TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            name          TEXT,
            created_at    TEXT
        )
    ''')
    # Add user_id column to entries if not exists
    try:
        cursor.execute(
            "ALTER TABLE entries ADD COLUMN user_id INTEGER DEFAULT 0"
        )
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.commit()
    conn.close()

def register_user(email, password, name):
    """
    Register a new user.
    Returns (True, user_id) on success, (False, error_message) on failure.
    """
    try:
        # Hash password with bcrypt (salt rounds=12)
        password_hash = bcrypt.hashpw(
            password.encode('utf-8'),
            bcrypt.gensalt(rounds=12)
        ).decode('utf-8')

        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (email, password_hash, name, created_at) "
            "VALUES (?, ?, ?, ?)",
            (email.lower().strip(), password_hash, name,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return True, user_id
    except sqlite3.IntegrityError:
        return False, "Email already registered. Please login instead."
    except Exception as e:
        return False, f"Registration failed: {str(e)}"

def login_user(email, password):
    """
    Verify login credentials.
    Returns (True, user_dict) on success, (False, error_message) on failure.
    """
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, email, password_hash, name FROM users WHERE email = ?",
            (email.lower().strip(),)
        )
        user = cursor.fetchone()
        conn.close()

        if not user:
            return False, "Email not found. Please sign up first."

        user_id, user_email, password_hash, name = user

        # Verify password with bcrypt
        if bcrypt.checkpw(password.encode('utf-8'),
                          password_hash.encode('utf-8')):
            return True, {
                "id":    user_id,
                "email": user_email,
                "name":  name
            }
        else:
            return False, "Incorrect password. Please try again."
    except Exception as e:
        return False, f"Login failed: {str(e)}"

def generate_token(user_id, email):
    """Generate JWT token valid for 24 hours."""
    payload = {
        "user_id": user_id,
        "email":   email,
        "exp":     datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(token):
    """
    Verify JWT token.
    Returns (True, payload) if valid, (False, error) if invalid/expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return True, payload
    except jwt.ExpiredSignatureError:
        return False, "Session expired. Please login again."
    except jwt.InvalidTokenError:
        return False, "Invalid session. Please login again."

def get_user_by_id(user_id):
    """Get user details by ID."""
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, email, name, created_at FROM users WHERE id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return {"id": row[0], "email": row[1],
                    "name": row[2], "created_at": row[3]}
        return None
    except Exception:
        return None

# Initialise table on module load
init_users_table()