from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import sqlite3
import hashlib
import secrets
from werkzeug.utils import secure_filename
import label_image

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ─── Database Setup ────────────────────────────────────────────────────────────

DB_PATH = os.path.join(app.root_path, 'users.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

init_db()

# ─── Helpers ──────────────────────────────────────────────────────────────────

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            # Return JSON error for AJAX/fetch requests instead of HTML redirect
            if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest' \
                    or request.path == '/predict':
                return jsonify({'error': 'Session expired. Please log in again.'}), 401
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

VITAMIN_INFO = {
    "Vitamin A": {
        "icon": "bi-eye",
        "color": "#f59e0b",
        "description": "Deficiency of Vitamin A is associated with significant morbidity and mortality from common childhood infections, and is the world's leading preventable cause of childhood blindness.",
        "symptoms": ["Night blindness", "Dry skin & eyes", "Frequent infections", "Growth delays"],
        "foods": ["Carrots", "Sweet potatoes", "Spinach", "Liver", "Eggs"]
    },
    "Vitamin B": {
        "icon": "bi-activity",
        "color": "#8b5cf6",
        "description": "Vitamin B12 deficiency may lead to a reduction in healthy red blood cells (anaemia). The nervous system may also be affected.",
        "symptoms": ["Fatigue & weakness", "Breathlessness", "Numbness in limbs", "Memory trouble", "Poor balance"],
        "foods": ["Meat & fish", "Eggs & dairy", "Fortified cereals", "Legumes", "Nuts & seeds"]
    },
    "Vitamin C": {
        "icon": "bi-shield-check",
        "color": "#10b981",
        "description": "Scurvy results from a severe lack of Vitamin C in the diet. Symptoms may not occur for months after dietary intake drops too low.",
        "symptoms": ["Bruising easily", "Bleeding gums", "Weakness & fatigue", "Skin rash", "Joint pain"],
        "foods": ["Citrus fruits", "Bell peppers", "Broccoli", "Strawberries", "Potatoes"]
    },
    "Vitamin D": {
        "icon": "bi-sun",
        "color": "#f97316",
        "description": "Vitamin D deficiency can lead to loss of bone density, contributing to osteoporosis and fractures. In children it can cause rickets.",
        "symptoms": ["Bone pain", "Muscle weakness", "Fatigue", "Depression", "Impaired healing"],
        "foods": ["Fatty fish", "Egg yolks", "Fortified milk", "Mushrooms", "Sunlight exposure"]
    },
    "Vitamin E": {
        "icon": "bi-heart-pulse",
        "color": "#ec4899",
        "description": "Vitamin E deficiency can cause nerve and muscle damage resulting in loss of feeling in arms and legs, muscle weakness, and vision problems.",
        "symptoms": ["Muscle weakness", "Vision problems", "Loss of body coordination", "Weakened immunity", "Numbness"],
        "foods": ["Sunflower seeds", "Almonds", "Spinach", "Avocado", "Vegetable oils"]
    }
}

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')

        errors = []
        if not name or len(name) < 2:
            errors.append('Name must be at least 2 characters.')
        if not email or '@' not in email:
            errors.append('Please enter a valid email address.')
        if len(password) < 6:
            errors.append('Password must be at least 6 characters.')
        if password != confirm:
            errors.append('Passwords do not match.')

        if errors:
            for e in errors:
                flash(e, 'danger')
            return render_template('register.html', name=name, email=email)

        try:
            with get_db() as conn:
                conn.execute(
                    'INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                    (name, email, hash_password(password))
                )
                conn.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('An account with this email already exists.', 'danger')
            return render_template('register.html', name=name, email=email)

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not email or not password:
            flash('Please fill in all fields.', 'danger')
            return render_template('login.html', email=email)

        with get_db() as conn:
            user = conn.execute(
                'SELECT * FROM users WHERE email = ? AND password = ?',
                (email, hash_password(password))
            ).fetchone()

        if user:
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            session['user_email'] = user['email']
            flash(f'Welcome back, {user["name"]}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')
            return render_template('login.html', email=email)

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(f.filename):
        return jsonify({'error': 'Only PNG and JPG files are allowed'}), 400

    filename = secure_filename(f.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(temp_path)

    try:
        # label_image.main() returns e.g. "vitamin a" (lowercase)
        raw_result = label_image.main(temp_path)

        # Normalize: "vitamin a" -> "Vitamin A"
        normalized = raw_result.strip().replace("_", " ").title()

        if os.path.exists(temp_path):
            os.remove(temp_path)

        if normalized in VITAMIN_INFO:
            info = VITAMIN_INFO[normalized]
            return jsonify({
                'vitamin': normalized,
                'description': info['description'],
                'symptoms': info['symptoms'],
                'foods': info['foods'],
                'icon': info['icon'],
                'color': info['color']
            })
        else:
            # Fuzzy match fallback
            for key in VITAMIN_INFO:
                if key.lower() in raw_result.lower() or raw_result.lower() in key.lower():
                    info = VITAMIN_INFO[key]
                    return jsonify({
                        'vitamin': key,
                        'description': info['description'],
                        'symptoms': info['symptoms'],
                        'foods': info['foods'],
                        'icon': info['icon'],
                        'color': info['color']
                    })
            return jsonify({
                'error': f'Unrecognized class: "{raw_result}". Check retrained_labels.txt.'
            }), 422

    except ValueError as ve:
        # Low confidence rejection
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(ve)}), 422

    except FileNotFoundError as fe:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(fe)}), 500

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/analysis')
@login_required
def analysis():
    return render_template('analysis.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
