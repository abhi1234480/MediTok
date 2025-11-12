import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import os, torch, tempfile, shutil
from werkzeug.utils import secure_filename
from ECG_report import GNNHybrid, predict, generate_ecg_report, DEVICE, MODEL_PATH, NUM_CLASSES

app = Flask(__name__)
app.secret_key = 'unmyeong'
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DATABASE = 'database.db'

# Load model once at startup
model = GNNHybrid(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            phone_number TEXT NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        phone_number = request.form['phone_number']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')

        conn = get_db_connection()
        user_check = conn.execute(
            'SELECT * FROM users WHERE username = ? OR email = ?',
            (username, email)
        ).fetchone()

        if user_check:
            flash('Username or email already exists. Please choose a different one.', 'error')
            conn.close()
            return render_template('register.html')
        hashed_password = generate_password_hash(password)

        try:
            conn.execute(
                'INSERT INTO users (username, email, phone_number, password) VALUES (?, ?, ?, ?)',
                (username, email, phone_number, hashed_password)
            )
            conn.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('An error occurred during registration. Please try again.', 'error')
        finally:
            conn.close()

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/home')
def home():
    if 'username' not in session:
        flash('Please log in to access the home page.', 'error')
        return redirect(url_for('login'))

    conn = get_db_connection()
    user = conn.execute(
        'SELECT username, email, phone_number FROM users WHERE username = ?',
        (session['username'],)
    ).fetchone()
    conn.close()

    return render_template('home.html', user=user)

@app.route('/predict', methods=['GET', 'POST'])
def predict_route():
    if request.method == 'POST':
        dat_file = request.files.get('dat_file')
        hea_file = request.files.get('hea_file')

        if not dat_file or not hea_file:
            return render_template("predict.html", error="Please upload both .dat and .hea files.")

        temp_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])

        try:
            dat_filename = secure_filename(dat_file.filename)
            hea_filename = secure_filename(hea_file.filename)

            dat_path = os.path.join(temp_dir, dat_filename)
            hea_path = os.path.join(temp_dir, hea_filename)

            dat_file.save(dat_path)
            hea_file.save(hea_path)
            basefile_noext_dat = os.path.splitext(dat_path)[0]
            basefile_noext_hea = os.path.splitext(hea_path)[0]

            if os.path.basename(basefile_noext_dat) != os.path.basename(basefile_noext_hea):
                return render_template("predict.html", error="The uploaded .dat and .hea files must have the same base name.")

            basefile_noext = basefile_noext_dat 
            res, preds = predict(basefile_noext, model)
            report_result = generate_ecg_report(preds, res)

            if report_result["success"]:
                report = report_result["content"]
                api_error = None
            else:
                report = None
                api_error = report_result["error"]

            if preds:
                conn = get_db_connection()
                conn.execute(
                    'INSERT INTO predictions (username, predicted_class) VALUES (?, ?)',
                    (session.get('username', 'guest'), preds[0])
                )
                conn.commit()
                conn.close()

            return render_template(
                "predict.html",
                probabilities=res,
                predicted=preds,
                report=report,
                api_error=api_error
            )

        finally:
            shutil.rmtree(temp_dir)

    return render_template("predict.html")

@app.route('/history')
def history():
    if 'username' not in session:
        flash("You must be logged in to view prediction history.", "error")
        return redirect(url_for('login'))

    conn = get_db_connection()
    rows = conn.execute(
        'SELECT predicted_class, created_at FROM predictions WHERE username = ? ORDER BY created_at DESC',
        (session['username'],)
    ).fetchall()
    conn.close()

    return render_template("history.html", predictions=rows)

@app.route('/analytics')
def analytics():
    if 'username' not in session:
        flash("You must be logged in to view analytics.", "error")
        return redirect(url_for('login'))

    conn = get_db_connection()
    rows = conn.execute(
        'SELECT predicted_class, COUNT(*) as count FROM predictions WHERE username = ? GROUP BY predicted_class',
        (session['username'],)
    ).fetchall()
    conn.close()

    labels = [row['predicted_class'] for row in rows]
    counts = [row['count'] for row in rows]

    return render_template("analytics.html", labels=labels, counts=counts)

@app.route('/datascience')
def datascience():
    return render_template('datascience.html')

@app.route('/exsisting')
def exsisting():
    return render_template('exsisting.html')

@app.route('/proposed')
def proposed():
    return render_template('proposed.html')

if __name__ == '__main__':
    app.run(debug=True)
