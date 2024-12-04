import os
import cv2
import sqlite3
from datetime import datetime
import time
import threading
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify
from functools import wraps
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Secret key for session management

# Path to your SQLite database file
DATABASE = "database.db"

# Dummy data for users and their roles
users = {
    'staff': {'password': 'staffpass', 'role': 'staff'},
    'kyle': {'password': 'kyle', 'role': 'admin'}
}

# Ensure screenshots folder exists
SCREENSHOTS_DIR = 'screenshots'
if not os.path.exists(SCREENSHOTS_DIR):
    os.makedirs(SCREENSHOTS_DIR)

# Counter for screenshots
screenshot_counter = 1

# Lock to ensure thread-safe operations
lock = threading.Lock()

# Initially empty camera list
cameras = []

# Define directories
REGISTERED_FACES_DIR = 'registered_faces'
TRAINER_FILE = 'trainer.yml'

# Ensure registered faces directory exists
if not os.path.exists(REGISTERED_FACES_DIR):
    os.makedirs(REGISTERED_FACES_DIR)

# Load OpenCV Haar Cascade and LBPH Recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Initialize video capture
video_capture = cv2.VideoCapture(0)


# Train the recognizer with registered faces
def train_recognizer():
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for filename in os.listdir(REGISTERED_FACES_DIR):
        if filename.endswith(('.jpg', '.png')):
            path = os.path.join(REGISTERED_FACES_DIR, filename)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            faces_detected = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces_detected:
                faces.append(image[y:y+h, x:x+w])
                labels.append(current_label)
            label_map[current_label] = filename.split('.')[0]
            current_label += 1

    if faces:
        recognizer.train(faces, np.array(labels))
        recognizer.save(TRAINER_FILE)
    return label_map

# Initialize recognizer and label map
label_map = train_recognizer() if os.listdir(REGISTERED_FACES_DIR) else {}

@app.route('/detect_intruder')
def detect_intruder():
    return render_template('detect_intruder.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_only')
def video_feed_only():
    return Response(generate_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_video_frames():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert the frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        # Yield the frame as a byte stream in the multipart format
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')


@app.route('/register_face', methods=['POST'])
def register_face():
    if not video_capture.isOpened():
        return jsonify({'status': 'error', 'message': 'Video capture not initialized.'})

    ret, frame = video_capture.read()
    if not ret:
        return jsonify({'status': 'error', 'message': 'Failed to capture image.'})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces_detected) == 0:
        return jsonify({'status': 'error', 'message': 'No face detected. Try again.'})

    # Save the largest face detected
    largest_face = max(faces_detected, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (150, 150))

    face_id = len(os.listdir(REGISTERED_FACES_DIR)) + 1
    img_path = os.path.join(REGISTERED_FACES_DIR, f'face_{face_id}.jpg')
    cv2.imwrite(img_path, face)

    global label_map
    label_map = train_recognizer()
    
    return jsonify({'status': 'success', 'message': 'Face registered successfully!'})

# Create a queue to handle saving images
def save_screenshot(frame):
    global screenshot_counter  # Keep track of the screenshot number

    # Save the screenshot as "screenshot_X.jpg"
    screenshot_filename = os.path.join(SCREENSHOTS_DIR, f'screenshot_{screenshot_counter}.jpg')
    cv2.imwrite(screenshot_filename, frame)
    screenshot_counter += 1  # Increment the screenshot counter
    print(f"Screenshot saved: {screenshot_filename}")

    # Email content setup
    sender_email = "carurucankylejustin@gmail.com"
    receiver_email = "cloudrazor38@gmail.com"
    subject = "Intruder Alert: CCTV Detection"
    body = "Your CCTV system has detected an intruder. Please find the attached screenshot."

    # Create the email message
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    # Add email body
    msg.attach(MIMEText(body, "plain"))

    # Add the image attachment
    attachment_path = screenshot_filename  # Use the newly saved screenshot
    try:
        with open(attachment_path, "rb") as attachment_file:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment_file.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename={attachment_path.split('/')[-1]}"
            )
            msg.attach(part)
    except FileNotFoundError:
        print(f"Error: File {attachment_path} not found. Skipping attachment.")

    # Sending the email
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()  # Secure the connection
            server.login(sender_email, "fyxx aaoc rqqr tzkt")  # App Password
            server.send_message(msg)
            print("Email sent successfully with the image attachment!")
    except Exception as e:
        print(f"Error sending email: {e}")


def generate_frames():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        
        for (x, y, w, h) in faces_detected:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (150, 150))

            if label_map:
                label, confidence = recognizer.predict(face)
            else:
                label, confidence = None, 100  # Default when no registered faces
            
            if label is not None and confidence < 70:  # Confidence threshold
                text = f"Authorized: {label_map.get(label, 'Unknown')} (Conf: {int(confidence)})"
                color = (0, 255, 0)  # Green for authorized
                print('Authorized')
            else:
                text = "Intruder"
                color = (0, 0, 255)  # Red for intruder
                print('Intruder')

                # If it's an intruder, save the image in a separate thread
                threading.Thread(target=intruder_save_image, args=(frame,)).start()

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')

def intruder_save_image(frame):
    # Use the lock to ensure thread-safe image saving
    with lock:
        time.sleep(5)
        save_screenshot(frame)

@app.route('/shutdown', methods=['POST'])
def shutdown():
    video_capture.release()
    cv2.destroyAllWindows()
    return jsonify({'status': 'success', 'message': 'Video capture released successfully.'})

# Login decorator to restrict access
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('You need to log in first!', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Admin-only decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session or users.get(session['username'], {}).get('role') != 'admin':
            flash('Admin access required!', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# Home route
@app.route('/')
def home():
    return redirect(url_for('login'))

# Login
def log_action(user, action, details=""):
    """Log an action in the AuditLogs table."""
    try:
        connection = sqlite3.connect(DATABASE)
        cursor = connection.cursor()

        # Insert log entry
        cursor.execute("""
            INSERT INTO AuditLogs (date, user, action, details)
            VALUES (?, ?, ?, ?)
        """, (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), user, action, details))

        connection.commit()
    except sqlite3.Error as e:
        print(f"Error logging action: {e}")
    finally:
        connection.close()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username]['password'] == password:
            session['username'] = username

            # Log successful login
            log_action(username, "Login", "User logged in successfully.")
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')

            # Log failed login attempt
            log_action(username if username else "Unknown", "Login Failed", "Invalid credentials provided.")

    return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
    user = session['username']
    log_action(user, "Logout", "User logged out successfully.")
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Dashboard route
@app.route('/dashboard')
@login_required
def dashboard():
    user = session['username']
    role = users[user]['role']
    return render_template('dashboard.html', username=user, role=role)

# Dashboard route
@app.route('/capture')
@login_required
def capture():
    user = session['username']
    role = users[user]['role']
    return render_template('capture.html', username=user, role=role)

# User Management route (Admin-only)
@app.route('/user_management', methods=['GET', 'POST'])
@admin_required
def user_management():
    if request.method == 'POST':
        action = request.form['action']
        username = request.form['username']
        
        if action == 'add':
            password = request.form.get('password', '')  # Default to empty if not provided
            role = request.form['role']
            if username in users:
                flash(f'User {username} already exists.', 'danger')
            elif role == 'user':  # Prevent adding a user with the "user" role
                flash(f'Cannot add user with "user" role.', 'danger')
            else:
                users[username] = {'password': password, 'role': role}
                flash(f'User {username} added successfully.', 'success')
        
        elif action == 'delete':
            if username in users:
                del users[username]
                flash(f'User {username} deleted successfully.', 'success')
            else:
                flash('User not found.', 'danger')
        
    return render_template('user_management.html', users=users)

# Camera Management route (Admin-only)
@app.route('/camera_management', methods=['GET', 'POST'])
@admin_required
def camera_management():
    if request.method == 'POST':
        action = request.form['action']
        camera_name = request.form['camera_name']
        
        if action == 'add':
            if camera_name in cameras:
                flash(f'Camera {camera_name} already exists.', 'danger')
            else:
                cameras.append(camera_name)
                flash(f'Camera {camera_name} added successfully.', 'success')
        elif action == 'delete' and camera_name in cameras:
            cameras.remove(camera_name)
            flash(f'Camera {camera_name} deleted successfully.', 'success')
        else:
            flash('Invalid action or camera not found.', 'danger')
        
    return render_template('camera_management.html', cameras=cameras)

# Camera Stream route (staff and admin can view)
@app.route('/camera_stream')
@login_required
def camera_stream():
    user_role = users[session['username']]['role']
    
    # Allow only staff and admins to view camera stream
    if user_role in ['staff', 'admin']:
        return render_template('camera_stream.html', cameras=cameras)
    else:
        flash('You do not have access to view the camera stream.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/camera_authorize')
@login_required
def camera_authorize():
    user_role = users[session['username']]['role']
    
    # Allow only staff and admins to view camera stream
    if user_role in ['staff', 'admin']:
        return render_template('camera_authorize.html', cameras=cameras)
    else:
        flash('You do not have access to view the camera stream.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/audit_trail')
@login_required
def audit_trail():
    user_role = users[session['username']]['role']
    
    # Allow only staff and admins to view camera stream
    if user_role in ['staff', 'admin']:
        return render_template('audit_trail.html', cameras=cameras)
    else:
        flash('You do not have access to view the camera stream.', 'danger')
        return redirect(url_for('dashboard'))

def get_audit_logs():
    """Retrieve all logs from the AuditLogs table."""
    try:
        connection = sqlite3.connect(DATABASE)
        connection.row_factory = sqlite3.Row  # Enables dictionary-like access to rows
        cursor = connection.cursor()

        # Fetch all records from the table
        cursor.execute("SELECT * FROM AuditLogs")
        rows = cursor.fetchall()

        # Convert rows to a list of dictionaries
        logs = [dict(row) for row in rows]
        return logs
    except sqlite3.Error as e:
        print(f"Error retrieving logs: {e}")
        return []
    finally:
        connection.close()

@app.route('/audit-logs', methods=['GET'])
def api_audit_logs():
    """API endpoint to retrieve all logs."""
    logs = get_audit_logs()
    return jsonify(logs)

if __name__ == '__main__':
    if os.path.exists(TRAINER_FILE):
        recognizer.read(TRAINER_FILE)
    app.run(debug=True)
