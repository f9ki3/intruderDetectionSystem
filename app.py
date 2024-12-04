from flask import Flask, render_template, request, redirect, url_for, session, flash
from functools import wraps
import os
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Secret key for session management

# Dummy data for users and their roles
users = {
    'staff': {'password': 'staffpass', 'role': 'staff'},
    'kyle': {'password': 'kyle', 'role': 'admin'}
}

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
            else:
                text = "Intruder"
                color = (0, 0, 255)  # Red for intruder
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')

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

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users and users[username]['password'] == password:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
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

if __name__ == '__main__':
    app.run(debug=True)
