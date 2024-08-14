from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify
import os,io
from readmrz import MrzDetector, MrzReader
from datetime import datetime
import pycountry
import numpy as np
import dlib
import cv2
from imutils import face_utils
import face_recognition
import base64
import random


def asfarray(a, dtype=np.float64):
    """
    Converts input to an array of the given data type.
    """
    return np.asarray(a, dtype=dtype)


np.asfarray = asfarray

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/id_photos'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
detector = MrzDetector()
reader = MrzReader()
detector1 = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

actions = ["Souriez", "Tournez à droite", "Tournez à gauche", "Levez les yeux", "Baissez les yeux", "Clignez des yeux", "Secouez la tête"]


def find_existing_face_image(face_filename):
    """
    Check if a face image file already exists in the upload folder.
    """
    face_file_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
    if os.path.isfile(face_file_path):
        return face_file_path
    else:
        return None


def normalize_date(date_str):
    """
    Normalizes a date string from 'yymmdd' format to 'dd/mm/yyyy'.
    """
    try:
        date_obj = datetime.strptime(date_str, '%y%m%d').date()
        return date_obj.strftime('%d/%m/%Y')
    except ValueError:
        return 'Invalid Date'


def normalize_country(country_code):
    """
    Normalizes a country code to its full country name.
    """
    try:
        country = pycountry.countries.get(alpha_3=country_code)
        return country.name if country else 'Unknown'
    except Exception:
        return 'Unknown'


def normalize_nationality(nationality_code):
    """
    Normalizes a nationality code to its full country name.
    """
    try:
        country = pycountry.countries.get(alpha_3=nationality_code)
        return country.name if country else 'Unknown'
    except Exception:
        return 'Unknown'


def normalize_document_id(doc_id):
    """
    Normalizes a document ID by removing non-alphanumeric characters and converting to uppercase.
    """
    return ''.join(filter(str.isalnum, doc_id)).upper()


def normalize_document_type(document_type):
    """
    Normalizes a document type to a readable string.
    """
    if document_type == 'ID' or document_type == 'I':
        return 'Identity Card'
    elif document_type == 'P' or document_type == 'PP':
        return 'Passport'


def normalize_sex(sex):
    """
    Normalizes a sex code to a readable string.
    """
    if sex == 'F':
        return 'Female'
    elif sex == 'M':
        return 'Male'


@app.route('/')
def Home():
    """
    Renders the home page.
    """
    return render_template('Page_Accueil.html', error=None)


@app.route('/home')
def index():
    """
    Renders the index page.
    """
    return render_template('index.html', error=None)


@app.route('/upload_id', methods=['POST'])
def upload_id():
    """
    Handles the upload of the front and back sides of an ID card or a passport, processes the MRZ data,
    and renders the appropriate profile or upload page accordingly.
    """
    error = None
    front_file = request.files.get('front_file')
    back_file = request.files.get('back_file')
    passport_file = request.files.get('passport_file')

    if not front_file and not passport_file:
        error = 'Please upload either the front and back sides of your ID card, or a passport.'
        return render_template('index.html', error=error)

    try:
        if passport_file:
            filename = "passport_" + secure_filename(passport_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            passport_file.save(file_path)
            image = detector.read(file_path)
            cropped = detector.crop_area(image)
            result = reader.process(cropped)

            os.remove(file_path)

            normalized_fields = process_mrz_data(result)
            face_filename = process_face_detection(image, passport_file.filename)

            return render_template('profile.html', face_filename=face_filename, mrz_data=normalized_fields, error=None)

        elif front_file and back_file:
            back_filename = "back_" + secure_filename(back_file.filename)
            back_file_path = os.path.join(app.config['UPLOAD_FOLDER'], back_filename)
            back_file.save(back_file_path)
            back_image = detector.read(back_file_path)
            cropped_back = detector.crop_area(back_image)
            result = reader.process(cropped_back)

            os.remove(back_file_path)

            normalized_fields = process_mrz_data(result)

            front_filename = "front_" + secure_filename(front_file.filename)
            front_file_path = os.path.join(app.config['UPLOAD_FOLDER'], front_filename)
            front_file.save(front_file_path)
            img = cv2.imread(front_file_path)
            if img is None:
                raise Exception(f"Error: Could not read image from '{front_file_path}'.")

            face_filename = process_face_detection(img, front_file.filename)
            os.remove(front_file_path)

            return render_template('profile.html', face_filename=face_filename, mrz_data=normalized_fields, error=None)

        else:
            error = 'Please upload both the front and back sides of your ID card.'
            return render_template('index.html', error=error)

    except Exception as e:
        error = str(e)
        return render_template('index.html', error=error)


def process_mrz_data(result):
    """
    Processes the MRZ data extracted from the document image.
    """
    normalized_fields = {}
    if 'optional_data' in result and result['optional_data'] is not None:
        normalized_fields['ID'] = result['optional_data']
    if 'document_number' in result:
        normalized_fields['Document number'] = normalize_document_id(result['document_number'])
    if 'document_type' in result:
        normalized_fields['Document type'] = normalize_document_type(result['document_type'])
    if 'surname' in result:
        normalized_fields['Last name'] = result['surname']
    if 'name' in result:
        normalized_fields['First name'] = result['name']
    if 'sex' in result:
        normalized_fields['Gender'] = normalize_sex(result['sex'])
    if 'birth_date' in result:
        normalized_fields['Birth date'] = normalize_date(result['birth_date'])
    if 'expiry_date' in result:
        normalized_fields['Expiry date'] = normalize_date(result['expiry_date'])
    if 'country' in result:
        normalized_fields['Country'] = normalize_country(result['country'])
    if 'nationality' in result:
        normalized_fields['Nationality'] = normalize_nationality(result['nationality'])

    return normalized_fields


def process_face_detection(image, filename):
    """
    Detects the face in the provided image and returns the filename of the cropped face image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        raise Exception("No faces detected in the image.")

    for (x, y, w, h) in faces:
        margin = int(0.2 * w)
        x_start = max(x - margin, 0)
        y_start = max(y - margin, 0)
        x_end = min(x + w + margin, image.shape[1])
        y_end = min(y + h + margin, image.shape[0])

        face_crop = image[y_start:y_end, x_start:x_end]
        mask = np.zeros((face_crop.shape[0], face_crop.shape[1]), dtype=np.uint8)
        center = (face_crop.shape[1] // 2, face_crop.shape[0] // 2)
        radius = min(center[0], center[1], face_crop.shape[1] - center[0], face_crop.shape[0] - center[1])
        cv2.circle(mask, center, radius, (255, 255, 255), -1)

        face_crop_masked = cv2.bitwise_and(face_crop, face_crop, mask=mask)
        b, g, r = cv2.split(face_crop_masked)
        alpha = mask
        face_crop_rgba = cv2.merge((b, g, r, alpha))

        face_filename = "face_" + secure_filename(filename).rsplit('.', 1)[0] + ".png"
        face_file_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
        cv2.imwrite(face_file_path, face_crop_rgba)

    return face_filename


@app.route('/compare_selfie', methods=['POST'])
def compare_selfie():
    """
    Compares faces if they are similar
    :return: similarity score and a comparison result
    """
    if 'selfie' not in request.files:
        return jsonify(error="Error: No file part."), 400

    file = request.files.get('selfie')
    if not file or file.filename == '':
        return jsonify(error="Error: No selected file."), 400

    try:
        selfie_filename = secure_filename(file.filename)
        selfie_path = os.path.join(app.config['UPLOAD_FOLDER'], selfie_filename)
        file.save(selfie_path)

        face_filename = request.form.get('face_filename')
        if not face_filename:
            return jsonify(error="Error: Face filename not provided."), 400

        face_file_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
        if not os.path.exists(face_file_path):
            return jsonify(error="Error: Face image not found."), 400

        selfie_image = face_recognition.load_image_file(selfie_path)
        selfie_encodings = face_recognition.face_encodings(selfie_image)
        if len(selfie_encodings) == 0:
            os.remove(selfie_path)
            return jsonify(error="Error: No faces detected in the selfie."), 400

        selfie_encoding = selfie_encodings[0]
        face_image = face_recognition.load_image_file(face_file_path)
        face_encodings = face_recognition.face_encodings(face_image)
        if len(face_encodings) == 0:
            os.remove(selfie_path)
            return jsonify(error="Error: No faces detected in the ID photo."), 400

        face_encoding = face_encodings[0]
        distance = np.linalg.norm(selfie_encoding - face_encoding)
        similarity_score = 1 - distance
        similarity_threshold = 0.6
        comparison_result = bool(distance < similarity_threshold)
        os.remove(selfie_path)

        return render_template('comparison_result.html', comparison_result=comparison_result,
                               similarity_score=similarity_score)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route('/verify_selfie')
def verify_selfie():
    """
    Receives a face as an argument and verify if is not none
    :return: the face filename or a json response as an error
    """
    face_filename = request.args.get('face_filename', default=None)
    if face_filename is None:
        return jsonify(error="Error: Face filename not provided."), 400

    return render_template('verify_selfie.html', face_filename=face_filename)


def get_random_actions(n=3):
    """
    Get a random selection of actions from the predefined list.
    :param n: Number of actions to select
    :return: List of random actions
    """
    return random.sample(actions, n)


@app.route('/initiate_liveness', methods=['GET'])
def initiate_liveness():
    """
    Endpoint to initiate liveness detection by selecting random actions.
    :return: JSON response with selected actions and delay time
    """
    selected_actions = get_random_actions()
    delay = 4
    return jsonify({"actions": selected_actions, "delay": delay})


def process_image(image_data, action):
    """
    Process the image to detect specified action.
    :param image_data: Base64 encoded image data
    :param action: Action to be detected
    :return: Result of action detection
    """
    image_data = base64.b64decode(image_data.split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    frame_resized = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    rects = detector1(gray, 1)
    landmarks = []

    if len(rects) == 0:
        return "Aucun visage détecté"

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        landmarks.append(shape)

    action_result = analyze_actions(landmarks, action, frame_resized)
    return action_result


def analyze_actions(landmarks, action, frame):
    """
    Analyse the user's action based on facial landmarks and head pose estimation.

    Parameters:
    - landmarks: List of facial landmarks detected in the frame.
    - action: The action to be detected (e.g., "Souriez", "Tournez à droite").
    - frame: The current video frame for processing.

    Returns:
    - A string indicating whether the action was detected or not.
    """
    if not landmarks:
        return f"{action} non"

    shape = landmarks[0]

    # Smile Detection
    if action == "Souriez":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.equalizeHist(gray)
        smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=35, minSize=(25, 25),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        mouth = shape[48:60]
        mouth_width = np.linalg.norm(mouth[0] - mouth[6])
        smile_threshold = 30  # Adjust based on validation
        if len(smiles) > 0 and mouth_width > smile_threshold:
            return "Sourire détecté"
        else:
            return "Souriez non"

    # Head Pose Estimation
    image_points = np.array([shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]], dtype="double")
    model_points = np.array(
        [(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
         (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)])
    size = frame.shape
    focal_length = size[1]
    center = (size[1] // 2, size[0] // 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                             dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return "Échec de l'estimation de la pose"

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(pose_matrix)
    yaw, pitch = eulerAngles[1, 0], eulerAngles[0, 0]

    turn_threshold = 15
    look_threshold = 15
    EAR_THRESHOLD = 0.25

    if action == "Tournez à gauche" and yaw < -turn_threshold:
        return "Tournez à gauche détecté"
    if action == "Tournez à droite" and yaw > turn_threshold:
        return "Tournez à droite détecté"
    if action == "Levez les yeux" and pitch > look_threshold:
        return "Levez les yeux détecté"
    if action == "Baissez les yeux" and pitch < -look_threshold:
        return "Baissez les yeux détecté"
    if action == "Clignez des yeux":
        eye_aspect_ratio = compute_ear(shape[36:42])
        if eye_aspect_ratio < EAR_THRESHOLD:
            return "Clignement des yeux détecté"
    if action == "Secouez la tête" and abs(yaw) > turn_threshold:
        return "Secouez la tête détecté"

    return f"{action} non"



def compute_ear(eye_landmarks):
    """
    Calculate the Eye Aspect Ratio (EAR) for blink detection.

    Parameters:
    - eye_landmarks: Array of eye landmarks.

    Returns:
    - The computed EAR value.
    """
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear


def eye_aspect_ratio(eye):
    a = np.linalg.norm(eye[1] - eye[5])
    b = np.linalg.norm(eye[2] - eye[4])
    c = np.linalg.norm(eye[0] - eye[3])
    ear = (a + b) / (2.0 * c)
    return ear


@app.route('/process_frame', methods=['POST'])
def process_frame():
    """
    Endpoint to process a frame from the video feed and check the liveness action.
    :return: JSON response with the result of the action detection and identity verification
    """
    data = request.json
    image_data = data['image']
    action = data['action']
    face_filename = data['face_filename']
    result = process_image(image_data, action)
    selfie_image = face_recognition.load_image_file(io.BytesIO(base64.b64decode(image_data.split(',')[1])))
    selfie_encodings = face_recognition.face_encodings(selfie_image)

    if len(selfie_encodings) == 0:
        return jsonify({"result": f"{action} non", "error": "No faces detected in the selfie."})

    selfie_encoding = selfie_encodings[0]

    face_file_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
    if not os.path.exists(face_file_path):
        return jsonify({"result": f"{action} non", "error": "Face image not found."})

    face_image = face_recognition.load_image_file(face_file_path)
    face_encodings = face_recognition.face_encodings(face_image)
    if len(face_encodings) == 0:
        return jsonify({"result": f"{action} non", "error": "No faces detected in the ID photo."})

    face_encoding = face_encodings[0]
    distance = np.linalg.norm(selfie_encoding - face_encoding)
    similarity_score = 1 - distance
    similarity_threshold = 0.6
    comparison_result = bool(distance < similarity_threshold)

    if comparison_result:
        result += " et même personne que l'ID"
    else:
        result += " mais pas la même personne que l'ID"

    return jsonify({"result": result, "similarity_score": similarity_score})


@app.route('/liveness_detection')
def liveness_detection():
    """
    Render the liveness detection page.
    :return: HTML template for liveness detection
    """
    face_filename = request.args.get('face_filename', default=None)
    return render_template('liveness_Test.html', face_filename=face_filename)


@app.route('/success')
def success():
    """
    Render the success page after successful liveness detection.
    :return: HTML template for success
    """
    return render_template('success.html')


if __name__ == '__main__':
    app.run(debug=True)