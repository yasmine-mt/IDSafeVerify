#Transform index.html into a dynamic page with after scanning the back
# and retrieving the info it demand to upload the front to retrieve only the photo
#and display it  so after i will do another process with this photo
#but it should been displayed in index.html
#if its passport we upload just the front (To scan the mrz and retrieve the img)
# else we upload the front(To retrieve the face) and the back (to scan the mrz)
#i still have a problem with the upload front animations line
#problem in verify by selfie +Button try again is not working + it takes time too much+wbcam not working
#use MTCNN to draw the face on the video because this library is very slow

from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify,Response
import os
from readmrz import MrzDetector, MrzReader
from datetime import datetime
import pycountry
import numpy as np
import dlib
import cv2
from imutils import face_utils
import face_recognition
import base64

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

# Initialize the shape predictor
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

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
    """4
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
    Renders the index page.
    """
    return render_template('Page_Accueil.html', error=None)

@app.route('/home')
def index():
    """
    Renders the index page.
    """
    return render_template('index.html', error=None)

@app.route('/upload_back', methods=['POST'])
def upload_back():
    """
    Handles the upload of the back side of an ID card or a passport, processes the MRZ data, and renders the upload front page or profile page accordingly.
    """
    error = None
    if 'file' not in request.files:
        error = 'No file part'
        return render_template('index.html', error=error)

    file = request.files['file']
    if file.filename == '':
        error = 'No selected file'
        return render_template('index.html', error=error)

    if file:
        try:
            filename = "back_" + secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            image = detector.read(file_path)
            cropped = detector.crop_area(image)
            result = reader.process(cropped)

            os.remove(file_path)

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

            if normalized_fields.get('Document type') == 'Passport':
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                if len(faces) == 0:
                    error = "No faces detected in the image."
                    return render_template('index.html', error=error)
                else:
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

                        face_filename = "face_" + secure_filename(file.filename).rsplit('.', 1)[0] + ".png"
                        face_file_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
                        cv2.imwrite(face_file_path, face_crop_rgba)

                    return render_template('profile.html', face_filename=face_filename, mrz_data=normalized_fields, error=None)
            else:
                return render_template('upload_front.html', mrz_data=normalized_fields, error=None)
        except Exception as e:
            error = str(e)
            return render_template('index.html', error=error)
    return render_template('index.html', error=error)

@app.route('/upload_front', methods=['POST'])
def upload_front():
    """
    Uploads the front image of the ID, detects the face, and returns a larger, circular cropped face image.
    """
    error = None
    face_filename = None

    if 'file' not in request.files:
        error = 'No file part'
        return render_template('upload_front.html', error=error, mrz_data=request.form.get('mrz_data'))

    file = request.files['file']
    if file.filename == '':
        error = 'No selected file'
        return render_template('upload_front.html', error=error, mrz_data=request.form.get('mrz_data'))

    try:
        filename = "front_" + secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(file_path)
        img = cv2.imread(file_path)
        if img is None:
            raise Exception(f"Error: Could not read image from '{file_path}'.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            error = "No faces detected in the image."
            return render_template('upload_front.html', error=error, mrz_data=request.form.get('mrz_data'))
        else:
            for (x, y, w, h) in faces:
                margin = int(0.2 * w)
                x_start = max(x - margin, 0)
                y_start = max(y - margin, 0)
                x_end = min(x + w + margin, img.shape[1])
                y_end = min(y + h + margin, img.shape[0])

                face_crop = img[y_start:y_end, x_start:x_end]
                mask = np.zeros((face_crop.shape[0], face_crop.shape[1]), dtype=np.uint8)
                center = (face_crop.shape[1] // 2, face_crop.shape[0] // 2)
                radius = min(center[0], center[1], face_crop.shape[1] - center[0], face_crop.shape[0] - center[1])
                cv2.circle(mask, center, radius, (255, 255, 255), -1)

                face_crop_masked = cv2.bitwise_and(face_crop, face_crop, mask=mask)
                b, g, r = cv2.split(face_crop_masked)
                alpha = mask
                face_crop_rgba = cv2.merge((b, g, r, alpha))

                face_filename = "face_" + secure_filename(file.filename).rsplit('.', 1)[0] + ".png"
                face_file_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
                cv2.imwrite(face_file_path, face_crop_rgba)

        os.remove(file_path)

    except Exception as e:
        error = str(e)
        return render_template('upload_front.html', error=error, mrz_data=request.form.get('mrz_data'))

    mrz_data = eval(request.form.get('mrz_data'))
    return render_template('profile.html', face_filename=face_filename, mrz_data=mrz_data, error=None)




@app.route('/compare_selfie', methods=['POST'])
def compare_selfie():
    """
    Compare Selfie to the photo on The ID Card
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

        return render_template('comparison_result.html', comparison_result=comparison_result, similarity_score=similarity_score)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route('/verify_selfie')
def verify_selfie():
    face_filename = request.args.get('face_filename', default=None)
    if face_filename is None:
        return jsonify(error="Error: Face filename not provided."), 400

    return render_template('verify_selfie.html', face_filename=face_filename)


@app.route('/liveness_detection', methods=['GET', 'POST'])
def liveness_detection():
    if request.method == 'POST':
        captured_image_data = request.form.get('captured_image')
        face_filename = request.form.get('face_filename')

        if not captured_image_data or not face_filename:
            return jsonify(result="Error: Missing data."), 400

        # Decode the base64 image data
        captured_image_data = captured_image_data.split(',')[1]
        captured_image = base64.b64decode(captured_image_data)
        captured_image_np = np.frombuffer(captured_image, dtype=np.uint8)
        captured_image_np = cv2.imdecode(captured_image_np, cv2.IMREAD_COLOR)

        # Load the ID photo
        face_file_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
        if not os.path.exists(face_file_path):
            return jsonify(result="Error: Face image not found."), 400
        face_image = face_recognition.load_image_file(face_file_path)
        face_encodings = face_recognition.face_encodings(face_image)
        if len(face_encodings) == 0:
            return jsonify(result="Error: No faces detected in the ID photo."), 400

        face_encoding = face_encodings[0]

        # Encode the captured image from webcam
        captured_encodings = face_recognition.face_encodings(captured_image_np)
        if len(captured_encodings) == 0:
            return jsonify(result="Error: No faces detected in the captured image."), 400

        captured_encoding = captured_encodings[0]

        # Compare the captured image with the ID photo
        distance = np.linalg.norm(captured_encoding - face_encoding)
        similarity_score = 1 - distance
        similarity_threshold = 0.6
        comparison_result = bool(distance < similarity_threshold)

        return jsonify(result="Comparison successful" if comparison_result else "Comparison failed",
                       similarity_score=similarity_score)
    else:
        face_filename = request.args.get('face_filename', default=None)
        if not face_filename:
            return jsonify(error="Error: Face filename not provided."), 400

        return render_template('liveness.html', face_filename=face_filename)


@app.route('/liveness_detection')
def liveness_detection_page():
    """
    Renders the liveness detection page.

    Returns:
        render_template: The HTML template for the liveness detection page.
    """
    face_filename = request.args.get('face_filename', default=None)
    if not face_filename:
        return jsonify(error="Error: Face filename not provided."), 400

    return render_template('liveness.html', face_filename=face_filename)


def detect_face_orientation(image):
    """
    Detects the orientation of the face in the given image and returns
    the coordinates of the face center and radius of the circle to be drawn.

    Args:
        image (np.ndarray): The input image in which to detect the face.

    Returns:
        tuple: A message indicating whether the face is detected,
               and the coordinates (center_x, center_y, radius) of the detected face.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector1(gray)
    if len(faces) == 0:
        return "No face detected", None

    face = faces[0]
    shape = predictor(gray, face)
    shape = face_utils.shape_to_np(shape)
    (x, y, w, h) = face.left(), face.top(), face.width(), face.height()
    center_x, center_y = x + w // 2, y + h // 2
    radius = max(w, h) // 2

    # Assume face is oriented correctly if detected
    return "Face detected", (center_x, center_y, radius)

def detect_smile(frame):
    """
    Detects if the person is smiling in the given image using OpenCV's pre-trained Haar cascade.

    Args:
        image (np.ndarray): The input image in which to detect the smile.

    Returns:
        bool: True if a smile is detected, False otherwise.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
        if len(smiles) > 0:
                return True
        return False
@app.route('/check_smile', methods=['POST'])
def check_smile():
        """
        Handles smile detection by processing the uploaded image and checking for a smile.

        Returns:
            jsonify: A JSON response indicating whether a smile was detected.
        """
        file = request.files.get('image')
        if not file:
            return jsonify(result="Error: No file uploaded."), 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)

        image = cv2.imread(file_path)
        if image is None:
            os.remove(file_path)
            return jsonify(result="Error: Could not read image."), 400

        smiling = detect_smile(image)

        os.remove(file_path)

        if smiling:
            return jsonify(result="Smile detected!")
        else:
            return jsonify(result="No smile detected."), 400


def generate_frames():
    cap = cv2.VideoCapture(0)  # 0 is the default camera
    while True:
        success, frame = cap.read()
        if not success:
            break

        smile_detected = detect_smile(frame)

        # Draw rectangle around detected faces
        if smile_detected:
            cv2.putText(frame, 'Smile detected!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No smile detected.', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/smile_detection')
def smile_detection():
    return render_template('liveness.html')

if __name__ == '__main__':
    app.run(debug=True)
