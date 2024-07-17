#Transform index.html into a dynamic page with after scanning the back
# and retrieving the info it demand to upload the front to retrieve only the photo
#and display it  so after i will do another process with this photo
#but it should been displayed in index.html
#if its passport we upload just the front (To scan the mrz and retrieve the img)
# else we upload the front(To retrieve the face) and the back (to scan the mrz)
#i still have a problem with the upload front animations line


from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
import os
from readmrz import MrzDetector, MrzReader
from datetime import datetime
import pycountry
import numpy as np
import cv2
import face_recognition

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
    if 'selfie' not in request.files:
        return render_template('comparison_result.html', error="Error: No file part.")

    selfie = request.files['selfie']
    if selfie.filename == '':
        return render_template('comparison_result.html', error="Error: No selected file.")

    if selfie:
        try:
            selfie_filename = secure_filename(selfie.filename)
            selfie_path = os.path.join(app.config['UPLOAD_FOLDER'], selfie_filename)
            selfie.save(selfie_path)

            face_filename = request.form.get('face_filename')
            if not face_filename:
                return render_template('comparison_result.html', error="Error: Face filename not provided.")

            face_file_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
            if not os.path.exists(face_file_path):
                return render_template('comparison_result.html', error="Error: Face image not found.")
            selfie_image = face_recognition.load_image_file(selfie_path)
            selfie_encodings = face_recognition.face_encodings(selfie_image)
            if len(selfie_encodings) == 0:
                return render_template('comparison_result.html', error="Error: No faces detected in the selfie.")
            selfie_encoding = selfie_encodings[0]
            face_image = face_recognition.load_image_file(face_file_path)
            face_encodings = face_recognition.face_encodings(face_image)
            if len(face_encodings) == 0:
                return render_template('comparison_result.html', error="Error: No faces detected in the ID photo.")
            face_encoding = face_encodings[0]
            distance = np.linalg.norm(selfie_encoding - face_encoding)
            similarity_score = 1 - distance
            similarity_threshold = 0.6
            comparison_result = distance < similarity_threshold
            selfie.close()
            os.remove(selfie_path)
            return render_template('comparison_result.html', face_filename=face_filename,
                                   comparison_result=comparison_result, similarity_score=similarity_score)
        except Exception as e:
            error = str(e)
            return render_template('comparison_result.html', error=error)

    return render_template('comparison_result.html', error="Error: Something went wrong.")

@app.route('/liveness_detection', methods=['GET'])
def liveness_detection():
    return render_template('liveness_detection.html')

if __name__ == '__main__':
    app.run(debug=True)

