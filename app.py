#Transform index.html into a dynamic page with after scanning the back
# and retrieving the info it demand to upload the front to retrieve only the photo
#and display it  so after i will do another process with this photo
#but it should been displayed in index.html
#you can use face recognition so it can detect easily the face on the id card
# Add a home page which ask the user to check the type of document passport or identity card
#if its passport we upload just the front (To scan the mrz and retrieve the img)
# else we upload the front(To retrieve the face) and the back (to scan the mrz)
#Add documentation

from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
import os
from readmrz import MrzDetector, MrzReader
from datetime import datetime
import pycountry
import numpy as np
import cv2

def asfarray(a, dtype=np.float64):
    return np.asarray(a, dtype=dtype)

np.asfarray = asfarray

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/id_photos'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
detector = MrzDetector()
reader = MrzReader()

def normalize_date(date_str):
    try:
        date_obj = datetime.strptime(date_str, '%y%m%d').date()
        return date_obj.strftime('%d/%m/%Y')
    except ValueError:
        return 'Invalid Date'

def normalize_country(country_code):
    try:
        country = pycountry.countries.get(alpha_3=country_code)
        return country.name if country else 'Unknown'
    except Exception:
        return 'Unknown'

def normalize_nationality(nationality_code):
    try:
        country = pycountry.countries.get(alpha_3=nationality_code)
        return country.name if country else 'Unknown'
    except Exception:
        return 'Unknown'

def normalize_document_id(doc_id):
    return ''.join(filter(str.isalnum, doc_id)).upper()

def normalize_document_type(document_type):
    if document_type == 'ID' or document_type == 'I':
        return 'Identity Card'
    elif document_type == 'P':
        return 'Passport'

def normalize_sex(sex):
    if sex == 'F':
        return 'Female'
    elif sex == 'M':
        return 'Male'

@app.route('/')
def index():
    return render_template('index.html', error=None)

@app.route('/upload_back', methods=['POST'])
def upload_back():
    error = None
    if 'file' not in request.files:
        error = 'No file part'
        return render_template('index.html', error=error)

    file = request.files['file']
    if file.filename == '':
        error = 'No selected file'
        return render_template('index.html', error=error)

    if file:
        filename = "back_" + secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        try:
            image = detector.read(file_path)
            cropped = detector.crop_area(image)
            result = reader.process(cropped)

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

            return render_template('upload_front.html', mrz_data=normalized_fields, error=None)
        except Exception as e:
            error = str(e)
            return render_template('index.html', error=error)
    return render_template('index.html', error=error)

@app.route('/upload_front', methods=['POST'])
def upload_front():
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

        existing_face_filename = find_existing_face_image(request.form.get('mrz_data'))
        if existing_face_filename:
            face_filename = existing_face_filename
        else:
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
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    face_crop = img[y:y + h, x:x + w]
                    face_filename = "face_" + secure_filename(file.filename)
                    face_file_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
                    cv2.imwrite(face_file_path, face_crop)

    except Exception as e:
        error = str(e)
        return render_template('upload_front.html', error=error, mrz_data=request.form.get('mrz_data'))

    mrz_data = eval(request.form.get('mrz_data'))
    return render_template('profile.html', face_filename=face_filename, mrz_data=mrz_data, error=None)

def find_existing_face_image(mrz_data_str):
    try:
        mrz_data = eval(mrz_data_str)
        if 'ID' in mrz_data:
            document_id = normalize_document_id(mrz_data['ID'])
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                if filename.startswith('face_') and filename.endswith('.jpg'):
                    if filename.startswith(f'face_{document_id}_'):
                        return filename
        return None
    except Exception as e:
        print(f"Error finding existing face image: {str(e)}")
        return None

if __name__ == '__main__':
    app.run(debug=True)
