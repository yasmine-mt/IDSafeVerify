#Transform index.html into a dynamic page with after scanning the back
# and retrieving the info it demand to upload the front to retrieve only the photo
#and display it  so after i will do another process with this photo
#but it should been displayed in index.html
#you can use face recognition so it can detect easily the face on the id card
# Add a home page which ask the user to check the type of document passport or identity card
#if its passport we upload just the front (To scan the mrz and retrieve the img)
# else we upload the front(To retrieve the face) and the back (to scan the mrz)


from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
import os
from readmrz import MrzDetector, MrzReader
from datetime import datetime
import pycountry
import cv2

from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash
import os
from readmrz import MrzDetector, MrzReader
from datetime import datetime
import pycountry
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/id_photos'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
detector = MrzDetector()
reader = MrzReader()


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
    if document_type in ['ID', 'I']:
        return 'Identity Card'
    elif document_type == 'P':
        return 'Passport'


def normalize_sex(sex):
    """
    Normalizes a sex code to a readable string.
    """
    if sex == 'F':
        return 'Female'
    elif sex == 'M':
        return 'Male'


def normalize_mrz_data(result):
    """
    Normalize MRZ data for rendering.
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


def process_passport_face(filepath):
    """
    Process the face from the passport.
    """
    try:
        image = cv2.imread(filepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Face detection with cascade classifier
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = image[y:y + h, x:x + w]
            face_filename = "face_" + secure_filename(filepath).rsplit('.', 1)[0] + ".png"
            face_file_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
            cv2.imwrite(face_file_path, roi_color)

            return face_file_path, None

        return None, "No faces detected in the image."

    except Exception as e:
        return None, str(e)


@app.route('/')
def index():
    """
    Renders the index page.
    """
    return render_template('index.html', error=None)


@app.route('/upload_back', methods=['POST'])
def upload_back():
    """
    Handles the upload of the back side of an ID card, processes the MRZ data, and renders the upload front page.
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
        filename = "back_" + secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Process MRZ data
            image = detector.read(file_path)
            cropped = detector.crop_area(image)
            result = reader.process(cropped)
            normalized_fields = normalize_mrz_data(result)

            # if normalized_fields.get('Document type') == 'Passport':
            #     # Process passport face
            #     face_filename, error = process_passport_face(file_path)
            #     if error:
            #         return render_template('upload_front.html', error=error)
            #
            #     return render_template('profile.html', face_filename=face_filename, mrz_data=normalized_fields, error=None)
            # else:
            return render_template('upload_front.html', mrz_data=normalized_fields, error=None)

        except Exception as e:
            error = str(e)
            return render_template('index.html', error=error)

    return render_template('index.html', error='Unknown error occurred')


@app.route('/upload_front', methods=['POST'])
def upload_front():
    """
    Handles the upload of the front side of an ID card, detects the face, and renders the profile page.
    """
    error = None
    face_filename = None

    if 'file' not in request.files:
        error = 'No file part'
        return render_template('upload_front.html', error=error)

    file = request.files['file']
    if file.filename == '':
        error = 'No selected file'
        return render_template('upload_front.html', error=error)

    try:
        filename = "front_" + secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Face detection with cascade classifier
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = image[y:y + h, x:x + w]
            face_filename = "face_" + secure_filename(file.filename).rsplit('.', 1)[0] + ".png"
            face_file_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
            cv2.imwrite(face_file_path, roi_color)

            return render_template('profile.html', face_filename=face_filename, error=None)

        return render_template('upload_front.html', error='No faces detected in the image.')

    except Exception as e:
        error = str(e)
        return render_template('upload_front.html', error=error)


@app.route('/verify_selfie', methods=['POST'])
def verify_selfie():
    """
    Handles the verification of the uploaded selfie against the ID photo.
    """
    error = None
    face_filename = request.form['face_filename']

    if 'selfie' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    selfie = request.files['selfie']
    if selfie.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    try:
        # Process uploaded selfie
        selfie_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(selfie.filename))
        selfie.save(selfie_path)

        # Perform face comparison logic (implement this part)
        #here i should add the logic to the face comparaison

        flash('Selfie verification successful!')

        return redirect(url_for('index'))

    except Exception as e:
        error = str(e)
        flash(error)
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.secret_key = 'super_secret_key'
    app.run(debug=True)


