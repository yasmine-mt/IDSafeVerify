#Transform index.html into a dynamic page with after scanning the back
# and retrieving the info it demand to upload the front to retrieve only the photo
#and display it  so after i will do another process with this photo
#but it should been displayed in index.html
#you can use face recognition so it can detect easily the face on the id card
# Add a home page which ask the user to check the type of document passport or identity card
#if its passport we upload just the front (To scan the mrz and retrieve the img)
# else we upload the front(To retrieve the face) and the back (to scan the mrz)
#Add documentation
#when upload show the image ploaded instead of its title

from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
import os
from readmrz import MrzDetector, MrzReader
from datetime import datetime
import pycountry
import numpy as np
import cv2


def asfarray(a, dtype=np.float64):
    """
    Converts input to an array of the given data type.

    Parameters:
    a: array_like
        Input data, in any form that can be converted to an array.
    dtype: data-type, optional
        Desired data-type for the array, default is np.float64.

    Returns:
    out: ndarray
        Array interpretation of `a`.
    """
    return np.asarray(a, dtype=dtype)


np.asfarray = asfarray

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/id_photos'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
detector = MrzDetector()
reader = MrzReader()


def normalize_date(date_str):
    """
    Normalizes a date string from 'yymmdd' format to 'dd/mm/yyyy'.

    Parameters:
    date_str: str
        Date string in 'yymmdd' format.

    Returns:
    str
        Date string in 'dd/mm/yyyy' format or 'Invalid Date' if parsing fails.
    """
    try:
        date_obj = datetime.strptime(date_str, '%y%m%d').date()
        return date_obj.strftime('%d/%m/%Y')
    except ValueError:
        return 'Invalid Date'


def normalize_country(country_code):
    """
    Normalizes a country code to its full country name.

    Parameters:
    country_code: str
        Alpha-3 country code.

    Returns:
    str
        Full country name or 'Unknown' if the code is invalid.
    """
    try:
        country = pycountry.countries.get(alpha_3=country_code)
        return country.name if country else 'Unknown'
    except Exception:
        return 'Unknown'


def normalize_nationality(nationality_code):
    """
    Normalizes a nationality code to its full country name.

    Parameters:
    nationality_code: str
        Alpha-3 nationality code.

    Returns:
    str
        Full country name or 'Unknown' if the code is invalid.
    """
    try:
        country = pycountry.countries.get(alpha_3=nationality_code)
        return country.name if country else 'Unknown'
    except Exception:
        return 'Unknown'


def normalize_document_id(doc_id):
    """
    Normalizes a document ID by removing non-alphanumeric characters and converting to uppercase.

    Parameters:
    doc_id: str
        Document ID string.

    Returns:
    str
        Normalized document ID.
    """
    return ''.join(filter(str.isalnum, doc_id)).upper()

def detect_face(image_path):
    """
    Detects faces in the given image using OpenCV's Haar Cascade classifier.

    Parameters:
    image_path: str
        Path to the image file.

    Returns:
    tuple
        (face_found: bool, face_coordinates: tuple or None)
        face_found: True if a face is detected, False otherwise.
        face_coordinates: Tuple of (x, y, w, h) coordinates of the detected face, or None if no face is detected.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        return True, (x, y, w, h)
    else:
        return False, None
def normalize_document_type(document_type):
    """
    Normalizes a document type to a readable string.

    Parameters:
    document_type: str
        Document type code ('ID', 'I', 'P').

    Returns:
    str
        'Identity Card' for 'ID' or 'I', 'Passport' for 'P'.
    """
    if document_type == 'ID' or document_type == 'I':
        return 'Identity Card'
    elif document_type == 'P' or document_type =='PP':
        return 'Passport'


def normalize_sex(sex):
    """
    Normalizes a sex code to a readable string.

    Parameters:
    sex: str
        Sex code ('F', 'M').

    Returns:
    str
        'Female' for 'F', 'Male' for 'M'.
    """
    if sex == 'F':
        return 'Female'
    elif sex == 'M':
        return 'Male'


@app.route('/')
def index():
    """
    Renders the index page.

    Returns:
    HTML template
        Rendered 'index.html'.
    """
    return render_template('index.html', error=None)


@app.route('/upload_back', methods=['POST'])
def upload_back():
    """
    Handles the upload of the back side of an ID card, processes the MRZ data,
    and decides whether to proceed to front upload or directly to profile page.
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

            if normalized_fields.get('Document type') == 'Passport':
                existing_face_filename = find_existing_face_image(str(normalized_fields))
                if existing_face_filename:
                    face_filename = existing_face_filename
                else:
                    img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                    if len(faces) == 0:
                        error = "No faces detected in the image."
                        return render_template('index.html', error=error)

                    for (x, y, w, h) in faces:
                        margin = int(0.2 * w)
                        x_start = max(x - margin, 0)
                        y_start = max(y - margin, 0)
                        x_end = min(x + w + margin, img.shape[1])
                        y_end = min(y + h + margin, img.shape[0])

                        face_crop = img[y_start:y_end, x_start:x_end]

                        # Create a circular mask
                        mask = np.zeros((face_crop.shape[0], face_crop.shape[1]), dtype=np.uint8)
                        center = (face_crop.shape[1] // 2, face_crop.shape[0] // 2)
                        radius = min(center[0], center[1], face_crop.shape[1] - center[0], face_crop.shape[0] - center[1])
                        cv2.circle(mask, center, radius, (255, 255, 255), -1)

                        # Apply the circular mask to the face crop
                        face_crop_masked = cv2.bitwise_and(face_crop, face_crop, mask=mask)

                        # Create an alpha channel
                        b, g, r = cv2.split(face_crop_masked)
                        alpha = mask
                        face_crop_rgba = cv2.merge((b, g, r, alpha))

                        face_filename = "face_" + secure_filename(file.filename).rsplit('.', 1)[0] + ".png"
                        face_file_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
                        cv2.imwrite(face_file_path, cv2.cvtColor(face_crop_rgba, cv2.COLOR_RGBA2BGRA))

                return render_template('profile.html', face_filename=face_filename, mrz_data=normalized_fields, error=None)
            else:
                return render_template('upload_front.html', mrz_data=str(normalized_fields), error=None)

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
                    margin = int(0.2 * w)
                    x_start = max(x - margin, 0)
                    y_start = max(y - margin, 0)
                    x_end = min(x + w + margin, img.shape[1])
                    y_end = min(y + h + margin, img.shape[0])

                    face_crop = img[y_start:y_end, x_start:x_end]

                    # Create a circular mask
                    mask = np.zeros((face_crop.shape[0], face_crop.shape[1]), dtype=np.uint8)
                    center = (face_crop.shape[1] // 2, face_crop.shape[0] // 2)
                    radius = min(center[0], center[1], face_crop.shape[1] - center[0], face_crop.shape[0] - center[1])
                    cv2.circle(mask, center, radius, (255, 255, 255), -1)

                    # Apply the circular mask to the face crop
                    face_crop_masked = cv2.bitwise_and(face_crop, face_crop, mask=mask)

                    # Create an alpha channel
                    b, g, r = cv2.split(face_crop_masked)
                    alpha = mask
                    face_crop_rgba = cv2.merge((b, g, r, alpha))

                    face_filename = "face_" + secure_filename(file.filename).rsplit('.', 1)[0] + ".png"
                    face_file_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
                    cv2.imwrite(face_file_path, face_crop_rgba)

    except Exception as e:
        error = str(e)
        return render_template('upload_front.html', error=error, mrz_data=request.form.get('mrz_data'))

    mrz_data = eval(request.form.get('mrz_data'))
    return render_template('profile.html', face_filename=face_filename, mrz_data=mrz_data, error=None)

def find_existing_face_image(mrz_data_str):
    """
    Finds an existing face image based on MRZ data.

    Parameters:
    mrz_data_str: str
        String representation of MRZ data.

    Returns:
    str
        Filename of the existing face image or None if not found.
    """
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
