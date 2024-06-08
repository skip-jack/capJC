from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from deepface import DeepFace
import base64
import os
import uuid
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files.get('file')
        image_data = request.form.get('imageData')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
        elif image_data:
            header, encoded = image_data.split(",", 1)
            data = base64.b64decode(encoded)
            filename = str(uuid.uuid4()) + ".png"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(file_path, "wb") as f:
                f.write(data)
        else:
            return "No valid image file provided", 400

        analysis_results = DeepFace.analyze(img_path=file_path, actions=['age', 'gender', 'race'], enforce_detection=False)

        # Handle the case where analysis_results is a list
        if isinstance(analysis_results, list):
            analysis = analysis_results[0]
        else:
            analysis = analysis_results

        # Extract gender from analysis
        gender = 'Male' if analysis['gender']['Man'] > analysis['gender']['Woman'] else 'Female'
        
        return render_template('results.html', filename=filename, analysis=analysis, gender=gender)
    except Exception as e:
        print("Error occurred: ", traceback.format_exc())
        return "Error processing the file", 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True)


