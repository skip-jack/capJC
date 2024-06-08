from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from deepface import DeepFace
import cv2
import os

app = Flask(__name__)

os.makedirs(os.path.join(app.static_folder, 'uploads'), exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
           # Save the uploaded file
            file = request.files['image']
            filename = secure_filename(file.filename)  # This ensures the filename is safe to use on any filesystem
            image_path = os.path.join(app.static_folder, 'uploads', filename)
            file.save(image_path)
            
            # Analyze the image using DeepFace
            analysis = DeepFace.analyze(cv2.imread(image_path), actions=['age', 'gender', 'race'])

            # Redirect to the results page with the analysis results
            return redirect(url_for('results', name=request.form['name'], 
                                    age=analysis['age'], 
                                    gender=analysis['gender'], 
                                    race=analysis['dominant_race'], 
                                    image_url=file.filename))
    return render_template('home.html')

@app.route('/results')
def results():
    # Extract the query parameters
    name = request.args.get('name')
    age = request.args.get('age')
    gender = request.args.get('gender')
    race = request.args.get('race')
    image_url = request.args.get('image_url')

    return render_template('results.html', 
                           name=name, 
                           age=age, 
                           gender=gender, 
                           race=race, 
                           image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)