import os
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import cv2  # Import the OpenCV library
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(APP_ROOT, 'uploads/')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Create an 'uploads' folder in your project directory.


@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/result/<filename>')
def uploaded_file(filename):
    # Serve the uploaded file
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the POST request has a file part
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']

        # If the user does not select a file, the browser submits an empty part without a filename
        if file.filename == '':
            return 'No selected file'

        # Save the uploaded file to the 'uploads' folder
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
            file.save(filename)
            return render_template('display_image.html', image = 'input.jpg')

    return render_template('upload.html')

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5001)))