import os
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, jsonify, send_file
import cv2
import time
from threading import Thread
from app import main

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(APP_ROOT, 'uploads/')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

MODULE_PATH = os.environ.get("EASYOCR_MODULE_PATH") or \
              os.environ.get("MODULE_PATH") or \
              os.path.expanduser("~/.EasyOCR/")

def threshold_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding (you can customize the thresholding method)
    _, thresholded = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Simulate a 90-second delay
    time.sleep(100)

    # Save the thresholded image
    thresholded_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
    cv2.imwrite(thresholded_path, thresholded)

    return 'result.jpg'


@app.route('/')
def home():
    return redirect(url_for('upload_file'))


@app.route('/result/<filename>')
def uploaded_file(filename):
    # Serve the uploaded file
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the POST request has a file part
        lang = request.form.get("language")
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
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
            error_path = os.path.join(app.config['UPLOAD_FOLDER'], 'log.txt')

            if os.path.exists(file_path):
                os.remove(file_path)
            if os.path.exists(error_path):
                os.remove(error_path)

            # Start a background thread to process the image
            # thread = Thread(target=threshold_image, args=(filename,))
            thread = Thread(target=main, args=(lang,))
            thread.start()

            # Return a JSON response indicating the processing has started
            # return jsonify({'status': 'processing', 'image': 'processing.jpg'})
            return render_template('display_image2.html', image='thresholded.jpg')

    print(MODULE_PATH)
    return render_template('upload.html')


@app.route('/processing_status')
def processing_status():
    thresholded_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
    error_path = os.path.join(app.config['UPLOAD_FOLDER'], 'log.txt')

    # Check if the thresholded image exists
    if os.path.exists(thresholded_path):
        return jsonify({'status': 'completed', 'image': 'result.jpg'})
    elif os.path.exists(error_path):
        # return jsonify({'status': 'processing', 'image': 'result.jpg'})
        return jsonify({'status': 'error', 'image': 'log.jpg'})
    else:
        return jsonify({'status': 'processing', 'image': 'result.jpg'})
    

@app.route('/download')
def download_image():
    # Use send_file to send the image file to the client
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5003)))
