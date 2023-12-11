import os
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, jsonify
import cv2
import time
from threading import Thread

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(APP_ROOT, 'uploads/')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


def threshold_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding (you can customize the thresholding method)
    _, thresholded = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
    if os.path.exists(file_path):
        os.remove(file_path)
    # Simulate a 90-second delay
    time.sleep(20)

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

            # Start a background thread to process the image
            thread = Thread(target=threshold_image, args=(filename,))
            thread.start()

            # Return a JSON response indicating the processing has started
            # return jsonify({'status': 'processing', 'image': 'processing.jpg'})
            return render_template('display_image2.html', image='result.jpg')


    return render_template('upload.html')


@app.route('/processing_status')
def processing_status():
    thresholded_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')

    # Check if the thresholded image exists
    if os.path.exists(thresholded_path):
        return jsonify({'status': 'completed', 'image': 'result.jpg'})
    else:
        return jsonify({'status': 'processing', 'image': 'result.jpg'})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5003)))
