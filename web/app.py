import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = r'static\images'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model = tf.keras.models.load_model('model3.h5')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def prediction(img):
    image = img.resize((256,256))
    image = image.convert('RGB')
    img_array = np.array(image)
    img = img_array/255.0
    original_shape = (256, 256, 3)
    new_shape = (1,) + original_shape
    new_array = np.empty(new_shape, dtype=np.uint8)
    
    y_pred = model.predict(new_array)
    
    result = y_pred[0]
    if result[0]==1:
        return ("Fetal abdomen")
    elif result[1]==1:
        return ("Fetal brain")
    elif result[2]==1:
        return ("Fetal femur")
    else:
        return ('Fetal thorax')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        
        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', message='No selected file')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img = Image.open(filepath)
            class_name = prediction(img)

            return render_template('index.html', message='File uploaded and processed!', image=filename, prediction=class_name)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
