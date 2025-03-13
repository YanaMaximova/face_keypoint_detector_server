import base64
import os
from flask import Flask, url_for, render_template, redirect
from flask_bootstrap import Bootstrap

from pathlib import Path
from werkzeug.utils import secure_filename
from forms import LoadPictureForm, SettingsForm
import cv2
from models import detect


app = Flask(__name__)
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'my_secret_key'
Bootstrap(app)

SAVE_PATH = os.path.join(Path().absolute(), '../pictures')
PROCESS_PATH = os.path.join(Path().absolute(), '../processed')
MODEL_PATH = os.path.join(Path().absolute(), "../model/facepoints_model.pt")

app.config['UPLOAD_FOLDER'] = SAVE_PATH
app.config['PROCESS_FOLDER'] = PROCESS_PATH



@app.route('/processing/<string:image_name>', methods=['GET', 'POST'])
def image_processing(image_name):
    settings_form = SettingsForm()
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    with open(image_path, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode()
    image_src = 'data:image/png;base64, {0}'.format(image_data)

    img = cv2.imread(image_path)

    components = detect(model_path=MODEL_PATH,
               filename=image_path)
    
    xx, yy = components[::2], components[1::2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for x, y in zip(xx, yy):
        cv2.circle(img_rgb, (int(x), int(y)), 3, (255, 0, 0), -1)

    processed_path = os.path.join(app.config['PROCESS_FOLDER'], image_name)
    cv2.imwrite(processed_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    with open(processed_path, 'rb') as processed_file:
        processed_data = base64.b64encode(processed_file.read()).decode()

    processed_src = 'data:image/png;base64, {0}'.format(processed_data)

    if settings_form.validate_on_submit():
        return redirect(url_for('index'))

    return render_template('process.html',
                           image_src=image_src, processed_src=processed_src,
                           settings_form=settings_form)
    '''
    processing = PictureProcess()
    output, components = processing.process(img)

    save_path = os.path.join(app.config['PROCESS_FOLDER'], image_name)
    cv2.imwrite(save_path, output)

    process_path = os.path.join(app.config['PROCESS_FOLDER'], image_name)
    try:
        with open(process_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode()
            processed_src = 'data:image/png;base64, {0}'.format(image_data)
    except Exception as e:
        print(f'Error {e}')
        processed_src = image_src

    if settings_form.validate_on_submit():
        return redirect(url_for('index'))

    return render_template('process.html',
                           image_src=image_src, processed_src=processed_src,
                           settings_form=settings_form, components=components)
    '''

@app.route('/', methods=['GET', 'POST'])
def index():
    load_picture_form = LoadPictureForm()
    try:
        if load_picture_form.validate_on_submit():
            file = load_picture_form.picture.data
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return redirect(url_for('image_processing', image_name=filename))

    except Exception as e:
        print(e)

    return render_template('index.html',
                           load_picture_form=load_picture_form)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
