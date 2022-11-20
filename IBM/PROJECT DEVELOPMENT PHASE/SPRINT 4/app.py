from flask import Flask, escape, request, render_template
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# model = load model("models/fruits.h5")
model = load_model("models/1")

class_name = [ 'APPLES',
'BANANA',
'ORANGE', 
'PINEAPPLE',
'WATERMELON ']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        f = request.files['fruit']
        filename= f.filename
        target = os.path.join(APP_ROOT, 'images/')
        # print(target)
        des = "/".join([target, filename])
        f.save(des)

        test_image = image.load_img('images\\'+filename,target_size=(300,300))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        prediction = model.predict(test_image)
        # print(prediction)

        predicted_class = class_name[np.argmax(prediction[0])]
        # print(predicted_class)
        confidence = round(np.max(prediction[0])*100)

        # print(confidence)

        return render_template('prediction.html', confidence= "chances -> "+str(confidence)+ "%", prediction = "prediction -> "+str(predicted_class))

    else:
        return render_template('prediction.html')




if __name__ == '__main__':
    app.debug = True
    app.run()