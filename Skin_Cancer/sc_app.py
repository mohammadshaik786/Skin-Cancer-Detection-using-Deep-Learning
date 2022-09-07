# importing the python and flask server libraries to create our web app.
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)

# creating a dictionary for our classes to be predicted into.
dic = {0: 'Actinic Keratoses', 1: 'Basal Cell Carcinoma', 2: 'Benign Keratosis', 3: 'Dermatofibroma',
       4: 'Melanoma', 5: 'Melanocytic nevi', 6: 'Vascular Skin Lesions'}
# loading the model of mobilenet which we will use to tes our images through the web app.
model = load_model('/Users/aqdus/desktop/skin_cancer/model.h5')

model.make_predict_function()

# defining the predict and preprocess function to test our input image.
def predict_label(img_path):
	i = image.load_img(img_path, target_size=(224,224))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 224,224,3)
	p = np.argmax(model.predict(i))
	return dic[p]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("main.html") #getting our data from main.html file.

@app.route("/about")
def about_page():
	return "Skin Cancer Detection using MobileNet Deep Learning Model"
 # using method and get and post to recieve the image from the webapp html file which is main.html.
@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename
		img.save(img_path)
		print(img_path)
		im=Image.open(img_path)
		data=io.BytesIO() #converting our image to bytes for simplicity and output.
		im.save(data,'JPEG')
		encoded_img_data=base64.b64encode(data.getvalue()) #encoding the image to display.
		p = predict_label(img_path)

	return render_template("main.html", prediction=p, img_data=encoded_img_data.decode('utf-8')) #passing the result of prediction and image back to the main.html file using render template function.


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
