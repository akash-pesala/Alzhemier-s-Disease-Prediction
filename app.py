import sys
import os
import glob
import re
import numpy as np

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from flask import Flask , render_template , request , url_for
import pickle

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

import tensorflow as tf
from tensorflow.keras import models, layers
import math
import matplotlib.pyplot as plt

from matplotlib.image import imread
import cv2
from PIL import Image

app = Flask('__name__')

SelectedRFModel_AD = pickle.load(open('SelectedRFModel_AD.pkl', 'rb'))
SelectedBoostModel = pickle.load(open('SelectedBoostModel.pkl', 'rb'))

###############################--- home page / Index page --#################################
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/server_home')
def server_home():
    return render_template('index.html')

@app.route('/blog-single')
def blog_single():
    return render_template('blog-single.html')

@app.route('/about')
def about():
    return render_template('404.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

###############################--- home page / Index page --#################################

@app.route('/Early_Alzhiemer')
def Early_Alzhiemer():
    return render_template('Early_Alzhiemer.html')

@app.route('/Early_predict', methods = ['GET', 'POST'])
def Early_predict():
    if(request.method == "POST"): 
        Gender = request.form['Gender'] 
        Age = request.form['Age'] 
        EDUC =request.form['EDUC'] 
        #SES = request.form['SES'] 
        SES = request.form['SES'] 
        MMSE = request.form['MMSE']
        eTIV = request.form['eTIV'] 
        nWBV = request.form['nWBV'] 
        ASF = request.form['ASF'] 

        Gender_values = ['Female','Male']
        G=Gender_values.index(Gender.title())

        arr=np.array([G,Age,EDUC,SES,MMSE,eTIV,nWBV,ASF])
        arr=arr.reshape(1,-1)

        res1 = SelectedRFModel_AD.predict(arr)
        res2 = SelectedBoostModel.predict(arr)

        if res2==0:
            output="Non-demented 0"
        elif res2==1:
            output="Demented 1"

        output = "Early Prediction of Alzheimer's diesease is: " + output

        confidence=90

        #name = str(res1[0]) + " ----- " + str(res2[0])
        return render_template('result.html', gender=Gender, age=Age, educ=EDUC, ses=SES, mmse=MMSE, etiv=eTIV, nwbv=nWBV, asf=ASF, prediction_output=output, confidence=confidence)

    else:
        return "Sorry!!"

###################################################################################

@app.route('/Alzhiemer_Disease')
def Alzhiemer_Disease():
    return render_template('Alzhiemer_Disease.html')

model_path = 'alzheimer_detection_model.h5'

model = load_model(model_path)

class_names = np.array(['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented'])

@app.route('/image_predict', methods = ['GET', 'POST'])
def image2():
    if(request.method == 'POST'):
        f=request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # # Make prediction
        # image=imread(file_path)
        # predicted_class, confidence,val  = predict1(model,image)

        # img = im.fromarray(image)


        # Load the image using PIL library
        image = Image.open(f)
        image = image.resize((128, 128))  # Resize the image to 128x128 pixels
        # Convert grayscale to RGB format
        image = image.convert("RGB")
        img = np.array(image).reshape(-1, 128, 128, 3)  # Reshape to (batch_size, height, width, channels)

        result = model.predict(img)

        predicted_class_index = np.argmax(result)

        predicted_class = class_names[predicted_class_index]
        print("Predicted class:", predicted_class)

        val = max(result[0])
        confidence = round(100 * (max(result[0])), 2)

        #res = 'The predicted stage is \''+ predicted_class +"\'" + "\n Confidence:" + str(confidence) + "\n Val:" + str(val) 
        #res=result

        l1 = 100 * (result[0][0])
        l2 = 100 * (result[0][1])
        l3 = 100 * (result[0][2])
        l4 = 100 * (result[0][3])

        l1f = "{:.9f}%".format(l1)
        l2f = "{:.9f}%".format(l2)
        l3f = "{:.9f}%".format(l3)
        l4f = "{:.9f}%".format(l4)

        #l1f,l2f,l3f,l4f = l1,l2,l3,l4

        #per = "\n Mild_Demented : "+str(l1f) + "\n Moderate_Demented : "+str(l2f) + "\n Non_Demented : "+str(l3f) + "\n Very_Mild_Demented : "+str(l4f)
        
        p1 = "\n Mild_Demented : "+str(l1f)
        p2 = "\n Moderate_Demented : "+str(l2f)
        p3 = "\n Non_Demented : "+str(l3f)
        p4 = "\n Very_Mild_Demented : "+str(l4f)

        res = 'The predicted stage is \''+ predicted_class +"\'" 

        confidence = 90
        '''predicted_class = class_names[predicted_class_index]
        print("Predicted class:", predicted_class)

        res = 'The predicted stage is \''+ predicted_class +"\'" + "\n Confidence:" + str(confidence) + "\n Val:" + str(val)'''
        #return render_template("open.html", n = res)

    return render_template("result2.html", n=res, p1=p1, p2=p2, p3=p3, p4=p4, confidence=confidence)

###################################################################################

if __name__ == "__main__":
    app.run(debug = True)