import re
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from flask import Flask, request, render_template
# from tensorflow.keras import modelspip 
from tensorflow.keras.preprocessing import image
# from tensorflow.python.ops.gen_array_ops import concat
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename 

#loading the model
app = Flask(__name__)

model = load_model(r"C:\\Users\\vemul\\OneDrive\Desktop\\flask\\uploads\\crime.h5",compile=False)
#home page
@app.route('/')
def home():
    return render_template('home.html')

#prediction page
# @app.route('/prediction')
# def prediction():
#     return render_template('predict.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    result = None


    if request.method == 'POST':
        #get the file from post request
        # try:
        f = request.files['image']  
        
        #save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename('f.filename'))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(64, 64))
        
        x = image.img_to_array(img) # converting image into array
        x = np.expand_dims(x,axis=0) # expanding  Dimensions
        pred = np.argmax(model.predict(x)) #predicting the higher probability index
        op = ['Fighting','Arrest','Vandalism','Assault','Stealing','Arson','NormalVideos','Burglary','Explosion','Robbery','Abuse','Shooting','Shoplifting','RoadAccident']
        result = op[pred]
        result = 'The predicted output is {}' .format(str(result))
        print(result)
    return render_template('predict.html', text=result)
    
        # except Exception as e:
        #     print("Error : ",str(e))
        #     result = "An error occured while processing the image"
        # return render_template('predict.html', text=result)


""" Running our application """
if __name__ == "__main__":
    app.run(debug=True)
        
        
        
        
        
        
        
        
        
        
        
        
        
        