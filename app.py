
# coding=utf-8
import sys
import os, shutil
import glob
import re
import numpy as np
import cv2

# Flask utils
from flask import Flask,flash, request, render_template,send_from_directory
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__, static_url_path='')
app.secret_key = os.urandom(24)

app.config['CARTOON_FOLDER'] = 'cartoon_images'
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route('/uploads/<filename>')
def upload_img(filename):
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/cartoon_images/<filename>')
def cartoon_img(filename):
    
    return send_from_directory(app.config['CARTOON_FOLDER'], filename)


def cartoonize_1(img, k):

    # Convert the input image to gray scale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Peform adaptive threshold
    edges  = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)

    # cv2.imshow('edges', edges)

    # Defining input data for clustering
    data = np.float32(img).reshape((-1, 3))

  

    # Defining criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    # Applying cv2.kmeans function
    _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    # print(center)

    # Reshape the output data to the size of input image
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    #cv2.imshow("result", result)

    # Smooth the result
    blurred = cv2.medianBlur(result, 3)

    # Combine the result and edges to get final cartoon effect
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

    return cartoon

def cartoonize_2(img):

    # Convert the input image to gray scale 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # stylization of image
    img_style = cv2.stylization(img, sigma_s=150,sigma_r=0.25)
    
    return img_style

def cartoonize_3(img):

    # Convert the input image to gray scale 
    
    
    # pencil sketch  of image
    
    imout_gray, imout = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    
    return imout_gray

def cartoonize_4(img):

    # Convert the input image to gray scale 
    
    
    # pencil sketch  of image
    
    imout_gray, imout = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    
    return imout

def cartoonize_5(img, k):

    # Convert the input image to gray scale 
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img1g=cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    img1b=cv2.medianBlur(img1g,3)
    #Clustering - (K-MEANS)
    imgf=np.float32(img1).reshape(-1,3)
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    compactness,label,center=cv2.kmeans(imgf,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center=np.uint8(center)
    final_img=center[label.flatten()]
    final_img=final_img.reshape(img1.shape)
    edges=cv2.adaptiveThreshold(img1b,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,3)
    final=cv2.bitwise_and(final_img,final_img,mask=edges)

    return final

def cartoonize_6(img):

    # Convert the input image to gray scale 
    
    
    # pencil sketch  of image
    
    dst = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    
    return dst


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        
        f = request.files['file']
        style = request.form.get('style')
        print(style)
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        
        f.save(file_path)
        file_name=os.path.basename(file_path)
        
        # reading the uploaded image
        
        img = cv2.imread(file_path)
        if style =="Style1":
            cart_fname =file_name + "_style1_cartoon.jpg"
            cartoonized = cartoonize_1(img, 8)
            cartoon_path = os.path.join(
                basepath, 'cartoon_images', secure_filename(cart_fname))
            fname=os.path.basename(cartoon_path)
            print(fname)
            cv2.imwrite(cartoon_path,cartoonized)
            return render_template('predict.html',file_name=file_name, cartoon_file=fname)
        elif style =="Style2":
            cart_fname =file_name + "_style2_cartoon.jpg"
            cartoonized = cartoonize_2(img)
            cartoon_path = os.path.join(
                basepath, 'cartoon_images', secure_filename(cart_fname))
            fname=os.path.basename(cartoon_path)
            print(fname)
            cv2.imwrite(cartoon_path,cartoonized)
            return render_template('predict.html',file_name=file_name, cartoon_file=fname)
        elif style=="Style3":
            cart_fname =file_name + "_style3_cartoon.jpg"
            cartoonized = cartoonize_3(img)
            cartoon_path = os.path.join(
                basepath, 'cartoon_images', secure_filename(cart_fname))
            fname=os.path.basename(cartoon_path)
            print(fname)
            cv2.imwrite(cartoon_path,cartoonized)
            return render_template('predict.html',file_name=file_name, cartoon_file=fname)
        elif style=="Style4":
            cart_fname =file_name + "_style4_cartoon.jpg"
            cartoonized = cartoonize_4(img)
            cartoon_path = os.path.join(
                basepath, 'cartoon_images', secure_filename(cart_fname))
            fname=os.path.basename(cartoon_path)
            print(fname)
            cv2.imwrite(cartoon_path,cartoonized)
            return render_template('predict.html',file_name=file_name, cartoon_file=fname)
        elif style=="Style5":
            cart_fname =file_name + "_style5_cartoon.jpg"
            cartoonized = cartoonize_5(img,5)
            cartoon_path = os.path.join(
                basepath, 'cartoon_images', secure_filename(cart_fname))
            fname=os.path.basename(cartoon_path)
            print(fname)
            cv2.imwrite(cartoon_path,cartoonized)
            return render_template('predict.html',file_name=file_name, cartoon_file=fname)
        elif style=="Style6":
            cart_fname =file_name + "_style6_cartoon.jpg"
            cartoonized = cartoonize_6(img)
            cartoon_path = os.path.join(
                basepath, 'cartoon_images', secure_filename(cart_fname))
            fname=os.path.basename(cartoon_path)
            print(fname)
            cv2.imwrite(cartoon_path,cartoonized)
            return render_template('predict.html',file_name=file_name, cartoon_file=fname)
        else:
             flash('Please select style')
             return render_template('index.html')
            
       
              
        
    return ""



if __name__ == '__main__':
        app.run(debug=True, host="localhost", port=8080)

