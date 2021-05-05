from __future__ import print_function
from flask import Flask, render_template, request, send_from_directory, send_file
import requests
from flask import jsonify
from PIL import Image
import numpy as np 
import os
from azureml.core import Workspace, Datastore, Dataset
import json

import sys
from joblib import dump, load
from PIL import Image
from skimage.transform import resize
import cv2
import time
import re


import tensorflow as tf
from tensorflow import keras



app = Flask(__name__)




########################
# FONCTIONS IMAGES
########################
def label2cat(label):       
    my_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:1, 8:1, 9:1, 10:1, 11:2,
              12:2, 13:2, 14:2, 15:2, 16:2, 17:3, 18:3, 19:3, 20:3, 21:4, 22:4,
              23:5, 24:6, 25:6, 26:7, 27:7, 28:7, 29:7, 30:7, 31:7, 32:7, 33:7}
    a = Image.open(label)
    #print (np.unique(a))    
    fct = np.vectorize(my_dict.get)
    img = fct(a)
    return img
    
def LayersToRGBImage(img):
    colors = [(0, 0, 0 ), (128, 64, 128), (150, 100, 100),
             (220, 220, 0), (107, 142, 35), (70, 130, 180),
             (220, 20, 60), (119, 11, 32)]

    nimg = np.zeros((img.shape[0], img.shape[1], 3))
    for i in range(img.shape[2]):
        c = img[:,:,i]
        col = colors[i]
        
        for j in range(3):
            nimg[:,:,j]+=col[j]*c
    nimg = nimg/255.0
    return nimg

def recolor_img(label_path):
    #img = Image.open(label_path)
    #print (img.size)
    #print(np.unique(img))
    img = label_path
    img2 = np.zeros((1024, 2048, 3), dtype=np.uint8)
        
    for y in range(img.shape[1]):
        for x in range(img.shape[0]):
            #z = img.getpixel((x,y))
            z = img[x ,y ]
            if z == 0:
                r, g, b =0, 0, 0
            elif z == 1:
                r, g, b = 128, 64, 128
            elif z == 2: 
                r, g, b = 150, 100, 100
            elif z == 3: 
                r, g, b = 220, 220, 0
            elif z == 4: 
                r, g, b = 107, 142, 35
            elif z == 5: 
                r, g, b = 70, 130, 180 
            elif z == 6: 
                r, g, b = 220, 20, 60
            else: 
                r, g, b = 119, 11, 32 
            img2[x, y] = (r, g, b)    
            #img2.putpixel((x,y),(r,g,b,255))            

    return img2


########################
# DASHBOARD
########################
@app.route('/')
def dashboard():
    print('>>>>>>>>>>>>>>> /')
    salutation = 'Thank for using flask-fundamentum!'
    return render_template('home.html', msg=salutation)

                                      


########################
# RETOURNE LISTE DES IMAGES  # retourne liste des noms des images en JSON
########################
@app.route("/api/get_img_list/",  methods= ['GET'])
def listimgs():
    print( '>>>>>>>>>>>>>>> /api/get_img_list')
    
    print( '\t >>>>>>>>>>>>>>> 1')
    val_files = np.load('static/val_files_frankfurt.npy')
    
    print(' \t >>>>>>>>>>>>>>> 2')
    response =  {'status': 'ok', 'data': val_files.tolist()}

    return jsonify(response)    



########################
# TELECHARGER IMAGE RAW 
########################
@app.route("/api/download_image/", methods=['POST'])
def download_image():
    print('>>>>>>>>>>> /api.download_image')
    
    print('\t >>>>>>>>>>>>>>> 1')
    data = request.get_json()
    image = data["image"]
      
    print('\t >>>>>>>>>>>>>>> 2')
    
    subscription_id = 'e7c8495b-647b-464c-8fc6-74fad7e47bb1'
    resource_group = 'gpe-ressource-lab1'
    workspace_name = 'test-workspace3' 

    ws = Workspace(subscription_id, resource_group, workspace_name)
    dstore = Datastore.get(ws, 'workspaceblobstore')  
    # on va chercher l'image brute dans le dossier Azure
    dset_raw = Dataset.File.from_files(path=[(dstore, ('data_folder2/leftImg8bit/val/frankfurt/' + image + '_leftImg8bit.png'))])
    dset_gt = Dataset.File.from_files(path=[(dstore, ('data_folder2/gtFine/val/' + image + '_gtFine_labelIds.png' ))])     
    print(dset_raw)
    print(dset_gt)
           
                     
    print('\t >>>>>>>>>>>>>>> 3')
    temp_dir_raw = dset_raw.download(os.getcwd()+ '/static/images/raw/', overwrite = True)
    temp_dir_gt = dset_gt.download(os.getcwd()+ '/static/images/mask/', overwrite = True)                                  
    
    print('\t >>>>>>>>>>>>>>> 4')
    data = {
        'raw':"./static/images/raw/" + image + '_leftImg8bit.png',
        'mask':"./static/images/mask/" + image + '_gtFine_labelIds.png'
        }
                                      
    response = {'status': 'ok', 'data':data}
    return jsonify(response)

###########
# PREDICT   # La requête predict renvoie l’adresse où l’image prédite est stoquée
###########
                                      
@app.route("/api/predict/", methods=['POST'])
def predict(): 
                                      
    print('>>>>>>>>>>> /api/predict')

    print('\t >>>>>>>>>>>>>>> 1')
    data = request.get_json()
    image = data["image"]
    
    
    print('\t >>>>>>>>>>>>>>> 2')
    image_path = "./static/images/raw/" + image + '_leftImg8bit.png_img.png'

    input_data = json.dumps({"data" : image })

    #headers = {'Content-Type': 'application/json', 'Cache-Control': 'no-cache'}
    headers = {'Content-Type': 'application/json'}
    # adresse du endpoint:
    resp = requests.post('http://3b4d0601-2904-4b29-a504-b173f8a49ad4.westeurope.azurecontainer.io/score', input_data, headers=headers)   
    print('resp: ', resp)
     
       
    print('\t >>>>>>>>>>>>>>> 3') 
        
    ##################
    ### MODEL VIA ENDPOINT
    ##################
    #print(resp.json())
    new_image = tf.keras.preprocessing.image.array_to_img(np.array(json.loads(resp.json())))  
    new_image.save('./static/images/predict/'+ image + '_predict.png', 'PNG') 
 
    
    # Repertoire static: stoque de manière temporaire le fichier segmenté.                                       
    print('\t >>>>>>>>>>>>>>> 4') 
    response = {'status': 'ok', 'data': "./static/images/predict/" + image + '_predict.png'}
    return jsonify(response)


###########
# CLEAN # La fonction clean supprime tous les répertoires
###########                                      
#@app.route('/api/clean/')
#def clean():                                      
                                              
    
if __name__ == "__main__":
    app.run()
    
 
    
    
