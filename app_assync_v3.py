#!flask/bin/python
from flask import Flask, jsonify, abort, make_response,request, json, redirect
import requests
from multiprocessing.pool import ThreadPool
pool = ThreadPool(processes=3)

import multipart as mp

try:
    from io import BytesIO
except ImportError:
    from StringIO import StringIO as BytesIO
import cv2
import os
import os.path
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

# Import Mask RCNN
  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import tensorflow as tf

from pdf2image import convert_from_path
from pdf2image import convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

graph = tf.get_default_graph()

ROOT_DIR = os.path.abspath("")

sys.path.append(ROOT_DIR)

app = Flask(__name__)

#logsOfError=''

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
SEAL_WEIGHTS_PATH = "mask_rcnn_seal_0030.h5"

class SealConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "seal"
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + seal
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.95

config = SealConfig()

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()


DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Or, load the last model you trained
weights_path = model.find_last()

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

def postChecking(masks, rois):

    ############################################################# Crop place where there is a mask ##############################################################################

    masks = masks[rois[0]:rois[2],rois[1]:rois[3]]
    masks=masks.reshape(masks.shape[0], masks.shape[1], 1)

    ############################################################# Create a ideal seal shape in the input matrix #########################################################
    
    ideal_seal = np.zeros((masks.shape[0],masks.shape[1]),  dtype=bool)
    ideal_seal = ideal_seal.reshape(masks.shape[0],masks.shape[1],1)
    for i in range(len(ideal_seal)):
        for j in range(len(ideal_seal[i])):
            if np.sqrt(pow((len(masks[0])/2-i),2)+pow((len(masks[0])/2-j),2))<=len(masks[0])/2: # add with Pythagorean theorem
                ideal_seal[i][j]=True
                
    ############################################################# Comparison input matrix and ideal seal matrix #########################################################
    
    res = np.subtract(ideal_seal, masks, dtype=np.int)

    ############################################################# Counting inappropriate cells in matrix between two matrix ############################################## 
    
    count=0
    for i, el in enumerate(res):
        for j in el:
            if j!=0:
                count+=1
    
    ############################################################# Counting percent of inappropriate cells in matrix ####################################################

    shape_arr = masks.shape[0]*masks.shape[1]*masks.shape[2]
    res_perc = count*100/shape_arr
    print(res_perc)
    if res_perc<20:
        return True
    else:
        return False


def detect_seal(image):

    results_rois=[]
    results_scores=[]


    results = model.detect([image], graph, verbose=1)
    

    w = image.shape[1]
    h = image.shape[0]
    r = results[0] 
    rois = r['rois'] # format r['rois']=[y1,x1,y2,x2]
    scores = r['scores']
    masks = r['masks']
    print("------------------------------------------------")
    print(rois)
    print(scores)
    for i, result in enumerate(scores):
        mask = masks[:,:,i]
        if result>0.95 and (rois[i][3]-rois[i][1])/(rois[i][2]-rois[i][0])>0.9: # probability results AND width to height ratio
            if postChecking(mask, rois[i]):
                image1 = cv2.rectangle(image, (rois[i][1],rois[i][0] ), (rois[i][3],rois[i][2]), (255, 0, 0))   
                cv2.imwrite("test.jpg", image1) 
                results_rois.append(rois[i])
                results_scores.append(result)

    print(results_rois)
    if len(results_rois)>0:
        return results_rois, h, w     
    else:
        return None, h, w

    

def async_processing(app_id, filename, url, js_for_id, logsOfError):
    #pages = convert_from_bytes(bytearr)  
    #poppler_path='venv/poppler-0.68.0/bin'
    found=[]
    try:       
        #path_pdf_file = '/applications/'+str(app_id) +'/'+ str(filename)
        path_pdf_file = str(app_id) +'/'+ str(filename)
        try:
            pages = convert_from_path(path_pdf_file)
        except Exception as e:
            print("2-" +str(e))
            logsOfError = logsOfError + "line-179/ error:"+str(e)+"; "
        
        for i,page in enumerate(pages):
            print()
            print('app_id='+str(app_id)+' filename='+str(filename)+' page='+str(i))
            img_array = np.array(page)
            try:
                result, h, w = detect_seal(img_array)
            except Exception as e:
                print("3-" +str(e))
                logsOfError = logsOfError +  "line-191/ error:"+str(e)+"; "
            if result is not None:
                for j in result:
                    res_dict = {'x':format(float(j[1]*100/w),'.3f'), 'y':format(float(j[0]*100/h),'.3f'), 'width':format(float(j[3]*100/w)-float(j[1]*100/w),'.3f'), 'height':format(float(j[2]*100/h)-float(j[0]*100/h),'.3f'), 'page':int(i)}
                    found.append(res_dict)
            
    except  Exception as e:
        print("4-" +str(e))
        logsOfError = logsOfError +  "line-171/ error:"+str(e)+"; "

    data ={'app_id':app_id, 'filename':filename, 'json':js_for_id, 'found': found, 'logs':str(logsOfError)}
    print(data) 
    try:
        requests.post(str(url), data=json.dumps(data), headers={"content-type" : "application/json"})
        print("send")
    except Exception as e:
        print(e)
       
    
    print(data)
    print(len(data['found']))
    #return data




@app.route('/detectSeal', methods=["POST"])
def index():
    if request.method == "POST":

        data = request.get_json(force=True)
        app_id = data['app_id']
        filename = data['filename']
        url = data['url']
        js_for_id = data['json']

        logsOfError=''

        filename_check = filename.split(".")
        if filename_check[-1]!='pdf':
            return "Wrong file format"
        #elif os.path.isfile('/applications/'+str(app_id) +'/'+ str(filename))==False:
        elif os.path.isfile(str(app_id) +'/'+ str(filename))==False:
            return "Don`t found this file"
        else:
            result = pool.apply_async(async_processing, args=(app_id, filename, url, js_for_id, logsOfError))
            #result = async_processing(app_id, filename, url, js_for_id, logsOfError)
            return "Received"
    else:
        return "Method is not post!"
      



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8839, threaded=True)
