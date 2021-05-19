import pandas as pd
import os
import io 
import cv2
import numpy as np
from keras.models import load_model
from .detector.face_detector import MTCNNFaceDetector
#from detector import face_detector
import tensorflow as tf


class Diagnose():
    def __init__(self):
        pass

        
    def Image_croping(self , image , detect_model_path):
        """ return left eye and right eye image cropped"""
       ## loading wieghts
        fd = MTCNNFaceDetector(sess=tf.compat.v1.keras.backend.get_session(), model_path= detect_model_path) # loading face detection model
        face, lms = fd.detect_face(image) # detect the number of faces 
        if len(face) == 1 : 
            left_eye_im, right_eye_im = fd.cropImage(image,lms)
            return left_eye_im , right_eye_im , face
        if len(face) > 1 :
            return 'Multiple faces detected' , 'Multiple faces detected'  , face
        return 'No face detected' ,  'No face detected' , face


    def Eyes_diagnosis(self, left_eye_im , right_eye_im , diagnosis_model_path):

        """ return description and probability of disease for each eye"""
        diagnosis_model_path = os.path.join(os.path.dirname(__file__), diagnosis_model_path)
        model = load_model(diagnosis_model_path) # load disease detection model 
        ############################# left eye #############################
        left_eye_im = cv2.resize(left_eye_im, (100, 100))  
        left_eye_im = left_eye_im.reshape(1 ,100 , 100 , -1)

        left_eye_im_diagnosis = model.predict(left_eye_im)

        if left_eye_im_diagnosis > 0.56:
            left_eye_im_desc =  ' Left Eye : Cataract detected'
        else :
            left_eye_im_desc = 'Left Eye : No Cataract detected'


        ############################# right eye #############################
        right_eye_im = cv2.resize(right_eye_im, (100, 100))
        right_eye_im = right_eye_im.reshape(1 ,100 , 100 , -1)
        right_eye_im_diagnosis  = model.predict(right_eye_im)

        if right_eye_im_diagnosis > 0.56:
            right_eye_im_desc =  'Right Eye : Cataract detected'
        else :
            right_eye_im_desc = 'Right Eye : No Cataract detected'

        return left_eye_im_desc ,left_eye_im_diagnosis[0], right_eye_im_desc , right_eye_im_diagnosis[0]

    def Diagnose_patient(self , image , detect_model_path , diagnosis_model_path):
        
        """ return croped eye image, diagnosis descripition and probability for both eyes ( 6 output items) """
    
        left_eye_im , right_eye_im , face = self.Image_croping(image , detect_model_path) # crop eyes & ensure image is good for diagnosis

        if len(face) != 1 :
            return left_eye_im , left_eye_im, left_eye_im , left_eye_im , left_eye_im , left_eye_im  

        left_eye_im_desc ,left_eye_im_diagnosis, right_eye_im_desc , right_eye_im_diagnosis = self.Eyes_diagnosis(left_eye_im , right_eye_im , diagnosis_model_path) # diagnosis 

        return left_eye_im , left_eye_im_desc , left_eye_im_diagnosis , right_eye_im , right_eye_im_desc , right_eye_im_diagnosis
