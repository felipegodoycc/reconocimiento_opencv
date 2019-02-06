import os
import cv2
import numpy
from utils import *

def train():
    # Crear una lista de imagenes y una lista de nombres correspondientes
    print("Leyendo archivos...")
    (images, lables, names, id) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(DIR_FACES):
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(DIR_FACES, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                lable = id
                images.append(cv2.imread(path, 0))
                lables.append(int(lable))
            id += 1
    # Crear una matriz Numpy de las dos listas anteriores
    (images, lables) = [numpy.array(lis) for lis in [images, lables]]

    print("Creando modelos")
    model = cv2.face.LBPHFaceRecognizer_create()
    print("Entrenando modelo")
    model.train(images, lables)
    print("Guardando modelo")
    model.save(MODEL_FACES+'model.yml')

train()