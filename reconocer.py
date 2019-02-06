# OpenCV module
import cv2
# Modulo para leer directorios y rutas de archivos
import os
# OpenCV trabaja con arreglos de numpy
import numpy
# Utlilidades para limpiar codigo
from utils import *

LEARN = True

names, id = {}, 0

for (subdirs, dirs, files) in os.walk(DIR_FACES):
    for subdir in dirs:
        names[id] = subdir
        id += 1

(im_width, im_height) = (112, 92)

# OpenCV entrena un modelo a partir de las imagenes
model = cv2.face.LBPHFaceRecognizer_create()
model.read(os.path.join(MODEL_FACES,"model.yml"))

# Parte 2: Utilizar el modelo entrenado en funcionamiento con la camara
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    # leemos un frame y lo guardamos
    rval, frame = cap.read()
    frame = cv2.flip(frame, 1, 0)

    # convertimos la imagen a blanco y negro
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # redimensionar la imagen
    mini = cv2.resize(gray, (int(gray.shape[1] / SIZE), int(gray.shape[0] / SIZE)))

    """buscamos las coordenadas de los rostros (si los hay) y
   guardamos su posicion"""
    faces = face_cascade.detectMultiScale(mini)

    for i in range(len(faces)):
        face_i = faces[i]
        (x, y, w, h) = [v * SIZE for v in face_i]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))

        # Intentado reconocer la cara
        prediction = model.predict(face_resize)

        # Dibujamos un rectangulo en las coordenadas del rostro
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Escribiendo el nombre de la cara reconocida
        # La variable cara tendra el nombre de la persona reconocida
        cara = '%s' % (names[prediction[0]])

        # Si la prediccion tiene una exactitud menor a 100 se toma como prediccion valida
        if prediction[1] < 100:
            # Ponemos el nombre de la persona que se reconoció
            cv2.putText(frame, '%s - %.0f' % (cara, prediction[1]), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 255, 0))
            if LEARN:
                nombre = names[prediction[0]]
                path = os.path.join(DIR_FACES, nombre)
                pin = sorted([int(n[:n.find('.')]) for n in os.listdir(path)
                              if n[0] != '.'] + [0])[-1] + 1

                # Metemos la foto en el directorio
                cv2.imwrite('%s/%s.bmp' % (path, pin), face_resize)

        # Si la prediccion es mayor a 100 no es un reconomiento con la exactitud suficiente
        elif prediction[1] > 101 and prediction[1] < 500:
            # Si la cara es desconocida, poner desconocido
            cv2.putText(frame, 'Desconocido', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

    # Mostramos la imagen
    cv2.imshow('OpenCV Reconocimiento facial', frame)

    # Si se presiona la tecla ESC se cierra el programa
    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyAllWindows()
        if LEARN:
            from train import train
            train()
        break
