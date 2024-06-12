import cv2
from keras.models import model_from_json
import numpy as np

# Load the model architecture from JSON file
with open("emotiondetector.json", "r", encoding="utf-8") as json_file:
    model_json = json_file.read()

# Load the model architecture from JSON string
model = model_from_json(model_json)

# Load the model weights
model.load_weights("emotiondetector.h5")

# Load the Haar cascade classifier for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features from image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Open webcam
webcam = cv2.VideoCapture(0)

# Emotion labels dictionary
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Main loop for capturing frames and predicting emotions
while True:
    i,im=webcam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(im,1.3,5)
    try: 
        for (p,q,r,s) in faces:
            image = gray[q:q+s,p:p+r]
            cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
            image = cv2.resize(image,(48,48))
            img = extract_features(image)
            #pred = model.predict(img)
            prediction_label = labels[model.predict(img).argmax()]
            print("Predicted Output:", prediction_label)
            cv2.putText(im,prediction_label)
            cv2.putText(im, '% s' %(prediction_label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))
        cv2.imshow("Output",im)
        cv2.waitKey(27)
    except cv2.error:
        pass

