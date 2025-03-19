from flask import Flask, Response
import cv2 as cv
import numpy as np
import os
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask
app = Flask(__name__)

# Initialize FaceNet and SVM Model
facenet = FaceNet()
faces_embeddings = np.load('models/faces_embeddings_done_4classes.npz')
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
model = pickle.load(open("models/svm_model_160x160.pkl", 'rb'))

# Initialize Camera
cap = cv.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
        for x, y, w, h in faces:
            img = rgb_img[y:y+h, x:x+w]
            img = cv.resize(img, (160, 160))
            img = np.expand_dims(img, axis=0)
            ypred = facenet.embeddings(img)
            face_name = model.predict(ypred)
            final_name = encoder.inverse_transform(face_name)
            
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
            cv.putText(frame, str(final_name[0]), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

        # Convert frame ke format JPEG
        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

