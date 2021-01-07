import numpy as np
import cv2

#Loading load_model from tensorflow.keras.models
from tensorflow.keras.models import load_model

#Loading the facenet model
embedding_model=load_model("Facenet_model.h5")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


images = {}

images['ben_afflek'] = cv2.imread('data/ben_afflek/1.jpg')
images['elton_john'] = cv2.imread('data/elton_john/1.jpg')
images['jerry_seinfeld'] = cv2.imread('data/jerry_seinfeld/1.jpg')
images['madonna'] = cv2.imread('data/madonna/1.jpg')
images['obama'] = cv2.imread('data/obama/1.png')
images['michelle_obama'] = cv2.imread('data/michelle_obama/1.png')


import numpy as np
def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))
    y = (x - mean) / std_adj
    return y

for name in images:
    img=images[name]
    img=cv2.resize(img,(160,160))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    images[name]=prewhiten(img)


def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

for name in images:
    img=images[name]
    img=np.reshape(img,(1,160,160,3))
    emb=embedding_model.predict(img)
    images[name]=l2_normalize(emb)

def predict_name(img):
    img=cv2.resize(img,(160,160))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=np.resize(img,(1,160,160,3))
    img=prewhiten(img)

    emb=embedding_model.predict(img)
    emb_norm=l2_normalize(emb)

    minimum_dis=9999
    person=None

    for name in images:
        dis=np.linalg.norm(emb_norm-images[name])
        if dis<minimum_dis:
            person=name
            minimum_dis=dis
    return person,minimum_dis

img = cv2.imread('obama_photo.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.2, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    face = img[y:y+h, x:x+w]
    name,_=predict_name(face)
    
    img = cv2.putText(img, name, (x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255)) 

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
