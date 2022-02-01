#importing libraries
import cv2
from albumentations import( Compose,Resize, ToFloat)
import numpy as np
from keras.models import load_model
from keras.metrics import mean_absolute_error
from google.colab import files
import matplotlib.pyplot as plt
import dlib
from flask import Flask, redirect, url_for, request, render_template



#augmentations
AUGMENTATIONS_TEST = Compose([
    Resize(200, 200, always_apply=False, p=1),
    ToFloat(max_value=255)])

checkpoint_path = "/age/content/content/My_model"
def mae_years(in_gt, in_pred):
    return mean_absolute_error(in_gt, in_pred)
model = load_model(checkpoint_path, custom_objects={'mae_years':mae_years})
cnn_face_detector = dlib.get_frontal_face_detector()
       
print('Model loaded')



app = Flask(__name__)

def model_predict(img_path, model):
    Image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
    # apply face detection (cnn)
    faces_cnn = cnn_face_detector(Image, 1)
    print(faces_cnn)
    if len(faces_cnn) >= 1:
      print('face deteted')
      face = faces_cnn[0]
      x1, y1= face.left(), face.top()
      x2, y2 = face.right(), face.bottom()
      Image = Image[y1:y2,x1:x2,:] 
    img_augmented = AUGMENTATIONS_TEST(image=Image)
    img = np.expand_dims(img_augmented['image'], axis=0)
    img.shape
    preds = model.predict(img, verbose = True)
    return preds

@app.route('/', methods=['GET']) 
def hello_word(): 
    return render_template('index.html') 
@app.route('/', methods=['POST']) 
def predict(): 
    imagefile= request.files['imagefile'] 
    image_path = "/age/templates/" + imagefile.filename 
    imagefile.save(image_path)  
    Image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    # apply face detection (cnn)
    faces_cnn = cnn_face_detector(Image, 1)
    print(faces_cnn)
    if len(faces_cnn) >= 1:
      print('face deteted')
      face = faces_cnn[0]
      x1, y1= face.left(), face.top()
      x2, y2 = face.right(), face.bottom()
      Image = Image[y1:y2,x1:x2,:] 
    img_augmented = AUGMENTATIONS_TEST(image=Image)
    img = np.expand_dims(img_augmented['image'], axis=0)
    img.shape
    preds = model.predict(img, verbose = True)
    age = preds[0][0]
    return render_template('index.html',prediction=round(age),user_image =  image_path)
    
if __name__ == '__main__':
    app.run()  # If address is in use, may need to terminate other sessions:
               # Runtime > Manage Sessions > Terminate Other Sessions
