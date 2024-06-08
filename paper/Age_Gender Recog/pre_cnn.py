import cv2
import numpy as np
import gdown
import tensorflow as tf
from deepface.basemodels import VGGFace
from deepface.commons import functions
import tarfile
import os
import csv

# Paths to pre-trained models for age and gender
age_model_path = 'E:/capJC/paper/Age_Gender Recog/age_net.caffemodel'
age_proto_path = 'E:/capJC/paper/Age_Gender Recog/age_deploy.prototxt'
gender_model_path = 'E:/capJC/paper/Age_Gender Recog/gender_net.caffemodel'
gender_proto_path = 'E:/capJC/paper/Age_Gender Recog/gender_deploy.prototxt'

# Paths to pre-trained model for face detection
face_model_path = 'E:/capJC/paper/Age_Gender Recog/opencv_face_detector_uint8.pb'
face_proto_path = 'E:/capJC/paper/Age_Gender Recog/opencv_face_detector.pbtxt'

# Age and Gender model mean values and classes
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Load the models
age_net = cv2.dnn.readNet(age_model_path, age_proto_path)
gender_net = cv2.dnn.readNet(gender_model_path, gender_proto_path)
face_net = cv2.dnn.readNet(face_model_path, face_proto_path)



# --------------------------
# pylint: disable=line-too-long
# --------------------------
# dependency configurations
tf_version = int(tf.__version__.split(".", maxsplit=1)[0])

if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import Convolution2D, Flatten, Activation
elif tf_version == 2:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Convolution2D, Flatten, Activation
# --------------------------
# Labels for the ethnic phenotypes that can be detected by the model.
labels = ["asian", "indian", "black", "white", "middle eastern", "latino hispanic"]


def loadModel(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/race_model_single_batch.h5",
):

    model = VGGFace.baseModel()

    # --------------------------

    classes = 6
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name="predictions")(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation("softmax")(base_model_output)

    # --------------------------

    race_model = Model(inputs=model.input, outputs=base_model_output)

    # --------------------------

    # load weights
    home = functions.get_deepface_home()

    if os.path.isfile(home + "/.deepface/weights/race_model_single_batch.h5") != True:
        print("race_model_single_batch.h5 will be downloaded...")

        output = home + "/.deepface/weights/race_model_single_batch.h5"
        gdown.download(url, output, quiet=False)

    race_model.load_weights(home + "/.deepface/weights/race_model_single_batch.h5")

    return race_model


# Function to predict age and gender
def predict_age_gender(face_img):
    if face_img.size == 0:
        return None, None  # Returning None if the face image is empty
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    # Predict gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    # Predict age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]
    return age, gender

# Function to detect faces
def get_faces(frame, conf_threshold=0.7):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    face_net.setInput(blob)
    detections = face_net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
    return face_boxes

# Extract the tar.gz file
with tarfile.open('C:\\Users\\hvbvm\\Downloads\\UTKface_inthewild-20231127T083138Z-001\\UTKface_inthewild\\part2.tar.gz', "r:gz") as tar:
    tar.extractall("extracted_images")

# Process each image, write to CSV and track files with no faces detected
results_csv = 'age_gender_predictions.csv'
no_faces_csv = 'no_faces_detected.csv'
with open(results_csv, mode='w', newline='') as results_file, open(no_faces_csv, mode='w', newline='') as no_faces_file:
    results_writer = csv.writer(results_file)
    no_faces_writer = csv.writer(no_faces_file)
    results_writer.writerow(['File', 'Face_Position', 'Age', 'Gender'])  # Writing the header
    no_faces_writer.writerow(['File'])  # Writing the header for no faces detected

    for root, dirs, files in os.walk('extracted_images'):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path)

                # Detect faces
                faces = get_faces(image)
                if not faces:
                    no_faces_writer.writerow([filename])  # Write filename to no_faces_detected.csv
                for (x1, y1, x2, y2) in faces:
                    face = image[y1:y2, x1:x2]

                    # Check if the face image is empty
                    if face.size != 0:
                        # Predict age and gender for each face
                        age, gender = predict_age_gender(face)
                        results_writer.writerow([filename, f"({x1}, {y1}, {x2}, {y2})", age, gender])  # Writing the data









import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input
from sklearn.model_selection import train_test_split

path = "E:/capJC/paper/part1"
images = []
age = []
gender = []
for img in os.listdir(path):
  ages = img.split("_")[0]
  genders = img.split("_")[1]
  img = cv2.imread(str(path)+"/"+str(img))
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  images.append(np.array(img))
  age.append(np.array(ages))
  gender.append(np.array(genders))
  
age = np.array(age,dtype=np.int64)
images = np.array(images)   #Forgot to scale image for my training. Please divide by 255 to scale. 
gender = np.array(gender,np.uint64)

x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(images, age, random_state=42)

x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(images, gender, random_state=42)

##################################################
#Define age model and train. 
##################################

age_model = Sequential()
age_model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(200,200,3)))
#age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))

age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
#age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))
              
age_model.add(Conv2D(256, kernel_size=3, activation='relu'))
#age_model.add(Conv2D(256, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))

age_model.add(Conv2D(512, kernel_size=3, activation='relu'))
#age_model.add(Conv2D(512, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))

age_model.add(Flatten())
age_model.add(Dropout(0.2))
age_model.add(Dense(512, activation='relu'))

age_model.add(Dense(1, activation='linear', name='age'))
              
age_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print(age_model.summary())              
                           
history_age = age_model.fit(x_train_age, y_train_age,
                        validation_data=(x_test_age, y_test_age), epochs=50)

age_model.save('age_model_50epochs.h5')

################################################################
#Define gender model and train
##################################################
gender_model = Sequential()

gender_model.add(Conv2D(36, kernel_size=3, activation='relu', input_shape=(200,200,3)))

gender_model.add(MaxPool2D(pool_size=3, strides=2))
gender_model.add(Conv2D(64, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))

gender_model.add(Conv2D(128, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))

gender_model.add(Conv2D(256, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))

gender_model.add(Conv2D(512, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))

gender_model.add(Flatten())
gender_model.add(Dropout(0.2))
gender_model.add(Dense(512, activation='relu'))
gender_model.add(Dense(1, activation='sigmoid', name='gender'))

gender_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_gender = gender_model.fit(x_train_gender, y_train_gender,
                        validation_data=(x_test_gender, y_test_gender), epochs=50)

gender_model.save('gender_model_50epochs.h5')


############################################################

history = history_age

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
#acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

####################################################################
from keras.models import load_model
#Test the model
my_model = load_model('gender_model_50epochs.h5', compile=False)


predictions = my_model.predict(x_test_gender)
y_pred = (predictions>= 0.5).astype(int)[:,0]

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_test_gender, y_pred))

#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm=confusion_matrix(y_test_gender, y_pred)  
sns.heatmap(cm, annot=True)




