# load_models and facerecog_service libraries
import datetime
import cv2
import os
import requests
from tqdm import tqdm
import sys
from PIL import Image, ImageOps, ImageEnhance

# facerecog_service libraries
import numpy
import shutil

# Training libraries
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras_vggface.vggface import VGGFace
import pickle

# Prediction libraries
from yoloface import face_analysis
from keras.preprocessing import image as kerasImagePreprocess
from keras_vggface import utils as kerasVGGFaceUtils
from keras.models import load_model

from ...core.logging import logger

CWD = os.getcwd()

class Models:
    def __init__(self):
        self.face_encodings = []
        self.known_face_encodings = []
        self.known_face_names = []

    def encodeFaces(self):
        # Update dataset before encoding
        self.updateDataset()

        # Encoding faces (Re-training for face detection algorithm)
        logger.info("[v3] Encoding Faces... (This may take a while)")
        
        # NOTE: UNCOMMENT THIS LINE IF YOU WANT TO USE GPU INSTEAD OF CPU
        # tf.config.list_physical_devices('gpu')

        DATASET_DIRECTORY = f"{CWD}/data/dataset"

        # Preprocess dataset
        trainDatagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        # Setup for dataset training
        trainGenerator = \
            trainDatagen.flow_from_directory(
            DATASET_DIRECTORY,
            target_size=(224,224),
            color_mode='rgb',
            batch_size=32,
            class_mode='categorical',
            shuffle=True)

        # Get list of classes
        trainGenerator.class_indices.values()
        NO_CLASSES = len(trainGenerator.class_indices.values())

        # Initiate training model
        baseModel = VGGFace(include_top=False,
        model='vgg16',
        input_shape=(224, 224, 3))
        # NOTE: IF ERROR, UNCOMMENT. IF NOT ERROR, DELETE.
        # baseModel.summary()

        # Setup first layers
        x = baseModel.output
        x = GlobalAveragePooling2D()(x)

        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)

        # Setup final layer with softmax activation
        preds = Dense(NO_CLASSES, activation='softmax')(x)

        # Create a new model with the base model's original input and the new model's output
        model = Model(inputs = baseModel.input, outputs = preds)
        model.summary()

        # Don't train the first 19 layers - 0..18
        for layer in model.layers[:19]:
            layer.trainable = False

        # Train the rest of the layers - 19 onwards
        for layer in model.layers[19:]:
            layer.trainable = True

        # Compling the model
        model.compile(optimizer='Adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        # MAIN TRAINING
        model.fit(trainGenerator,
        batch_size = 1,
        verbose = 1,
        epochs = 20)

        # Create HDF5 file
        today = datetime.datetime.now().strftime("%Y%m%d")
        model.save(f'{CWD}/ml-models/training-models/{today}-trained.h5')

        classDictionary = trainGenerator.class_indices
        classDictionary = {
            value:key for key, value in classDictionary.items()
        }

        # Save the class dictionary to pickle
        faceLabelFilename = f'{CWD}/ml-models/training-models/face-labels.pickle'
        with open(faceLabelFilename, 'wb') as f:
            pickle.dump(classDictionary, f)
        
        logger.info("[v3] Encoding Done!")

    def recog(self, filename: str, requestFolderCount: int):
        logger.info("[v3] Recognizing faces into user IDs")

        # Set the dimensions of the image
        imageWidth, imageHeight = (224, 224)

        # load the training labels
        faceLabelFilename = f'{CWD}/ml-models/training-models/face-labels.pickle'
        with open(faceLabelFilename, "rb") as \
            f: class_dictionary = pickle.load(f)

        class_list = [value for _, value in class_dictionary.items()]

        # Detecting faces
        facecascade = cv2.CascadeClassifier(f'{CWD}/ml-models/haarcascade/haarcascade_frontalface_default.xml')

        # Load the image
        imgtest = cv2.imread(filename, cv2.IMREAD_COLOR)
        image_array = numpy.array(imgtest, "uint8")

        # Get the faces detected in the image
        faces = facecascade.detectMultiScale(imgtest, 
            scaleFactor=1.1, minNeighbors=5)

        # Load model
        today = datetime.datetime.now().strftime("%Y%m%d")
        trainedFilename = f'{CWD}/ml-models/training-models/{today}-trained.h5'
        if not os.path.exists(trainedFilename):
            logger.warning("[v3] PROGRAM IS ENCODING WHEN SOMEONE IS SENDING REQUEST.")
            self.encodeFaces()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        model = load_model(trainedFilename)

        facesDetected = []
        frames = []

        count = 1
        for (face_x, face_y, face_w, face_h) in faces:
            # Resize the detected face to 224 x 224
            size = (imageWidth, imageHeight)
            roi = image_array[face_y: face_y + face_h, face_x: face_x + face_w]
            resized_image = cv2.resize(roi, size)

            frame = f"{CWD}/data/output/{today}/{requestFolderCount}/frame"
            if not os.path.exists(frame):
                os.mkdir(frame)
                
            frame += f"/frame{count.zfill(3)}.jpg"
            
            cv2.imwrite(frame)

            frames.append(frame.split("output/")[1])

            # Preparing the image for prediction
            x = kerasImagePreprocess.img_to_array(resized_image)
            x = numpy.expand_dims(x, axis=0)
            x = kerasVGGFaceUtils.preprocess_input(x, version=1)

            # Predicting
            predicted_prob = model.predict(x)
            facesDetected.append(class_list[predicted_prob[0].argmax()])

        return facesDetected, frames

    def updateDataset(self):
        logger.info("[v3] Updating datasets... (This may took a while)")

        APITOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InJlemFhckBrYXplZS5pZCIsImlhdCI6MTY3NTgyMTY2Mn0.eprZiRQUjiWjbfZYlbziT6sXG-34f2CnQCSy3yhAh6I"
        r = requests.get("http://103.150.87.245:3001/api/profile/list-photo", headers={'Authorization': 'Bearer ' + APITOKEN})

        datas = r.json()["data"]

        for data in tqdm(datas, file=sys.stdout):
            userID = data["user_id"]
            url = data["photo"]

            r = requests.get(url)

            foldername = f'{CWD}/data/dataset/{userID}'

            if not os.path.exists(foldername):
                os.mkdir(foldername)

            filename = f"{foldername}/{userID}.jpg"
            
            # Save grabbed image to {CWD}/data/faces/
            with open(filename, 'wb') as f:
                f.write(r.content)
            
            self.imgAugmentation(filename)

        logger.info("[v3] Datasets updated!")

    def imgAugmentation(img):
        try:
            face = face_analysis()
            frame = cv2.imread(img)
            _,boxes,conv = face.face_detection(frame_arr=frame,frame_status=True,model='tiny')
            if len(boxes) > 1:
                print("More than 1 face detected. Only choosing the first face that got detected")
            if len(boxes) != 0:
                box = boxes[0]
                x,y,w,h = box[0],box[1],box[2],box[3]
                cropped_face = frame[y:y + w, x:x + h]
                cv2.imwrite(img, cropped_face)
        except Exception as e:
            logger.error(f"[v3] ERROR - {str(e)}. Filename: {img}")

        # Read image
        input_img = Image.open(img)
        input_img = input_img.convert('RGB')
        # Flip Image
        img_flip = ImageOps.flip(input_img)
        img_flip.save(f"{img.split('.jpeg')[0]}-flipped.jpeg")
        # Mirror Image 
        img_mirror = ImageOps.mirror(input_img)
        img_mirror.save(f"{img.split('.jpeg')[0]}-mirrored.jpeg")
        # Rotate Image
        img_rot1 = input_img.rotate(30)
        img_rot1.save(f"{img.split('.jpeg')[0]}-rotated1.jpeg")
        img_rot2 = input_img.rotate(330)
        img_rot2.save(f"{img.split('.jpeg')[0]}-rotated2.jpeg")
        # Adjust Brightness
        enhancer = ImageEnhance.Brightness(input_img)
        im_darker = enhancer.enhance(0.5)
        im_darker.save(f"{img.split('.jpeg')[0]}-darker1.jpeg")
        im_darker2 = enhancer.enhance(0.7)
        im_darker2.save(f"{img.split('.jpeg')[0]}-darker2.jpeg")
        enhancer = ImageEnhance.Brightness(input_img)
        im_darker = enhancer.enhance(1.2)
        im_darker.save(f"{img.split('.jpeg')[0]}-brighter1.jpeg")
        im_darker2 = enhancer.enhance(1.5)
        im_darker2.save(f"{img.split('.jpeg')[0]}-brighter2.jpeg")