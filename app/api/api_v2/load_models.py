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
        today = datetime.datetime.now()
        if os.path.exists(f"{CWD}/ml-models/training-models/{today.strftime('%Y%m%d')}-trained.h5"):
            logger.info("ENCODING AND UPDATE SKIPPED. MODEL EXISTS.")
            return None
        else:
            try:
                today = today - datetime.timedelta(days=1)
                os.remove(f"{CWD}/ml-models/training-models/{today.strftime('%Y%m%d')}-trained.h5")
            except FileNotFoundError:
                logger.info("First time training, creating new initial train file.")

        # Update dataset before encoding
        self.updateDataset()

        # Encoding faces (Re-training for face detection algorithm)
        logger.info("Encoding Faces... (This may take a while)")
        
        # NOTE: UNCOMMENT THIS LINE IF YOU WANT TO USE GPU INSTEAD OF CPU
        # tf.config.list_physical_devices('gpu')

        DATASET_DIRECTORY = f"{CWD}/data/api_v2/dataset"

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
        
        logger.info("Encoding Done!")

    def recog(self, filename: str, requestFolderCount: int):
        logger.info("Recognizing faces into user IDs")

        # Set the dimensions of the image
        imageWidth, imageHeight = (224, 224)

        # load the training labels
        faceLabelFilename = f'{CWD}/ml-models/training-models/face-labels.pickle'
        with open(faceLabelFilename, "rb") as \
            f: class_dictionary = pickle.load(f)

        class_list = [value for _, value in class_dictionary.items()]

        # Detecting faces
        detector = cv2.FaceDetectorYN.create(f"{CWD}/ml-models/face_detection_yunet/face_detection_yunet_2022mar.onnx", "", (224, 224))

        # Load the image
        imgtest = cv2.imread(filename, cv2.IMREAD_COLOR)
        image_array = numpy.array(imgtest, "uint8")

        # Get the faces detected in the image
        height, width, channels = imgtest.shape
        detector.setInputSize((width, height))
        channel, faces = detector.detect(imgtest)
        faces = faces if faces is not None else []
        boxes = []

        # Load model
        today = datetime.datetime.now().strftime("%Y%m%d")
        trainedFilename = f'{CWD}/ml-models/training-models/{today}-trained.h5'
        if not os.path.exists(trainedFilename):
            logger.warning("PROGRAM IS ENCODING WHEN SOMEONE IS SENDING REQUEST.")
            self.encodeFaces()
        
        model = load_model(trainedFilename)

        facesDetected = []
        frames = []

        count = 1
        for face in faces:
            box = list(map(int, face[:4]))
            boxes.append(box)
            face_x = box[0]
            face_y = box[1]
            face_w = box[2]
            face_h = box[3]
            # Resize the detected face to 224 x 224
            size = (imageWidth, imageHeight)
            roi = image_array[face_y: face_y + face_w, face_x: face_x + face_h]
            resized_image = cv2.resize(roi, size)

            frame = f"{CWD}/data/api_v2/output/{today}/{requestFolderCount}/frame"
            if not os.path.exists(frame):
                os.mkdir(frame)
                
            frame += f"/frame{str(count).zfill(3)}.jpg"
            
            cv2.imwrite(frame, resized_image)

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
        logger.info("Updating datasets... (This may took a while)")

        APITOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InJlemFhckBrYXplZS5pZCIsImlhdCI6MTY3NTgyMTY2Mn0.eprZiRQUjiWjbfZYlbziT6sXG-34f2CnQCSy3yhAh6I"
        r = requests.get("http://103.150.87.245:3001/api/profile/list-photo", headers={'Authorization': 'Bearer ' + APITOKEN})

        datas = r.json()["data"]

        for data in tqdm(datas, file=sys.stdout):
            userID = data["user_id"]
            url = data["photo"]

            r = requests.get(url)

            foldername = f'{CWD}/data/api_v2/dataset/{userID}'

            if not os.path.exists(foldername):
                os.mkdir(foldername)

            filename = f"{foldername}/{userID}.jpg"
            
            # Save grabbed image to {CWD}/data/api_v2/faces/
            with open(filename, 'wb') as f:
                f.write(r.content)
            
            self.imgAugmentation(filename)

        logger.info("Datasets updated!")

    def imgAugmentation(self, img):
        try:
            face = face_analysis()
            frame = Image.open(img)
            frame = frame.convert("RGB")
            frame = numpy.array(frame)
            detector = cv2.FaceDetectorYN.create(f"{CWD}/ml-models/face_detection_yunet/face_detection_yunet_2022mar.onnx", "", (224, 224))
            height, width, channels = frame.shape
            detector.setInputSize((width, height))
            channel, faces = detector.detect(frame)
            faces = faces if faces is not None else []
            boxes = []
            for face in faces:
                box = list(map(int, face[:4]))
                boxes.append(box)
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                faceCropped = frame[y:y + h, x:x + w]
            if len(boxes) > 1:
                print("More than 1 face detected. Only choosing the first face that got detected")
            if len(boxes) != 0:
                cv2.imwrite(img, cv2.cvtColor(faceCropped, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.error(f"ERROR - {str(e)}. Filename: {img}")

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