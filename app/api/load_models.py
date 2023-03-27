# for load machine learning models
import datetime
import cv2
import face_recognition
import os
import shutil
import numpy
import math
import requests
from tqdm import tqdm
import sys
from PIL import Image, ImageOps, ImageEnhance
import json

# Fixing vggface import error
filename = "/usr/local/lib/python3.7/site-packages/keras_vggface/models.py"
text = open(filename).read()
open(filename, "w+").write(text.replace('keras.engine.topology', 'keras.utils.layer_utils'))

from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras_vggface.vggface import VGGFace
import pickle

from yoloface import face_analysis
from keras.preprocessing import image as kerasImagePreprocess
from keras_vggface import utils as kerasVGGFaceUtils
from keras.models import load_model

from ..core.logging import logger

# testing pr

CWD = os.getcwd()

class Models:
    def __init__(self):
        self.face_encodings = []
        self.known_face_encodings = []
        self.known_face_names = []

    def v1Encode(self):
        ''' v1 API encode using YuNet + face-recognition library '''
        logger.info("[v1] Encoding is starting... (This might take a while)")

        today = datetime.datetime.now()
        if os.path.exists(f"{CWD}/ml-models/training-models/{today.strftime('%Y%m%d')}-v1trained.json"):
            logger.info("V1 ENCODING AND UPDATE SKIPPED. MODEL EXISTS.")
            with open(f"{CWD}/ml-models/training-models/{today.strftime('%Y%m%d')}-v1trained.json", "r") as f:
                trainedv1 = json.loads(f.read().replace("'", '"'))
            self.known_face_encodings = [numpy.array(i) for i in trainedv1['known_face_encodings']]
            self.known_face_names = trainedv1['known_face_names']
            return None
        else:
            try:
                today = today - datetime.timedelta(days=1)
                os.remove(f"{CWD}/ml-models/training-models/{today.strftime('%Y%m%d')}-v1trained.json")
            except FileNotFoundError:
                logger.info("First time v1 training, creating new initial train file.")

        # Update dataset before encoding
        self.imgCropV1()

        pbar = tqdm(total=len(os.listdir(f'{CWD}/data/api_v1/dataset')), bar_format="v1Encode Progress: {percentage:3.0f}% | ETA: {remaining}", file=sys.stdout)

        # Encoding faces (Re-training for face detection algorithm)
        for i, image in enumerate(os.listdir(f'{CWD}/data/api_v1/dataset')):
            face_image = face_recognition.load_image_file(f'{CWD}/data/api_v1/dataset/{image}')
            try:
                face_encoding = face_recognition.face_encodings(face_image)[0]
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(image)
            except IndexError:
                pass

            try:
                if i % (len(os.listdir(f'{CWD}/data/api_v1/dataset'))//20) == 0:
                    pbar.update(len(os.listdir(f'{CWD}/data/api_v1/dataset'))//20)
                    sys.stdout.flush()
            except:
                pass

        today = datetime.datetime.now()

        face_encodings = [i.tolist() for i in self.face_encodings]
        known_face_encodings = [i.tolist() for i in self.known_face_encodings]
        known_face_names = self.known_face_names

        output = {'face_encodings': face_encodings, 'known_face_encodings': known_face_encodings, 'known_face_names': known_face_names}
        with open(f"{CWD}/ml-models/training-models/{today.strftime('%Y%m%d')}-v1trained.json", "w") as f:
            f.write(f"{output}")

        logger.info("[v1] Encoding Done!")
    
    def v2Encode(self):
        ''' v2 API encode using YuNet + VGGFace '''
        logger.info("[v2] Encoding is starting... (This might take a while)")

        today = datetime.datetime.now()
        if os.path.exists(f"{CWD}/ml-models/training-models/{today.strftime('%Y%m%d')}-v2trained.h5"):
            logger.info("V2 ENCODING AND UPDATE SKIPPED. MODEL EXISTS.")
            return None
        else:
            try:
                today = today - datetime.timedelta(days=1)
                os.remove(f"{CWD}/ml-models/training-models/{today.strftime('%Y%m%d')}-v2trained.h5")
            except FileNotFoundError:
                logger.info("First time v2 training, creating new initial train file.")

        # Update dataset before encoding
        self.imgCropV2()
        
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
        model.save(f'{CWD}/ml-models/training-models/{today}-v2trained.h5')

        classDictionary = trainGenerator.class_indices
        classDictionary = {
            value:key for key, value in classDictionary.items()
        }

        # Save the class dictionary to pickle
        faceLabelFilename = f'{CWD}/ml-models/training-models/face-labels.pickle'
        with open(faceLabelFilename, 'wb') as f:
            pickle.dump(classDictionary, f)
        
        logger.info("Encoding Done!")
        
        logger.info("[v2] Encoding Done!")
    
    def v3Encode(self):
        ''' v3 API encode using YoloFace + VGGFace '''
        logger.info("[v3] Encoding is starting... (This might take a while)")

        today = datetime.datetime.now()
        if os.path.exists(f"{CWD}/ml-models/training-models/{today.strftime('%Y%m%d')}-v3trained.h5"):
            logger.info("V3 ENCODING AND UPDATE SKIPPED. MODEL EXISTS.")
            return None
        else:
            try:
                today = today - datetime.timedelta(days=1)
                os.remove(f"{CWD}/ml-models/training-models/{today.strftime('%Y%m%d')}-v3trained.h5")
            except FileNotFoundError:
                logger.info("First time v3 training, creating new initial train file.")

        # Update dataset before encoding
        self.imgCropV3()

        DATASET_DIRECTORY = f"{CWD}/data/api_v3/dataset"

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
        model.save(f'{CWD}/ml-models/training-models/{today}-v3trained.h5')

        classDictionary = trainGenerator.class_indices
        classDictionary = {
            value:key for key, value in classDictionary.items()
        }

        # Save the class dictionary to pickle
        faceLabelFilename = f'{CWD}/ml-models/training-models/face-labels.pickle'
        with open(faceLabelFilename, 'wb') as f:
            pickle.dump(classDictionary, f)
        
        logger.info("Encoding Done!")
        
        logger.info("[v3] Encoding Done!")

    def grabRawDatasets(self):
        ''' Saving dataset that is served from external API and saving it onto local storage '''        
        logger.info("Raw datasets is getting grabbed...")

        APIURL = "http://103.150.87.245:3001/api/profile/list-photo"
        APITOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InJlemFhckBrYXplZS5pZCIsImlhdCI6MTY3NTgyMTY2Mn0.eprZiRQUjiWjbfZYlbziT6sXG-34f2CnQCSy3yhAh6I"
        
        r = requests.get(APIURL, headers={'Authorization': 'Bearer ' + APITOKEN})

        datas = r.json()["data"]

        pbar = tqdm(total=len(datas), bar_format="grabRawDatasets Progress: {percentage:3.0f}% | ETA: {remaining}", file=sys.stdout)

        for i, data in enumerate(datas):
            userID = data["user_id"]
            url = data["photo"]

            r = requests.get(url)

            filename = f'{CWD}/data/raw_dataset/{userID}.jpg'
            
            # Save grabbed image to {CWD}/data/raw_dataset/
            with open(filename, 'wb') as f:
                f.write(r.content)

            try:
                if i % (len(data)//20) == 0:
                    pbar.update(len(data)//20)
                    sys.stdout.flush()
            except:
                pass

        logger.info("All raw datasets grabbed!")

    def imgCropV1(self):
        ''' Image augmentation using YuNet with FaceRecogLib format '''
        rawDatasetPath = f"{CWD}/data/raw_dataset"
        rawDatasetImages = os.listdir(rawDatasetPath)
        # rawDatasetImages.remove(".gitkeep")

        pbar = tqdm(total=len(rawDatasetImages), bar_format="imgAugmentationV1 Progress: {percentage:3.0f}% | ETA: {remaining}", file=sys.stdout)

        for i, rawDatasetImage in enumerate(rawDatasetImages):
            rawImg = rawDatasetPath + "/" + rawDatasetImage
            img = f"{CWD}/data/api_v1/dataset/{rawDatasetImage}"

            try:
                # Detect face
                frame = Image.open(rawImg)
                frame = frame.convert("RGB")
                cv2_input = numpy.array(frame)
                detector = cv2.FaceDetectorYN.create(f"{CWD}/ml-models/face_detection_yunet/face_detection_yunet_2022mar.onnx", "", (320, 320))
                height, width, channels = cv2_input.shape
                detector.setInputSize((width, height))

                # Get face arrays
                channel, faces = detector.detect(cv2_input)
                faces = faces if faces is not None else []
                boxes = []
                for face in faces:
                    box = list(map(int, face[:4]))
                    boxes.append(box)

                faces = boxes

                if len(faces) > 1:
                    print(f"Found more than 1 face in dataset image. Selecting only the first one. Image path: {rawImg}")
                    faces = [faces[0]]
                if len(faces) == 0:
                    print(f"No face found in dataset image. Image path: {rawImg}", flush=True)
                    cv2.imwrite(img, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    pass

                # Get face coordinates
                x = faces[0][0]
                y = faces[0][1]
                w = faces[0][2]
                h = faces[0][3]

                faceCropped = cv2_input[y:y + h, x:x + w]
                    
                cv2.imwrite(img, cv2.cvtColor(faceCropped, cv2.COLOR_BGR2RGB))
                
                self.imgAugment(img)
            except Exception as e:
                pass

            try:
                if i % (len(rawDatasetImages)//20) == 0:
                    pbar.update(len(rawDatasetImages)//20)
                    sys.stdout.flush()
            except:
                pass

    def imgCropV2(self):
        ''' Image augmentation using YuNet with VGGFace format '''
        rawDatasetPath = f"{CWD}/data/raw_dataset"
        rawDatasetImages = os.listdir(rawDatasetPath)
        # rawDatasetImages.remove(".gitkeep")

        pbar = tqdm(total=len(rawDatasetImages), bar_format="imgAugmentationYunet Progress: {percentage:3.0f}% | ETA: {remaining}", file=sys.stdout)

        for i, rawDatasetImage in enumerate(rawDatasetImages):
            rawImg = rawDatasetPath + "/" + rawDatasetImage
            img = f"{CWD}/data/api_v2/dataset/{rawDatasetImage.split('.')[0]}"

            if not os.path.exists(img):
                os.mkdir(img)

            img += f"/{rawDatasetImage}"

            try:
                # Detect face
                frame = Image.open(rawImg)
                frame = frame.convert("RGB")
                cv2_input = numpy.array(frame)
                detector = cv2.FaceDetectorYN.create(f"{CWD}/ml-models/face_detection_yunet/face_detection_yunet_2022mar.onnx", "", (320, 320))
                height, width, channels = cv2_input.shape
                detector.setInputSize((width, height))

                # Get face arrays
                channel, faces = detector.detect(cv2_input)
                faces = faces if faces is not None else []
                boxes = []
                for face in faces:
                    box = list(map(int, face[:4]))
                    boxes.append(box)

                faces = boxes

                if len(faces) > 1:
                    print(f"Found more than 1 face in dataset image. Selecting only the first one. Image path: {rawImg}")
                    faces = [faces[0]]
                if len(faces) == 0:
                    print(f"No face found in dataset image. Image path: {rawImg}", flush=True)
                    cv2.imwrite(img, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    pass

                # Get face coordinates
                x = faces[0][0]
                y = faces[0][1]
                w = faces[0][2]
                h = faces[0][3]

                faceCropped = cv2_input[y:y + h, x:x + w]
                    
                cv2.imwrite(img, cv2.cvtColor(faceCropped, cv2.COLOR_BGR2RGB))
            except Exception as e:
                logger.error(f"Something happened - {str(e)}. Image saved without cropping. Filename: {img}")
                frame.save(img)
            
            self.imgAugment(img)
            
            try:
                if i % (len(rawDatasetImages)//20) == 0:
                    pbar.update(len(rawDatasetImages)//20)
                    sys.stdout.flush()
            except:
                pass

    def imgCropV3(self):
        ''' Image augmentation using YoloFace with VGGFace format '''
        rawDatasetPath = f"{CWD}/data/raw_dataset"
        rawDatasetImages = os.listdir(rawDatasetPath)
        # rawDatasetImages.remove(".gitkeep")

        pbar = tqdm(total=len(rawDatasetImages), bar_format="imgAugmentationV3 Progress: {percentage:3.0f}% | ETA: {remaining}", file=sys.stdout)

        for i, rawDatasetImage in enumerate(rawDatasetImages):
            rawImg = rawDatasetPath + "/" + rawDatasetImage
            img = f"{CWD}/data/api_v3/dataset/{rawDatasetImage.split('.')[0]}"

            if not os.path.exists(img):
                os.mkdir(img)

            img += f"/{rawDatasetImage}"

            try:
                face = face_analysis()
                frame = Image.open(rawImg)
                frame = frame.convert("RGB")
                cv2_input = numpy.array(frame)
                _,boxes,conv = face.face_detection(frame_arr=cv2_input,frame_status=True,model='tiny')
                if len(boxes) > 1:
                    print(f"Found more than 1 face in dataset image. Selecting only the first one. Image path: {rawImg}", flush=True)
                if len(boxes) == 0:
                    print(f"No face found in dataset image. Image path: {rawImg}", flush=True)
                    cv2.imwrite(img, cv2.cvtColor(cv2_input, cv2.COLOR_BGR2RGB))
                    pass
                
                box = boxes[0]
                x,y,w,h = box[0],box[1],box[2],box[3]
                cropped_face = cv2_input[y:y + w, x:x + h]
                cv2.imwrite(img, cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))

            except Exception as e:
                logger.error(f"Something happened - {str(e)}. Image saved without cropping. Filename: {img}")
                frame.save(img)

            self.imgAugment(img)

            try:
                if i % (len(rawDatasetImages)//20) == 0:
                    pbar.update(len(rawDatasetImages)//20)
                    sys.stdout.flush()
            except:
                pass

    def imgAugment(self, filename: str):
        input_img = Image.open(filename);
        input_img = input_img.convert('RGB')

        # Flip Image
        img_flip = ImageOps.flip(input_img)
        img_flip.save(f"{filename.split('.jpg')[0]}-flipped.jpg")

        # Mirror Image 
        img_mirror = ImageOps.mirror(input_img)
        img_mirror.save(f"{filename.split('.jpg')[0]}-mirrored.jpg")

        # Rotate Image
        img_rot1 = input_img.rotate(30)
        img_rot1.save(f"{filename.split('.jpg')[0]}-rotated1.jpg")
        img_rot2 = input_img.rotate(330)
        img_rot2.save(f"{filename.split('.jpg')[0]}-rotated2.jpg")

        # Adjust Brightness
        enhancer = ImageEnhance.Brightness(input_img)
        im_darker = enhancer.enhance(0.5)
        im_darker.save(f"{filename.split('.jpg')[0]}-darker1.jpg")
        im_darker2 = enhancer.enhance(0.7)
        im_darker2.save(f"{filename.split('.jpg')[0]}-darker2.jpg")
        enhancer = ImageEnhance.Brightness(input_img)
        im_darker = enhancer.enhance(1.2)
        im_darker.save(f"{filename.split('.jpg')[0]}-brighter1.jpg")
        im_darker2 = enhancer.enhance(1.5)
        im_darker2.save(f"{filename.split('.jpg')[0]}-brighter2.jpg")

    def inputRecog(self, model: str, filename: str, filenameDatas: dict):
        modelsAccepted = ["v1", "v2", "v3"]
        timeNow = filenameDatas["timeNow"]
        id = filenameDatas["id"]
        
        if model not in modelsAccepted:
            logger.error(f"Model inputRecog not valid. Parameter passed: {model}. Parameters accepted: {modelsAccepted}")
            return None
        if model == "v1":
            ### V1 - YuNet Detection ###
            logger.info("[v1] Grabbing faces detected from input image")

            detector = cv2.FaceDetectorYN.create(f"{CWD}/ml-models/face_detection_yunet/face_detection_yunet_2022mar.onnx", "", (320, 320))

            frame = cv2.imread(filename)

            height, width, channels = frame.shape

            # Set input size
            detector.setInputSize((width, height))
            # Getting detections
            channel, faces = detector.detect(frame)
            faces = faces if faces is not None else []

            boxes = []
            detectConfidences = []
            filenames = []
            count = 1
            
            for face in faces:
                box = list(map(int, face[:4]))
                boxes.append(box)
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                faceCropped = frame[y:y + h, x:x + w]

                ### SEMENTARA MASIH TANPA FILTERING MINIMUM PIXEL SHAPE 50
                # if w >= 50 and h >= 50 and x >= 0 and y >= 0:
                filename = f"{CWD}/data/api_v1/output/{timeNow}/{id}/frame/frame{str(count).zfill(3)}.jpg"
                if not os.path.exists(f"{CWD}/data/api_v1/output/{timeNow}/{id}/frame/"):
                    os.mkdir(f"{CWD}/data/api_v1/output/{timeNow}/{id}/frame/")
                filenames.append(filename.split("output/")[1])
                cv2.imwrite(filename, faceCropped)
                cv2.imwrite(filename, self.resize(filename, 360))
                count += 1
                    
                confidence = face[-1]
                confidence = "{:.2f}%".format(confidence*100)

                detectConfidences.append(confidence)

            ### V1 - Face Recognition Library Recognition ###
            facesDetected = []

            logger.info(f"[v1] Recognizing faces into user IDs. Filename: {filename}")

            for frameFilename in filenames:
                currentFrame = self.resize(f"{CWD}/data/api_v1/output/{frameFilename}", 480)
                currentFrame = self.convertBGRtoRGB(currentFrame)

                faceNames = list(self.getFaceNames(currentFrame))

                tmpFaceNames = []
                for i in faceNames:
                    IDdetected = i.split("-")[0]
                    if IDdetected == "Unknown (0%)":
                        IDdetected = "Unknown"
                        confidence = 0
                    else:
                        confidence = i.split("jpg (")[1].split("%")[0]
                    # Threshold confidence of 85% for the API to return
                    if float(confidence) > 85 or IDdetected == "Unknown":
                        tmpFaceNames.append([IDdetected, f"{confidence}%"])
                faceNames = tmpFaceNames

                facesDetected.append({"frame_path": frameFilename, "face_detected": faceNames[0][0].split(".jpg")[0], "confidence": faceNames[0][1]})

            result = {"face_count": len(filenames), "result": facesDetected}

            return result
        elif model == "v2":
            ### V2 - YuNet Detection ###
            logger.info("[v2] Grabbing faces detected from input image")

            timeNow = filenameDatas["timeNow"]
            id = filenameDatas["id"]
            detector = cv2.FaceDetectorYN.create(f"{CWD}/ml-models/face_detection_yunet/face_detection_yunet_2022mar.onnx", "", (320, 320))

            frame = cv2.imread(filename)

            height, width, channels = frame.shape

            # Set input size
            detector.setInputSize((width, height))
            # Getting detections
            channel, faces = detector.detect(frame)
            faces = faces if faces is not None else []

            boxes = []
            filenames = []
            count = 1
            
            for face in faces:
                box = list(map(int, face[:4]))
                boxes.append(box)
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                faceCropped = frame[y:y + h, x:x + w]

                ### SEMENTARA MASIH TANPA FILTERING MINIMUM PIXEL SHAPE 50
                # if w >= 50 and h >= 50 and x >= 0 and y >= 0:
                filename = f"{CWD}/data/api_v2/output/{timeNow}/{id}/frame/frame{str(count).zfill(3)}.jpg"
                if not os.path.exists(f"{CWD}/data/api_v2/output/{timeNow}/{id}/frame/"):
                    os.mkdir(f"{CWD}/data/api_v2/output/{timeNow}/{id}/frame/")
                filenames.append(filename.split("output/")[1])
                cv2.imwrite(filename, faceCropped)
                cv2.imwrite(filename, self.resize(filename, 360))
                count += 1

            ### V2 - Load variables for recognition ###
            # Load the training labels
            faceLabelFilename = f'{CWD}/ml-models/training-models/face-labels.pickle'
            with open(faceLabelFilename, "rb") as \
                f: class_dictionary = pickle.load(f)

            class_list = [value for _, value in class_dictionary.items()]

            # Load model
            today = datetime.datetime.now().strftime("%Y%m%d")
            trainedFilename = f'{CWD}/ml-models/training-models/{today}-v2trained.h5'
            if not os.path.exists(trainedFilename):
                logger.warning("PROGRAM IS ENCODING WHEN SOMEONE IS SENDING REQUEST.")
                self.encodeFaces()
            
            v2model = load_model(trainedFilename)

            ### V2 - VGGFace Recognition ###
            facesDetected = {}

            count = 1
            for frameFilename in filenames:
                currentFrame = cv2.imread(f"{CWD}/data/api_v2/output/{frameFilename}")
                resized_image = cv2.resize(currentFrame, (224, 224))

                frame = f"{CWD}/data/api_v2/output/{timeNow}/{id}/frame"
                if not os.path.exists(frame):
                    os.mkdir(frame)
                    
                frame += f"/frame{str(count).zfill(3)}.jpg"
                
                cv2.imwrite(frame, resized_image)

                # Preparing the image for prediction
                x = kerasImagePreprocess.img_to_array(resized_image)
                x = numpy.expand_dims(x, axis=0)
                x = kerasVGGFaceUtils.preprocess_input(x, version=1)

                # Predicting
                predicted_prob = v2model.predict(x)
                faceDetected = class_list[predicted_prob[0].argmax()]

                facesDetected.update({frameFilename: faceDetected})
                count += 1

            result = {"faceCount":len(filenames), "result": facesDetected}

            return result
        elif model == "v3":
            ### V3 - YoloFace Detection ###
            logger.info("Recognizing faces into user IDs")

            # Set the dimensions of the image
            imageWidth, imageHeight = (224, 224)

            # load the training labels
            faceLabelFilename = f'{CWD}/ml-models/training-models/face-labels.pickle'
            with open(faceLabelFilename, "rb") as \
                f: class_dictionary = pickle.load(f)

            class_list = [value for _, value in class_dictionary.items()]

            # Detecting faces
            face = face_analysis()

            # Load the image
            imgtest = cv2.imread(filename, cv2.IMREAD_COLOR)
            image_array = numpy.array(imgtest, "uint8")

            # Get the faces detected in the image
            _,box,conf=face.face_detection(frame_arr=imgtest,frame_status=True,model='tiny')

            # Load model
            today = datetime.datetime.now().strftime("%Y%m%d")
            trainedFilename = f'{CWD}/ml-models/training-models/{today}-v3trained.h5'
            if not os.path.exists(trainedFilename):
                logger.warning("PROGRAM IS ENCODING WHEN SOMEONE IS SENDING REQUEST.")
                self.encodeFaces()
            
            v3model = load_model(trainedFilename)

            filenames = []

            count = 1
            for (face_x, face_y, face_w, face_h) in box:
                faceCropped = image_array[face_y: face_y + face_w, face_x: face_x + face_h]
                filename = f"{CWD}/data/api_v3/output/{timeNow}/{id}/frame/frame{str(count).zfill(3)}.jpg"
                if not os.path.exists(f"{CWD}/data/api_v3/output/{timeNow}/{id}/frame/"):
                    os.mkdir(f"{CWD}/data/api_v3/output/{timeNow}/{id}/frame/")
                filenames.append(filename.split("output/")[1])
                cv2.imwrite(filename, faceCropped)
                cv2.imwrite(filename, self.resize(filename, 360))
                count += 1
                
            ### V3 - VGGFace Recognition ###
            facesDetected = {}

            count = 1
            for frameFilename in filenames:
                currentFrame = cv2.imread(f"{CWD}/data/api_v3/output/{frameFilename}")
                resized_image = cv2.resize(currentFrame, (224, 224))

                frame = f"{CWD}/data/api_v3/output/{timeNow}/{id}/frame"
                if not os.path.exists(frame):
                    os.mkdir(frame)
                    
                frame += f"/frame{str(count).zfill(3)}.jpg"
                
                cv2.imwrite(frame, resized_image)

                # Preparing the image for prediction
                x = kerasImagePreprocess.img_to_array(resized_image)
                x = numpy.expand_dims(x, axis=0)
                x = kerasVGGFaceUtils.preprocess_input(x, version=1)

                # Predicting
                predicted_prob = v3model.predict(x)
                faceDetected = class_list[predicted_prob[0].argmax()]

                facesDetected.update({frameFilename: faceDetected})
                count += 1

            result = {"faceCount":len(filenames), "result": facesDetected}

            return result

    def getFaceNames(self, frame):
        # Find all the faces and face encodings in the image
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = '0%'

            # Calculate the shortest distance to face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            best_match_index = numpy.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = self.faceConfidence(face_distances[best_match_index])
            
            face_names.append(f'{name} ({confidence})')

        return face_names

    def convertBGRtoRGB(self, frame):
        return frame[:, :, ::-1]

    def faceConfidence(self, face_distance, face_match_threshold=0.6):
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)

        if face_distance > face_match_threshold:
            return str(round(linear_val * 100, 2)) + '%'
        else:
            value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
            return str(round(value, 2)) + '%'

    # def getFaceCoordinates(self, frame, filenameDatas, model: str):
    #     ''' Detect input face '''
    #     if model not in ['yunet', 'yoloface']:
    #         logger.error("getFaceCoordinates model not found. Only accepting yunet/yoloface")
    #         return None

    #     if model == "yunet":
    #         logger.info("[YuNet] Grabbing faces detected from input image")

    #         timeNow = filenameDatas["timeNow"]
    #         id = filenameDatas["id"]
    #         detector = cv2.FaceDetectorYN.create(f"{CWD}/ml-models/face_detection_yunet/face_detection_yunet_2022mar.onnx", "", (320, 320))

    #         height, width, channels = frame.shape

    #         # Set input size
    #         detector.setInputSize((width, height))
    #         # Getting detections
    #         channel, faces = detector.detect(frame)
    #         faces = faces if faces is not None else []

    #         boxes = []
    #         confidences = []
    #         filenames = []
    #         count = 1
            
    #         for face in faces:
    #             box = list(map(int, face[:4]))
    #             boxes.append(box)
    #             x = box[0]
    #             y = box[1]
    #             w = box[2]
    #             h = box[3]
    #             faceCropped = frame[y:y + h, x:x + w]

    #             ### SEMENTARA MASIH TANPA FILTERING MINIMUM PIXEL SHAPE 50
    #             # if w >= 50 and h >= 50 and x >= 0 and y >= 0:
    #             filename = f"{CWD}/data/api_v1/output/{timeNow}/{id}/frame/frame{str(count).zfill(3)}.jpg"
    #             if not os.path.exists(f"{CWD}/data/api_v1/output/{timeNow}/{id}/frame/"):
    #                 os.mkdir(f"{CWD}/data/api_v1/output/{timeNow}/{id}/frame/")
    #             filenames.append(filename.split("output/")[1])
    #             cv2.imwrite(filename, faceCropped)
    #             cv2.imwrite(filename, self.resize(filename, 360))
    #             count += 1
                    
    #             confidence = face[-1]
    #             confidence = "{:.2f}%".format(confidence*100)

    #             confidences.append(confidence)

    #     if model == "yoloface":
    #         logger.info("[YoloFace] Grabbing faces detected from input image")

    #         timeNow = filenameDatas["timeNow"]
    #         id = filenameDatas["id"]

    #         face = face_analysis()
    #         _,faces,_ = face.face_detection(frame_arr=frame,frame_status=True,model='tiny')

    #         cv2_input = numpy.array(frame)

    #         filename = f"{CWD}/data/api_v1/output/{timeNow}/{id}/frame/frame{str(count).zfill(3)}.jpg"
    #         if not os.path.exists(f"{CWD}/data/api_v1/output/{timeNow}/{id}/frame/"):
    #             os.mkdir(f"{CWD}/data/api_v1/output/{timeNow}/{id}/frame/")
    #         filenames.append(filename.split("output/")[1])
    #         cv2.imwrite(filename, faceCropped)
    #         cv2.imwrite(filename, self.resize(filename, 360))
            
    #         box = boxes[0]
    #         x,y,w,h = box[0],box[1],box[2],box[3]
    #         cropped_face = cv2_input[y:y + w, x:x + h]
    #         cv2.imwrite(img, cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))

    #     if model == "yunet":
    #         model = "YuNet"
    #     elif model == "yoloface":
    #         model = "YoloFace"

    #     logger.info(f"[{model}] Face grab success. Got total faces of {len(filenames)}")
    #     return (filenames, confidences)
    
    def resize(self, filename: str, resolution: int):
        frame = cv2.imread(filename)
        if frame.shape[0] != resolution or frame.shape[1] != resolution:
            return cv2.resize(frame, (0, 0), fx=1-(frame.shape[1]-resolution)/frame.shape[1], fy=1-(frame.shape[1]-resolution)/frame.shape[1])
        else:
            return frame

# models = Models()
# models.grabRawDatasets()