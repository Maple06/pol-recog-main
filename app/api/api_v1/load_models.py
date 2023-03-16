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

from ...core.logging import logger

# testing pr

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
        logger.info("[v1] Encoding Faces... (This may take a while)")
        for image in tqdm(os.listdir(f'{CWD}/data/api_v1/dataset'), file=sys.stdout):
            face_image = face_recognition.load_image_file(f'{CWD}/data/api_v1/dataset/{image}')
            try:
                face_encoding = face_recognition.face_encodings(face_image)[0]
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(image)
            except IndexError:
                pass
        
        logger.info("[v1] Encoding Done!")

    def updateDataset(self):
        logger.info("[v1] Updating datasets... (This may took a while)")

        APITOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InJlemFhckBrYXplZS5pZCIsImlhdCI6MTY3NTgyMTY2Mn0.eprZiRQUjiWjbfZYlbziT6sXG-34f2CnQCSy3yhAh6I"
        r = requests.get("http://103.150.87.245:3001/api/profile/list-photo", headers={'Authorization': 'Bearer ' + APITOKEN})

        datas = r.json()["data"]

        for data in tqdm(datas, file=sys.stdout):
            userID = data["user_id"]
            url = data["photo"]

            r = requests.get(url)

            filename = f'{CWD}/data/api_v1/dataset/{userID}.jpg'
            
            # Save grabbed image to {CWD}/data/faces/
            with open(filename, 'wb') as f:
                f.write(r.content)
            
            self.imgAugmentation(filename)

        logger.info("[v1] Datasets updated!")

    def imgAugmentation(self, img):
        # Zoom to face
        try :
            frame = Image.open(img)
            frame = frame.convert("RGB")
            cv2_input = numpy.array(frame)
            detector = cv2.FaceDetectorYN.create(f"{CWD}/ml-models/face_detection_yunet/face_detection_yunet_2022mar.onnx", "", (320, 320))
            height, width, channels = cv2_input.shape
            detector.setInputSize((width, height))
            channel, faces = detector.detect(cv2_input)
            faces = faces if faces is not None else []
            boxes = []
            for face in faces:
                box = list(map(int, face[:4]))
                boxes.append(box)
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                faceCropped = cv2_input[y:y + h, x:x + w]
            if len(boxes) == 1:
                cv2.imwrite(img , cv2.cvtColor(faceCropped, cv2.COLOR_BGR2RGB))
        except :
           pass
        input_img = Image.open(img)
        input_img = input_img.convert('RGB')
        # Flip Image
        img_flip = ImageOps.flip(input_img)
        img_flip.save(f"{img.split('.jpg')[0]}-flipped.jpg")
        # Mirror Image 
        img_mirror = ImageOps.mirror(input_img)
        img_mirror.save(f"{img.split('.jpg')[0]}-mirrored.jpg")
        # Rotate Image
        img_rot1 = input_img.rotate(30)
        img_rot1.save(f"{img.split('.jpg')[0]}-rotated1.jpg")
        img_rot2 = input_img.rotate(330)
        img_rot2.save(f"{img.split('.jpg')[0]}-rotated2.jpg")
        # Adjust Brightness
        enhancer = ImageEnhance.Brightness(input_img)
        im_darker = enhancer.enhance(0.5)
        im_darker.save(f"{img.split('.jpg')[0]}-darker1.jpg")
        im_darker2 = enhancer.enhance(0.7)
        im_darker2.save(f"{img.split('.jpg')[0]}-darker2.jpg")
        enhancer = ImageEnhance.Brightness(input_img)
        im_darker = enhancer.enhance(1.2)
        im_darker.save(f"{img.split('.jpg')[0]}-brighter1.jpg")
        im_darker2 = enhancer.enhance(1.5)
        im_darker2.save(f"{img.split('.jpg')[0]}-brighter2.jpg")

    def getFaceCoordinates(self, frame, filenameDatas):
        logger.info("[v1] Grabbing faces detected from input image")

        timeNow = filenameDatas["timeNow"]
        id = filenameDatas["id"]
        detector = cv2.FaceDetectorYN.create(f"{CWD}/ml-models/face_detection_yunet/face_detection_yunet_2022mar.onnx", "", (320, 320))

        height, width, channels = frame.shape

        # Set input size
        detector.setInputSize((width, height))
        # Getting detections
        channel, faces = detector.detect(frame)
        faces = faces if faces is not None else []

        boxes = []
        confidences = []
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

            confidences.append(confidence)

        logger.info(f"[v1] Face grab success. Got total faces of {len(filenames)}")
        return (filenames, confidences)
    
    def resize(self, filename: str, resolution: int):
        frame = cv2.imread(filename)
        if frame.shape[0] != resolution or frame.shape[1] != resolution:
            return cv2.resize(frame, (0, 0), fx=1-(frame.shape[1]-resolution)/frame.shape[1], fy=1-(frame.shape[1]-resolution)/frame.shape[1])
        else:
            return frame