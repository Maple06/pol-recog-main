from ....core.logging import logger
from ..load_models import cv2, face_recognition, os, shutil, datetime, numpy as np, math, requests, tqdm, sys, Image, ImageOps, ImageEnhance
from ..load_models import Models

CWD = os.getcwd()

models = Models()
models.encodeFaces()

# Module specific business logic (will be use for endpoints)
class RecogService:
    def __init__(self):
        pass

    def process(self, image):
        # Get time now for filename
        timeNow = self.getTimeNow()

        count = 1
        filename = f"{CWD}/data/output/{timeNow}/{count}/data/input.jpg"

        tmpcount = 1
        while os.path.exists(filename):
            filename = f"{CWD}/data/output/{timeNow}/{tmpcount}/data/input.jpg"
            count = tmpcount
            tmpcount += 1

        if not os.path.exists(f"{CWD}/data/output/{timeNow}/"):
            os.mkdir(f"{CWD}/data/output/{timeNow}/")
        if not os.path.exists(f"{CWD}/data/output/{timeNow}/{count}/"):
            os.mkdir(f"{CWD}/data/output/{timeNow}/{count}/")
        if not os.path.exists(f"{CWD}/data/output/{timeNow}/{count}/data/"):
            os.mkdir(f"{CWD}/data/output/{timeNow}/{count}/data/")

        # Save the image that is sent from the request and reject if filename is not valid
        with open(filename, "wb") as f:
            if image.filename.split(".")[-1].lower() not in ["jpg", "png", "jpeg", "heif"]:
                logger.warning(f"[v1] Filename not supported. Filename: {filename}")
                return {"path_frame": None, "path_result": None, "result": None, "error_message": "Filename not supported", "status": 0}
            else:
                shutil.copyfileobj(image.file, f)
                logger.info(f"[v1] Saving image to {filename}")

        frame = cv2.imread(filename)

        filenameDatas = {"timeNow": timeNow, "id": filename.split(f"{timeNow}/")[1].split("/data")[0]}

        filenames, confidences = models.getFaceCoordinates(frame, filenameDatas)

        if len(filenames) == 0:
            logger.info("[v1] API return success with exception: No face detected. Files removed")
            os.remove(filename)
            return {"path_frame": None, "path_result": None, "result": None, "error_message": "No face detected", "status": 0}
        
        resultRaw = []
        for currentFilename in filenames:
            try:
                resultRaw.append(self.recog(f"{CWD}/data/output/{currentFilename}"))
            except:
                resultRaw.append(["Unknown", "0%"])

        frameNames = (i.split("/")[-1].split(".")[0] for i in filenames)
        
        result = {}
        for i, frameName in enumerate(frameNames):
            userDetected = resultRaw[i][0]
            confidence = resultRaw[i][1]
            if userDetected == "Unknown":
                confidence = confidences[i]
            result.update({frameName: f"{userDetected} :: {confidence}"})
        
        JSONFilename = f"{CWD}/data/output/{timeNow}/{count}/data/face.json"

        with open(JSONFilename, "w") as f:
            f.write(str(result))

        logger.info("[v1] API return success. Request fulfilled.")
        return {"path_frame": filenames, "path_result": JSONFilename.split("output/")[1], "result": result, "status": 1}

    def getTimeNow(self):
        # before: %d-%b-%y.%H-%M-%S
        return datetime.datetime.now().strftime("%Y%m%d")

    def recog(self, filename: str):
        logger.info(f"[v1] Recognizing faces into user IDs. Filename: {filename}")
        # Read image as cv2
        frame = cv2.imread(filename)

        frame = models.resize(filename, 480)
        frame = self.convertBGRtoRGB(frame)

        faceNames = list(self.getFaceNames(frame))

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

        return [faceNames[0][0], faceNames[0][1]]

    def getFaceNames(self, frame):
        # Find all the faces and face encodings in the image
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(models.known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = '0%'

            # Calculate the shortest distance to face
            face_distances = face_recognition.face_distance(models.known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = models.known_face_names[best_match_index]
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
        
recogService = RecogService()