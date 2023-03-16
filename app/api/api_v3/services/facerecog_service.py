from ....core.logging import logger
from ..load_models import cv2, os, shutil, datetime, numpy as np, pickle, load_model, kerasImagePreprocess, kerasVGGFaceUtils 
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
        filename = f"{CWD}/data/api_v3/output/{timeNow}/{count}/data/input.jpg"

        tmpcount = 1
        while os.path.exists(filename):
            filename = f"{CWD}/data/api_v3/output/{timeNow}/{tmpcount}/data/input.jpg"
            count = tmpcount
            tmpcount += 1

        if not os.path.exists(f"{CWD}/data/api_v3/output/{timeNow}/"):
            os.mkdir(f"{CWD}/data/api_v3/output/{timeNow}/")
        if not os.path.exists(f"{CWD}/data/api_v3/output/{timeNow}/{count}/"):
            os.mkdir(f"{CWD}/data/api_v3/output/{timeNow}/{count}/")
        if not os.path.exists(f"{CWD}/data/api_v3/output/{timeNow}/{count}/data/"):
            os.mkdir(f"{CWD}/data/api_v3/output/{timeNow}/{count}/data/")

        # Save the image that is sent from the request and reject if filename is not valid
        with open(filename, "wb") as f:
            if image.filename.split(".")[-1].lower() not in ["jpg", "png", "jpeg", "heif"]:
                logger.warning(f"[v3] Filename not supported. Filename: {filename}")
                return {"path_frame": None, "path_result": None, "result": None, "error_message": "Filename not supported", "status": 0}
            else:
                shutil.copyfileobj(image.file, f)
                logger.info(f"[v3] Saving image to {filename}")

        facesDetected, frameNames = models.recog(filename, count)

        if len(facesDetected) == 0:
            logger.info("[v3] API return success with exception: No face detected. Files removed")
            os.remove(filename)
            return {"path_frame": None, "path_result": None, "result": None, "error_message": "No face detected", "status": 0}
        
        result = {}
        for i, frameName in enumerate(frameNames):
            result.update({frameName: facesDetected[i]})

        JSONFilename = f"{CWD}/data/api_v3/output/{timeNow}/{count}/data/face.json"

        with open(JSONFilename, "w") as f:
            f.write(str(result))

        logger.info("[v3] API return success. Request fulfilled.")
        return {"path_frame": facesDetected, "path_result": JSONFilename.split("output/")[1], "result": result, "status": 1}

    def getTimeNow(self):
        # before: %d-%b-%y.%H-%M-%S
        return datetime.datetime.now().strftime("%Y%m%d")
        
recogService = RecogService()