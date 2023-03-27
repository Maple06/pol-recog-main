from ....core.logging import logger
from ...load_models import cv2, face_recognition, os, shutil, datetime, numpy as np, math
from ...load_models import Models

CWD = os.getcwd()

models = Models()
models.v1Encode()

# Module specific business logic (will be use for endpoints)
class RecogService:
    def __init__(self):
        pass

    def process(self, image):
        # Get time now for filename
        timeNow = self.getTimeNow()

        count = 1
        filename = f"{CWD}/data/api_v1/output/{timeNow}/{count}/data/input.jpg"

        tmpcount = 1
        while os.path.exists(filename):
            filename = f"{CWD}/data/api_v1/output/{timeNow}/{tmpcount}/data/input.jpg"
            count = tmpcount
            tmpcount += 1

        logger.info(f"[v1] API request received. Image path: {filename}")

        if not os.path.exists(f"{CWD}/data/api_v1/output/{timeNow}/"):
            os.mkdir(f"{CWD}/data/api_v1/output/{timeNow}/")
        if not os.path.exists(f"{CWD}/data/api_v1/output/{timeNow}/{count}/"):
            os.mkdir(f"{CWD}/data/api_v1/output/{timeNow}/{count}/")
        if not os.path.exists(f"{CWD}/data/api_v1/output/{timeNow}/{count}/data/"):
            os.mkdir(f"{CWD}/data/api_v1/output/{timeNow}/{count}/data/")

        # Save the image that is sent from the request and reject if filename is not valid
        with open(filename, "wb") as f:
            if image.filename.split(".")[-1].lower() not in ["jpg", "png", "jpeg", "heif"]:
                logger.warning(f"[v1] Filename not supported. Filename: {filename}")
                return {"path_frame": None, "path_result": None, "result": None, "error_message": "Filename not supported", "status": 0}
            else:
                shutil.copyfileobj(image.file, f)
                logger.info(f"[v1] Saving image to {filename}")

        filenameDatas = {"timeNow": timeNow, "id": filename.split(f"{timeNow}/")[1].split("/data")[0]}

        output = models.inputRecog("v1", filename, filenameDatas)

        JSONFilename = f"{CWD}/data/api_v1/output/{timeNow}/{count}/data/face.json"

        with open(JSONFilename, "w") as f:
            f.write(str(output))

        logger.info("[v1] API return success. Request fulfilled.")
        return {"path_frame": [i["frame_path"] for i in output["result"]], "path_result": JSONFilename.split("output/")[1], "result": output["result"], "status": 1}

    def getTimeNow(self):
        # before: %d-%b-%y.%H-%M-%S
        return datetime.datetime.now().strftime("%Y%m%d")
        
recogService = RecogService()