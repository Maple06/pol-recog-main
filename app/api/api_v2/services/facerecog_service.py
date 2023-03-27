from ....core.logging import logger
from ...load_models import os, shutil, datetime, cv2
from ...load_models import Models

CWD = os.getcwd()

models = Models()
models.v2Encode()

# Module specific business logic (will be use for endpoints)
class RecogService:
    def __init__(self):
        pass

    def process(self, image):
        # Get time now for filename
        timeNow = self.getTimeNow()

        count = 1
        filename = f"{CWD}/data/api_v2/output/{timeNow}/{count}/data/input.jpg"

        tmpcount = 1
        while os.path.exists(filename):
            filename = f"{CWD}/data/api_v2/output/{timeNow}/{tmpcount}/data/input.jpg"
            count = tmpcount
            tmpcount += 1

        logger.info(f"[v2] API request received. Image path: {filename}")

        if not os.path.exists(f"{CWD}/data/api_v2/output/{timeNow}/"):
            os.mkdir(f"{CWD}/data/api_v2/output/{timeNow}/")
        if not os.path.exists(f"{CWD}/data/api_v2/output/{timeNow}/{count}/"):
            os.mkdir(f"{CWD}/data/api_v2/output/{timeNow}/{count}/")
        if not os.path.exists(f"{CWD}/data/api_v2/output/{timeNow}/{count}/data/"):
            os.mkdir(f"{CWD}/data/api_v2/output/{timeNow}/{count}/data/")

        # Save the image that is sent from the request and reject if filename is not valid
        with open(filename, "wb") as f:
            if image.filename.split(".")[-1].lower() not in ["jpg", "png", "jpeg", "heif"]:
                logger.warning("Filename not supported")
                return {"path_frame": None, "path_result": None, "result": None, "error_message": "Filename not supported", "status": 0}
            else:
                shutil.copyfileobj(image.file, f)
                logger.info(f"Saving image to {filename}")

        frame = cv2.imread(filename)

        filenameDatas = {"timeNow": timeNow, "id": filename.split(f"{timeNow}/")[1].split("/data")[0]}

        output = models.inputRecog("v2", filename, filenameDatas)

        JSONFilename = f"{CWD}/data/api_v2/output/{timeNow}/{count}/data/face.json"

        with open(JSONFilename, "w") as f:
            f.write(str(output))

        logger.info("[v2] API return success. Request fulfilled.")
        return {"path_frame": list(output["result"].keys()), "path_result": JSONFilename.split("output/")[1], "result": output["result"], "status": 1}

    def getTimeNow(self):
        # before: %d-%b-%y.%H-%M-%S
        return datetime.datetime.now().strftime("%Y%m%d")
        
recogService = RecogService()