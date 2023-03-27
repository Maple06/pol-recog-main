from ....core.logging import logger
from ..services.facerecog_service import recogService

# Module of an endpoint
class Recog:
    def __init__(self):
        pass

    def get_prediction(self, image):
        try:
            result = recogService.process(image)
            return result

        except Exception as e:
            logger.error('[v2] Error analysing an image :', e)
            return {"faceDetected": None, "confidence": None, "match-status": False, "error-status": 1, "error-message": f"Error analysing an image: {e}"}