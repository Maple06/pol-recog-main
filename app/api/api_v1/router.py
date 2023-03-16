# API core module for all endpoints
from fastapi import APIRouter
from .endpoints.facerecog_endpoint import Recog
from fastapi import UploadFile, File

routerv1 = APIRouter(
    prefix='/api/v1',
    responses = {
        404: {'description': 'Not Found'}
    }
)

@routerv1.post('/')
async def faceRecog(file: UploadFile = File(...)):
    recog = Recog()
    result = recog.get_prediction(file)

    return result