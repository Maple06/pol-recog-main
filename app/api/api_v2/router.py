# API core module for all endpoints
from fastapi import APIRouter
from .endpoints.facerecog_endpoint import Recog
from fastapi import UploadFile, File

routerv2 = APIRouter(
    prefix='/api/v2',
    responses = {
        404: {'description': 'Not Found'}
    }
)

@routerv2.post('/')
async def faceRecog(file: UploadFile = File(...)):
    recog = Recog()
    result = recog.get_prediction(file)

    return result