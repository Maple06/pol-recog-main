# API core module for all endpoints
from fastapi import APIRouter
from .api_v1.endpoints.facerecog_endpoint import Recog as EndpointV1
from .api_v2.endpoints.facerecog_endpoint import Recog as EndpointV2
from .api_v3.endpoints.facerecog_endpoint import Recog as EndpointV3
from fastapi import UploadFile, File

routerv1 = APIRouter(
    prefix='/api/v1',
    responses = {
        404: {'description': 'Not Found'}
    }
)
routerv2 = APIRouter(
    prefix='/api/v2',
    responses = {
        404: {'description': 'Not Found'}
    }
)
routerv3 = APIRouter(
    prefix='/api/v3',
    responses = {
        404: {'description': 'Not Found'}
    }
)

@routerv1.post('/')
async def faceRecog(file: UploadFile = File(...)):
    recog = EndpointV1()
    result = recog.get_prediction(file)

    return result

@routerv2.post('/')
async def faceRecog(file: UploadFile = File(...)):
    recog = EndpointV2()
    result = recog.get_prediction(file)

    return result

@routerv3.post('/')
async def faceRecog(file: UploadFile = File(...)):
    recog = EndpointV3()
    result = recog.get_prediction(file)

    return result