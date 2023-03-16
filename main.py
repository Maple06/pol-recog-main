import uvicorn
from fastapi import FastAPI
from app.api.api_v1.router import routerv1
from app.api.api_v2.router import routerv2
from app.api.api_v3.router import routerv3
from app.core.scheduler import dailyEncodev1, dailyEncodev2, dailyEncodev3

app = FastAPI()
app.include_router(routerv1)
app.include_router(routerv2)
app.include_router(routerv3)

# Default root path
@app.get('/')
async def root():

    message = {
        'message': 'This is face recognition POLDA API v1.0 v2.0 and v3.0.',
        'v1.0': 'YuNet (Detection) with Face recognition library (Recognition)',
        'v2.0': 'YuNet (Detection) with VGGFace (Recognition)',
        'v3.0': 'YoloFace (Detection) with VGGFace (Recognition)'
    }

    return message

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3344)