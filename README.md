![Repo card](https://repository-images.githubusercontent.com/614725734/2a524003-eb75-4b06-945c-3ef03ad65a34)

# Face Recognition API using FastAPI, OpenCV, YuNet face detection (v1 and v2), YoloFace detection (v3), face-recognition (v1) library and VGGFace recognition (v2 and v3).

### Usage
- Using docker <br>
`docker compose up --build`
    - When container is running successfully, it will take several minutes until localhost is available and usable. Just wait until FastAPI shows "Application startup complete" in the logs.

- Native <br>
`uvicorn main:app --host 0.0.0.0 --port 3344`
    - This runs the app on localhost port 3344

Send a post request to the main directory "/api/v\<version\>" (localhost:3344) that include 1 body requests, "file" which is an image upload/image binary string.

This API updates then re-train datasets on 01:00 a.m. local time

### Outputs
- v1 (/api/v1)
```
{
    "path_frame": [
        "yyyyMMdd/uniqueID/frame/frame001.jpg",
        "yyyyMMdd/uniqueID/frame/frame002.jpg"
    ],
    "path_result": "yyyyMMdd/uniqueID/data/face.json",
    "result": [
        {
            "frame_name": "frame001.jpg",
            "face_detected": "userIDdetected",
            "confidence": "float%"
        },
        {
            "frame_name": "frame002.jpg",
            "face_detected": "userIDdetected",
            "confidence": "float%"
        }
    ],
    "status": 1
}
```

- v2 and v3 (/api/v2 /api/v3)
```
{
    "path_frame": [
        "yyyyMMdd/uniqueID/frame/frame001.jpg",
        "yyyyMMdd/uniqueID/frame/frame002.jpg"
    ],
    "path_result": "yyyyMMdd/uniqueID/data/face.json",
    "result": {
        "frame001.jpg": "userIDdetected",
        "frame002.jpg": "userIDdetected"
    }
    "status": 1
}
```

##### v2 and v3 does not support confidence score since VGGFace does not support confidence score.

### This is a ready for deployment module by interns at PT Kazee Digital Indonesia for private company usage.
