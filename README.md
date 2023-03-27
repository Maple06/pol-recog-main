# Face Recognition API using FastAPI, OpenCV, and face-recognition library.

### Usage
- Using docker <br>
`docker compose up --build`
    - When container is running successfully, it will take several minutes until localhost is available and usable. Just wait until FastAPI shows "Application startup complete" in the logs.

- Native <br>
`uvicorn main:app --host 0.0.0.0 --port 3344`
    - This runs the app on localhost port 3344

Send a post request to the main directory "/" (localhost:3344) that include 1 body requests, "file" which is an image upload/image binary string.

This API updates then re-train datasets on 01:00 a.m. local time

### Outputs
- JSON format: return {"path_frame": ["path/to/frame001.jpg", "path/to/frame002.jpg"], "path_result": "path/to/face.json", "result": {'frame001': 'userID-1 :: confidence%', 'frame002': 'userID-2 :: confidence%'}, "status": <0/1>}
- Error JSON format: {"error-message": <string>, "status": 0} (Errors always have status code 0)
1. "Filename not supported"
    - API only accepts jpg, png, jpeg, and heif. Set this in app/api/api_v1/services/facerecog_service.py line 36.
2. "No face detected"
    - API success. No face is detected.
3. No error-message and neither path_frame, path_result, nor result is null
    - API success and outputs JSON

### This is a ready for deployment module by interns at PT Kazee Digital Indonesia for private company usage.