FROM python:3.10

COPY ./ ./

RUN python -m pip install -U pip wheel cmake

RUN python -m pip install -r ./requirements/base.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3344", "--workers", "5"]