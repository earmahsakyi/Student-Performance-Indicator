FROM python:3.10-slim-bookworm
WORKDIR /app
COPY . /app
COPY artifacts/ /app/artifacts/

RUN apt update -y && apt install awscli -y

RUN pip install -r requirements.txt
CMD ["python3", "app.py"]
