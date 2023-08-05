FROM python:3.11
WORKDIR /app
COPY ["model.pth",\
"inference.py",\
"train.py",\
"README.md",\
"requirements.txt",\
"."]
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
