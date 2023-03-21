FROM python:3.8-slim-buster
WORKDIR /aml-spam-classification-1
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install -r requirements.txt
COPY app.py app.py
COPY score.py score.py
COPY . /aml-spam-classification-1
#CMD [ "python", -m" , "flask", "run", "--host=0.0.0.0"]
CMD ["python", "app.py"]