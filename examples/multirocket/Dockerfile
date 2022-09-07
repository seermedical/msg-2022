#FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04
FROM ubuntu:20.04

RUN apt-get update &&\
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3 &&\
    apt-get install -y pip &&\
    ln -sf /usr/bin/python3.8 /usr/bin/python &&\
    rm -rf /var/lib/apt/lists/*

# ADITIONAL PYTHON DEPENDENCIES (if you have them)
COPY requirements_prod.txt ./
RUN pip install -r requirements_prod.txt

WORKDIR /app


# COPY WHATEVER OTHER SCRIPTS YOU MAY NEED
COPY trained_model /trained_model
COPY submission.py train_model.py multirocket.py ./

# SPECIFY THE ENTRYPOINT SCRIPT
CMD ["python", "-u", "submission.py"]