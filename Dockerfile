# ##############################################################################
# BASE IMAGE
# ##############################################################################
# Use this one for submissions that dont require GPU
FROM ubuntu:20.04

# Use this one for submissions that require GPU processing
FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04


# ##############################################################################
# PYTHON
# ##############################################################################
# Comment out this entire section if you are not using python

# INSTALL PYTHON 3.8 (If using one of the base images provided above)
RUN apt-get update &&\
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3 &&\
    apt-get install -y pip &&\
    ln -sf /usr/bin/python3.8 /usr/bin/python &&\
    rm -rf /var/lib/apt/lists/*

# INSTALL TENSORFLOW
RUN python -m pip install tensorflow==2.9.1

# ADITIONAL PYTHON DEPENDENCIES (if you have them)
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN python -m pip install numba
RUN python -m pip install psutil
RUN python -m pip install sklearn


# ##############################################################################
# YOUR UNIQUE SETUP CODE HERE
# ##############################################################################
WORKDIR /app

# COPY WHATEVER OTHER SCRIPTS YOU MAY NEED
#COPY submission.py ./
COPY demo_submission_multirocket.py demo_submission_multirocket_per_patient.py ./
COPY models ./models

# SPECIFY THE ENTRYPOINT SCRIPT
CMD python demo_submission_multirocket_per_patient.py