# ##############################################################################
# BASE IMAGE
# ##############################################################################
# Use this one for submissions that dont require GPU
FROM ubuntu:20.04

# Use this one for submissions that require GPU processing
# FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04


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

# INSTALL DIFFERENT VERSION OF PYTHON (change 3.10 to the version you want)
# RUN apt-get update &&\
#    DEBIAN_FRONTEND=noninteractive apt-get install software-properties-common -y &&\
#    add-apt-repository -y ppa:deadsnakes/ppa &&\
#     apt-get install -y python3.10  &&\
#     ln -sf /usr/bin/python3.10 /usr/bin/python &&\
#     ln -sf /usr/bin/python3.10 /usr/bin/python3 &&\
#     apt-get install -y curl &&\
#     curl -sS https://bootstrap.pypa.io/get-pip.py | python &&\
#     rm -rf /var/lib/apt/lists/*

# INSTALL PYTORCH (with specific version known to work with cuda driver 11.2)
# RUN python -m pip install --no-cache-dir \
#     torch==1.12.1 \
#     torchaudio==0.12.1 \
#     torchvision==0.13.1 \
#     --extra-index-url https://download.pytorch.org/whl/cu112

# INSTALL TENSORFLOW (with specific version known to work with cuda driver 11.2)
# RUN python -m pip install tensorflow==2.9.1

# ADITIONAL PYTHON DEPENDENCIES (if you have them)
COPY requirements_prod.txt ./
RUN pip install -r requirements_prod.txt


# ##############################################################################
# R
# ##############################################################################
# # INSTALL R
# RUN apt-get update &&\
#     DEBIAN_FRONTEND=noninteractive apt-get install -y r-base &&\
#     rm -rf /var/lib/apt/lists/*

# # INSTALL THE R PACKAGES YOU NEED
# RUN R -e "install.packages('caret',dependencies=TRUE, repos='http://cran.rstudio.com/')" &&\
#     R -e "install.packages('data.table',dependencies=TRUE, repos='http://cran.rstudio.com/')" &&\
#     R -e "install.packages('xgboost',dependencies=TRUE, repos='http://cran.rstudio.com/')"


# ##############################################################################
# YOUR UNIQUE SETUP CODE HERE
# ##############################################################################
WORKDIR /app


# COPY WHATEVER OTHER SCRIPTS YOU MAY NEED
COPY submission.py ./
COPY train_model.py ./
COPY minirocket.py ./

# RUN WHATEVER OTHER COMMANDS YOU MAY NEED TO SET UP THE SYSTEM
# RUN mycommand1 &&\
#     mycommand2 &&\
#     mycommand3

# SPECIFY THE ENTRYPOINT SCRIPT
CMD ["python", "-u", "submission.py"]
