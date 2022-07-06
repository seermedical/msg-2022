FROM python:3.8.11
WORKDIR /app

# OPTIONALLY INSTALL PYTHON DEPENDENCIES IF YOU HAVE THEM
COPY requirements.txt ./
RUN pip install -r requirements.txt

# COPY WHATEVER OTHER SCRIPTS YOU MAY NEED
# COPY scriptA.py scriptB.py scriptC.py ./
# COPY submission.py dummy.py ./

# RUN WHATEVER OTHER COMMANDS YOU MAY NEED
# RUN mycommand1 &&\
#     mycommand2 &&\
#     mycommand3

# SPECIFY THE ENTRYPOINT SRIPT
COPY dummy.py ./
CMD python dummy.py
