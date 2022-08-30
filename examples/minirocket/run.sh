sudo docker build -f Dockerfile -t msg-docker-submission .
sudo docker build -f Dockerfile_train -t msg-docker-train .
sudo docker run --gpus all -v `pwd`/../../../data:/dataset -v `pwd`/trained_model:/trained_model msg-docker-train
sudo docker run --gpus all -v `pwd`/../../../data:/dataset -v `pwd`/submission:/submission msg-docker-submission
