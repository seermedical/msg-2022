docker build -f Dockerfile -t msg-docker-submission .
docker build -f Dockerfile_train -t msg-docker-train .
docker run --gpus all -v `pwd`/../../../../data/msg:/dataset -v `pwd`/trained_model:/trained_model msg-docker-train
docker run --gpus all -v `pwd`/../../../../data/msg:/dataset -v `pwd`/submission:/submission msg-docker-submission
