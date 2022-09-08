docker build -f Dockerfile_train -t msg-docker-train .
#docker run --gpus all -v $(pwd)/../../../../../data/msg:/dataset -v $(pwd)/trained_model:/trained_model msg-docker-train
docker run -v /mnt/d/Dataset/msg_contest_data:/dataset -v $(pwd)/trained_model:/trained_model msg-docker-train

docker build -f Dockerfile -t msg-docker-submission .
#docker run --gpus all -v $(pwd)/../../../../../data/msg:/dataset -v $(pwd)/submission:/submission msg-docker-submission
docker run -v ../../dataset/dummy_test:/dataset -v $(pwd)/submission:/submission msg-docker-submission
