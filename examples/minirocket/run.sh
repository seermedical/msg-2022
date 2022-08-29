sudo docker build -f Dockerfile_gpu -t msg-docker-test .
sudo docker run --gpus all -v `pwd`/../../data:/dataset -v `pwd`:/submission msg-docker-test
