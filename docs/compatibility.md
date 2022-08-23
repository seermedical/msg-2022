# Compatibility

- [home](README.md)

## GPU compatibility

If you want to make use of the GPU resources that are on the cloud infrastructure, then you should use a docker image that contains Nvidia `cuda`, and `cudnn` drivers. 

The cuda driver installed in the docker container must be `<=11.4`. However, for maximum compatibility with different deep learning libraries, we recommend `11.2`. 

The sample `Dockerfile` in this repo provides a recomended base image to build from that contains the recommended cuda libraries installed.


## Python Library Versions

For maximum compatibility, it is recomended to use `python3.8`, and pin your python libraries to the following suggested versions (but only if you actually need those libraries).

```bash
numpy==1.23.2
pandas==1.4.0
scipy==1.9.0

scikit-learn==1.1.2

tensorflow==2.9.1

torch==1.12.1
torchaudio==0.12.1
torchvision==0.13.1
```
