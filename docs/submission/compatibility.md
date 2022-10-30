# Compatibility

- [Go back to Main](../README.md)
- [Go back to Submission](submission.md)
- [Go back to Creating Docker Image](create_docker.md)

## GPU Compatibility

If you want to make use of the GPU resources that are on the cloud infrastructure, then you should use a docker image that contains Nvidia `cuda`, and `cudnn` drivers. 

The cuda driver installed in the docker container must be `<=11.4`. However, for maximum compatibility with different deep learning libraries, we recommend `11.2`. 

The sample `Dockerfile` in this repo provides a recomended base image to build from that contains the recommended cuda libraries installed.

## Computational Limitations

The submission you make must take the following computational resource limits into consideration and not exceed them. 

- **CPU :** 4 cpu cores will be available.
- **RAM :** Approx 16Gb of memory will be available.
- **GPU :** 1 GPU, with 16Gb of memory will be available.
- **Run Time :** Your submission should run within approx 10 minutes or less. If submissions take too long to run, they will fail.
    - There are approximately `2,500` inputs in the test dataset. So your code should be capable of processing at least `4.2` inputs per second in order to stay within the time limit.

## Python Library Versions

For maximum compatibility, it is recomended to use `python3.8`, and pin your python libraries to the following suggested versions (but only if you actually need those libraries).

```bash
numpy==1.23.2
pandas==1.4.0
scipy==1.9.0

scikit-learn==1.1.2

# TENSORFLOW VERSION KNOWN TO BE COMPATIBLE WITH THE CUDA DRIVERS ON THE SERVERS
tensorflow==2.9.1

# PYTORCH VERSION KNOWN TO BE COMPATIBLE WITH THE CUDA DRIVERS ON THE SERVERS
torch==1.12.1
torchaudio==0.12.1
torchvision==0.13.1
```
