# My Seizure Gauge Forecasting Challenge 2022

- Sign up for competition
- Data [[See instructional video](https://drive.google.com/file/d/1Er4xBvZnZgtM_1ARyQ4gfCB6BVtdki5k/view)]
  - [Get training data](get_data.md)
  - [Expected Directory Structure](directory_structure.md)
- Training
  - Set up local environment
  - Load data
  - Train your model using prefered machine learning library
  - [Evaluate performance](train/evaluate.md)

- Prepare code for submission [[See instructional video](https://drive.google.com/file/d/1hEBGSeBxeZFrnLshuTyWMqCBczp8swts/view)]
  1. [Create code that generates predictions](submission/create_code.md)
      - Notes: [Input data for docker container](submission/inputs.md)
      - Notes: [Required outputs](submission/outputs.md)
      - Notes: [Computational restrictions](submission/restrictions.md)
  2. [Create Docker Image](submission/create_docker.md)
      - Notes: [Compatibility issues](submission/compatibility.md)
      - Notes: [Dockerfile Tips](submission/dockerfile_tips.md)
  3. [Test docker container locally](submission/run_locally.md) (Strongly Recomended)

- Make Submision [[See instructional video](https://drive.google.com/file/d/18M4mD56DDFhxGt20WVD8zG2AE6OhUX4Y/view?usp=sharing)]
  1. [Setup Eval.ai CLI](submission/prepare.md)
  2. [Make Submision](submission/submit.md)


## Examples

- [Example submission code](../examples/inceptiontime) using an `InceptionTime` deep learning model.
