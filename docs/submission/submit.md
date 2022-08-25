# Docker Submission - Submit

- [Go back to Main](../README.md)
- [Go back to Submission](submission.md)

## 1. Preparation

1. Sign up for [the competition on Eval AI](https://eval.ai/web/challenges/challenge-page/1693/overview).
2. Install evalai cli tool

    ```bash
    pip install evalai
    ```

3. Set your eval AI authentication token on the cli tool. You can get the token by going to your [eval.ai profile](https://eval.ai/web/profile)

    ```bash
    evalai set_token XXXXXXXXXXXX
    ```

## 2. Make the actual submission

Make sure you substitute `evalai-submission:latest` with the actual docker image name and tag you built.

```bash
# DEVELOPMENT PHASE
# evalai push MY_DOCKER_IMAGE:MY_TAG --phase my-dev-1693
evalai push evalai-submission:latest --phase my-dev-1693

# ACTUAL SUBMISSION PHASE DURING COMPETITION
# evalai push MY_DOCKER_IMAGE:MY_TAG --phase my-test-1693
evalai push evalai-submission:latest --phase my-test-1693
```

This will upload your docker image, and trigger an evaluation. 

Note, the first time you submit, it might take a while to upload. Subsequent uploads will be quicker if the base layers of your docker container are the same as the previous submission.

You can now monitor the progress of the submission in the [my submissions](https://eval.ai/web/challenges/challenge-page/1693/my-submission) section in the competition dashboard.
