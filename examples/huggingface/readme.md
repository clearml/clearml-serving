# Example Huggingface on ClearML Serving

Technically, the underlying NVIDIA Triton inference engine can handle almost any type of model, including Pytorch models which is how many Huggingface models are shipped out of the box.

But in order to get better serving speeds, check out this [repository](https://github.com/ELS-RD/transformer-deploy), their [docs](https://els-rd.github.io/transformer-deploy/) and the excellent accompanying [blogpost](https://medium.com/towards-data-science/hugging-face-transformer-inference-under-1-millisecond-latency-e1be0057a51c) to convert huggingface models first into ONNX and then into TensorRT optimized binaries.

## Model vs Tokenizer

Most Huggingface NLP models ship with a tokenizer as well. We don’t want to leave it to the end user to embed their own inputs. The blogpost above uses an ensemble endpoint in Triton that first runs some python code that contains the tokenizer and then sends the result to a second endpoint which contains the actual model.

This is a good approach, but the tokenizer is CPU based and not independently scalable from the GPU based transformer model. With ClearML serving, we can move the tokenization step to the preprocessing script that we provide to the ClearML serving inference container, which will make this step completely autoscalable.

## Getting the right TensorRT <> Triton versions

Chances are very high that the transformer-deploy image has a different triton version than what ClearML serving uses, which will give issues later on. Triton is very harsh on its version requirements. Please check the triton version we are using in `clearml_serving/engines/triton/Dockerfile` and compare it to the main Dockerfile from the `transformers-deploy` repo. Check [this](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) page for more information about which TensorRT version is shipped in which Triton container.

If they don't match up, either rebuild the ClearML triton image locally with the right triton version and make sure it is picked up by compose, or build the `transformers-deploy` image locally with the correct version and use it to run the model conversion. Your model has to be optimized using the exact same TensorRT version or it will not serve!

## Setting up for the example

At the time of this writing, compiling a huggingface model from the `transformers-deploy` main branch means it is compiled using tensorRT version 8.4.1, which corresponds to Triton version 22.07.

To get ClearML running on 22.07, all we need to do is change the base image name in the `docker-compose-triton-gpu.yml` file, the the correct version.

```diff
...
clearml-serving-triton:
-   image: allegroai/clearml-serving-triton:latest
+   image: allegroai/clearml-serving-triton:1.2.0-22.07
    container_name: clearml-serving-triton
    restart: unless-stopped
    # optimize perforamnce
    security_opt:
      - seccomp:unconfined
...
```
Or you can build your own correct version by adapting the dockerfile in `clearml_serving/engines/triton/Dockerfile`, building it and making sure the triton compose yaml uses it instead.


## Setting up the serving service

### Get the repository (with the example)

Clone the serving repository if you haven’t already.

```bash
git clone https://github.com/allegroai/clearml-serving.git
cd clearml-serving
```

### Launch the serving task to clearml

Install `clearml-serving` either via pip or from the repository. Create serving Service:

```bash
clearml-serving create --name "huggingface serving example"
```

(write down the service ID, this is the service ID that is in your env file as well)

### Setting up the docker-compose serving stack
Setup the `docker/example.env` file with your ClearML credentials, then add an extra line to install 3rd party packages. In this case, we want to also install the `transformers` package because we’re going to run the tokenizer in the inference container

```bash
CLEARML_WEB_HOST="https://app.clear.ml"
CLEARML_API_HOST="https://api.clear.ml"
CLEARML_FILES_HOST="https://files.clear.ml"
CLEARML_API_ACCESS_KEY="<>"
CLEARML_API_SECRET_KEY="<>"
CLEARML_SERVING_TASK_ID="<>"
# Add this to install necessary packages
CLEARML_EXTRA_PYTHON_PACKAGES=transformers
# Change this depending on your machine and performance needs
CLEARML_USE_GUNICORN=1
CLEARML_SERVING_NUM_PROCESS=8
# Restarts if the serving process crashes
CLEARML_SERVING_RESTART_ON_FAILURE=1
```

Huggingface models require Triton engine support, please use `docker-compose-triton.yml` / `docker-compose-triton-gpu.yml` or if running on Kubernetes, the matching helm chart to set things up. Check the repository main readme documentation if you need help.

To run with the correct version of Triton for this example, do:
```bash
docker compose --env-file docker/example.env -f docker/docker-compose-triton-gpu.yml -f examples/huggingface/docker-compose-override.yml  up --force-recreate
```
This should get you a running ClearML stack with Triton which is reporting to a ClearML task in a project called `DevOps`.

### Getting the sample model

If you didn’t use the transformers-deploy repository on your own model, you can run this single command to get a tensorRT binary of an example classification model. 

Please make sure you have properly installed docker and nvidia-container-toolkit, so it can be run on GPU. The command will download a `model.bin` file to the local directory for you to serve.

```bash
curl https://clearml-public.s3.amazonaws.com/models/model_onnx.bin -o model.bin
```

### Setup

1. Upload the TensorRT model (write down the model ID)

```bash
clearml-serving --id <your_service_ID> model upload --name "Transformer ONNX" --project "Hugginface Serving" --path model.bin
```

2. Create a model endpoint:

```bash
# Without dynamic batching
clearml-serving --id <your_service_ID> model add --engine triton --endpoint "transformer_model" --model-id <your_model_ID> --preprocess examples/huggingface/preprocess.py --input-size "[-1, -1]" "[-1, -1]" "[-1, -1]" --input-type int32 int32 int32 --input-name "input_ids" "token_type_ids" "attention_mask" --output-size "[-1, 2]" --output-type float32 --output-name "output" --aux-config platform=\"tensorrt_plan\" default_model_filename=\"model.bin\"

# With dynamic batching
clearml-serving --id <your_service_ID> model add --engine triton --endpoint "transformer_model" --model-id <your_model_ID> --preprocess examples/huggingface/preprocess.py --input-size "[-1]" "[-1]" "[-1]" --input-type int32 int32 int32 --input-name "input_ids" "token_type_ids" "attention_mask" --output-size "[2]" --output-type float32 --output-name "output" --aux-config platform=\"onnxruntime_onnx\" default_model_filename=\"model.bin\" dynamic_batching.preferred_batch_size="[1,2,4,8,16,32,64]" dynamic_batching.max_queue_delay_microseconds=5000000 max_batch_size=64
```

> Note the backslashes for string values! `platform=\"tensorrt_plan\" default_model_filename=\"model.bin\"`

> **INFO**: the model input and output parameters are usually in a `config.pbtxt` file next to the model itself. 

1. Make sure you have the `clearml-serving` `docker-compose-triton.yml` (or `docker-compose-triton-gpu.yml`) running, it might take it a minute or two to sync with the new endpoint.
2. Test new endpoint (do notice the first call will trigger the model pulling, so it might take longer, from here on, it's all in memory):

> ***Notice:***
 You can also change the serving service while it is already running! This includes adding/removing endpoints, adding canary model routing etc. by default new endpoints/models will be automatically updated after 1 minute
> 

## Running Inference

After waiting a little bit for the stack to detect your new endpoint and load it, you can use curl to send a request:

```bash
curl -X POST "http://127.0.0.1:8080/serve/transformer_model" -H "accept: application/json" -H "Content-Type: application/json" -d '{"text": "This is a ClearML example to show how Triton binaries are deployed."}'
```

Or use the notebook in this example folder to run it using python `requests`

The inference request will be sent to the ClearML inference service first, which will run the raw request through the `preprocessing.py` file, which takes out the `text` value, runs it through the tokenizer and then sends the result to Triton, which runs the model and sends the output back to the same `preprocessing.py` file but in the postprocessing function this time, whose result is returned to the user.

## Benchmarking

To run a load test on your endpoint to check its performance, use the following commands:
```bash
ab -l -n 8000 -c 128  -H "accept: application/json" -H "Content-Type: application/json" -T "application/json" -p examples/huggingface/example_payload.json  "http://127.0.0.1:8080/serve/transformer_model"
```