# Deploy vLLM model

## setting up the serving service

1. Create serving Service: `clearml-serving create --name "serving example"` (write down the service ID)
2. Make sure to add any required additional packages (for your custom model) to the [docker-compose.yml](https://github.com/allegroai/clearml-serving/blob/826f503cf4a9b069b89eb053696d218d1ce26f47/docker/docker-compose.yml#L97) (or as environment variable to the `clearml-serving-inference` container), by defining for example: `CLEARML_EXTRA_PYTHON_PACKAGES="vllm==0.5.4"`
3. Create model endpoint: 
`clearml-serving --id <service_id> model add --engine vllm --endpoint "test_vllm" --preprocess "examples/vllm/preprocess.py" --name "test vllm" --project "serving examples"`

Or auto update 

`clearml-serving --id <service_id> model auto-update --engine vllm --endpoint "test_vllm" --preprocess "examples/vllm/preprocess.py" --name "test vllm" --project "serving examples" --max-versions 2`

Or add Canary endpoint

`clearml-serving --id <service_id> model canary --endpoint "test_vllm" --weights 0.1 0.9 --input-endpoint-prefix test_vllm`

4. If you already have the `clearml-serving` docker-compose running, it might take it a minute or two to sync with the new endpoint.

Or you can run the clearml-serving container independently `docker run -v ~/clearml.conf:/root/clearml.conf -p 8080:8080 -e CLEARML_SERVING_TASK_ID=<service_id> clearml-serving:latest`

5. Test new endpoint (do notice the first call will trigger the model pulling, so it might take longer, from here on, it's all in memory):

```python

import openai
openai.api_key = "dummy"
openai.api_base = f"http://serving.apps.okd.mts.ai/clearml/v1"


r0 = await openai.ChatCompletion.acreate(
    model=vllm_endpoint,
    messages=[{"role": "system", "content": ""}, {"role": "user", "content": "Hi there, goodman!"}],
    temperature=1.0,
    max_tokens=1024,
    top_p=1.0,
    request_timeout=10000,
)

print(f"ChatCompletion: {r0['choices'][0]['message']}")

r1 = await openai.Completion.acreate(
    model=vllm_endpoint,
    prompt="Hi there, goodman!",
    temperature=1.0,
    max_tokens=256,
)

print(f"Completion: \n {r1['choices'][0]['text']}")

```
