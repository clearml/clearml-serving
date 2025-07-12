# Deploy vLLM model

## setting up the serving service

1. Create serving Service: `clearml-serving create --name "serving example"` (write down the service ID)

2. Add vLLM engine parameters in `VLLM_ENGINE_ARGS` variable as it was done in [this file](/docker/docker-compose-gpu.yml#L108). Make sure to add any required additional packages (for your custom model) to the [requirements.txt](/clearml_serving/serving/requirements.txt) or [docker-compose.yml](https://github.com/allegroai/clearml-serving/blob/826f503cf4a9b069b89eb053696d218d1ce26f47/docker/docker-compose.yml#L97) (or as environment variable to the `clearml-serving-inference` container), by defining for example: `CLEARML_EXTRA_PYTHON_PACKAGES="vllm==0.9.2 prometheus_client==0.21.1"`

3. Create model endpoint: 
    ```
    clearml-serving --id <service_id> model add --model-id <model_id> --engine vllm --endpoint "test_vllm" --preprocess "examples/vllm/preprocess.py"
    ```

4. If you already have the `clearml-serving` docker-compose running, it might take it a minute or two to sync with the new endpoint. To run docker-compose, see [docker-compose instructions](/README.md#nail_care-initial-setup), p. 8 (and use [docker-compose-gpu.yml](/docker/docker-compose-gpu.yml) file for vllm on gpu and [docker-compose.yml](/docker/docker-compose.yml) otherwise)

5. Test new endpoint (do notice the first call will trigger the model pulling, so it might take longer, from here on, it's all in memory):

    ```bash
    python examples/vllm/test_openai_api.py
    ```

    **Available routes**:

    + /v1/completions
    + /v1/chat/completions
    + /v1/models
    + /v1/audio/transcriptions
    + /v1/audio/translations
    + /v1/embeddings
    + /pooling
    + /classify
    + /score
    + /rerank

    see [test_openai_api.py](test_openai_api.py) for more information.

6. Check metrics using grafana (You have to select Prometheus as data source, all of vLLM metrics have "vllm:" prefix). For more information, see [Model monitoring and performance metrics](/README.md#bar_chart-model-monitoring-and-performance-metrics-bell)

NOTE!

If you want to use send_request method, keep in mind that you have to pass "completions" or "chat/completions" in entrypoint (and pass model as a part of "data" parameter) and use it for non-streaming models:

```python
prompt = "Hi there, goodman!"
result = self.send_request(endpoint="chat/completions", version=None, data={"model": "test_vllm", "messages": [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": prompt}]})
answer = result.choises[0].message.content
```
OR
If you want to use send_request method, use openai client instead