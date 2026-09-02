# Train and Deploy Scikit-Learn model

## training mock logistic regression model

Run the mock python training code
```bash
pip install -r examples/sklearn/requirements.txt 
python examples/sklearn/train_model.py
```

The output will be a model created on the project "serving examples", by the name "train sklearn model - sklearn-model"

## setting up the serving service

1. Create serving Service: `clearml-serving create --name "serving example"` (write down the service ID)
2. Create model endpoint: 
`clearml-serving --id <service_id> model add --engine sklearn --endpoint "test_model_sklearn" --preprocess "examples/sklearn/preprocess.py" --name "train sklearn model - sklearn-model" --project "serving examples"`

Or auto update 

`clearml-serving --id <service_id> model auto-update --engine sklearn --endpoint "test_model_sklearn_auto" --preprocess "examples/sklearn/preprocess.py" --name "train sklearn model - sklearn-model" --project "serving examples" --max-versions 2`

Or add Canary endpoint

`clearml-serving --id <service_id> model canary --endpoint "test_model_sklearn_auto" --weights 0.1 0.9 --input-endpoint-prefix test_model_sklearn_auto`

3. If you already have the `clearml-serving` docker-compose running, it might take it a minute or two to sync with the new endpoint.

Or you can run the clearml-serving container independently `docker run -v ~/clearml.conf:/root/clearml.conf -p 8080:8080 -e CLEARML_SERVING_TASK_ID=<service_id> clearml-serving:latest`

4. Test new endpoint (do notice the first call will trigger the model pulling, so it might take longer, from here on, it's all in memory): `curl -X POST "http://127.0.0.1:8080/serve/test_model_sklearn" -H "accept: application/json" -H "Content-Type: application/json" -d '{"x0": 1, "x1": 2}'`


> **_Notice:_**  You can also change the serving service while it is already running!
This includes adding/removing endpoints, adding canary model routing etc.
by default new endpoints/models will be automatically updated after 1 minute
