from typing import Any, Optional, Callable, Union


# Preprocess class Must be named "Preprocess"
# No need to inherit or to implement all methods
class Preprocess(object):
    """
    Preprocess class Must be named "Preprocess"
    Otherwise there are No limitations, No need to inherit or to implement all methods
    Notice! This is not thread safe! the same instance may be accessed from multiple threads simultaneously
    to store date in a safe way push it into the `state` dict argument of preprocessing/postprocessing functions

    Notice the execution flows is synchronous as follows:

    1. RestAPI(...) -> body: Union[bytes, dict]
    2. preprocess(body: Union[bytes, dict], ...) -> data: Any
    3. process(data: Any, ...) -> data: Any
    4. postprocess(data: Any, ...) -> result: dict
    5. RestAPI(result: dict) -> returned request
    """

    def __init__(self):
        # set internal state, this will be called only once. (i.e. not per request)
        # it will also set the internal model_endpoint to reference the specific model endpoint object being served
        self.model_endpoint = None  # type: clearml_serving.serving.endpoints.ModelEndpoint

    def load(self, local_file_name: str) -> Any:  # noqa
        """
        OPTIONAL: provide loading method for the model
        useful if we need to load a model in a specific way for the prediction engine to work

        REMOVE FUNCTION IF NOT USED

        Notice! When used with specific engines (i.e. not Custom)
        The returned object will be passed as is to the inference engine,
        this means it must not be None, otherwise the endpoint will be ignored!

        :param local_file_name: file name / path to read load the model from

        :return: Object that will be called with .predict() method for inference.
        """
        pass

    def unload(self) -> None:
        """
        OPTIONAL: provide unloading method for the model
        For example:
        ```py
        import torch
        torch.cuda.empty_cache()
        ```
        """
        pass

    def preprocess(
            self,
            body: Union[bytes, dict],
            state: dict,
            collect_custom_statistics_fn: Optional[Callable[[dict], None]],
    ) -> Any:  # noqa
        """
        Optional: do something with the request data, return any type of object.
        The returned object will be passed as is to the inference engine

        :param body: dictionary or bytes as recieved from the RestAPI
        :param state: Use state dict to store data passed to the post-processing function call.
            This is a per-request state dict (meaning a new empty dict will be passed per request)
            Usage example:
            >>> def preprocess(..., state):
                    state['preprocess_aux_data'] = [1,2,3]
            >>> def postprocess(..., state):
                    print(state['preprocess_aux_data'])
        :param collect_custom_statistics_fn: Optional, if provided allows to send a custom set of key/values
            to the statictics collector servicd.
            None is passed if statiscs collector is not configured, or if the current request should not be collected

            Usage example:
            >>> print(body)
            {"x0": 1, "x1": 2}
            >>> if collect_custom_statistics_fn:
            >>>   collect_custom_statistics_fn({"x0": 1, "x1": 2})

        :return: Object to be passed directly to the model inference
        """
        return body

    def postprocess(
            self,
            data: Any,
            state: dict,
            collect_custom_statistics_fn: Optional[Callable[[dict], None]],
    ) -> dict:  # noqa
        """
        Optional: post process the data returned from the model inference engine
        returned dict will be passed back as the request result as is.

        :param data: object as recieved from the inference model function
        :param state: Use state dict to store data passed to the post-processing function call.
            This is a per-request state dict (meaning a dict instance per request)
            Usage example:
            >>> def preprocess(..., state):
                    state['preprocess_aux_data'] = [1,2,3]
            >>> def postprocess(..., state):
                    print(state['preprocess_aux_data'])
        :param collect_custom_statistics_fn: Optional, if provided allows to send a custom set of key/values
            to the statictics collector servicd.
            None is passed if statiscs collector is not configured, or if the current request should not be collected

            Usage example:
            >>> if collect_custom_statistics_fn:
            >>>   collect_custom_statistics_fn({"y": 1})

        :return: Dictionary passed directly as the returned result of the RestAPI
        """
        return data

    def process(
            self,
            data: Any,
            state: dict,
            collect_custom_statistics_fn: Optional[Callable[[dict], None]],
    ) -> Any:  # noqa
        """
        OPTIONAL: do something with the actual data, return any type of object.
        The returned object will be passed as is to the postprocess function engine

        REMOVE FUNCTION IF NOT USED

        :param data: object as recieved from the preprocessing function
        :param state: Use state dict to store data passed to the post-processing function call.
            This is a per-request state dict (meaning a dict instance per request)
            Usage example:
            >>> def preprocess(..., state):
                    state['preprocess_aux_data'] = [1,2,3]
            >>> def postprocess(..., state):
                    print(state['preprocess_aux_data'])
        :param collect_custom_statistics_fn: Optional, if provided allows to send a custom set of key/values
            to the statictics collector servicd.
            None is passed if statiscs collector is not configured, or if the current request should not be collected

            Usage example:
            >>> if collect_custom_statistics_fn:
            >>>   collect_custom_statistics_fn({"type": "classification"})

        :return: Object to be passed tp the post-processing function
        """
        return data

    def send_request(  # noqa
            self,
            endpoint: str,
            version: Optional[str] = None,
            data: Optional[dict] = None
    ) -> Optional[dict]:
        """
        NOTICE! This method will be replaced in runtime, by the inference service

        Helper method to send model inference requests to the inference service itself.
        This is designed to help with model ensemble, model pipelines, etc.
        On request error return None, otherwise the request result data dictionary

        Usage example:

        >>> x0, x1 = 1, 2
        >>> result = self.send_request(endpoint="test_model_sklearn", version="1", data={"x0": x0, "x1": x1})
        >>> y = result["y"]
        """
        pass
