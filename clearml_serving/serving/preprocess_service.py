import os
import json
import sys
import threading
import traceback
import warnings
from pathlib import Path
from typing import Optional, Any, Callable, List

import numpy as np
from clearml import Task, Model
from clearml.binding.artifacts import Artifacts
from clearml.storage.util import sha256sum
from requests import post as request_post

from .endpoints import ModelEndpoint

class Singleton(object):
    _instance = None
    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance

class BasePreprocessRequest(object):
    __preprocessing_lookup = {}
    __preprocessing_modules = set()
    _grpc_env_conf_prefix = "CLEARML_GRPC_"
    _default_serving_base_url = "http://127.0.0.1:8080/serve/"
    _server_config = {}  # externally configured by the serving inference service
    _timeout = None  # timeout in seconds for the entire request, set in __init__
    is_preprocess_async = False
    is_process_async = False
    is_postprocess_async = False

    def __init__(
            self,
            model_endpoint: ModelEndpoint,
            task: Task = None,
    ):
        """
        Notice this object is not be created per request, but once per Process
        Make sure it is always thread-safe
        """
        self.model_endpoint = model_endpoint
        self._preprocess = None
        self._model = None
        if self._timeout is None:
            self._timeout = int(float(os.environ.get('GUNICORN_SERVING_TIMEOUT', 600)) * 0.8)

        # load preprocessing code here
        if self.model_endpoint.preprocess_artifact:
            if not task or self.model_endpoint.preprocess_artifact not in task.artifacts:
                raise ValueError("Error: could not find preprocessing artifact \'{}\' on Task id={}".format(
                    self.model_endpoint.preprocess_artifact, task.id))
            else:
                try:
                    self._instantiate_custom_preprocess_cls(task)
                except Exception as ex:
                    raise ValueError("Error: Failed loading preprocess code for \'{}\': {}\n\n{}".format(
                        self.model_endpoint.preprocess_artifact, ex, traceback.format_exc()))

    def _instantiate_custom_preprocess_cls(self, task: Task) -> None:
        path = task.artifacts[self.model_endpoint.preprocess_artifact].get_local_copy(extract_archive=False)
        if not path or not Path(path).exists():
            raise ValueError("Artifact '{}' could not be downloaded".format(self.model_endpoint.preprocess_artifact))

        # check file content hash, should only happen once?!
        # noinspection PyProtectedMember
        file_hash, _ = sha256sum(path, block_size=Artifacts._hash_block_size)
        if file_hash != task.artifacts[self.model_endpoint.preprocess_artifact].hash:
            print("INFO: re-downloading artifact '{}' hash changed".format(
                self.model_endpoint.preprocess_artifact))
            path = task.artifacts[self.model_endpoint.preprocess_artifact].get_local_copy(
                extract_archive=True,
                force_download=True,
            )
        else:
            # extract zip if we need to, otherwise it will be the same
            path = task.artifacts[self.model_endpoint.preprocess_artifact].get_local_copy(
                extract_archive=True,
            )

        import importlib.util
        if Path(path).is_file():
            spec = importlib.util.spec_from_file_location("Preprocess", path)
            _preprocess = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_preprocess)
        else:
            submodules_path = [Path(path).as_posix()] + sys.path
            module_name = str(self.model_endpoint.preprocess_artifact).replace(".", "_")
            spec = importlib.util.spec_from_file_location(
                module_name, location=(Path(path) / "__init__.py").as_posix(),
                submodule_search_locations=submodules_path,
            )
            _preprocess = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = _preprocess
            spec.loader.exec_module(_preprocess)

        class PreprocessDelWrapper(_preprocess.Preprocess):
            def __del__(self):
                super_ = super(PreprocessDelWrapper, self)
                if callable(getattr(super_, "unload", None)):
                    try:
                        super_.unload()
                    except Exception as ex:
                        print("Failed unloading model: {}".format(ex))
                if callable(getattr(super_, "__del__", None)):
                    super_.__del__()

        Preprocess = PreprocessDelWrapper # noqa
        # override `send_request` method
        Preprocess.send_request = BasePreprocessRequest._preprocess_send_request
        # create preprocess class
        self._preprocess = Preprocess()
        # update the model endpoint on the instance we created
        self._preprocess.model_endpoint = self.model_endpoint
        # custom model load callback function
        if callable(getattr(self._preprocess, 'load', None)):
            self._model = self._preprocess.load(self._get_local_model_file())

    def preprocess(
            self,
            request: dict,
            state: dict,
            collect_custom_statistics_fn: Callable[[dict], None] = None,
    ) -> Optional[Any]:
        """
        Raise exception to report an error
        Return value will be passed to serving engine

        :param request: dictionary as recieved from the RestAPI
        :param state: Use state dict to store data passed to the post-processing function call.
            Usage example:
            >>> def preprocess(..., state):
                    state['preprocess_aux_data'] = [1,2,3]
            >>> def postprocess(..., state):
                    print(state['preprocess_aux_data'])
        :param collect_custom_statistics_fn: Optional, allows to send a custom set of key/values
            to the statictics collector servicd

            Usage example:
            >>> print(request)
            {"x0": 1, "x1": 2}
            >>> collect_custom_statistics_fn({"x0": 1, "x1": 2})

        :return: Object to be passed directly to the model inference
        """
        if self._preprocess is not None and hasattr(self._preprocess, 'preprocess'):
            return self._preprocess.preprocess(request, state, collect_custom_statistics_fn)
        return request

    def postprocess(
            self,
            data: Any,
            state: dict,
            collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Optional[dict]:
        """
        Raise exception to report an error
        Return value will be passed to serving engine

        :param data: object as recieved from the inference model function
        :param state: Use state dict to store data passed to the post-processing function call.
            Usage example:
            >>> def preprocess(..., state):
                    state['preprocess_aux_data'] = [1,2,3]
            >>> def postprocess(..., state):
                    print(state['preprocess_aux_data'])
        :param collect_custom_statistics_fn: Optional, allows to send a custom set of key/values
            to the statictics collector servicd

            Usage example:
            >>> collect_custom_statistics_fn({"y": 1})

        :return: Dictionary passed directly as the returned result of the RestAPI
        """
        if self._preprocess is not None and hasattr(self._preprocess, 'postprocess'):
            return self._preprocess.postprocess(data, state, collect_custom_statistics_fn)
        return data

    def process(
            self,
            data: Any,
            state: dict,
            collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        """
        The actual processing function. Can be sent to external service

        :param data: object as recieved from the preprocessing function
        :param state: Use state dict to store data passed to the post-processing function call.
            Usage example:
            >>> def preprocess(..., state):
                    state['preprocess_aux_data'] = [1,2,3]
            >>> def postprocess(..., state):
                    print(state['preprocess_aux_data'])
        :param collect_custom_statistics_fn: Optional, allows to send a custom set of key/values
            to the statictics collector servicd

            Usage example:
            >>> collect_custom_statistics_fn({"type": "classification"})

        :return: Object to be passed tp the post-processing function
        """
        pass

    def _get_local_model_file(self):
        if not self.model_endpoint.model_id:
            return None
        model_repo_object = Model(model_id=self.model_endpoint.model_id)
        return model_repo_object.get_local_copy()

    @classmethod
    def set_server_config(cls, server_config: dict) -> None:
        cls._server_config = server_config

    @classmethod
    def get_server_config(cls) -> dict:
        return cls._server_config

    @classmethod
    def validate_engine_type(cls, engine: str) -> bool:
        return engine in cls.__preprocessing_lookup

    @classmethod
    def get_engine_cls(cls, engine: str) -> Callable:
        return cls.__preprocessing_lookup.get(engine)

    @staticmethod
    def register_engine(engine_name: str, modules: Optional[List[str]] = None) -> Callable:
        """
        A decorator to register an annotation type name for classes deriving from Annotation
        """

        def wrapper(cls):
            cls.__preprocessing_lookup[engine_name] = cls
            return cls

        if modules:
            BasePreprocessRequest.__preprocessing_modules |= set(modules)

        return wrapper

    @staticmethod
    def load_modules() -> None:
        for m in BasePreprocessRequest.__preprocessing_modules:
            try:
                # silently fail
                import importlib
                importlib.import_module(m)
            except (ImportError, TypeError):
                pass

    @staticmethod
    def _preprocess_send_request(_, endpoint: str, version: str = None, data: dict = None) -> Optional[dict]:
        endpoint = "{}/{}".format(endpoint.strip("/"), version.strip("/")) if version else endpoint.strip("/")
        base_url = BasePreprocessRequest.get_server_config().get("base_serving_url")
        base_url = (base_url or BasePreprocessRequest._default_serving_base_url).strip("/")
        url = "{}/{}".format(base_url, endpoint.strip("/"))
        return_value = request_post(url, json=data, timeout=BasePreprocessRequest._timeout)
        if not return_value.ok:
            return None
        return return_value.json()


@BasePreprocessRequest.register_engine("triton", modules=["grpc", "tritonclient"])
class TritonPreprocessRequest(BasePreprocessRequest):
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        _content_lookup = {
            getattr(np, 'int', int): 'int_contents',
            np.uint8: 'uint_contents',
            np.int8: 'int_contents',
            np.int64: 'int64_contents',
            np.uint64: 'uint64_contents',
            np.int32: 'int_contents',
            np.uint: 'uint_contents',
            getattr(np, 'bool', bool): 'bool_contents',
            np.float32: 'fp32_contents',
            np.float64: 'fp64_contents',
        }
    _default_grpc_address = "127.0.0.1:8001"
    _default_grpc_compression = False
    _ext_grpc = None
    _ext_np_to_triton_dtype = None
    _ext_service_pb2 = None
    _ext_service_pb2_grpc = None
    is_preprocess_async = False
    is_process_async = True
    is_postprocess_async = False

    def __init__(self, model_endpoint: ModelEndpoint, task: Task = None):
        super(TritonPreprocessRequest, self).__init__(
            model_endpoint=model_endpoint, task=task)

        # load Triton Module
        if self._ext_grpc is None:
            from tritonclient.grpc import grpc  # noqa
            self._ext_grpc = grpc

        if self._ext_np_to_triton_dtype is None:
            from tritonclient.utils import np_to_triton_dtype  # noqa
            self._ext_np_to_triton_dtype = np_to_triton_dtype

        if self._ext_service_pb2 is None:
            from tritonclient.grpc.aio import service_pb2, service_pb2_grpc  # noqa
            self._ext_service_pb2 = service_pb2
            self._ext_service_pb2_grpc = service_pb2_grpc

        self._grpc_stub = {}

    async def process(
            self,
            data: Any,
            state: dict,
            collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        """
        The actual processing function.
        Detect gRPC server and send the request to it

        :param data: object as recieved from the preprocessing function
            If multiple inputs are needed, data is a list of numpy array
        :param state: Use state dict to store data passed to the post-processing function call.
            Usage example:
            >>> def preprocess(..., state):
                    state['preprocess_aux_data'] = [1,2,3]
            >>> def postprocess(..., state):
                    print(state['preprocess_aux_data'])
        :param collect_custom_statistics_fn: Optional, allows to send a custom set of key/values
            to the statictics collector servicd

            Usage example:
            >>> collect_custom_statistics_fn({"type": "classification"})

        :return: Object to be passed tp the post-processing function
        """
        # allow overriding the process method
        if self._preprocess is not None and hasattr(self._preprocess, "process"):
            return await self._preprocess.process(data, state, collect_custom_statistics_fn)

        # Create gRPC stub for communicating with the server
        triton_server_address = self._server_config.get("triton_grpc_server") or self._default_grpc_address
        if not triton_server_address:
            raise ValueError("External Triton gRPC server is not configured!")

        tid = threading.get_ident()
        if self._grpc_stub.get(tid):
            grpc_stub = self._grpc_stub.get(tid)
        else:
            channel_opt = []
            for k, v in os.environ.items():
                if str(k).startswith(self._grpc_env_conf_prefix):
                    try:
                        v = int(v)
                    except:  # noqa
                        try:
                            v = float(v)
                        except:  # noqa
                            pass
                    channel_opt.append(('grpc.{}'.format(k[len(self._grpc_env_conf_prefix):]), v))

            try:
                channel = self._ext_grpc.aio.insecure_channel(triton_server_address, options=channel_opt or None)
                grpc_stub = self._ext_service_pb2_grpc.GRPCInferenceServiceStub(channel)
                self._grpc_stub[tid] = grpc_stub
            except Exception as ex:
                raise ValueError("External Triton gRPC server misconfigured [{}]: {}".format(triton_server_address, ex))

        use_compression = self._server_config.get("triton_grpc_compression", self._default_grpc_compression)

        # Generate the request
        request = self._ext_service_pb2.ModelInferRequest()
        if self.model_endpoint.version:
            request.model_name = "{}_{}".format(
                self.model_endpoint.serving_url, self.model_endpoint.version).strip("/")
        else:
            request.model_name = str(self.model_endpoint.serving_url).strip("/")

        # we do not use the Triton model versions, we just assume a single version per endpoint
        request.model_version = "1"

        # make sure that if we have only one input we maintain backwards compatibility
        list_data = [data] if len(self.model_endpoint.input_name) == 1 else data

        # Populate the inputs in inference request
        for i_data, m_name, m_type, m_size in zip(
                list_data, self.model_endpoint.input_name,
                self.model_endpoint.input_type, self.model_endpoint.input_size
        ):
            # take the input data
            input_data = np.array(i_data, dtype=m_type)

            input0 = request.InferInputTensor()
            input0.name = m_name
            input_dtype = np.dtype(m_type).type
            input0.datatype = self._ext_np_to_triton_dtype(input_dtype)
            input0.shape.extend(input_data.shape)

            # to be inferred
            input_func = self._content_lookup.get(input_dtype)
            if not input_func:
                raise ValueError("Input type nt supported {}".format(input_dtype))
            input_func = getattr(input0.contents, input_func)
            input_func[:] = input_data.flatten()

            # push into request
            request.inputs.extend([input0])

        # Populate the outputs in the inference request
        for m_name in self.model_endpoint.output_name:
            output0 = request.InferRequestedOutputTensor()
            output0.name = m_name
            request.outputs.extend([output0])

        # send infer request over gRPC
        compression = None
        try:
            compression = self._ext_grpc.Compression.Gzip if use_compression \
                else self._ext_grpc.Compression.NoCompression
            response = await grpc_stub.ModelInfer(request, compression=compression, timeout=self._timeout)
        except Exception as ex:
            print("Exception calling Triton RPC function: "
                  "request_inputs={}, ".format([(r.name, r.shape, r.datatype) for r in (request.inputs or [])]) +
                  f"triton_address={triton_server_address}, compression={compression}, timeout={self._timeout}:\n{ex}")
            raise

        # process result
        output_results = []
        index = 0
        for i, output in enumerate(response.outputs):
            shape = []
            for value in output.shape:
                shape.append(value)
            output_results.append(
                np.frombuffer(
                    response.raw_output_contents[index],
                    dtype=self.model_endpoint.output_type[min(i, len(self.model_endpoint.output_type)-1)]
                )
            )
            output_results[-1] = np.resize(output_results[-1], shape)
            index += 1

        # if we have a single matrix, return it as is
        return output_results[0] if index == 1 else output_results


@BasePreprocessRequest.register_engine("sklearn", modules=["joblib", "sklearn"])
class SKLearnPreprocessRequest(BasePreprocessRequest):
    def __init__(self, model_endpoint: ModelEndpoint, task: Task = None):
        super(SKLearnPreprocessRequest, self).__init__(
            model_endpoint=model_endpoint, task=task)
        if self._model is None:
            # get model
            import joblib  # noqa
            self._model = joblib.load(filename=self._get_local_model_file())

    def process(self, data: Any, state: dict, collect_custom_statistics_fn: Callable[[dict], None] = None) -> Any:
        """
        The actual processing function.
        We run the model in this context
        """
        return self._model.predict(data)


@BasePreprocessRequest.register_engine("xgboost", modules=["xgboost"])
class XGBoostPreprocessRequest(BasePreprocessRequest):
    def __init__(self, model_endpoint: ModelEndpoint, task: Task = None):
        super(XGBoostPreprocessRequest, self).__init__(
            model_endpoint=model_endpoint, task=task)
        if self._model is None:
            # get model
            import xgboost  # noqa
            self._model = xgboost.Booster()
            self._model.load_model(self._get_local_model_file())

    def process(self, data: Any, state: dict, collect_custom_statistics_fn: Callable[[dict], None] = None) -> Any:
        """
        The actual processing function.
        We run the model in this context
        """
        return self._model.predict(data)


@BasePreprocessRequest.register_engine("lightgbm", modules=["lightgbm"])
class LightGBMPreprocessRequest(BasePreprocessRequest):
    def __init__(self, model_endpoint: ModelEndpoint, task: Task = None):
        super(LightGBMPreprocessRequest, self).__init__(
            model_endpoint=model_endpoint, task=task)
        if self._model is None:
            # get model
            import lightgbm  # noqa
            self._model = lightgbm.Booster(model_file=self._get_local_model_file())

    def process(self, data: Any, state: dict, collect_custom_statistics_fn: Callable[[dict], None] = None) -> Any:
        """
        The actual processing function.
        We run the model in this context
        """
        return self._model.predict(data)


@BasePreprocessRequest.register_engine("custom")
class CustomPreprocessRequest(BasePreprocessRequest):
    def __init__(self, model_endpoint: ModelEndpoint, task: Task = None):
        super(CustomPreprocessRequest, self).__init__(
            model_endpoint=model_endpoint, task=task)

    def process(self, data: Any, state: dict, collect_custom_statistics_fn: Callable[[dict], None] = None) -> Any:
        """
        The actual processing function.
        We run the process in this context
        """
        if self._preprocess is not None and hasattr(self._preprocess, 'process'):
            return self._preprocess.process(data, state, collect_custom_statistics_fn)
        return None


@BasePreprocessRequest.register_engine("custom_async")
class CustomAsyncPreprocessRequest(BasePreprocessRequest):
    is_preprocess_async = True
    is_process_async = True
    is_postprocess_async = True
    asyncio_to_thread = None

    def __init__(self, model_endpoint: ModelEndpoint, task: Task = None):
        super(CustomAsyncPreprocessRequest, self).__init__(
            model_endpoint=model_endpoint, task=task)
        # load asyncio only when needed, basically python < 3.10 does not supported to_thread
        if CustomAsyncPreprocessRequest.asyncio_to_thread is None:
            from asyncio import to_thread as asyncio_to_thread
            CustomAsyncPreprocessRequest.asyncio_to_thread = asyncio_to_thread
        # override `send_request` method with the async version
        self._preprocess.__class__.send_request = CustomAsyncPreprocessRequest._preprocess_send_request

    async def preprocess(
            self,
            request: dict,
            state: dict,
            collect_custom_statistics_fn: Callable[[dict], None] = None,
    ) -> Optional[Any]:
        """
        Raise exception to report an error
        Return value will be passed to serving engine

        :param request: dictionary as recieved from the RestAPI
        :param state: Use state dict to store data passed to the post-processing function call.
            Usage example:
            >>> def preprocess(..., state):
                    state['preprocess_aux_data'] = [1,2,3]
            >>> def postprocess(..., state):
                    print(state['preprocess_aux_data'])
        :param collect_custom_statistics_fn: Optional, allows to send a custom set of key/values
            to the statictics collector servicd

            Usage example:
            >>> print(request)
            {"x0": 1, "x1": 2}
            >>> collect_custom_statistics_fn({"x0": 1, "x1": 2})

        :return: Object to be passed directly to the model inference
        """
        if self._preprocess is not None and hasattr(self._preprocess, 'preprocess'):
            return await self._preprocess.preprocess(request, state, collect_custom_statistics_fn)
        return request

    async def postprocess(
            self,
            data: Any,
            state: dict,
            collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Optional[dict]:
        """
        Raise exception to report an error
        Return value will be passed to serving engine

        :param data: object as recieved from the inference model function
        :param state: Use state dict to store data passed to the post-processing function call.
            Usage example:
            >>> def preprocess(..., state):
                    state['preprocess_aux_data'] = [1,2,3]
            >>> def postprocess(..., state):
                    print(state['preprocess_aux_data'])
        :param collect_custom_statistics_fn: Optional, allows to send a custom set of key/values
            to the statictics collector servicd

            Usage example:
            >>> collect_custom_statistics_fn({"y": 1})

        :return: Dictionary passed directly as the returned result of the RestAPI
        """
        if self._preprocess is not None and hasattr(self._preprocess, 'postprocess'):
            return await self._preprocess.postprocess(data, state, collect_custom_statistics_fn)
        return data

    async def process(self, data: Any, state: dict, collect_custom_statistics_fn: Callable[[dict], None] = None) -> Any:
        """
        The actual processing function.
        We run the process in this context
        """
        if self._preprocess is not None and hasattr(self._preprocess, 'process'):
            return await self._preprocess.process(data, state, collect_custom_statistics_fn)
        return None

    @staticmethod
    async def _preprocess_send_request(_, endpoint: str, version: str = None, data: dict = None) -> Optional[dict]:
        endpoint = "{}/{}".format(endpoint.strip("/"), version.strip("/")) if version else endpoint.strip("/")
        base_url = BasePreprocessRequest.get_server_config().get("base_serving_url")
        base_url = (base_url or BasePreprocessRequest._default_serving_base_url).strip("/")
        url = "{}/{}".format(base_url, endpoint.strip("/"))
        return_value = await CustomAsyncPreprocessRequest.asyncio_to_thread(
            request_post, url, json=data, timeout=BasePreprocessRequest._timeout)
        if not return_value.ok:
            return None
        return return_value.json()


class VllmEngine(Singleton):
    # is_already_loaded = False

    def __init__(self) -> None:
        from vllm.logger import init_logger
        self.logger = init_logger('vllm.entrypoints.openai.api_server')

        import socket
        import prometheus_client
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if not s.connect_ex(('localhost', 8000)) == 0:
                prometheus_client.start_http_server(8000)

    def load_engine(
        self,
        name: str,
        model_path: str,
        vllm_model_config: dict,
        chat_settings: dict
    ) -> None:

        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.entrypoints.logger import RequestLogger
        from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
        from vllm.entrypoints.openai.serving_classification import ServingClassification
        from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
        from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
        from vllm.entrypoints.openai.serving_engine import OpenAIServing
        from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
        from vllm.entrypoints.openai.serving_pooling import OpenAIServingPooling
        from vllm.entrypoints.openai.serving_score import ServingScores
        from vllm.entrypoints.openai.serving_tokenization import OpenAIServingTokenization
        from vllm.entrypoints.openai.serving_transcription import (
            OpenAIServingTranscription,
            OpenAIServingTranslation
        )
        from vllm.entrypoints.chat_utils import (
            load_chat_template,
            resolve_hf_chat_template,
            resolve_mistral_chat_template
        )
        from vllm.transformers_utils.tokenizer import MistralTokenizer
        from vllm.usage.usage_lib import UsageContext
        from vllm import envs

        # if self.is_already_loaded:
        #     self.add_models(name=name, model_path=model_path)
        #     return None

        vllm_engine_config = json.loads(os.environ.get("VLLM_ENGINE_ARGS").replace("'", ""))
        vllm_engine_config["model"] = model_path
        vllm_engine_config["served_model_name"] = name
        engine_args = AsyncEngineArgs(**vllm_engine_config)
        if envs.VLLM_USE_V1:
            async_engine_client = AsyncLLM.from_engine_args(
                engine_args,
                usage_context=UsageContext.OPENAI_API_SERVER
            )
        else:
            async_engine_client = AsyncLLMEngine.from_engine_args(
                engine_args,
                usage_context=UsageContext.OPENAI_API_SERVER
            )
        model_config = async_engine_client.engine.get_model_config()
        request_logger = RequestLogger(
            max_log_len=vllm_model_config["max_log_len"]
        )
        resolved_chat_template = load_chat_template(vllm_model_config["chat_template"])
        if resolved_chat_template is not None:
            # Get the tokenizer to check official template
            tokenizer = async_engine_client.engine.get_tokenizer()

            if isinstance(tokenizer, MistralTokenizer):
                # The warning is logged in resolve_mistral_chat_template.
                resolved_chat_template = resolve_mistral_chat_template(
                    chat_template=resolved_chat_template)
            else:
                hf_chat_template = resolve_hf_chat_template(
                    tokenizer=tokenizer,
                    chat_template=None,
                    tools=None,
                    model_config=model_config,
                )

                if hf_chat_template != resolved_chat_template:
                    self.logger.warning(
                        "Using supplied chat template: %s\n"
                        "It is different from official chat template '%s'. "
                        "This discrepancy may lead to performance degradation.",
                        resolved_chat_template, model_path)
        self.openai_serving_models = OpenAIServingModels(
            async_engine_client,
            model_config,
            [BaseModelPath(name=name, model_path=model_path)],
            lora_modules=vllm_model_config["lora_modules"],
            prompt_adapters=vllm_model_config["prompt_adapters"],
        )

        import asyncio
        asyncio.create_task(self.openai_serving_models.init_static_loras())

        self.openai_serving = OpenAIServing(
            async_engine_client,
            model_config,
            self.openai_serving_models,
            request_logger=request_logger,
            return_tokens_as_token_ids=vllm_model_config["return_tokens_as_token_ids"],
            enable_force_include_usage=chat_settings["enable_force_include_usage"]
        )
        self.openai_serving_chat = OpenAIServingChat(
            async_engine_client,
            model_config,
            self.openai_serving_models,
            response_role=vllm_model_config["response_role"],
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=chat_settings["chat_template_content_format"],
            return_tokens_as_token_ids=vllm_model_config["return_tokens_as_token_ids"],
            reasoning_parser=chat_settings["reasoning_parser"],
            enable_auto_tools=chat_settings["enable_auto_tools"],
            expand_tools_even_if_tool_choice_none=chat_settings["expand_tools_even_if_tool_choice_none"],
            tool_parser=chat_settings["tool_parser"],
            enable_prompt_tokens_details=chat_settings["enable_prompt_tokens_details"],
            enable_force_include_usage=chat_settings["enable_force_include_usage"]
        ) if model_config.runner_type == "generate" else None
        self.openai_serving_completion = OpenAIServingCompletion(
            async_engine_client,
            model_config,
            self.openai_serving_models,
            request_logger=request_logger,
            return_tokens_as_token_ids=vllm_model_config["return_tokens_as_token_ids"],
            enable_force_include_usage=chat_settings["enable_force_include_usage"]
        ) if model_config.runner_type == "generate" else None
        self.openai_serving_pooling = OpenAIServingPooling(
            async_engine_client,
            model_config,
            self.openai_serving_models,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=chat_settings["chat_template_content_format"]
        ) if model_config.runner_type == "pooling" else None
        self.openai_serving_embedding = OpenAIServingEmbedding(
            async_engine_client,
            model_config,
            self.openai_serving_models,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=chat_settings["chat_template_content_format"]
        ) if model_config.task == "embed" else None
        self.openai_serving_classification = ServingClassification(
            async_engine_client,
            model_config,
            self.openai_serving_models,
            request_logger=request_logger
        ) if model_config.task == "classify" else None
        enable_serving_reranking = (model_config.task == "classify" and getattr(model_config.hf_config, "num_labels", 0) == 1)
        # self.jinaai_serving_reranking = ServingScores(
        #     async_engine_client,
        #     model_config,
        #     self.openai_serving_models,
        #     request_logger=request_logger
        # ) if enable_serving_reranking else None
        self.openai_serving_scores = ServingScores(
            async_engine_client,
            model_config,
            self.openai_serving_models,
            request_logger=request_logger
        ) if (model_config.task == "embed" or enable_serving_reranking) else None
        self.openai_serving_tokenization = OpenAIServingTokenization(
            async_engine_client,
            model_config,
            self.openai_serving_models,
            request_logger=request_logger,
            chat_template=vllm_model_config["chat_template"],
            chat_template_content_format=chat_settings["chat_template_content_format"]
        )
        self.openai_serving_transcription = OpenAIServingTranscription(
            async_engine_client,
            model_config,
            self.openai_serving_models,
            request_logger=request_logger
        ) if model_config.runner_type == "transcription" else None
        self.openai_serving_translation = OpenAIServingTranslation(
            async_engine_client,
            model_config,
            self.openai_serving_models,
            request_logger=request_logger,
        ) if model_config.runner_type == "transcription" else None
        self.task = model_config.task
        self.enable_server_load_tracking = False
        # state.server_load_metrics = 0
        self.logger.info("vLLM Engine was successfully initialized")
        # self.is_already_loaded = True
        return None

    def add_models(self, name: str, model_path: str) -> None:
        from vllm.entrypoints.openai.serving_models import BaseModelPath
        self.openai_serving_models.base_model_paths.append(
            BaseModelPath(
                name=name, model_path=model_path
            )
        )
        self.logger.info("Model {} was added to vllm engine".format(name))
        # TODO: RESTART ENGINE
        return None
    
    def remove_model(self, name: str) -> None:
        self.openai_serving_models.base_model_paths = [
            model for model in self.openai_serving_models.base_model_paths
            if model.name != name
        ]
        self.logger.info("Model {} was removed from vllm engine".format(name))
        # TODO: RESTART ENGINE
        return None

    async def tokenize(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        from vllm.entrypoints.openai.protocol import ErrorResponse, TokenizeResponse
        from fastapi.responses import JSONResponse
        from typing_extensions import assert_never
        request, raw_request = data["request"], data["raw_request"]
        handler = self.openai_serving_tokenization
        if handler is None:
            return self.openai_serving.create_error_response(
                message="The model does not support Tokenization API"
            )
        generator = await handler.create_tokenize(request=request, raw_request=raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        elif isinstance(generator, TokenizeResponse):
            return JSONResponse(content=generator.model_dump())
        assert_never(generator)

    async def detokenize(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        from vllm.entrypoints.openai.protocol import ErrorResponse, DetokenizeResponse
        from fastapi.responses import JSONResponse
        from typing_extensions import assert_never
        request, raw_request = data["request"], data["raw_request"]
        handler = self.openai_serving_tokenization
        if handler is None:
            return self.openai_serving.create_error_response(
                message="The model does not support Detokenization API"
            )
        generator = await handler.create_detokenize(request=request, raw_request=raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        elif isinstance(generator, DetokenizeResponse):
            return JSONResponse(content=generator.model_dump())
        assert_never(generator)

    async def models(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        from fastapi.responses import JSONResponse
        models_ = await self.openai_serving_models.show_available_models()
        return JSONResponse(content=models_.model_dump())

    async def show_version(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        from vllm.version import __version__ as VLLM_VERSION
        from fastapi.responses import JSONResponse
        ver = {"version": VLLM_VERSION}
        return JSONResponse(content=ver)

    async def completions(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        from vllm.entrypoints.openai.protocol import ErrorResponse, CompletionResponse
        from fastapi.responses import JSONResponse, StreamingResponse
        request, raw_request = data["request"], data["raw_request"]
        handler = self.openai_serving_completion
        if handler is None:
            return self.openai_serving.create_error_response(
                message="The model does not support Completions API"
            )
        generator = await handler.create_completion(request=request, raw_request=raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        elif isinstance(generator, CompletionResponse):
            return JSONResponse(content=generator.model_dump())
        return StreamingResponse(content=generator, media_type="text/event-stream")

    async def chat_completions(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        from vllm.entrypoints.openai.protocol import ErrorResponse, ChatCompletionResponse
        from fastapi.responses import JSONResponse, StreamingResponse
        request, raw_request = data["request"], data["raw_request"]
        handler = self.openai_serving_chat
        if handler is None:
            return self.openai_serving.create_error_response(
                message="The model does not support Chat Completions API"
            )
        generator = await handler.create_chat_completion(request=request, raw_request=raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        elif isinstance(generator, ChatCompletionResponse):
            return JSONResponse(content=generator.model_dump())
        return StreamingResponse(content=generator, media_type="text/event-stream")

    async def embedding(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        from vllm.entrypoints.openai.protocol import ErrorResponse, EmbeddingResponse
        from fastapi.responses import JSONResponse
        from typing_extensions import assert_never
        request, raw_request = data["request"], data["raw_request"]
        handler = self.openai_serving_embedding
        if handler is None:
            return self.openai_serving.create_error_response(
                message="The model does not support Embeddings API"
            )
        generator = await handler.create_embedding(request=request, raw_request=raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        elif isinstance(generator, EmbeddingResponse):
            return JSONResponse(content=generator.model_dump())
        assert_never(generator)

    async def pooling(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        from vllm.entrypoints.openai.protocol import ErrorResponse, PoolingResponse
        from fastapi.responses import JSONResponse
        from typing_extensions import assert_never
        request, raw_request = data["request"], data["raw_request"]
        handler = self.openai_serving_pooling
        if handler is None:
            return self.openai_serving.create_error_response(
                message="The model does not support Pooling API"
            )
        generator = await handler.create_pooling(request=request, raw_request=raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        elif isinstance(generator, PoolingResponse):
            return JSONResponse(content=generator.model_dump())
        assert_never(generator)

    async def classify(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        from vllm.entrypoints.openai.protocol import ErrorResponse, ClassificationResponse
        from fastapi.responses import JSONResponse
        from typing_extensions import assert_never
        request, raw_request = data["request"], data["raw_request"]
        handler = self.openai_serving_classification
        if handler is None:
            return self.openai_serving.create_error_response(
                message="The model does not support Classification API"
            )
        generator = await handler.create_classify(request=request, raw_request=raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        elif isinstance(generator, ClassificationResponse):
            return JSONResponse(content=generator.model_dump())
        assert_never(generator)

    async def score(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        from vllm.entrypoints.openai.protocol import ErrorResponse, ScoreResponse
        from fastapi.responses import JSONResponse
        from typing_extensions import assert_never
        request, raw_request = data["request"], data["raw_request"]
        handler = self.openai_serving_scores
        if handler is None:
            return self.openai_serving.create_error_response(
                message="The model does not support Score API"
            )
        generator = await handler.create_score(request=request, raw_request=raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        elif isinstance(generator, ScoreResponse):
            return JSONResponse(content=generator.model_dump())
        assert_never(generator)

    async def audio_transcriptions(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        from vllm.entrypoints.openai.protocol import ErrorResponse, TranscriptionResponse
        from fastapi.responses import JSONResponse, StreamingResponse
        request, raw_request = data["request"], data["raw_request"]
        handler = self.openai_serving_transcription
        if handler is None:
            return self.openai_serving.create_error_response(
                message="The model does not support Transcriptions API"
            )
        audio_data = await request.file.read()
        generator = await handler.create_transcription(audio_data, request, raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        elif isinstance(generator, TranscriptionResponse):
            return JSONResponse(content=generator.model_dump())
        return StreamingResponse(content=generator, media_type="text/event-stream")

    async def audio_translations(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        from vllm.entrypoints.openai.protocol import ErrorResponse, TranslationResponse
        from fastapi.responses import JSONResponse, StreamingResponse
        request, raw_request = data["request"], data["raw_request"]
        handler = self.openai_serving_translation
        if handler is None:
            return self.openai_serving.create_error_response(
                message="The model does not support Translations API"
            )
        audio_data = await request.file.read()
        generator = await handler.create_translation(audio_data, request, raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        elif isinstance(generator, TranslationResponse):
            return JSONResponse(content=generator.model_dump())
        return StreamingResponse(content=generator, media_type="text/event-stream")

    async def rerank(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        from vllm.entrypoints.openai.protocol import ErrorResponse, RerankResponse
        from fastapi.responses import JSONResponse
        from typing_extensions import assert_never
        request, raw_request = data["request"], data["raw_request"]
        handler = self.openai_serving_scores
        if handler is None:
            return self.openai_serving.create_error_response(
                message="The model does not support Rerank (Score) API"
            )
        generator = await handler.do_rerank(request=request, raw_request=raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        elif isinstance(generator, RerankResponse):
            return JSONResponse(content=generator.model_dump())
        assert_never(generator)

@BasePreprocessRequest.register_engine("vllm", modules=["vllm", "fastapi"])
class VllmPreprocessRequest(BasePreprocessRequest):
    is_preprocess_async = True
    is_process_async = True
    is_postprocess_async = True
    asyncio_to_thread = None
    _vllm_engine = None

    def __init__(self, model_endpoint: ModelEndpoint, task: Task = None):
        super(VllmPreprocessRequest, self).__init__(
            model_endpoint=model_endpoint, task=task)
        self._vllm_engine = VllmEngine()
        self._vllm_engine.load_engine(
            name=model_endpoint.serving_url,
            model_path=self._get_local_model_file(),
            **self._model
        )

        if VllmPreprocessRequest.asyncio_to_thread is None:
            from asyncio import to_thread as asyncio_to_thread
            VllmPreprocessRequest.asyncio_to_thread = asyncio_to_thread

        # override `send_request` method with the async version
        self._preprocess.__class__.send_request = VllmPreprocessRequest._preprocess_send_request

    async def preprocess(
            self,
            request: dict,
            state: dict,
            collect_custom_statistics_fn: Callable[[dict], None] = None,
    ) -> Optional[Any]:
        """
        Raise exception to report an error
        Return value will be passed to serving engine

        :param request: dictionary as recieved from the RestAPI
        :param state: Use state dict to store data passed to the post-processing function call.
            Usage example:
            >>> def preprocess(..., state):
                    state['preprocess_aux_data'] = [1,2,3]
            >>> def postprocess(..., state):
                    print(state['preprocess_aux_data'])
        :param collect_custom_statistics_fn: Optional, allows to send a custom set of key/values
            to the statictics collector servicd

            Usage example:
            >>> print(request)
            {"x0": 1, "x1": 2}
            >>> collect_custom_statistics_fn({"x0": 1, "x1": 2})

        :return: Object to be passed directly to the model inference
        """
        if self._preprocess is not None and hasattr(self._preprocess, 'preprocess'):
            return await self._preprocess.preprocess(request, state, collect_custom_statistics_fn)
        return request

    async def postprocess(
            self,
            data: Any,
            state: dict,
            collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Optional[dict]:
        """
        Raise exception to report an error
        Return value will be passed to serving engine

        :param data: object as recieved from the inference model function
        :param state: Use state dict to store data passed to the post-processing function call.
            Usage example:
            >>> def preprocess(..., state):
                    state['preprocess_aux_data'] = [1,2,3]
            >>> def postprocess(..., state):
                    print(state['preprocess_aux_data'])
        :param collect_custom_statistics_fn: Optional, allows to send a custom set of key/values
            to the statictics collector servicd

            Usage example:
            >>> collect_custom_statistics_fn({"y": 1})

        :return: Dictionary passed directly as the returned result of the RestAPI
        """
        if self._preprocess is not None and hasattr(self._preprocess, 'postprocess'):
            return await self._preprocess.postprocess(data, state, collect_custom_statistics_fn)
        return data

    async def tokenize(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        return await self._vllm_engine.tokenize(
            data=data,
            state=state,
            collect_custom_statistics_fn=collect_custom_statistics_fn
        )

    async def detokenize(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        return await self._vllm_engine.detokenize(
            data=data,
            state=state,
            collect_custom_statistics_fn=collect_custom_statistics_fn
        )

    async def v1_models(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        return await self._vllm_engine.models(
            data=data,
            state=state,
            collect_custom_statistics_fn=collect_custom_statistics_fn
        )

    async def version(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        return await self._vllm_engine.show_version(
            data=data,
            state=state,
            collect_custom_statistics_fn=collect_custom_statistics_fn
        )

    async def v1_chat_completions(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        return await self._vllm_engine.chat_completions(
            data=data,
            state=state,
            collect_custom_statistics_fn=collect_custom_statistics_fn
        )

    async def v1_completions(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        return await self._vllm_engine.completions(
            data=data,
            state=state,
            collect_custom_statistics_fn=collect_custom_statistics_fn
        )

    async def v1_embeddings(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        return await self._vllm_engine.embedding(
            data=data,
            state=state,
            collect_custom_statistics_fn=collect_custom_statistics_fn
        )

    async def pooling(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        return await self._vllm_engine.pooling(
            data=data,
            state=state,
            collect_custom_statistics_fn=collect_custom_statistics_fn
        )
    
    async def classify(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        return await self._vllm_engine.classify(
            data=data,
            state=state,
            collect_custom_statistics_fn=collect_custom_statistics_fn
        )

    async def score(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        return await self._vllm_engine.score(
            data=data,
            state=state,
            collect_custom_statistics_fn=collect_custom_statistics_fn
        )

    async def v1_audio_transcriptions(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        return await self._vllm_engine.audio_transcriptions(
            data=data,
            state=state,
            collect_custom_statistics_fn=collect_custom_statistics_fn
        )

    async def v1_audio_translations(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        return await self._vllm_engine.audio_translations(
            data=data,
            state=state,
            collect_custom_statistics_fn=collect_custom_statistics_fn
        )

    async def rerank(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Callable[[dict], None] = None
    ) -> Any:
        return await self._vllm_engine.rerank(
            data=data,
            state=state,
            collect_custom_statistics_fn=collect_custom_statistics_fn
        )

    @staticmethod
    async def _preprocess_send_request(_, endpoint: str, version: str = None, data: dict = None) -> Optional[dict]:
        endpoint = "/openai/{}".format(endpoint.strip("/"))
        base_url = BasePreprocessRequest.get_server_config().get("base_serving_url")
        base_url = (base_url or BasePreprocessRequest._default_serving_base_url).strip("/")
        url = "{}/{}".format(base_url, endpoint.strip("/"))
        return_value = await VllmPreprocessRequest.asyncio_to_thread(
            request_post, url, json=data, timeout=BasePreprocessRequest._timeout)
        if not return_value.ok:
            return None
        return return_value.json()
