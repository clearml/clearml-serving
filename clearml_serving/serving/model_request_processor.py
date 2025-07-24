import json
import os
import gc
import uuid
from collections import deque
from pathlib import Path
from random import random
from time import sleep, time
from typing import Optional, Union, Dict, List, TYPE_CHECKING
import itertools
import threading
from multiprocessing import Lock
import asyncio
from numpy.random import choice

from clearml import Task, Model, InputModel
from clearml.utilities.dicts import merge_dicts, cast_str_to_bool
from clearml.storage.util import hash_dict
from .preprocess_service import BasePreprocessRequest
from .endpoints import ModelEndpoint, ModelMonitoring, CanaryEP, EndpointMetricLogging

try:
    import torch
except ImportError:
    torch = None


class ModelRequestProcessorException(Exception):
    def __init__(self, message):
        super().__init__(message)


class EndpointNotFoundException(ModelRequestProcessorException):
    def __init__(self, message):
        super().__init__(message)


class EndpointModelLoadException(ModelRequestProcessorException):
    def __init__(self, message):
        super().__init__(message)


class EndpointBackendEngineException(ModelRequestProcessorException):
    def __init__(self, message):
        super().__init__(message)


class ServingInitializationException(Exception):
    def __init__(self, message):
        super().__init__(message)


class MetricLoggingException(Exception):
    def __init__(self, message):
        super().__init__(message)


class FastWriteCounter(object):
    def __init__(self):
        self._counter_inc = itertools.count()
        self._counter_dec = itertools.count()

    def inc(self):
        next(self._counter_inc)

    def dec(self):
        next(self._counter_dec)

    def value(self):
        return next(self._counter_inc) - next(self._counter_dec)


class FastSimpleQueue(object):
    _default_wait_timeout = 10

    def __init__(self):
        self._deque = deque()
        # Notify not_empty whenever an item is added to the queue; a
        # thread waiting to get is notified then.
        self._not_empty = threading.Event()
        self._last_notify = time()

    def put(self, a_object, block=True):
        self._deque.append(a_object)
        if time() - self._last_notify > self._default_wait_timeout:
            self._not_empty.set()
            self._last_notify = time()

    def get(self, block=True):
        while True:
            try:
                return self._deque.popleft()
            except IndexError:
                if not block:
                    return None
                # wait until signaled
                try:
                    if self._not_empty.wait(timeout=self._default_wait_timeout):
                        self._not_empty.clear()
                except Exception as ex:  # noqa
                    pass


class ModelRequestProcessor(object):
    _system_tag = "serving-control-plane"
    _kafka_topic = "clearml_inference_stats"
    _config_key_serving_base_url = "serving_base_url"
    _config_key_triton_grpc = "triton_grpc_server"
    _config_key_triton_compression = "triton_grpc_compression"
    _config_key_kafka_stats = "kafka_service_server"
    _config_key_def_metric_freq = "metric_logging_freq"

    def __init__(
            self,
            task_id: Optional[str] = None,
            update_lock_guard: Optional[Lock] = None,
            name: Optional[str] = None,
            project: Optional[str] = None,
            tags: Optional[List[str]] = None,
            force_create: bool = False
    ) -> None:
        """
        ModelRequestProcessor constructor

        :param task_id: Optional specify existing Task ID of the ServingService
        :param update_lock_guard: If provided use external (usually multiprocess) lock guard for updates
        :param name: Optional name current serving service
        :param project: Optional select project for the current serving service
        :param tags: Optional add tags to the serving service
        :param force_create: force_create if provided, ignore task_id and create a new serving Task
        """
        self._task = self._create_task(name=name, project=project, tags=tags) \
            if force_create else self._get_control_plane_task(task_id=task_id, name=name, project=project, tags=tags)
        self._endpoints = dict()  # type: Dict[str, ModelEndpoint]
        self._model_monitoring = dict()  # type: Dict[str, ModelMonitoring]
        self._model_monitoring_versions = dict()  # type: Dict[str, Dict[int, str]]
        self._model_monitoring_endpoints = dict()  # type: Dict[str, ModelEndpoint]
        self._model_monitoring_update_request = True
        # Dict[base_serve_url, Dict[version, model_id]]
        self._canary_endpoints = dict()  # type: Dict[str, CanaryEP]
        self._canary_route = dict()  # type: Dict[str, dict]
        self._engine_processor_lookup = dict()  # type: Dict[str, BasePreprocessRequest]
        self._metric_logging = dict()  # type: Dict[str, EndpointMetricLogging]
        self._endpoint_metric_logging = dict()  # type: Dict[str, EndpointMetricLogging]
        self._last_update_hash = None
        self._sync_daemon_thread = None
        self._stats_sending_thread = None
        self._stats_queue = FastSimpleQueue()
        # this is used for Fast locking mechanisms (so we do not actually need to use Locks)
        self._update_lock_flag = False
        self._request_processing_state = FastWriteCounter()
        self._update_lock_guard = update_lock_guard or threading.Lock()
        self._instance_task = None
        # serving server config
        self._configuration = {}
        # deserialized values go here
        self._kafka_stats_url = None
        self._triton_grpc = None
        self._triton_grpc_compression = None
        self._serving_base_url = None
        self._metric_log_freq = None
        self._endpoint_telemetry = {}
        self._enable_endpoint_telemetry = os.environ.get("CLEARML_ENABLE_ENDPOINT_TELEMETRY", "1") != "0"

    def on_request_endpoint_telemetry(self, base_url=None, version=None):
        try:
            if not self._enable_endpoint_telemetry:
                return

            url = self._normalize_endpoint_url(base_url, version)
            endpoint_telemetry = self._endpoint_telemetry.get(url)
            if endpoint_telemetry:
                endpoint_telemetry.on_request()
        except Exception:
            pass

    def on_response_endpoint_telemetry(self, base_url=None, version=None):
        try:
            if not self._enable_endpoint_telemetry:
                return

            url = self._normalize_endpoint_url(base_url, version)
            endpoint_telemetry = self._endpoint_telemetry.get(url)
            if endpoint_telemetry:
                endpoint_telemetry.on_response()
        except Exception:
            pass

    def _start_model_endpoint_telemetry(self, endpoint):
        try:
            if not self._enable_endpoint_telemetry:
                return

            from clearml.router.endpoint_telemetry import EndpointTelemetry
            from clearml.utilities.networking import get_private_ip

            url = self._normalize_endpoint_url(endpoint.serving_url, endpoint.version)
            try:
                private_ip = get_private_ip()
            except Exception:
                private_ip = "localhost"
            http_url = "http://{}:{}/serve/{}".format(private_ip, os.environ.get("SERVING_PORT", "8080"), url)
            model = InputModel(model_id=endpoint.model_id)
            model_url = "{}/projects/{}/models/{}/output/general".format(
                self._task._get_app_server().rstrip("/"), model.project, model.id
            )
            if url not in self._endpoint_telemetry:
                container_id = str(uuid.uuid4()).replace("-", "")
                self._endpoint_telemetry[url] = EndpointTelemetry(
                    endpoint_name=model.name,
                    model_name=model.name,
                    model=model.id,
                    model_url=model_url,
                    model_source="ClearML",
                    model_version=endpoint.version,
                    tags=model.tags,
                    system_tags=model.system_tags,
                    input_size=endpoint.input_size,
                    input_type=endpoint.input_type,
                    endpoint_url=http_url,
                    preprocess_artifact=endpoint.preprocess_artifact,
                    container_id=container_id,
                    force_register=True
                )
            else:
                self._endpoint_telemetry[url].update(
                    endpoint_name=model.name,
                    model_name=model.name,
                    model=model.id,
                    model_url=model_url,
                    model_version=endpoint.version,
                    tags=model.tags,
                    system_tags=model.system_tags,
                    input_size=endpoint.input_size,
                    input_type=endpoint.input_type,
                    endpoint_url=http_url,
                    preprocess_artifact=endpoint.preprocess_artifact
                )
        except Exception as e:
            print("Error: Could not start endpoint telemetry '{}'".format(e))

    def _stop_model_endpoint_telemetry(self, endpoint):
        try:
            if not self._enable_endpoint_telemetry:
                return
            url = self._normalize_endpoint_url(endpoint.serving_url, endpoint.version)
            endpoint_telemetry = self._endpoint_telemetry.pop(url, None)
            if endpoint_telemetry:
                endpoint_telemetry.stop()
        except Exception as e:
            print("Error: Could not stop endpoint telemetry '{}'".format(e))

    async def process_request(self, base_url: str, version: str, request_body: dict, serve_type: str) -> dict:
        """
        Process request coming in,
        Raise Value error if url does not match existing endpoints
        """
        self._request_processing_state.inc()
        # check if we need to stall
        if self._update_lock_flag:
            self._request_processing_state.dec()
            while self._update_lock_flag:
                await asyncio.sleep(0.5+random())
            # retry to process
            return await self.process_request(
                base_url=base_url,
                version=version,
                request_body=request_body,
                serve_type=serve_type
            )

        processor = None
        url = None
        try:
            # normalize url and version
            url = self._normalize_endpoint_url(base_url, version)

            # check canary
            canary_url = self._process_canary(base_url=url)
            if canary_url:
                url = canary_url

            ep = self._endpoints.get(url, None) or self._model_monitoring_endpoints.get(url, None)
            if not ep:
                raise EndpointNotFoundException("Model inference endpoint '{}' not found".format(url))

            processor = self._engine_processor_lookup.get(url)
            if not processor:
                processor_cls = BasePreprocessRequest.get_engine_cls(ep.engine_type)
                processor = processor_cls(model_endpoint=ep, task=self._task)
                self._engine_processor_lookup[url] = processor

            return_value = await self._process_request(
                processor=processor,
                url=url,
                body=request_body,
                serve_type=serve_type
            )
        finally:
            if url and processor is not None and processor is not self._engine_processor_lookup.get(url):
                gc.collect()
            self._request_processing_state.dec()

        return return_value

    def _process_canary(self, base_url: str) -> Optional[dict]:
        canary = self._canary_route.get(base_url)
        if not canary:
            return None
        # random choice
        draw = choice(canary['endpoints'], 1, p=canary['weights'])
        # the new endpoint to use
        return draw[0]

    def configure(
            self,
            external_serving_base_url: Optional[str] = None,
            external_triton_grpc_server: Optional[str] = None,
            external_triton_grpc_compression: Optional[bool] = None,
            external_kafka_service_server: Optional[str] = None,
            default_metric_log_freq: Optional[float] = None,
    ):
        """
        Set ModelRequestProcessor configuration arguments.

        :param external_serving_base_url: Set the external base http endpoint for the serving service
            This URL will be passed to user custom preprocess class,
            allowing it to concatenate and combine multiple model requests into one
        :param external_triton_grpc_server: set the external grpc tcp port of the Nvidia Triton clearml container.
            Used by the clearml triton engine class to send inference requests
        :param external_triton_grpc_compression: set gRPC compression (default: False, no compression)
        :param external_kafka_service_server: Optional, Kafka endpoint for the statistics controller collection.
        :param default_metric_log_freq: Default request metric logging (0 to 1.0, 1. means 100% of requests are logged)
        """
        if external_serving_base_url is not None:
            self._task.set_parameter(
                name="General/{}".format(self._config_key_serving_base_url),
                value=str(external_serving_base_url),
                value_type="str",
                description="external base http endpoint for the serving service"
            )
        if external_triton_grpc_server is not None:
            self._task.set_parameter(
                name="General/{}".format(self._config_key_triton_grpc),
                value=str(external_triton_grpc_server),
                value_type="str",
                description="external grpc tcp port of the Nvidia Triton ClearML container running"
            )
        if external_triton_grpc_compression is not None:
            self._task.set_parameter(
                name="General/{}".format(self._config_key_triton_compression),
                value=str(external_triton_grpc_compression),
                value_type="bool",
                description="use external grpc tcp compression"
            )
        if external_kafka_service_server is not None:
            self._task.set_parameter(
                name="General/{}".format(self._config_key_kafka_stats),
                value=str(external_kafka_service_server),
                value_type="str",
                description="external Kafka service url for the statistics controller server"
            )
        if default_metric_log_freq is not None:
            self._task.set_parameter(
                name="General/{}".format(self._config_key_def_metric_freq),
                value=str(default_metric_log_freq),
                value_type="float",
                description="Request metric logging frequency"
            )

    def get_configuration(self) -> dict:
        return dict(**self._configuration)

    def add_endpoint(
            self,
            endpoint: Union[ModelEndpoint, dict],
            preprocess_code: Optional[str] = None,
            model_name: Optional[str] = None,
            model_project: Optional[str] = None,
            model_tags: Optional[List[str]] = None,
            model_published: Optional[bool] = None,
    ) -> str:
        """
        Return the unique name of the endpoint (endpoint + version)
        Overwrite existing endpoint if already exists  (outputs a warning)

        :param endpoint: New endpoint to register (overwrite existing endpoint if exists)
        :param preprocess_code: If provided upload local code as artifact
        :param model_name: If model-id not provided on, search based on model name
        :param model_project: If model-id not provided on, search based on model project
        :param model_tags: If model-id not provided on, search based on model tags
        :param model_published: If model-id not provided on, search based on model published state
        """
        if not isinstance(endpoint, ModelEndpoint):
            endpoint = ModelEndpoint(**endpoint)

        # make sure we have everything configured
        self._validate_model(endpoint)

        url = self._normalize_endpoint_url(endpoint.serving_url, endpoint.version)
        if url in self._endpoints:
            print("Warning: Model endpoint \'{}\' overwritten".format(url))

        if not endpoint.model_id and any([model_project, model_name, model_tags]):
            model_query = dict(
                project_name=model_project,
                model_name=model_name,
                tags=model_tags,
                only_published=bool(model_published),
                include_archived=False,
            )
            models = Model.query_models(max_results=2, **model_query)
            if not models:
                raise EndpointModelLoadException("Could not find any Model to serve {}".format(model_query))
            if len(models) > 1:
                print("Warning: Found multiple Models for \'{}\', selecting id={}".format(model_query, models[0].id))
            endpoint.model_id = models[0].id
        elif not endpoint.model_id and endpoint.engine_type != "custom":
            # if the "engine_type" is "custom" it might be there is no model_id attached
            print("Warning: No Model provided for \'{}\'".format(url))

        # upload as new artifact
        if preprocess_code:
            if not Path(preprocess_code).exists():
                raise EndpointModelLoadException("Preprocessing code \'{}\' could not be found".format(preprocess_code))
            preprocess_artifact_name = "py_code_{}".format(url.replace("/", "_"))
            self._task.upload_artifact(
                name=preprocess_artifact_name, artifact_object=Path(preprocess_code), wait_on_upload=True)
            endpoint.preprocess_artifact = preprocess_artifact_name

        # register the model
        self._add_registered_input_model(endpoint_url=endpoint.serving_url, model_id=endpoint.model_id)

        self._endpoints[url] = endpoint
        return url

    def add_model_monitoring(
            self,
            monitoring: Union[ModelMonitoring, dict],
            preprocess_code: Optional[str] = None,
    ) -> str:
        """
        Return the unique name of the endpoint (endpoint + version)
        Overwrite existing endpoint if already exists  (outputs a warning)

        :param monitoring: Model endpoint monitor (overwrite existing endpoint if exists)
        :param preprocess_code: If provided upload local code as artifact
        :return: Unique model monitoring ID (base_model_url)
        """
        if not isinstance(monitoring, ModelMonitoring):
            monitoring = ModelMonitoring(**monitoring)

        # make sure we actually have something to monitor
        if not any([monitoring.monitor_project, monitoring.monitor_name, monitoring.monitor_tags]):
            raise EndpointModelLoadException(
                "Model monitoring requires at least a project / name / tag to monitor, none were provided.")

        # make sure we have everything configured
        self._validate_model(monitoring)

        name = monitoring.base_serving_url
        if name in self._model_monitoring:
            print("Warning: Model monitoring \'{}\' overwritten".format(name))

        # upload as new artifact
        if preprocess_code:
            if not Path(preprocess_code).exists():
                raise EndpointModelLoadException(
                    "Preprocessing code \'{}\' could not be found".format(preprocess_code))
            preprocess_artifact_name = "py_code_{}".format(name.replace("/", "_"))
            self._task.upload_artifact(
                name=preprocess_artifact_name, artifact_object=Path(preprocess_code), wait_on_upload=True)
            monitoring.preprocess_artifact = preprocess_artifact_name

        self._model_monitoring[name] = monitoring
        return name

    def remove_model_monitoring(self, model_base_url: str) -> bool:
        """
        Remove model monitoring, use base_model_url as unique identifier
        """
        if model_base_url not in self._model_monitoring:
            return False
        self._model_monitoring.pop(model_base_url, None)
        return True

    def remove_endpoint(self, endpoint_url: str, version: Optional[str] = None) -> bool:
        """
        Remove specific model endpoint, use base_model_url as unique identifier
        """
        endpoint_url = self._normalize_endpoint_url(endpoint_url, version)
        if endpoint_url not in self._endpoints:
            return False
        self._endpoints.pop(endpoint_url, None)
        self._remove_registered_input_model(endpoint_url)
        return True

    def add_canary_endpoint(
            self,
            canary: Union[CanaryEP, dict],
    ) -> str:
        """
        Return the unique name of the endpoint (endpoint + version)
        Overwrite existing endpoint if already exists  (outputs a warning)

        :param canary: Canary endpoint router (overwrite existing endpoint if exists)
        :return: Unique canary ID (base_model_url)
        """
        if not isinstance(canary, CanaryEP):
            canary = CanaryEP(**canary)
        if canary.load_endpoints and canary.load_endpoint_prefix:
            raise EndpointModelLoadException(
                "Could not add canary endpoint with both "
                "prefix ({}) and fixed set of endpoints ({})".format(
                    canary.load_endpoints, canary.load_endpoint_prefix))
        name = canary.endpoint
        if name in self._canary_endpoints:
            print("Warning: Model monitoring \'{}\' overwritten".format(name))

        self._canary_endpoints[name] = canary
        return name

    def remove_canary_endpoint(self, endpoint_url: str) -> bool:
        """
        Remove specific canary model endpoint, use base_model_url as unique identifier
        """
        if endpoint_url not in self._canary_endpoints:
            return False
        self._canary_endpoints.pop(endpoint_url, None)
        return True

    def add_metric_logging(self, metric: Union[EndpointMetricLogging, dict], update: bool = False) -> bool:
        """
        Add metric logging to a specific endpoint
        Valid metric variable are any variables on the request or response dictionary,
        or a custom preprocess reported variable

        When overwriting and existing monitored variable, output a warning.

        :param metric: Metric variable to monitor
        :param update: If True update the current metric with the new one, otherwise overwrite if exists
        :return: True if successful
        """
        if not isinstance(metric, EndpointMetricLogging):
            metric = EndpointMetricLogging(**metric)

        name = str(metric.endpoint).strip("/")
        metric.endpoint = name

        if name not in self._endpoints and not name.endswith('*'):
            raise MetricLoggingException("Metric logging \'{}\' references a nonexistent endpoint".format(name))

        if name in self._metric_logging:
            print("Warning: Metric logging \'{}\' {}".format(name, "updated" if update else "overwritten"))

        if update and name in self._metric_logging:
            metric_dict = metric.as_dict()
            cur_metric_dict = self._metric_logging[name].as_dict()
            metric_dict = merge_dicts(cur_metric_dict, metric_dict)
            self._metric_logging[name] = EndpointMetricLogging(**metric_dict)
        else:
            self._metric_logging[name] = metric
        return True

    def remove_metric_logging(
            self,
            endpoint: str,
            variable_name: str = None,
    ) -> bool:
        """
        Remove existing logged metric variable. Use variable name and endpoint as unique identifier

        :param endpoint: Endpoint name (including version, e.g. "model/1" or "model/*")
        :param variable_name: Variable name (str), pass None to remove the entire endpoint logging

        :return: True if successful
        """

        name = str(endpoint).strip("/")

        if name not in self._metric_logging or \
                (variable_name and variable_name not in self._metric_logging[name].metrics):
            return False

        if not variable_name:
            self._metric_logging.pop(name, None)
        else:
            self._metric_logging[name].metrics.pop(variable_name, None)

        return True

    def list_metric_logging(self) -> Dict[str, EndpointMetricLogging]:
        """
        List existing logged metric variables.

        :return: Dictionary, key='endpoint/version' value=EndpointMetricLogging
        """

        return dict(**self._metric_logging)

    def list_endpoint_logging(self) -> Dict[str, EndpointMetricLogging]:
        """
        List endpoints (fully synced) current  metric logging state.

        :return: Dictionary, key='endpoint/version' value=EndpointMetricLogging
        """

        return dict(**self._endpoint_metric_logging)

    def deserialize(
            self,
            task: Task = None,
            prefetch_artifacts: bool = False,
            skip_sync: bool = False,
            update_current_task: bool = True
    ) -> bool:
        """
        Restore ModelRequestProcessor state from Task
        return True if actually needed serialization, False nothing changed

        :param task: Load data from Task
        :param prefetch_artifacts: If True prefetch artifacts requested by the endpoints
        :param skip_sync: If True do not update the canary/monitoring state
        :param update_current_task: is not skip_sync, and is True,
            update the current Task with the configuration synced from the serving service Task
        """
        if not task:
            task = self._task

        configuration = task.get_parameters_as_dict().get("General") or {}
        endpoints = task.get_configuration_object_as_dict(name='endpoints') or {}
        canary_ep = task.get_configuration_object_as_dict(name='canary') or {}
        model_monitoring = task.get_configuration_object_as_dict(name='model_monitoring') or {}
        metric_logging = task.get_configuration_object_as_dict(name='metric_logging') or {}

        task_artifacts = task.artifacts
        artifacts_hash = [
            task_artifacts[m["preprocess_artifact"]].hash
            for m in list(endpoints.values()) + list(model_monitoring.values())
            if m.get("preprocess_artifact") and m.get("preprocess_artifact") in task_artifacts
        ]

        hashed_conf = hash_dict(
            dict(endpoints=endpoints,
                 canary_ep=canary_ep,
                 model_monitoring=model_monitoring,
                 metric_logging=metric_logging,
                 configuration=configuration,
                 artifacts_hash=artifacts_hash)
        )
        if self._last_update_hash == hashed_conf and not self._model_monitoring_update_request:
            return False
        print("Info: syncing model endpoint configuration, state hash={}".format(hashed_conf))
        self._last_update_hash = hashed_conf

        endpoints = {
            k: ModelEndpoint(**{i: j for i, j in v.items() if hasattr(ModelEndpoint.__attrs_attrs__, i)})
            for k, v in endpoints.items()
        }
        model_monitoring = {
            k: ModelMonitoring(**{i: j for i, j in v.items() if hasattr(ModelMonitoring.__attrs_attrs__, i)})
            for k, v in model_monitoring.items()
        }
        canary_endpoints = {
            k: CanaryEP(**{i: j for i, j in v.items() if hasattr(CanaryEP.__attrs_attrs__, i)})
            for k, v in canary_ep.items()
        }
        metric_logging = {
            k: EndpointMetricLogging(**{i: j for i, j in v.items()
                                        if hasattr(EndpointMetricLogging.__attrs_attrs__, i)})
            for k, v in metric_logging.items()
        }

        # if there is no need to sync Canary and Models we can just leave
        if skip_sync:
            self._endpoints = endpoints
            self._model_monitoring = model_monitoring
            self._canary_endpoints = canary_endpoints
            self._metric_logging = metric_logging
            self._deserialize_conf_dict(configuration)
            return True

        # make sure we only have one stall request at any given moment
        with self._update_lock_guard:
            prev_endpoints = dict(**self._endpoints)

            # download artifacts
            # todo: separate into two, download before lock, and overwrite inside lock
            if prefetch_artifacts:
                for item in list(endpoints.values()) + list(model_monitoring.values()):
                    if item.preprocess_artifact:
                        # noinspection PyBroadException
                        try:
                            self._task.artifacts[item.preprocess_artifact].get_local_copy(
                                extract_archive=True,
                            )
                        except Exception:
                            pass

            # stall all requests
            self._update_lock_flag = True
            # wait until we have no request processed
            while self._request_processing_state.value() != 0:
                sleep(1)

            self._endpoints = endpoints
            self._model_monitoring = model_monitoring
            self._canary_endpoints = canary_endpoints
            self._metric_logging = metric_logging
            self._deserialize_conf_dict(configuration)

            # if we have models we need to sync, now is the time
            self._sync_monitored_models()

            self._update_canary_lookup()

            self._sync_metric_logging()

            # release stall lock
            self._update_lock_flag = False

            # update the state on the inference task
            if update_current_task and Task.current_task() and Task.current_task().id != self._task.id:
                self.serialize(task=Task.current_task())

            for endpoint in self._endpoints.values():
                self._start_model_endpoint_telemetry(endpoint)
            for endpoint_name, endpoint in prev_endpoints.items():
                if endpoint_name not in self._endpoints:
                    self._stop_model_endpoint_telemetry(endpoint)

        return True

    def reload(self) -> None:
        """
        Reload the serving session state from the backend
        """
        self._task.reload()
        self.deserialize(prefetch_artifacts=False, skip_sync=False, update_current_task=False)

    def serialize(self, task: Optional[Task] = None) -> None:
        """
        Store ModelRequestProcessor state into Task
        """
        if not task:
            task = self._task
        config_dict = {k: v.as_dict(remove_null_entries=True) for k, v in self._endpoints.items()}
        task.set_configuration_object(name='endpoints', config_dict=config_dict)
        config_dict = {k: v.as_dict(remove_null_entries=True) for k, v in self._canary_endpoints.items()}
        task.set_configuration_object(name='canary', config_dict=config_dict)
        config_dict = {k: v.as_dict(remove_null_entries=True) for k, v in self._model_monitoring.items()}
        task.set_configuration_object(name='model_monitoring', config_dict=config_dict)
        config_dict = {k: v.as_dict(remove_null_entries=True) for k, v in self._metric_logging.items()}
        task.set_configuration_object(name='metric_logging', config_dict=config_dict)
        # store our version
        from ..version import __version__
        # noinspection PyProtectedMember
        if task._get_runtime_properties().get("version") != str(__version__):
            # noinspection PyProtectedMember
            task._set_runtime_properties(runtime_properties=dict(version=str(__version__)))

    def get_version(self) -> str:
        """
        :return: version number (string) of the ModelRequestProcessor (clearml-serving session)
        """
        default_version = "1.0.0"
        if not self._task:
            return default_version
        # noinspection PyProtectedMember
        return self._task._get_runtime_properties().get("version", default_version)

    def _update_canary_lookup(self):
        canary_route = {}
        for k, v in self._canary_endpoints.items():
            if v.load_endpoint_prefix and v.load_endpoints:
                print("Warning: Canary has both prefix and fixed endpoints, ignoring canary endpoint")
                continue
            if v.load_endpoints:
                if len(v.load_endpoints) != len(v.weights):
                    print("Warning: Canary \'{}\' weights [{}] do not match number of endpoints [{}], skipping!".format(
                        k, v.weights, v.load_endpoints))
                    continue
                endpoints = []
                weights = []
                for w, ep in zip(v.weights, v.load_endpoints):
                    if ep not in self._endpoints and ep not in self._model_monitoring_endpoints:
                        print("Warning: Canary \'{}\' endpoint \'{}\' could not be found, skipping".format(k, ep))
                        continue
                    endpoints.append(ep)
                    weights.append(float(w))
                # normalize weights
                sum_weights = sum(weights)
                weights = [w/sum_weights for w in weights]
                canary_route[k] = dict(endpoints=endpoints, weights=weights)
            elif v.load_endpoint_prefix:
                endpoints = [ep for ep in list(self._endpoints.keys()) + list(self._model_monitoring_endpoints.keys())
                             if str(ep).startswith(v.load_endpoint_prefix)]
                endpoints = sorted(
                    endpoints,
                    reverse=True,
                    key=lambda x: '{}/{:0>9}'.format('/'.join(x.split('/')[:-1]), x.split('/')[-1]) if '/' in x else x
                )
                endpoints = endpoints[:len(v.weights)]
                weights = v.weights[:len(endpoints)]
                # normalize weights
                sum_weights = sum(weights)
                weights = [w/sum_weights for w in weights]
                canary_route[k] = dict(endpoints=endpoints, weights=weights)
                self._report_text(
                    "Info: Canary endpoint \'{}\' selected [{}]".format(k, canary_route[k])
                )

        # update back
        self._canary_route = canary_route

    def _sync_monitored_models(self, force: bool = False) -> bool:
        if not force and not self._model_monitoring_update_request:
            return False
        dirty = False

        for serving_base_url, versions_model_id_dict in self._model_monitoring_versions.items():
            # find existing endpoint versions
            for ep_base_url in list(self._model_monitoring_endpoints.keys()):
                # skip over endpoints that are not our own
                if not ep_base_url.startswith(serving_base_url+"/"):
                    continue
                # find endpoint version
                _, version = ep_base_url.split("/", 1)
                if int(version) not in versions_model_id_dict:
                    # remove old endpoint
                    self._model_monitoring_endpoints.pop(ep_base_url, None)
                    dirty = True
                    continue

            # add new endpoint
            for version, model_id in versions_model_id_dict.items():
                url = "{}/{}".format(serving_base_url, version)
                if url in self._model_monitoring_endpoints:
                    continue
                model = self._model_monitoring.get(serving_base_url)
                if not model:
                    # this should never happen
                    continue
                model_endpoint_config = {
                    i: j for i, j in model.as_dict(remove_null_entries=True).items()
                    if hasattr(ModelEndpoint.__attrs_attrs__, i)
                }
                model_endpoint_config["serving_url"] = serving_base_url
                model_endpoint_config["model_id"] = model_id
                model_endpoint_config["version"] = str(version)
                ep = ModelEndpoint(**model_endpoint_config)
                self._model_monitoring_endpoints[url] = ep
                dirty = True

        # filter out old model monitoring endpoints
        for ep_url in list(self._model_monitoring_endpoints.keys()):
            if not any(True for url in self._model_monitoring_versions if ep_url.startswith(url+"/")):
                self._model_monitoring_endpoints.pop(ep_url, None)
                self._remove_registered_input_model(ep_url)
                dirty = True

        # reset flag
        self._model_monitoring_update_request = False

        if dirty:
            config_dict = {k: v.as_dict(remove_null_entries=True) for k, v in self._model_monitoring_endpoints.items()}
            self._task.set_configuration_object(name='model_monitoring_eps', config_dict=config_dict)
            for m in self._model_monitoring_endpoints.values():
                # log us on the main task
                self._add_registered_input_model(endpoint_url=m.serving_url, model_id=m.model_id)

        return dirty

    def _update_monitored_models(self):
        for model in self._model_monitoring.values():
            current_served_models = self._model_monitoring_versions.get(model.base_serving_url, {})
            # To Do: sort by updated time ?
            models = Model.query_models(
                project_name=model.monitor_project or None,
                model_name=model.monitor_name or None,
                tags=model.monitor_tags or None,
                only_published=model.only_published,
                max_results=model.max_versions,
                include_archived=False,
            )

            # check what we already have:
            current_model_id_version_lookup = dict(
                zip(list(current_served_models.values()), list(current_served_models.keys()))
            )
            versions = sorted(current_served_models.keys(), reverse=True)

            # notice, most updated model first
            # select only the new models
            model_ids = [m.id for m in models]

            # we want last updated model to be last (so it gets the highest version number)
            max_v = 1 + (versions[0] if versions else 0)
            versions_model_ids = []
            for m_id in reversed(model_ids):
                v = current_model_id_version_lookup.get(m_id)
                if v is None:
                    v = max_v
                    max_v += 1
                versions_model_ids.append((v, m_id))

            # remove extra entries (old models)
            versions_model_ids_dict = dict(versions_model_ids[:model.max_versions])

            # mark dirty if something changed:
            if versions_model_ids_dict != current_served_models:
                self._model_monitoring_update_request = True

            # update model serving state
            self._model_monitoring_versions[model.base_serving_url] = versions_model_ids_dict

        if not self._model_monitoring_update_request:
            return False

        self._report_text("INFO: Monitored Models updated: {}".format(
            json.dumps(self._model_monitoring_versions, indent=2))
        )
        return True

    def _sync_metric_logging(self, force: bool = False) -> bool:
        if not force and not self._metric_logging:
            return False

        fixed_metric_endpoint = {
            k: v for k, v in self._metric_logging.items() if "*/" not in k
        }
        prefix_metric_endpoint = {k.split("*/")[0]: v for k, v in self._metric_logging.items() if "*/" in k}

        endpoint_metric_logging = {}
        for k, ep in list(self._endpoints.items()) + list(self._model_monitoring_endpoints.items()):
            if k in fixed_metric_endpoint:
                if k not in endpoint_metric_logging:
                    endpoint_metric_logging[k] = fixed_metric_endpoint[k]

                continue
            for p, v in prefix_metric_endpoint.items():
                if k.startswith(p):
                    if k not in endpoint_metric_logging:
                        endpoint_metric_logging[k] = v

                    break

        self._endpoint_metric_logging = endpoint_metric_logging
        return True

    def launch(self, poll_frequency_sec=300):
        """
        Launch the background synchronization thread and monitoring thread
        (updating runtime process based on changes on the Task, and monitoring model changes in the system)
        :param poll_frequency_sec: Sync every X seconds (default 300 seconds)
        """
        if self._sync_daemon_thread:
            return

        # read state
        self.deserialize(self._task, prefetch_artifacts=True)
        # model monitoring sync
        if self._update_monitored_models():
            # update endpoints
            self.deserialize(self._task, prefetch_artifacts=True)

        # get the serving instance (for visibility and monitoring)
        self._instance_task = Task.current_task()

        # start the background thread
        with self._update_lock_guard:
            if self._sync_daemon_thread:
                return
            self._sync_daemon_thread = threading.Thread(
                target=self._sync_daemon, args=(poll_frequency_sec, ), daemon=True)
            self._stats_sending_thread = threading.Thread(
                target=self._stats_send_loop, daemon=True)

            self._sync_daemon_thread.start()
            self._stats_sending_thread.start()

        # we return immediately

    def _sync_daemon(self, poll_frequency_sec: float = 300) -> None:
        """
        Background thread, syncing model changes into request service.
        """
        from clearml.backend_api.services import tasks

        poll_frequency_sec = float(poll_frequency_sec)
        # force mark started on the main serving service task
        self._task.mark_started(force=True)
        self._report_text("Launching - configuration sync every {} sec".format(poll_frequency_sec))
        cleanup = False
        model_monitor_update = False
        self._update_serving_plot()
        while True:
            # noinspection PyBroadException
            try:
                # make sure we let the serving session know we are alive
                self._task.send(tasks.PingRequest(self._task.id))
            except Exception:
                # noinspection PyBroadException
                try:
                    self._report_text("Failed pinging serving session id={}".format(self._task.id))
                except Exception as ex:
                    pass

            try:
                # this should be the only place where we call deserialize
                self._task.reload()
                if self.deserialize(self._task):
                    self._report_text("New configuration updated")
                    # mark clean up for next round
                    cleanup = True
                # model monitoring sync
                if self._update_monitored_models():
                    self._report_text("Model monitoring synced")
                    # update endpoints
                    self.deserialize(self._task)
                    # mark clean up for next round
                    model_monitor_update = True
                # update serving layout plot
                if cleanup or model_monitor_update:
                    self._update_serving_plot()
                if cleanup:
                    gc.collect()
                    self._engine_processor_lookup = dict()
            except Exception as ex:
                print("Exception occurred in monitoring thread: {}".format(ex))
            sleep(poll_frequency_sec)
            try:
                # we assume that by now all old deleted endpoints requests already returned
                call_gc_collect = False
                if model_monitor_update and not cleanup:
                    for k in list(self._engine_processor_lookup.keys()):
                        if k not in self._endpoints:
                            self._stop_model_endpoint_telemetry(self._engine_processor_lookup[k].model_endpoint)
                            # atomic
                            self._engine_processor_lookup.pop(k, None)
                            call_gc_collect = True
                if call_gc_collect:
                    gc.collect()
                cleanup = False
                model_monitor_update = False
            except Exception as ex:
                print("Exception occurred in monitoring thread: {}".format(ex))

    def _stats_send_loop(self) -> None:
        """
        Background thread for sending stats to Kafka service
        """
        if not self._kafka_stats_url:
            print("No Kafka Statistics service configured, shutting down statistics report")
            return

        print("Starting Kafka Statistics reporting: {}".format(self._kafka_stats_url))

        from kafka import KafkaProducer  # noqa
        import kafka.errors as Errors  # noqa

        while True:
            try:
                producer = KafkaProducer(
                    bootstrap_servers=self._kafka_stats_url,  # ['localhost:9092'],
                    value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                    compression_type='lz4',  # requires python lz4 package
                )
                break
            except Exception as ex:
                print("Error: failed opening Kafka consumer [{}]: {}".format(self._kafka_stats_url, ex))
                print("Retrying in 30 seconds")
                sleep(30)

        while True:
            try:
                stats_list_dict = [self._stats_queue.get(block=True)]
                while True:
                    v = self._stats_queue.get(block=False)
                    if v is None:
                        break
                    stats_list_dict.append(v)
            except Exception as ex:
                print("Warning: Statistics thread exception: {}".format(ex))
                break

            left_overs = []
            while stats_list_dict or left_overs:
                if not stats_list_dict:
                    stats_list_dict = left_overs
                    left_overs = []

                # send into kafka service
                try:
                    producer.send(self._kafka_topic, value=stats_list_dict).get()
                    stats_list_dict = []
                except Errors.MessageSizeTooLargeError:
                    # log.debug("Splitting Kafka message in half [{}]".format(len(stats_list_dict)))
                    # split in half - message is too long for kafka to send
                    left_overs += stats_list_dict[len(stats_list_dict)//2:]
                    stats_list_dict = stats_list_dict[:len(stats_list_dict)//2]
                    continue
                except Exception as ex:
                    print("Warning: Failed to send statistics packet to Kafka service: {}".format(ex))
                    break

    def get_id(self) -> str:
        return self._task.id

    def get_endpoints(self) -> Dict[str, ModelEndpoint]:
        endpoints = dict(**self._endpoints)
        endpoints.update(**self._model_monitoring_endpoints)
        return endpoints

    def get_synced_endpoints(self) -> Dict[str, ModelEndpoint]:
        self._task.reload()
        _endpoints = self._task.get_configuration_object_as_dict(name='endpoints') or {}
        _monitor_endpoints = self._task.get_configuration_object_as_dict(name='model_monitoring_eps') or {}
        endpoints = {
            k: ModelEndpoint(**{i: j for i, j in v.items() if hasattr(ModelEndpoint.__attrs_attrs__, i)})
            for k, v in _endpoints.items()}
        endpoints.update({
            k: ModelEndpoint(**{i: j for i, j in v.items() if hasattr(ModelEndpoint.__attrs_attrs__, i)})
            for k, v in _monitor_endpoints.items()
        })
        return endpoints

    def get_canary_endpoints(self) -> dict:
        return self._canary_endpoints

    def get_model_monitoring(self) -> dict:
        return self._model_monitoring

    def _get_instance_id(self) -> Optional[str]:
        return self._instance_task.id if self._instance_task else None

    def _report_text(self, text) -> Optional[str]:
        return self._task.get_logger().report_text("Instance [{}, pid={}]: {}".format(
            self._get_instance_id(), os.getpid(), text))

    def _update_serving_plot(self) -> None:
        """
        Update the endpoint serving graph on the serving instance Task
        """
        if not self._instance_task:
            return

        # Generate configuration table and details
        endpoints = list(self._endpoints.values()) + list(self._model_monitoring_endpoints.values())
        if not endpoints:
            # clear plot if we had any
            return

        # noinspection PyProtectedMember
        model_link_template = "{}/projects/*/models/{{model}}/".format(self._task._get_app_server().rstrip("/"))

        endpoints = [e.as_dict() for e in endpoints]
        table_values = [list(endpoints[0].keys())]
        table_values += [
            [
                e.get(c) or "" if c != "model_id" else "<a href=\"{}\"> {} </a>".format(
                    model_link_template.format(model=e["model_id"]), e["model_id"])
                for c in table_values[0]
            ] for e in endpoints
        ]
        self._instance_task.get_logger().report_table(
            title='Serving Endpoint Configuration', series='Details', iteration=0, table_plot=table_values,
            extra_layout={"title": "Model Endpoints Details"})

        # generate current endpoint view
        sankey_node = dict(
            label=[],
            color=[],
            customdata=[],
            hovertemplate='%{customdata}<extra></extra>',
            hoverlabel={"align": "left"},
        )
        sankey_link = dict(
            source=[],
            target=[],
            value=[],
            hovertemplate='<extra></extra>',
        )
        # root
        sankey_node['color'].append("mediumpurple")
        sankey_node['label'].append('{}'.format('external'))
        sankey_node['customdata'].append("")

        sankey_node_idx = {}

        # base_url = self._task._get_app_server() + '/projects/*/models/{model_id}/general'

        # draw all static endpoints
        # noinspection PyProtectedMember
        for i, ep in enumerate(endpoints):
            serve_url = ep['serving_url']
            full_url = '{}/{}'.format(serve_url, ep['version'] or "")
            sankey_node['color'].append("blue")
            sankey_node['label'].append("/{}/".format(full_url.strip("/")))
            sankey_node['customdata'].append(
                "model id: {}".format(ep['model_id'])
            )
            sankey_link['source'].append(0)
            sankey_link['target'].append(i + 1)
            sankey_link['value'].append(1. / len(self._endpoints))
            sankey_node_idx[full_url] = i + 1

        # draw all model monitoring
        sankey_node['color'].append("mediumpurple")
        sankey_node['label'].append('{}'.format('monitoring models'))
        sankey_node['customdata'].append("")
        monitoring_root_idx = len(sankey_node['customdata']) - 1

        for i, m in enumerate(self._model_monitoring.values()):
            serve_url = m.base_serving_url
            sankey_node['color'].append("purple")
            sankey_node['label'].append('{}'.format(serve_url))
            sankey_node['customdata'].append(
                "project: {}<br />name: {}<br />tags: {}".format(
                    m.monitor_project or '', m.monitor_name or '', m.monitor_tags or '')
            )
            sankey_link['source'].append(monitoring_root_idx)
            sankey_link['target'].append(monitoring_root_idx + i + 1)
            sankey_link['value'].append(1. / len(self._model_monitoring))

            # add links to the current models
            serve_url = serve_url.rstrip("/") + "/"
            for k in sankey_node_idx:
                if k.startswith(serve_url):
                    sankey_link['source'].append(monitoring_root_idx + i + 1)
                    sankey_link['target'].append(sankey_node_idx[k])
                    sankey_link['value'].append(1.0 / m.max_versions)

        # add canary endpoints
        # sankey_node['color'].append("mediumpurple")
        # sankey_node['label'].append('{}'.format('Canary endpoints'))
        # sankey_node['customdata'].append("")
        canary_root_idx = len(sankey_node['customdata']) - 1

        # sankey_link['source'].append(0)
        # sankey_link['target'].append(canary_root_idx)
        # sankey_link['value'].append(1.)

        for i, c in enumerate(self._canary_endpoints.values()):
            serve_url = c.endpoint
            sankey_node['color'].append("green")
            sankey_node['label'].append('CANARY: /{}/'.format(serve_url.strip("/")))
            sankey_node['customdata'].append(
                "outputs: {}".format(
                    c.load_endpoints or c.load_endpoint_prefix)
            )
            sankey_link['source'].append(0)
            sankey_link['target'].append(canary_root_idx + i + 1)
            sankey_link['value'].append(1. / len(self._canary_endpoints))

            # add links to the current models
            if serve_url not in self._canary_route:
                continue
            for ep, w in zip(self._canary_route[serve_url]['endpoints'], self._canary_route[serve_url]['weights']):
                idx = sankey_node_idx.get(ep)
                if idx is None:
                    continue
                sankey_link['source'].append(canary_root_idx + i + 1)
                sankey_link['target'].append(idx)
                sankey_link['value'].append(w)

        # create the sankey graph
        dag_flow = dict(
            link=sankey_link,
            node=sankey_node,
            textfont=dict(color='rgba(0,0,0,255)', size=10),
            type='sankey',
            orientation='h'
        )
        fig = dict(data=[dag_flow], layout={'xaxis': {'visible': False}, 'yaxis': {'visible': False}})

        self._instance_task.get_logger().report_plotly(
            title='Serving Endpoints Layout', series='', iteration=0, figure=fig)

    def _deserialize_conf_dict(self, configuration: dict) -> None:
        self._configuration = configuration

        # deserialized values go here
        self._kafka_stats_url = \
            configuration.get(self._config_key_kafka_stats) or \
            os.environ.get("CLEARML_DEFAULT_KAFKA_SERVE_URL")
        self._triton_grpc = \
            configuration.get(self._config_key_triton_grpc) or \
            os.environ.get("CLEARML_DEFAULT_TRITON_GRPC_ADDR")
        self._triton_grpc_compression = \
            cast_str_to_bool(str(configuration.get(
                self._config_key_triton_compression, os.environ.get("CLEARML_DEFAULT_TRITON_GRPC_COMPRESSION", '0')
            )))
        self._serving_base_url = \
            configuration.get(self._config_key_serving_base_url) or \
            os.environ.get("CLEARML_DEFAULT_BASE_SERVE_URL")
        self._metric_log_freq = \
            float(configuration.get(self._config_key_def_metric_freq,
                                    os.environ.get("CLEARML_DEFAULT_METRIC_LOG_FREQ", 1.0)))
        # update back configuration
        self._configuration[self._config_key_kafka_stats] = self._kafka_stats_url
        self._configuration[self._config_key_triton_grpc] = self._triton_grpc
        self._configuration[self._config_key_triton_compression] = self._triton_grpc_compression
        self._configuration[self._config_key_serving_base_url] = self._serving_base_url
        self._configuration[self._config_key_def_metric_freq] = self._metric_log_freq
        # update preprocessing classes
        BasePreprocessRequest.set_server_config(self._configuration)

    async def _process_request(self, processor: BasePreprocessRequest, url: str, body: dict, serve_type: str) -> dict:
        # collect statistics for this request
        stats_collect_fn = None
        collect_stats = False
        custom_stats = dict()
        freq = 1
        # decide if we are collecting the stats
        metric_endpoint = self._metric_logging.get(url)
        if self._kafka_stats_url:
            freq = metric_endpoint.log_frequency if metric_endpoint and metric_endpoint.log_frequency is not None \
                else self._metric_log_freq

            if freq and (freq >= 1 or random() <= freq):
                stats_collect_fn = custom_stats.update
                collect_stats = True

        tic = time()
        state = dict()
        # noinspection PyUnresolvedReferences
        preprocessed = await processor.preprocess(body, state, stats_collect_fn) \
            if processor.is_preprocess_async \
            else processor.preprocess(body, state, stats_collect_fn)
        processed_func = getattr(processor, serve_type.replace("/", "_"))
        # noinspection PyUnresolvedReferences
        processed = await processed_func(preprocessed, state, stats_collect_fn) \
            if processor.is_process_async \
            else processed_func(preprocessed, state, stats_collect_fn)
        # noinspection PyUnresolvedReferences
        return_value = await processor.postprocess(processed, state, stats_collect_fn) \
            if processor.is_postprocess_async \
            else processor.postprocess(processed, state, stats_collect_fn)
        tic = time() - tic
        if collect_stats:
            stats = dict(
                _latency=round(tic, 4),  # 10th of a millisecond should be enough
                _count=int(1.0/freq),
                _url=url
            )

            if custom_stats:
                stats.update(custom_stats)

            if metric_endpoint:
                metric_keys = set(metric_endpoint.metrics.keys())
                # collect inputs
                if body and isinstance(body, dict):
                    keys = set(body.keys()) & metric_keys
                    stats.update({k: body[k] for k in keys})
                # collect outputs
                if return_value:
                    keys = set(return_value.keys()) & metric_keys
                    stats.update({k: return_value[k] for k in keys})

            # send stats in background, push it into a thread queue
            # noinspection PyBroadException
            try:
                self._stats_queue.put(stats, block=False)
            except Exception:
                pass

        return return_value

    @classmethod
    def list_control_plane_tasks(
            cls,
            name: Optional[str] = None,
            project: Optional[str] = None,
            tags: Optional[List[str]] = None
    ) -> List[dict]:

        # noinspection PyProtectedMember
        tasks = Task.query_tasks(
            task_name=name or None,
            project_name=project or None,
            tags=tags or None,
            additional_return_fields=["id", "name", "project", "tags"],
            task_filter={'type': ['service'],
                         'status': ["created", "in_progress"],
                         'system_tags': [cls._system_tag]}
        )  # type: List[dict]
        if not tasks:
            return []

        for t in tasks:
            # noinspection PyProtectedMember
            t['project'] = Task._get_project_name(t['project'])

        return tasks

    @classmethod
    def _get_control_plane_task(
            cls,
            task_id: Optional[str] = None,
            name: Optional[str] = None,
            project: Optional[str] = None,
            tags: Optional[List[str]] = None,
            disable_change_state: bool = False,
    ) -> Task:
        if task_id:
            task = Task.get_task(task_id=task_id)
            if not task:
                raise ServingInitializationException("Could not find Control Task ID={}".format(task_id))
            task_status = task.status
            if task_status not in ("created", "in_progress",):
                if disable_change_state:
                    raise ServingInitializationException(
                        "Could Control Task ID={} status [{}] "
                        "is not valid (only 'draft', 'running' are supported)".format(task_id, task_status))
                else:
                    task.mark_started(force=True)
            return task

        # noinspection PyProtectedMember
        tasks = Task.query_tasks(
            task_name=name or None,
            project_name=project or None,
            tags=tags or None,
            task_filter={'type': ['service'],
                         'status': ["created", "in_progress"],
                         'system_tags': [cls._system_tag]}
        )
        if not tasks:
            raise ServingInitializationException("Could not find any valid Control Tasks")

        if len(tasks) > 1:
            print("Warning: more than one valid Controller Tasks found, using Task ID={}".format(tasks[0]))

        return Task.get_task(task_id=tasks[0])

    @classmethod
    def _create_task(
            cls,
            name: Optional[str] = None,
            project: Optional[str] = None,
            tags: Optional[List[str]] = None
    ) -> Task:
        task = Task.create(
            project_name=project or "DevOps",
            task_name=name or "Serving Service",
            task_type="service",
        )
        task.set_system_tags([cls._system_tag])
        if tags:
            task.set_tags(tags)
        return task

    @classmethod
    def _normalize_endpoint_url(cls, endpoint: str, version: Optional[str] = None) -> str:
        return "{}/{}".format(endpoint.rstrip("/"), version or "").rstrip("/")

    @classmethod
    def _validate_model(cls, endpoint: Union[ModelEndpoint, ModelMonitoring]) -> bool:
        """
        Raise exception if validation fails, otherwise return True
        """
        if endpoint.engine_type in ("triton", ):
            if endpoint.auxiliary_cfg:
                aux_config_dict = {}

                if isinstance(endpoint.auxiliary_cfg, dict):
                    aux_config_dict = endpoint.auxiliary_cfg
                elif isinstance(endpoint.auxiliary_cfg, str):
                    from clearml.utilities.pyhocon import ConfigFactory

                    # noinspection PyBroadException
                    try:
                        aux_config_dict = ConfigFactory.parse_string(endpoint.auxiliary_cfg)
                    except Exception:
                        # we failed parsing the auxiliary pbtxt
                        aux_config_dict = {}

                if aux_config_dict.get("input", None) or aux_config_dict.get("output", None):
                    # noinspection PyBroadException
                    try:
                        from ..engines.triton.triton_helper import TritonHelper

                        suggested_cli_in = {"name": [], "dims": [], "data_type": []}
                        suggested_cli_out = {"name": [], "dims": [], "data_type": []}
                        for layer in aux_config_dict.get("input", None) or []:
                            suggested_cli_in["name"] += ['"{}"'.format(layer["name"])]
                            suggested_cli_in["data_type"] += [
                                TritonHelper.triton_to_np_dtype(layer["data_type"].replace("TYPE_", "", 1)).__name__]
                            suggested_cli_in["dims"] += ['"{}"'.format(layer["dims"])]

                        for layer in aux_config_dict.get("output", None) or []:
                            suggested_cli_out["name"] += ['"{}"'.format(layer["name"])]
                            suggested_cli_out["data_type"] += [
                                TritonHelper.triton_to_np_dtype(layer["data_type"].replace("TYPE_", "", 1)).__name__]
                            suggested_cli_out["dims"] += ['"{}"'.format(layer["dims"])]

                        suggested_cli = "Add to your command line: "\
                            "--input-name {} --input-type {} --input-size {} " \
                            "--output-name {} --output-type {} --output-size {} ".format(
                                " ".join(suggested_cli_in["name"]),
                                " ".join(suggested_cli_in["data_type"]),
                                " ".join(suggested_cli_in["dims"]),
                                " ".join(suggested_cli_out["name"]),
                                " ".join(suggested_cli_out["data_type"]),
                                " ".join(suggested_cli_out["dims"]),
                            )
                    except Exception:
                        suggested_cli = "?"

                    raise EndpointBackendEngineException(
                        "Triton engine requires *manual* input/output specification, "
                        "You input/output in your pbtxt, please remove them and specify manually.\n"
                        "{}".format(suggested_cli)
                    )

                if aux_config_dict.get("default_model_filename", None):
                    raise EndpointBackendEngineException(
                        "ERROR: You have `default_model_filename` in your config pbtxt, "
                        "please remove it. It will be added automatically by the system.")

            # verify we have all the info we need
            d = endpoint.as_dict()
            missing = [
                k for k in [
                    'input_type', 'input_size', 'input_name',
                    'output_type', 'output_size', 'output_name',
                ] if not d.get(k)
            ]

            if missing:
                raise EndpointBackendEngineException(
                    "Triton engine requires input description - missing values in {}".format(missing))
        return True

    def _add_registered_input_model(self, endpoint_url: str, model_id: str) -> bool:
        """
        Add registered endpoint url, return True if successful
        """
        if not self._task or not model_id or not endpoint_url:
            return False

        # noinspection PyBroadException
        try:
            self._task.set_input_model(model_id=model_id, name=endpoint_url.strip("/"))
        except Exception:
            return False

        return True

    def _remove_registered_input_model(self, endpoint_url: str) -> bool:
        """
        Remove registered endpoint url, return True if successful
        """
        if not self._task or not endpoint_url:
            return False

        # noinspection PyBroadException
        try:
            # we assume we have the API version ot support it
            from clearml.backend_api.services import tasks
            self._task.send(tasks.DeleteModelsRequest(
                task=self._task.id, models=[dict(name=endpoint_url.strip("/"), type=tasks.ModelTypeEnum.input)]
            ))
        except Exception:
            return False

        return True
