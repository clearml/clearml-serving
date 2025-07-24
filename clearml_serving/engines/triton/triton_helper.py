import os
import re
import shutil
import subprocess
from argparse import ArgumentParser
from time import time
from typing import Optional

import numpy as np
from clearml import Task, Logger, InputModel
from clearml.backend_api.utils import get_http_session_with_retry
from clearml.utilities.pyhocon import ConfigFactory, ConfigTree, HOCONConverter
from pathlib import Path

from clearml_serving.serving.endpoints import ModelEndpoint
from clearml_serving.serving.model_request_processor import ModelRequestProcessor


class TritonHelper(object):
    _metric_line_parsing = r"(\w+){(gpu_uuid=\"[\w\W]*\",)?model=\"(\w+)\",\s*version=\"(\d+)\"}\s*([0-9.]*)"
    _default_metrics_port = 8002

    def __init__(
            self,
            args,  # Any
            task,  # type: Task
            serving_id,  # type: str
            metric_host=None,  # type: Optional[str]
            metric_port=None,  # type: int
    ):
        # type: (...) -> None
        self._http_session = get_http_session_with_retry()
        self.args = dict(**args.__dict__) if args else {}
        self.task = task
        self._serving_service_task_id = serving_id
        self._serving_service_task = None  # type: Optional[ModelRequestProcessor]
        self._current_endpoints = {}
        self.metric_host = metric_host or '0.0.0.0'
        self.metric_port = metric_port or self._default_metrics_port
        self._parse_metric = re.compile(self._metric_line_parsing)
        self._timestamp = time()
        self._last_update_step = None
        print('String Triton Helper service\n{}\n'.format(self.args))

    def report_metrics(self, remote_logger):
        # type: (Optional[Logger]) -> bool
        # iterations are seconds from start
        iteration = int(time() - self._timestamp)

        report_msg = "reporting metrics: relative time {} sec".format(iteration)
        self.task.get_logger().report_text(report_msg)
        if remote_logger:
            remote_logger.report_text(report_msg, print_console=False)

        # noinspection PyBroadException
        try:
            # this is inside the container
            request = self._http_session.get('http://{}:{}/metrics'.format(self.metric_host, self.metric_port))  # noqa
            if not request.ok:
                return False
            content = request.content.decode().split('\n')
        except Exception:
            return False

        for line in content:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # noinspection PyBroadException
            try:
                metric, gpu_uuid, variant, version, value = self._parse_metric.match(line).groups()
                value = float(value)
            except Exception:
                continue
            self.task.get_logger().report_scalar(
                title=metric,
                series='{}.v{}'.format(variant, version),
                iteration=iteration,
                value=value
            )
            # on the remote logger we add our own Task ID (unique ID),
            # to support multiple servers reporting to the same service controller
            if remote_logger:
                remote_logger.report_scalar(
                    title=metric,
                    series='{}.v{}.{}'.format(variant, version, self.task.id),
                    iteration=iteration,
                    value=value
                )

    def model_service_update_step(self, model_repository_folder=None, verbose=True):
        # type: (Optional[str], bool) -> bool

        if not self._serving_service_task:
            return False

        active_endpoints = self._serving_service_task.get_synced_endpoints()

        self._last_update_step = time()

        # nothing to do
        if self._current_endpoints == active_endpoints:
            return False

        if not model_repository_folder:
            model_repository_folder = '/models/'

        if verbose:
            print('Updating local model folder: {}'.format(model_repository_folder))

        for url, endpoint in active_endpoints.items():

            # Triton model folder structure reference:
            # https://github.com/triton-inference-server/server/blob/r22.07/docs/model_repository.md#model-repository

            # skip if there is no change
            if url in self._current_endpoints and self._current_endpoints.get(url) == endpoint:
                continue

            # skip if this is not a triton engine endpoint:
            if endpoint.engine_type != "triton":
                continue

            url = url.replace("/", "_")

            folder = Path(model_repository_folder) / url
            folder.mkdir(parents=True, exist_ok=True)

            config_pbtxt = folder / 'config.pbtxt'
            # download model versions
            version = 1
            model_id = endpoint.model_id

            model_folder = folder / str(version)

            model_folder.mkdir(parents=True, exist_ok=True)
            model = None
            # noinspection PyBroadException
            try:
                model = InputModel(model_id)
                local_path = model.get_local_copy()
            except Exception:
                local_path = None

            if not local_path or not model:
                print("Error retrieving model ID {} []".format(model_id, model.url if model else ''))
                continue

            local_path = Path(local_path)

            # prepare config.pbtxt
            self.create_config_pbtxt(
                endpoint, target_pbtxt_file=config_pbtxt.as_posix(), platform=model.framework
            )

            if verbose:
                print('Update model v{} in {}'.format(version, model_folder))

            framework = str(model.framework).lower()

            # if this is a folder copy every and delete the temp folder
            if local_path.is_dir() and model and ("tensorflow" in framework or "keras" in framework):
                # we assume we have a `tensorflow.savedmodel` folder
                model_folder /= 'model.savedmodel'
                self._extract_folder(local_path, model_folder, verbose, remove_existing=True)
            elif "torch" in framework and local_path.is_file():
                # single file should be moved
                self._extract_single_file(local_path, model_folder / "model.pt", verbose)
            elif "onnx" in framework and local_path.is_dir():
                # just unzip both model.bin & model.xml into the model folder
                self._extract_folder(local_path, model_folder, verbose)
            elif ("tensorflow" in framework or "keras" in framework) and local_path.is_file():
                # just rename the single file to "model.graphdef"
                self._extract_single_file(local_path, model_folder / "model.graphdef", verbose)
            elif "tensorrt" in framework and local_path.is_file():
                # just rename the single file to "model.plan"
                self._extract_single_file(local_path, model_folder / "model.plan", verbose)
            elif local_path.is_file():
                # generic model will be stored as 'model.bin'
                self._extract_single_file(local_path, model_folder / "model.bin", verbose)
            elif local_path.is_dir():
                # generic model will be stored into the model folder
                self._extract_folder(local_path, model_folder, verbose)
            else:
                print("Model type could not be inferred skipping", model.id, model.framework, model.name)
                continue

        # todo: trigger triton model reloading (instead of relaying on current poll mechanism)
        # based on the model endpoint changes

        # update current state
        self._current_endpoints = active_endpoints

        return True

    @staticmethod
    def _extract_single_file(local_path, target_path, verbose):
        old_file = None
        if target_path.exists():
            old_file = target_path.parent / '.old.{}'.format(target_path.name)
            target_path.replace(old_file)
        if verbose:
            print('copy model into {}'.format(target_path))
        shutil.move(local_path.as_posix(), target_path.as_posix())
        if old_file:
            old_file.unlink()

    @staticmethod
    def _extract_folder(local_path, model_folder, verbose, remove_existing=False):
        model_folder.mkdir(parents=True, exist_ok=True)
        # rename to old
        old_folder = None
        if remove_existing and model_folder.exists():
            old_folder = model_folder.parent / '.old.{}'.format(model_folder.name)
            model_folder.replace(old_folder)
        if verbose:
            print('copy model into {}'.format(model_folder))
        shutil.copytree(
            local_path.as_posix(), model_folder.as_posix(), symlinks=False, dirs_exist_ok=True
        )
        if old_folder:
            shutil.rmtree(path=old_folder.as_posix())
        # delete temp folder
        shutil.rmtree(local_path.as_posix())

    def maintenance_daemon(
            self,
            local_model_repo='/models',  # type: str
            update_frequency_sec=60.0,  # type: float
            metric_frequency_sec=60.0  # type: float
    ):
        # type: (...) -> None

        Path(local_model_repo).mkdir(parents=True, exist_ok=True)

        self._serving_service_task = ModelRequestProcessor(task_id=self._serving_service_task_id)
        self.model_service_update_step(model_repository_folder=local_model_repo, verbose=True)

        # noinspection PyProtectedMember
        remote_logger = self._serving_service_task._task.get_logger()

        # todo: log triton server outputs when running locally

        # we assume we can run the triton server
        cmd = [
            'tritonserver',
            '--model-control-mode=poll',
            '--model-repository={}'.format(local_model_repo),
            '--repository-poll-secs={}'.format(update_frequency_sec),
            '--metrics-port={}'.format(self._default_metrics_port),
            '--allow-metrics=true',
            '--allow-gpu-metrics=true',
        ]
        for k, v in self.args.items():
            if not v or not str(k).startswith('t_'):
                continue
            cmd.append('--{}={}'.format(k.replace("t_", "", 1).replace("_","-"), v))

        print('Starting server: {}'.format(cmd))
        try:
            proc = subprocess.Popen(cmd)
        except FileNotFoundError:
            raise ValueError(
                "Triton Server Engine (tritonserver) could not be found!\n"
                "Verify you running inside the `nvcr.io/nvidia/tritonserver` docker container")
        base_freq = min(update_frequency_sec, metric_frequency_sec)
        metric_tic = update_tic = time()
        while True:
            try:
                error_code = proc.wait(timeout=base_freq)
                if error_code == 0:
                    print("triton-server process ended with error code {}".format(error_code))
                    return
                raise ValueError("triton-server process ended with error code {}".format(error_code))
            except subprocess.TimeoutExpired:
                pass
            pass

            # update models
            if time() - update_tic > update_frequency_sec:
                print("Info: syncing models from main serving service")
                if self.model_service_update_step(model_repository_folder=local_model_repo, verbose=True):
                    print("Info: Models updated from main serving service")
                update_tic = time()

            # update stats
            if time() - metric_tic > metric_frequency_sec:
                metric_tic = time()
                self.report_metrics(remote_logger)

    @classmethod
    def create_config_pbtxt(cls, endpoint, target_pbtxt_file, platform=None):
        # type: (ModelEndpoint, str, Optional[str]) -> bool
        """
        Full spec available here:
        https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md
        """

        def _convert_lists(config):
            if isinstance(config, list):
                return [_convert_lists(i) for i in config]

            if not isinstance(config, ConfigTree):
                return config

            for k in list(config.keys()):
                v = config[k]
                # try to convert to list
                if isinstance(v, (ConfigTree, list)):
                    # noinspection PyBroadException
                    try:
                        a_list = config.get_list(k, [])
                        if a_list:
                            config[k] = _convert_lists(a_list)
                            continue
                    except Exception:
                        pass

                config[k] = _convert_lists(v)

            return config

        final_config_pbtxt = ""
        config_dict = dict()

        if endpoint.auxiliary_cfg and isinstance(endpoint.auxiliary_cfg, str):
            final_config_pbtxt = endpoint.auxiliary_cfg + "\n"
        elif endpoint.auxiliary_cfg and isinstance(endpoint.auxiliary_cfg, dict):
            config_dict = dict(**endpoint.auxiliary_cfg)

        config_dict = ConfigFactory.from_dict(config_dict)

        # The framework for the model. Possible values are:
        #   "tensorrt_plan", "tensorflow_graphdef",
        #   "tensorflow_savedmodel", "onnxruntime_onnx",
        #   "pytorch_libtorch".
        # Default for TF: "tensorflow_savedmodel"

        # replace ": [{" with ": [{" (currently not needed)
        # pattern = re.compile(r"(?P<key>\w+)(?P<space>\s+)(?P<bracket>(\[)|({))")

        for i, s in enumerate(endpoint.input_size or []):
            config_dict.put("input.{}.dims".format(i), s)

        for i, s in enumerate(endpoint.output_size or []):
            config_dict.put("output.{}.dims".format(i), s)

        for i, s in enumerate(endpoint.input_type or []):
            input_type = "TYPE_" + cls.np_to_triton_dtype(np.dtype(s))
            config_dict.put("input.{}.data_type".format(i), input_type)

        for i, s in enumerate(endpoint.output_type or []):
            output_type = "TYPE_" + cls.np_to_triton_dtype(np.dtype(s))
            config_dict.put("output.{}.data_type".format(i), output_type)

        for i, s in enumerate(endpoint.input_name or []):
            config_dict.put("input.{}.name".format(i), "\"{}\"".format(s))

        for i, s in enumerate(endpoint.output_name or []):
            config_dict.put("output.{}.name".format(i), "\"{}\"".format(s))

        # check if we have platform in the auxiliary config pbtxt
        if platform and final_config_pbtxt:
            # noinspection PyBroadException
            try:
                final_config_pbtxt_dict = ConfigFactory.parse_string(final_config_pbtxt)
                # if we found it, null the requested platform and use the auxiliary config pbtxt platform `value`
                if final_config_pbtxt_dict.get("platform", None):
                    print(
                        "WARNING: ignoring auto-detecetd `platform={}` "
                        "and using auxiliary pbtxt `platform={}`".format(
                            str(platform).lower(), final_config_pbtxt_dict.get("platform")))
                    platform = None
            except Exception:
                # we failed parsing the auxiliary pbtxt
                pass

        if platform and not config_dict.get("platform", None) and not config_dict.get("backend", None):
            platform = str(platform).lower()
            if platform.startswith("tensorflow") or platform.startswith("keras"):
                config_dict["platform"] = "\"tensorflow_savedmodel\""
            elif platform.startswith("pytorch") or platform.startswith("caffe"):
                config_dict["backend"] = "\"pytorch\""
            elif platform.startswith("onnx"):
                config_dict["platform"] = "\"onnxruntime_onnx\""

        # convert to lists anything that we can:
        if config_dict:
            config_dict = _convert_lists(config_dict)
            # Convert HOCON standard to predefined message format
            config_pbtxt = "\n" + HOCONConverter.to_hocon(config_dict). \
                replace("=", ":").replace(" : ", ": ")

            # conform types (remove string quotes)
            config_pbtxt = config_pbtxt.replace("\\\"", "<DQUOTE>").\
                replace("\\\'", "<QUOTE>").replace("\"", "").replace("\'", "").\
                replace("<DQUOTE>", "\"").replace("<QUOTE>", "\'")
        else:
            config_pbtxt = ""

        # merge the two
        final_config_pbtxt += config_pbtxt
        print("INFO: target config.pbtxt file for endpoint '{}':\n{}\n".format(
            endpoint.serving_url, final_config_pbtxt))

        with open(target_pbtxt_file, "w") as config_file:
            config_file.write(final_config_pbtxt)

        return True

    @staticmethod
    def np_to_triton_dtype(np_dtype):
        # type (np.dtype) -> str
        """
        copied from tritonclientutils import np_to_triton_dtype
        """
        if np_dtype == bool:
            return "BOOL"
        elif np_dtype == np.int8:
            return "INT8"
        elif np_dtype == np.int16:
            return "INT16"
        elif np_dtype == np.int32:
            return "INT32"
        elif np_dtype == np.int64:
            return "INT64"
        elif np_dtype == np.uint8:
            return "UINT8"
        elif np_dtype == np.uint16:
            return "UINT16"
        elif np_dtype == np.uint32:
            return "UINT32"
        elif np_dtype == np.uint64:
            return "UINT64"
        elif np_dtype == np.float16:
            return "FP16"
        elif np_dtype == np.float32:
            return "FP32"
        elif np_dtype == np.float64:
            return "FP64"
        elif np_dtype == str:
            return "STRING"
        elif np_dtype == np.object_ or np_dtype.type == np.bytes_:
            return "BYTES"
        return None

    @staticmethod
    def triton_to_np_dtype(dtype):
        if dtype == "BOOL":
            return bool
        elif dtype == "INT8":
            return np.int8
        elif dtype == "INT16":
            return np.int16
        elif dtype == "INT32":
            return np.int32
        elif dtype == "INT64":
            return np.int64
        elif dtype == "UINT8":
            return np.uint8
        elif dtype == "UINT16":
            return np.uint16
        elif dtype == "UINT32":
            return np.uint32
        elif dtype == "UINT64":
            return np.uint64
        elif dtype == "FP16":
            return np.float16
        elif dtype == "FP32":
            return np.float32
        elif dtype == "FP64":
            return np.float64
        elif dtype == "BYTES":
            return np.object_
        return None


def main():
    title = 'clearml-serving - Nvidia Triton Engine Controller'
    print(title)
    parser = ArgumentParser(prog='clearml-serving', description=title)
    parser.add_argument(
        '--serving-id', default=os.environ.get('CLEARML_SERVING_TASK_ID'), type=str,
        help='Specify main serving service Task ID')
    parser.add_argument(
        '--project', default=None, type=str,
        help='Optional specify project for the serving engine Task')
    parser.add_argument(
        '--name', default='triton engine', type=str,
        help='Optional specify task name for the serving engine Task')
    parser.add_argument(
        '--update-frequency', default=os.environ.get('CLEARML_TRITON_POLL_FREQ') or 10., type=float,
        help='Model update frequency in minutes')
    parser.add_argument(
        '--metric-frequency', default=os.environ.get('CLEARML_TRITON_METRIC_FREQ') or 1., type=float,
        help='Metric reporting update frequency in minutes')
    parser.add_argument(
        '--inference-task-id', default=None, type=str,
        help='Optional: Specify the inference Task ID to report to. default: create a new one')
    parser.add_argument(
        '--t-http-port', type=str, help='<integer> The port for the server to listen on for HTTP requests')
    parser.add_argument(
        '--t-http-thread-count', type=str, help='<integer> Number of threads handling HTTP requests')
    parser.add_argument(
        '--t-allow-grpc', type=str, help='<integer> Allow the server to listen for GRPC requests')
    parser.add_argument(
        '--t-grpc-port', type=str, help='<integer> The port for the server to listen on for GRPC requests')
    parser.add_argument(
        '--t-grpc-infer-allocation-pool-size', type=str,
        help='<integer> The maximum number of inference request/response objects that remain '
             'allocated for reuse. As long as the number of in-flight requests doesn\'t exceed '
             'this value there will be no allocation/deallocation of request/response objects')
    parser.add_argument(
        '--t-pinned-memory-pool-byte-size', type=str,
        help='<integer> The total byte size that can be allocated as pinned system '
             'memory. If GPU support is enabled, the server will allocate pinned '
             'system memory to accelerate data transfer between host and devices '
             'until it exceeds the specified byte size. This option will not affect '
             'the allocation conducted by the backend frameworks. Default is 256 MB')
    parser.add_argument(
        '--t-cuda-memory-pool-byte-size', type=str,
        help='<<integer>:<integer>> The total byte size that can be allocated as CUDA memory for '
             'the GPU device. If GPU support is enabled, the server will allocate '
             'CUDA memory to minimize data transfer between host and devices '
             'until it exceeds the specified byte size. This option will not affect '
             'the allocation conducted by the backend frameworks. The argument '
             'should be 2 integers separated by colons in the format <GPU device'
             'ID>:<pool byte size>. This option can be used multiple times, but only '
             'once per GPU device. Subsequent uses will overwrite previous uses for '
             'the same GPU device. Default is 64 MB')
    parser.add_argument(
        '--t-min-supported-compute-capability', type=str,
        help='<float> The minimum supported CUDA compute capability. GPUs that '
             'don\'t support this compute capability will not be used by the server')
    parser.add_argument(
        '--t-buffer-manager-thread-count', type=str,
        help='<integer> The number of threads used to accelerate copies and other'
             'operations required to manage input and output tensor contents.'
             'Default is 0')
    parser.add_argument(
        '--t-log-verbose', type=str,
        help='<integer> Triton server logging verbosity (default disabled)')
    parser.add_argument(
        '--t-exit-on-error', type=bool, default=True,
        help='Exits the inference server if any error occurs during initialization.'
             'Recommended to set to True to catch any unanticipated errors.'
             'False prevents single models breaking the whole tritonserver.'
    )

    args = parser.parse_args()

    # check Args OS overrides
    prefix = "CLEARML_TRITON_"
    for k, v in os.environ.items():
        if not k.startswith(prefix):
            continue
        args_var = k.replace(prefix, "", 1).replace("-", "_").lower()
        if args_var in args.__dict__:
            # casting
            t = type(getattr(args, args_var, None))
            setattr(args, args_var, type(t)(v) if t is not None else v)

    # noinspection PyProtectedMember
    serving_task = ModelRequestProcessor._get_control_plane_task(task_id=args.inference_task_id)

    task = Task.init(
        project_name=args.project or serving_task.get_project_name() or "serving",
        task_name="{} - {}".format(serving_task.name, args.name),
        task_type=Task.TaskTypes.inference,
        continue_last_task=args.inference_task_id or None
    )
    print("configuration args: {}".format(args))
    helper = TritonHelper(args, task, serving_id=args.serving_id)

    # safe casting
    try:
        update_frequency_sec = float(args.update_frequency) * 60.0
    except (ValueError, TypeError):
        update_frequency_sec = 600
    try:
        metric_frequency_sec = float(args.metric_frequency) * 60.0
    except (ValueError, TypeError):
        metric_frequency_sec = 60

    # this function will never return
    helper.maintenance_daemon(
        local_model_repo='/models',
        update_frequency_sec=update_frequency_sec,
        metric_frequency_sec=metric_frequency_sec,
    )


if __name__ == '__main__':
    main()
