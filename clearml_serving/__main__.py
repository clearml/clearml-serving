import json
import os.path
import sys
from argparse import ArgumentParser
from pathlib import Path

from clearml.model import Framework

from clearml_serving.serving.endpoints import ModelMonitoring, ModelEndpoint, EndpointMetricLogging
from clearml_serving.serving.model_request_processor import ModelRequestProcessor, CanaryEP
from clearml_serving.version import __version__

# noinspection PyBroadException
try:
    from clearml.backend_api import Session
    Session.add_client(__package__.partition(".")[0].replace("_", "-"), __version__)  # noqa
except Exception:
    pass

verbosity = False
answer_yes = False


def verify_session_version(request_processor):
    current_v = float('.'.join(str(__version__).split(".")[:2]))
    stored_v = float('.'.join(str(request_processor.get_version()).split(".")[:2]))
    if stored_v != current_v:
        print(
            "WARNING: You are about to edit clearml-serving session ID={}\n"
            "It was created with a different version ({}), you are currently using version {}".format(
                request_processor.get_id(), stored_v, current_v))
        print("Do you want to continue [n]/y? ", end="")
        if answer_yes:
            print("y")
        else:
            should_continue = input().lower()
            if should_continue not in ("y", "yes"):
                print("INFO: If you wish to downgrade your clearml-serving CLI use "
                      "`pip3 install clearml-serving=={}`".format(request_processor.get_version()))
                sys.exit(0)


def safe_ModelRequestProcessor(*args, **kwargs):
    request_processor = ModelRequestProcessor(*args, **kwargs)
    verify_session_version(request_processor)
    return request_processor


def func_metric_ls(args):
    request_processor = ModelRequestProcessor(task_id=args.id)
    print("List endpoint metrics, control task id={}".format(request_processor.get_id()))
    request_processor.deserialize(skip_sync=True)
    print("Logged Metrics:\n{}".format(
        json.dumps({k: v.as_dict() for k, v in request_processor.list_metric_logging().items()}, indent=2)))


def func_metric_rm(args):
    request_processor = safe_ModelRequestProcessor(task_id=args.id)
    print("Serving service Task {}, Removing metrics from endpoint={}".format(
        request_processor.get_id(), args.endpoint))
    verify_session_version(request_processor)
    request_processor.deserialize(skip_sync=True)
    if not args.variable:
        if request_processor.remove_metric_logging(endpoint=args.endpoint):
            print("Removing metric endpoint: {}".format(args.endpoint))
        else:
            raise ValueError("Could not remove metric endpoint {}".format(args.endpoint))
    else:
        for v in args.variable:
            if request_processor.remove_metric_logging(endpoint=args.endpoint, variable_name=v):
                print("Removing metric endpoint: {} / {}".format(args.endpoint, v))
            else:
                raise ValueError("Could not remove metric {} from endpoint {}".format(v, args.endpoint))
    print("Updating serving service")
    request_processor.serialize()


def func_metric_add(args):
    request_processor = safe_ModelRequestProcessor(task_id=args.id)
    print("Serving service Task {}, Adding metric logging endpoint \'/{}/\'".format(
        request_processor.get_id(), args.endpoint))
    request_processor.deserialize(skip_sync=True)
    metric = EndpointMetricLogging(endpoint=args.endpoint)
    if args.log_freq is not None:
        metric.log_frequency = float(args.log_freq)
    for v in (args.variable_scalar or []):
        if '=' not in v:
            raise ValueError("Variable '{}' should be in the form of <name>=<buckets> "
                             "example: x1=0,1,2,3,4,5".format(v))
        name, buckets = v.split('=', 1)
        if name in metric.metrics:
            print("Warning: {} defined twice".format(name))
        if '/' in buckets:
            b_min, b_max, b_step = [float(b.strip()) for b in buckets.split('/', 2)]
            buckets = list(range(b_min, b_max, b_step))
        else:
            buckets = [float(b.strip()) for b in buckets.split(',')]
        metric.metrics[name] = dict(type="scalar", buckets=buckets)

    for v in (args.variable_enum or []):
        if '=' not in v:
            raise ValueError("Variable '{}' should be in the form of <name>=<buckets> "
                             "example: x1=cat,dog,sheep".format(v))
        name, buckets = v.split('=', 1)
        if name in metric.metrics:
            print("Warning: {} defined twice".format(name))
        buckets = [str(b.strip()) for b in buckets.split(',')]
        metric.metrics[name] = dict(type="enum", buckets=buckets)

    for v in (args.variable_value or []):
        name = v.strip()
        if name in metric.metrics:
            print("Warning: {} defined twice".format(name))
        metric.metrics[name] = dict(type="value", buckets=None)

    if not request_processor.add_metric_logging(metric=metric, update=True):
        raise ValueError("Could not add metric logging endpoint {}".format(args.endpoint))

    print("Updating serving service")
    request_processor.serialize()


def func_model_upload(args):
    if not args.path and not args.url:
        raise ValueError("Either --path or --url must be specified")
    if args.path and args.url:
        raise ValueError("Either --path or --url but not both")
    if args.path and not os.path.exists(args.path):
        raise ValueError("--path='{}' could not be found".format(args.path))
    if not args.id:
        raise ValueError("Serving Service ID must be provided, use --id <serving_id>")
    from clearml import Task, OutputModel
    from clearml.backend_interface.util import get_or_create_project
    # todo: make it look nice
    t = Task.get_task(task_id=args.id)
    print("Creating new Model name='{}' project='{}' tags={}".format(args.name, args.project, args.tags or ""))
    model = OutputModel(task=t, name=args.name, tags=args.tags or None, framework=args.framework)
    destination = args.destination or t.get_output_destination() or t.get_logger().get_default_upload_destination()
    model.set_upload_destination(uri=destination)
    if args.path:
        print("Uploading model file \'{}\' to {}".format(args.path, destination))
    else:
        print("Registering model file \'{}\'".format(args.url))
    model.update_weights(weights_filename=args.path, register_uri=args.url, auto_delete_file=False)
    t.flush(wait_for_uploads=True)
    if args.project:
        # noinspection PyProtectedMember
        model._base_model.update(
            project_id=get_or_create_project(session=t.session, project_name=args.project)
        )
    print("Model created and registered, new Model ID={}".format(model.id))
    if args.publish:
        model.publish()
        print("Published Model ID={}".format(model.id))


def func_model_ls(args):
    request_processor = ModelRequestProcessor(task_id=args.id)
    print("List model serving and endpoints, control task id={}".format(request_processor.get_id()))
    request_processor.deserialize(skip_sync=True)
    print("Endpoints:\n{}".format(
        json.dumps({k: v.as_dict() for k, v in request_processor.get_endpoints().items()}, indent=2)))
    print("Model Monitoring:\n{}".format(
        json.dumps({k: v.as_dict() for k, v in request_processor.get_model_monitoring().items()}, indent=2)))
    print("Canary:\n{}".format(
        json.dumps({k: v.as_dict() for k, v in request_processor.get_canary_endpoints().items()}, indent=2)))


def func_create_service(args):
    request_processor = ModelRequestProcessor(
        force_create=True, name=args.name, project=args.project, tags=args.tags or None)
    request_processor.serialize()
    print("New Serving Service created: id={}".format(request_processor.get_id()))


def func_config_service(args):
    request_processor = safe_ModelRequestProcessor(task_id=args.id)
    print("Configure serving service id={}".format(request_processor.get_id()))
    request_processor.deserialize(skip_sync=True)
    if args.base_serving_url:
        print("Configuring serving service [id={}] base_serving_url={}".format(
            request_processor.get_id(), args.base_serving_url))
        request_processor.configure(external_serving_base_url=args.base_serving_url)
    if args.triton_grpc_server:
        print("Configuring serving service [id={}] triton_grpc_server={}".format(
            request_processor.get_id(), args.triton_grpc_server))
        request_processor.configure(external_triton_grpc_server=args.triton_grpc_server)
    if args.kafka_metric_server:
        request_processor.configure(external_kafka_service_server=args.kafka_metric_server)
    if args.metric_log_freq is not None:
        pass


def func_list_services(_):
    running_services = ModelRequestProcessor.list_control_plane_tasks()
    print("Currently running Serving Services:\n")
    if not running_services:
        print("No running services found")
    else:
        for s in running_services:
            # Create ordered dictionary with consistent field order
            ordered_service = {
                'name': s.get('name', ''),
                'tags': s.get('tags', []),
                'id': s.get('id', ''),
                'project': s.get('project', '')
            }
            print(ordered_service)


def func_model_remove(args):
    request_processor = safe_ModelRequestProcessor(task_id=args.id)
    print("Serving service Task {}, Removing Model endpoint={}".format(request_processor.get_id(), args.endpoint))
    request_processor.deserialize(skip_sync=True)
    if request_processor.remove_endpoint(endpoint_url=args.endpoint):
        print("Removing static endpoint: {}".format(args.endpoint))
    elif request_processor.remove_model_monitoring(model_base_url=args.endpoint):
        print("Removing model monitoring endpoint: {}".format(args.endpoint))
    elif request_processor.remove_canary_endpoint(endpoint_url=args.endpoint):
        print("Removing model canary endpoint: {}".format(args.endpoint))
    else:
        raise ValueError("Could not find base endpoint URL: {}".format(args.endpoint))

    print("Updating serving service")
    request_processor.serialize()


def func_canary_add(args):
    request_processor = safe_ModelRequestProcessor(task_id=args.id)
    print("Serving service Task {}, Adding canary endpoint \'/{}/\'".format(
        request_processor.get_id(), args.endpoint))
    request_processor.deserialize(skip_sync=True)
    if not request_processor.add_canary_endpoint(
            canary=CanaryEP(
                endpoint=args.endpoint,
                weights=args.weights,
                load_endpoints=args.input_endpoints,
                load_endpoint_prefix=args.input_endpoint_prefix,
            )
    ):
        raise ValueError("Could not add canary endpoint URL: {}".format(args.endpoint))

    print("Updating serving service")
    request_processor.serialize()


def func_model_auto_update_add(args):
    request_processor = safe_ModelRequestProcessor(task_id=args.id)
    print("Serving service Task {}, Adding Model monitoring endpoint: \'/{}/\'".format(
        request_processor.get_id(), args.endpoint))

    if args.aux_config:
        if len(args.aux_config) == 1 and Path(args.aux_config[0]).exists():
            aux_config = Path(args.aux_config[0]).read_text()
        else:
            from clearml.utilities.pyhocon import ConfigFactory
            aux_config = ConfigFactory.parse_string(
                '\n'.join(args.aux_config).replace("\"", "\\\"").replace("'", "\\\'")
            ).as_plain_ordered_dict()
    else:
        aux_config = None

    request_processor.deserialize(skip_sync=True)
    if not request_processor.add_model_monitoring(
        ModelMonitoring(
            base_serving_url=args.endpoint,
            engine_type=args.engine,
            monitor_project=args.project,
            monitor_name=args.name,
            monitor_tags=args.tags or None,
            only_published=args.published,
            max_versions=args.max_versions,
            input_size=args.input_size,
            input_type=args.input_type,
            input_name=args.input_name,
            output_size=args.output_size,
            output_type=args.output_type,
            output_name=args.output_name,
            auxiliary_cfg=aux_config,
        ),
        preprocess_code=args.preprocess
    ):
        raise ValueError("Could not find base endpoint URL: {}".format(args.endpoint))

    print("Updating serving service")
    request_processor.serialize()


def func_model_endpoint_add(args):
    request_processor = safe_ModelRequestProcessor(task_id=args.id)
    print("Serving service Task {}, Adding Model endpoint \'/{}/\'".format(
        request_processor.get_id(), args.endpoint))
    request_processor.deserialize(skip_sync=True)

    if args.aux_config:
        if len(args.aux_config) == 1 and Path(args.aux_config[0]).exists():
            aux_config = Path(args.aux_config[0]).read_text()
        else:
            from clearml.utilities.pyhocon import ConfigFactory
            aux_config = ConfigFactory.parse_string(
                '\n'.join(args.aux_config).replace("\"", "\\\"").replace("'", "\\\'")
            ).as_plain_ordered_dict()
    else:
        aux_config = None

    if not request_processor.add_endpoint(
        ModelEndpoint(
            engine_type=args.engine,
            serving_url=args.endpoint,
            version=args.version,
            model_id=args.model_id,
            input_size=args.input_size,
            input_type=args.input_type,
            input_name=args.input_name,
            output_size=args.output_size,
            output_type=args.output_type,
            output_name=args.output_name,
            auxiliary_cfg=aux_config,
        ),
        preprocess_code=args.preprocess,
        model_name=args.name,
        model_project=args.project,
        model_tags=args.tags or None,
        model_published=args.published,
    ):
        raise ValueError("Could not find base endpoint URL: {}".format(args.endpoint))

    print("Updating serving service")
    request_processor.serialize()


def cli():
    title = 'clearml-serving - CLI for launching ClearML serving engine'
    print(title)
    parser = ArgumentParser(prog='clearml-serving', description=title)
    parser.add_argument('--debug', action='store_true', help='Print debug messages')
    parser.add_argument('--yes', action='store_true', help='Always answer YES on interactive inputs')
    parser.add_argument(
        '--id', type=str,
        help='Control plane Task ID to configure '
             '(if not provided automatically detect the running control plane Task)')
    subparsers = parser.add_subparsers(help='Serving engine commands', dest='command')

    parser_list = subparsers.add_parser('list', help='List running Serving Service')
    parser_list.set_defaults(func=func_list_services)

    parser_create = subparsers.add_parser('create', help='Create a new Serving Service')
    parser_create.add_argument(
        '--name', type=str,
        help='[Optional] name the new serving service. Default: Serving-Service')
    parser_create.add_argument(
        '--tags', type=str, nargs='+',
        help='[Optional] Specify tags for the new serving service')
    parser_create.add_argument(
        '--project', type=str,
        help='[Optional] Specify project for the serving service. Default: DevOps')
    parser_create.set_defaults(func=func_create_service)

    parser_metrics = subparsers.add_parser('metrics', help='Configure inference metrics Service')
    parser_metrics.set_defaults(func=parser_metrics.print_help)

    metric_cmd = parser_metrics.add_subparsers(help='model metric command help')

    parser_metrics_add = metric_cmd.add_parser('add', help='Add/modify metric for a specific endpoint')
    parser_metrics_add.add_argument(
        '--endpoint', type=str, required=True,
        help='metric endpoint name including version, e.g. "model/1" or a prefix "model/*" '
             'Notice: it will override any previous endpoint logged metrics')
    parser_metrics_add.add_argument(
        '--log-freq', type=float,
        help='Optional: logging request frequency, between 0.0 to 1.0 '
             'example: 1.0 means all requests are logged, 0.5 means half of the requests are logged '
             'if not specified, use global logging frequency, see `config --metric-log-freq`')
    parser_metrics_add.add_argument(
        '--variable-scalar', type=str, nargs='+',
        help='Add float (scalar) argument to the metric logger, '
             '<name>=<histogram> example with specific buckets: "x1=0,0.2,0.4,0.6,0.8,1" or '
             'with min/max/num_buckets "x1=0.0/1.0/5"')
    parser_metrics_add.add_argument(
        '--variable-enum', type=str, nargs='+',
        help='Add enum (string) argument to the metric logger, '
             '<name>=<optional_values> example: "detect=cat,dog,sheep"')
    parser_metrics_add.add_argument(
        '--variable-value', type=str, nargs='+',
        help='Add non-samples scalar argument to the metric logger, '
             '<name> example: "latency"')
    parser_metrics_add.set_defaults(func=func_metric_add)

    parser_metrics_rm = metric_cmd.add_parser('remove', help='Remove metric from a specific endpoint')
    parser_metrics_rm.add_argument(
        '--endpoint', type=str, help='metric endpoint name including version, e.g. "model/1" or a prefix "model/*"')
    parser_metrics_rm.add_argument(
        '--variable', type=str, nargs='*',
        help='Remove (scalar/enum) argument from the metric logger, <name> example: "x1"')
    parser_metrics_rm.set_defaults(func=func_metric_rm)

    parser_metrics_ls = metric_cmd.add_parser('list', help='list metrics logged on all endpoints')
    parser_metrics_ls.set_defaults(func=func_metric_ls)

    parser_config = subparsers.add_parser('config', help='Configure a new Serving Service')
    parser_config.add_argument(
        '--base-serving-url', type=str,
        help='External base serving service url. example: http://127.0.0.1:8080/serve')
    parser_config.add_argument(
        '--triton-grpc-server', type=str,
        help='External ClearML-Triton serving container gRPC address. example: 127.0.0.1:9001')
    parser_config.add_argument(
        '--kafka-metric-server', type=str,
        help='External Kafka service url. example: 127.0.0.1:9092')
    parser_config.add_argument(
        '--metric-log-freq', type=float,
        help='Set default metric logging frequency. 1.0 is 100%% of all requests are logged')
    parser_config.set_defaults(func=func_config_service)

    parser_model = subparsers.add_parser('model', help='Configure Model endpoints for an already running Service')
    parser_model.set_defaults(func=parser_model.print_help)

    model_cmd = parser_model.add_subparsers(help='model command help')

    parser_model_ls = model_cmd.add_parser('list', help='List current models')
    parser_model_ls.set_defaults(func=func_model_ls)

    parser_model_rm = model_cmd.add_parser('remove', help='Remove model by it`s endpoint name')
    parser_model_rm.add_argument(
        '--endpoint', type=str, help='model endpoint name')
    parser_model_rm.set_defaults(func=func_model_remove)

    parser_model_upload = model_cmd.add_parser('upload', help='Upload and register model files/folder')
    parser_model_upload.add_argument(
        '--name', type=str, required=True,
        help='Specifying the model name to be registered in')
    parser_model_upload.add_argument(
        '--tags', type=str, nargs='+',
        help='Optional: Add tags to the newly created model')
    parser_model_upload.add_argument(
        '--project', type=str, required=True,
        help='Specifying the project for the model to be registered in')
    parser_model_upload.add_argument(
        '--framework', type=str,
        choices=[p for p in Framework.__dict__.keys()
                 if not p.startswith("_") and not callable(getattr(Framework, p))]+["custom"],
        help='[Optional] Specify the model framework: "scikit-learn", "xgboost", "lightgbm", "tensorflow", "pytorch"')
    parser_model_upload.add_argument(
        '--publish', action='store_true',
        help='[Optional] Publish the newly created model '
             '(change model state to "published" i.e. locked and ready to deploy')
    parser_model_upload.add_argument(
        '--path', type=str,
        help='Specifying a model file/folder to be uploaded and registered/')
    parser_model_upload.add_argument(
        '--url', type=str,
        help='Optional, Specifying an already uploaded model url '
             '(e.g. s3://bucket/model.bin, gs://bucket/model.bin, azure://bucket/model.bin, '
             'https://domain/model.bin)')
    parser_model_upload.add_argument(
        '--destination', type=str,
        help='Optional, Specifying the target destination for the model to be uploaded '
             '(e.g. s3://bucket/folder/, gs://bucket/folder/, azure://bucket/folder/)')
    parser_model_upload.set_defaults(func=func_model_upload)

    parser_model_lb = model_cmd.add_parser('canary', help='Add model Canary/A/B endpoint')
    parser_model_lb.add_argument(
        '--endpoint', type=str, help='model canary serving endpoint name (e.g. my_model/latest)')
    parser_model_lb.add_argument(
        '--weights', type=float, nargs='+', help='model canary weights (order matching model ep), (e.g. 0.2 0.8)')
    parser_model_lb.add_argument(
        '--input-endpoints', type=str, nargs='+',
        help='Model endpoint prefixes, can also include version (e.g. my_model, my_model/v1)')
    parser_model_lb.add_argument(
        '--input-endpoint-prefix', type=str,
        help='Model endpoint prefix, lexicographic order or by version <int> (e.g. my_model/1, my_model/v1) '
             'where the first weight matches the last version.')
    parser_model_lb.set_defaults(func=func_canary_add)

    parser_model_monitor = model_cmd.add_parser('auto-update', help='Add/Modify model auto update service')
    parser_model_monitor.add_argument(
        '--endpoint', type=str,
        help='Base Model endpoint (must be unique)')
    parser_model_monitor.add_argument(
        '--engine', type=str, required=True,
        help='Model endpoint serving engine (triton, sklearn, xgboost, lightgbm, custom, custom_async)')
    parser_model_monitor.add_argument(
        '--max-versions', type=int, default=1,
        help='max versions to store (and create endpoints) for the model. highest number is the latest version')
    parser_model_monitor.add_argument(
        '--name', type=str,
        help='Specify Model Name to be selected and auto updated '
             '(notice regexp selection use \"$name^\" for exact match)')
    parser_model_monitor.add_argument(
        '--tags', type=str, nargs='+',
        help='Specify Tags to be selected and auto updated')
    parser_model_monitor.add_argument(
        '--project', type=str,
        help='Specify Model Project to be selected and auto updated')
    parser_model_monitor.add_argument(
        '--published', action='store_true',
        help='Only select published Model for the auto updated')
    parser_model_monitor.add_argument(
        '--preprocess', type=str,
        help='Specify Pre/Post processing code to be used with the model (point to local file / folder) '
             '- this should hold for all the models'
    )
    parser_model_monitor.add_argument(
        '--input-size', nargs='+', type=json.loads,
        help='Optional: Specify the model matrix input size [Rows x Columns X Channels etc ...] '
             'if multiple inputs are required specify using json notation e.g.: '
             '\"[dim0, dim1, dim2, ...]\" \"[dim0, dim1, dim2, ...]\"'
    )
    parser_model_monitor.add_argument(
        '--input-type', nargs='+',
        help='Optional: Specify the model matrix input type, examples: uint8, float32, int16, float16 etc. '
             'if multiple inputs are required pass multiple values: float32, float32,'
    )
    parser_model_monitor.add_argument(
        '--input-name', nargs='+',
        help='Optional: Specify the model layer pushing input into, examples: layer_0 '
             'if multiple inputs are required pass multiple values: layer_0, layer_1,'
    )
    parser_model_monitor.add_argument(
        '--output-size', nargs='+', type=json.loads,
        help='Optional: Specify the model matrix output size [Rows x Columns X Channels etc ...] '
             'if multiple outputs are required specify using json notation e.g.: '
             '\"[dim0, dim1, dim2, ...]\" \"[dim0, dim1, dim2, ...]\"'
    )
    parser_model_monitor.add_argument(
        '--output-type', nargs='+',
        help='Optional: Specify the model matrix output type, examples: uint8, float32, int16, float16 etc. '
             'if multiple outputs are required pass multiple values: float32, float32,'
    )
    parser_model_monitor.add_argument(
        '--output-name', nargs='+',
        help='Optional: Specify the model layer pulling results from, examples: layer_99 '
             'if multiple outputs are required pass multiple values: layer_98, layer_99,'
    )
    parser_model_monitor.add_argument(
        '--aux-config', nargs='+',
        help='Specify additional engine specific auxiliary configuration in the form of key=value. '
             'Examples: platform=\\"onnxruntime_onnx\\" response_cache.enable=true max_batch_size=8 '
             'input.0.format=FORMAT_NCHW output.0.format=FORMAT_NCHW '
             'Remarks: (1) string must be quoted (e.g. key=\\"a_string\\") '
             '(2) instead of key/value pairs, you can also pass a full configuration file (e.g. "./config.pbtxt")'
    )
    parser_model_monitor.set_defaults(func=func_model_auto_update_add)

    parser_model_add = model_cmd.add_parser('add', help='Add/Update model')
    parser_model_add.add_argument(
        '--engine', type=str, required=True,
        help='Model endpoint serving engine (triton, sklearn, xgboost, lightgbm)')
    parser_model_add.add_argument(
        '--endpoint', type=str, required=True,
        help='Model endpoint (must be unique)')
    parser_model_add.add_argument(
        '--version', type=str, default=None,
        help='Model endpoint version (default: None)')
    parser_model_add.add_argument(
        '--model-id', type=str,
        help='Specify a Model ID to be served')
    parser_model_add.add_argument(
        '--preprocess', type=str,
        help='Specify Pre/Post processing code to be used with the model (point to local file / folder)'
    )
    parser_model_add.add_argument(
        '--input-size', nargs='+', type=json.loads,
        help='Optional: Specify the model matrix input size [Rows x Columns X Channels etc ...] '
             'if multiple inputs are required specify using json notation e.g.: '
             '\"[dim0, dim1, dim2, ...]\" \"[dim0, dim1, dim2, ...]\"'
    )
    parser_model_add.add_argument(
        '--input-type', nargs='+',
        help='Optional: Specify the model matrix input type, examples: uint8, float32, int16, float16 etc. '
             'if multiple inputs are required pass multiple values: float32, float32,'
    )
    parser_model_add.add_argument(
        '--input-name', nargs='+',
        help='Optional: Specify the model layer pushing input into, examples: layer_0 '
             'if multiple inputs are required pass multiple values: layer_0, layer_1,'
    )
    parser_model_add.add_argument(
        '--output-size', nargs='+', type=json.loads,
        help='Optional: Specify the model matrix output size [Rows x Columns X Channels etc ...] '
             'if multiple outputs are required specify using json notation e.g.: '
             '\"[dim0, dim1, dim2, ...]\" \"[dim0, dim1, dim2, ...]\"'
    )
    parser_model_add.add_argument(
        '--output-type', nargs='+',
        help='Optional: Specify the model matrix output type, examples: uint8, float32, int16, float16 etc. '
             'if multiple outputs are required pass multiple values: float32, float32,'
    )
    parser_model_add.add_argument(
        '--output-name', nargs='+',
        help='Optional: Specify the model layer pulling results from, examples: layer_99 '
             'if multiple outputs are required pass multiple values: layer_98, layer_99,'
    )
    parser_model_add.add_argument(
        '--aux-config', nargs='+',
        help='Specify additional engine specific auxiliary configuration in the form of key=value. '
             'Examples: platform=\\"onnxruntime_onnx\\" response_cache.enable=true max_batch_size=8 '
             'input.0.format=FORMAT_NCHW output.0.format=FORMAT_NCHW '
             'Remarks: (1) string must be quoted (e.g. key=\\"a_string\\") '
             '(2) instead of key/value pairs, you can also pass a full configuration file (e.g. "./config.pbtxt")'
    )
    parser_model_add.add_argument(
        '--name', type=str,
        help='[Optional] Instead of specifying model-id select based on Model Name')
    parser_model_add.add_argument(
        '--tags', type=str, nargs='+',
        help='[Optional] Instead of specifying model-id select based on Model Tags')
    parser_model_add.add_argument(
        '--project', type=str,
        help='[Optional] Instead of specifying model-id select based on Model project')
    parser_model_add.add_argument(
        '--published', action='store_true',
        help='[Optional] Instead of specifying model-id select based on Model published')
    parser_model_add.set_defaults(func=func_model_endpoint_add)

    args = parser.parse_args()
    global verbosity, answer_yes
    verbosity = args.debug
    answer_yes = args.yes

    if args.command:
        if args.command not in ("create", "list") and not args.id:
            print("Notice! serving service ID not provided, selecting the first active service")

        try:
            args.func(args)
        except AttributeError:
            args.func()
    else:
        parser.print_help()


def main():
    global verbosity
    try:
        cli()
    except KeyboardInterrupt:
        print('\nUser aborted')
    except Exception as ex:
        print('\nError: {}'.format(ex))
        if verbosity:
            raise ex
        exit(1)


if __name__ == '__main__':
    main()