import os
import shlex
import traceback
import gzip
import asyncio
import time
import uuid

from fastapi import FastAPI, Request, Response, APIRouter, HTTPException, Depends
from fastapi.routing import APIRoute
from fastapi.responses import PlainTextResponse, JSONResponse
from grpc.aio import AioRpcError

from http import HTTPStatus

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest

from starlette.background import BackgroundTask

from typing import Optional, Dict, Any, Callable, Union

from clearml_serving.version import __version__
from clearml_serving.serving.init import setup_task
from clearml_serving.serving.model_request_processor import (
    ModelRequestProcessor,
    EndpointNotFoundException,
    EndpointBackendEngineException,
    EndpointModelLoadException,
    ServingInitializationException,
)
from clearml_serving.serving.utils import parse_grpc_errors


class GzipRequest(Request):
    async def body(self) -> bytes:
        if not hasattr(self, "_body"):
            body = await super().body()
            if "gzip" in self.headers.getlist("Content-Encoding"):
                body = gzip.decompress(body)
            self._body = body  # noqa
        return self._body


class GzipRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            request = GzipRequest(request.scope, request.receive)
            return await original_route_handler(request)

        return custom_route_handler


# process Lock, so that we can have only a single process doing the model reloading at a time
singleton_sync_lock = None  # Lock()
# shared Model processor object
processor = None  # type: Optional[ModelRequestProcessor]

# create clearml Task and load models
serving_service_task_id, session_logger, instance_id = setup_task()

# Health check tracking variables
startup_time = time.time()
service_instance_id = str(uuid.uuid4())[:8]

# polling frequency
model_sync_frequency_secs = 5
try:
    model_sync_frequency_secs = float(os.environ.get("CLEARML_SERVING_POLL_FREQ", model_sync_frequency_secs))
except (ValueError, TypeError):
    pass


grpc_aio_ignore_errors = parse_grpc_errors(shlex.split(os.environ.get("CLEARML_SERVING_AIO_RPC_IGNORE_ERRORS", "")))
grpc_aio_verbose_errors = parse_grpc_errors(shlex.split(os.environ.get("CLEARML_SERVING_AIO_RPC_VERBOSE_ERRORS", "")))


class CUDAException(Exception):
    def __init__(self, exception: str):
        self.exception = exception


# start FastAPI app
app = FastAPI(title="ClearML Serving Service", version=__version__, description="ClearML Service Service router")


@app.on_event("startup")
async def startup_event():
    global processor

    if processor:
        print(
            "ModelRequestProcessor already initialized [pid={}] [service_id={}]".format(
                os.getpid(), serving_service_task_id
            )
        )
    else:
        print("Starting up ModelRequestProcessor [pid={}] [service_id={}]".format(os.getpid(), serving_service_task_id))
        processor = ModelRequestProcessor(
            task_id=serving_service_task_id,
            update_lock_guard=singleton_sync_lock,
        )
        print("ModelRequestProcessor [id={}] loaded".format(processor.get_id()))
        processor.launch(poll_frequency_sec=model_sync_frequency_secs * 60)


@app.on_event("shutdown")
def shutdown_event():
    print("RESTARTING INFERENCE SERVICE!")


async def exit_app():
    loop = asyncio.get_running_loop()
    loop.stop()


@app.exception_handler(CUDAException)
async def cuda_exception_handler(request, exc):
    task = BackgroundTask(exit_app)
    return PlainTextResponse("CUDA out of memory. Restarting service", status_code=500, background=task)

def check_cuda_oom_exception(ex: Exception):
    if "CUDA out of memory. " in str(ex) or "NVML_SUCCESS == r INTERNAL ASSERT FAILED" in str(ex):
        if os.environ.get("CLEARML_SERVING_DEV_CUDAEXCEPTION", "0") != "0":
            raise CUDAException(exception=ex)
        # can't always recover from this - prefer to exit the program such that it can be restarted
        os._exit(1)
    else:
        raise HTTPException(status_code=422, detail="Error [{}] processing request: {}".format(type(ex), ex))

async def process_with_exceptions(
    base_url: str,
    version: Optional[str],
    request: Union[bytes, Dict[Any, Any]],
    serve_type: str
):
    processor.on_request_endpoint_telemetry(base_url=base_url, version=version)
    try:
        return_value = await processor.process_request(
            base_url=base_url,
            version=version,
            request_body=request,
            serve_type=serve_type
        )
    except EndpointNotFoundException as ex:
        raise HTTPException(status_code=404, detail="Error processing request, endpoint was not found: {}".format(ex))
    except (EndpointModelLoadException, EndpointBackendEngineException) as ex:
        session_logger.report_text(
            "[{}] Exception [{}] {} while processing request: {}\n{}".format(
                instance_id, type(ex), ex, request, "".join(traceback.format_exc())
            )
        )
        raise HTTPException(status_code=422, detail="Error [{}] processing request: {}".format(type(ex), ex))
    except ServingInitializationException as ex:
        session_logger.report_text(
            "[{}] Exception [{}] {} while loading serving inference: {}\n{}".format(
                instance_id, type(ex), ex, request, "".join(traceback.format_exc())
            )
        )
        raise HTTPException(status_code=500, detail="Error [{}] processing request: {}".format(type(ex), ex))
    except ValueError as ex:
        session_logger.report_text(
            "[{}] Exception [{}] {} while processing request: {}\n{}".format(
                instance_id, type(ex), ex, request, "".join(traceback.format_exc())
            )
        )
        check_cuda_oom_exception(ex)
    except AioRpcError as ex:
        if grpc_aio_verbose_errors and ex.code() in grpc_aio_verbose_errors:
            session_logger.report_text(
                "[{}] Exception [AioRpcError] {} while processing request: {}".format(instance_id, ex, request)
            )
        elif not grpc_aio_ignore_errors or ex.code() not in grpc_aio_ignore_errors:
            session_logger.report_text("[{}] Exception [AioRpcError] status={} ".format(instance_id, ex.code()))
        raise HTTPException(
            status_code=500, detail="Error [AioRpcError] processing request: status={}".format(ex.code())
        )
    except Exception as ex:
        session_logger.report_text(
            "[{}] Exception [{}] {} while processing request: {}\n{}".format(
                instance_id, type(ex), ex, request, "".join(traceback.format_exc())
            )
        )
        check_cuda_oom_exception(ex)
    processor.on_response_endpoint_telemetry(base_url=base_url, version=version)
    return return_value


# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """
    Basic health check endpoint.
    Returns 200 OK when service is running.
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "clearml-serving",
            "version": __version__,
            "timestamp": time.time(),
            "instance_id": service_instance_id,
        },
    )


@app.get("/readiness")
async def readiness_check():
    """
    Readiness check endpoint.
    Returns 200 if ready to serve requests, 503 if not ready.
    Checks if ModelRequestProcessor is initialized and models are loaded.
    """
    global processor

    if not processor:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "reason": "Processor not initialized",
                "timestamp": time.time(),
            },
        )

    try:
        # Check if models are loaded
        models_loaded = processor.get_loaded_endpoints()
        if not models_loaded:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "reason": "No models loaded",
                    "timestamp": time.time(),
                },
            )

        # Check GPU availability if applicable
        gpu_available = False
        try:
            import torch

            gpu_available = torch.cuda.is_available()
        except (ImportError, ModuleNotFoundError, AttributeError):
            # torch not installed or CUDA not available
            pass

        return JSONResponse(
            status_code=200,
            content={
                "status": "ready",
                "models_loaded": len(models_loaded),
                "gpu_available": gpu_available,
                "timestamp": time.time(),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "reason": f"Error checking readiness: {str(e)}",
                "timestamp": time.time(),
            },
        )


@app.get("/liveness")
async def liveness_check():
    """
    Liveness check endpoint.
    Lightweight check for container orchestration.
    Returns 200 OK if process is responsive.
    """
    return JSONResponse(
        status_code=200, content={"status": "alive", "timestamp": time.time()}
    )


@app.get("/metrics")
async def metrics_endpoint():
    """
    Detailed metrics endpoint.
    Returns service metrics including uptime, request count, GPU usage, etc.
    """
    global processor

    uptime_seconds = time.time() - startup_time

    metrics = {
        "uptime_seconds": round(uptime_seconds, 2),
        "total_requests": 0,
        "last_prediction_timestamp": None,
        "models": [],
    }

    if processor:
        try:
            metrics["total_requests"] = processor.get_request_count()
            metrics["last_prediction_timestamp"] = processor.get_last_prediction_time()

            # Get loaded models info
            loaded_endpoints = processor.get_loaded_endpoints()
            for endpoint_name in loaded_endpoints:
                metrics["models"].append({"endpoint": endpoint_name, "loaded": True})
        except AttributeError:
            # If methods don't exist yet, continue with basic metrics
            pass

    # Try to get GPU metrics
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        metrics["gpu_memory_used_mb"] = round(info.used / 1024 / 1024, 2)
        metrics["gpu_memory_total_mb"] = round(info.total / 1024 / 1024, 2)
        pynvml.nvmlShutdown()
    except (ImportError, ModuleNotFoundError, AttributeError, OSError):
        # GPU metrics not available (pynvml not installed, no GPU, or driver issues)
        metrics["gpu_memory_used_mb"] = None
        metrics["gpu_memory_total_mb"] = None

    return JSONResponse(status_code=200, content=metrics)

router = APIRouter(
    prefix=f"/{os.environ.get("CLEARML_DEFAULT_SERVE_SUFFIX", "serve")}",
    tags=["models"],
    responses={404: {"description": "Model Serving Endpoint Not found"}},
    route_class=GzipRoute,  # mark-out to remove support for GZip content encoding
)


@router.post("/{model_id}/{version}")
@router.post("/{model_id}/")
@router.post("/{model_id}")
async def base_serve_model(
    model_id: str,
    version: Optional[str] = None,
    request: Union[bytes, Dict[Any, Any]] = None
):
    return_value = await process_with_exceptions(
        base_url=model_id,
        version=version,
        request=request,
        serve_type="process"
    )
    return return_value


async def validate_json_request(raw_request: Request):
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise HTTPException(
            status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported Media Type: Only 'application/json' is allowed"
        )

@router.post("/openai/{endpoint_type:path}", dependencies=[Depends(validate_json_request)])
@router.get("/openai/{endpoint_type:path}", dependencies=[Depends(validate_json_request)])
async def openai_serve_model(
    endpoint_type: str,
    request: Union[CompletionRequest, ChatCompletionRequest],
    raw_request: Request
):
    combined_request = {"request": request, "raw_request": raw_request}
    return_value = await process_with_exceptions(
        base_url=request.model,
        version=None,
        request=combined_request,
        serve_type=endpoint_type
    )
    return return_value

app.include_router(router)
