"""Hugginface preprocessing module for ClearML Serving."""
from typing import Any, Optional
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_tokenization import OpenAIServingTokenization
from vllm.usage.usage_lib import UsageContext
from vllm.entrypoints.openai.serving_engine import LoRAModulePath, PromptAdapterPath


class Preprocess:
    """Processing class will be run by the ClearML inference services before and after each request."""

    def __init__(self):
        """Set internal state, this will be called only once. (i.e. not per request)."""
        self.model_endpoint = None

    def load(self, local_file_name: str) -> Optional[Any]:  # noqa
        vllm_engine_config = {
            "model": local_file_name,
            "tokenizer": local_file_name,
            "disable_log_requests": True,
            "disable_log_stats": False,
            "gpu_memory_utilization": 0.9,
            "quantization": None,
            "enforce_eager": True,
            "served_model_name": "test_vllm",
            "dtype": "float16"
        }
        vllm_model_config = {
            "lora_modules": None, # [LoRAModulePath(name=a, path=b)]
            "prompt_adapters": None, # [PromptAdapterPath(name=a, path=b)]
            "response_role": "assistant",
            "chat_template": None,
            "return_tokens_as_token_ids": False,
            "max_log_len": None
        }
        self._model = {}
        engine_args = AsyncEngineArgs(**vllm_engine_config)
        async_engine_client = AsyncLLMEngine.from_engine_args(engine_args, usage_context=UsageContext.OPENAI_API_SERVER)
        model_config = async_engine_client.engine.get_model_config()
        request_logger = RequestLogger(max_log_len=vllm_model_config["max_log_len"])
        self._model["openai_serving_chat"] = OpenAIServingChat(
            async_engine_client,
            model_config,
            served_model_names=[vllm_engine_config["served_model_name"]],
            response_role=vllm_model_config["response_role"],
            lora_modules=vllm_model_config["lora_modules"],
            prompt_adapters=vllm_model_config["prompt_adapters"],
            request_logger=request_logger,
            chat_template=vllm_model_config["chat_template"],
            return_tokens_as_token_ids=vllm_model_config["return_tokens_as_token_ids"]
        )
        self._model["openai_serving_completion"] = OpenAIServingCompletion(
            async_engine_client,
            model_config,
            served_model_names=[vllm_engine_config["served_model_name"]],
            lora_modules=vllm_model_config["lora_modules"],
            prompt_adapters=vllm_model_config["prompt_adapters"],
            request_logger=request_logger,
            return_tokens_as_token_ids=vllm_model_config["return_tokens_as_token_ids"]
        )
        self._model["openai_serving_embedding"] = OpenAIServingEmbedding(
            async_engine_client,
            model_config,
            served_model_names=[vllm_engine_config["served_model_name"]],
            request_logger=request_logger
        )
        self._model["openai_serving_tokenization"] = OpenAIServingTokenization(
            async_engine_client,
            model_config,
            served_model_names=[vllm_engine_config["served_model_name"]],
            lora_modules=vllm_model_config["lora_modules"],
            request_logger=request_logger,
            chat_template=vllm_model_config["chat_template"]
        )
        return self._model

    def remove_extra_system_prompts(self, messages: List) -> List:
        system_messages_indices = []
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                system_messages_indices.append(i)
            else:
                break
        if len(system_messages_indices) > 1:
            last_system_index = system_messages_indices[-1]
            messages = [msg for i, msg in enumerate(messages) if msg["role"] != "system" or i == last_system_index]
        return messages

    def preprocess(
        self,
        body: Union[bytes, dict],
        state: dict,
        collect_custom_statistics_fn: Optional[Callable[[dict], None]],
    ) -> Any:  # noqa
        if "messages" in body:
            body["messages"] = self.remove_extra_system_prompts(body["messages"])
        return body
