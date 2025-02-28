"""Hugginface preprocessing module for ClearML Serving."""
from typing import Any


# Notice Preprocess class Must be named "Preprocess"
class Preprocess:
    """Processing class will be run by the ClearML inference services before and after each request."""

    def __init__(self):
        """Set internal state, this will be called only once. (i.e. not per request)."""
        self.model_endpoint = None

    def load(self, local_file_name: str) -> Optional[Any]:  # noqa
        vllm_engine_config = {
            "model":f"{local_file_name}/model",
            "tokenizer":f"{local_file_name}/tokenizer",
            "disable_log_requests": True,
            "disable_log_stats": False,
            "gpu_memory_utilization": 0.9,
            "quantization": None,
            "enforce_eager": True,
            "served_model_name": "ai_operator_hyp22v4"
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
        self._model["engine_args"] = AsyncEngineArgs(**vllm_engine_config)
        self._model["async_engine_client"] = AsyncLLMEngine.from_engine_args(self.engine_args, usage_context=UsageContext.OPENAI_API_SERVER)


        self._model["model_config"] = self.async_engine_client.engine.get_model_config()

        self._model["request_logger"] = RequestLogger(max_log_len=vllm_model_config["max_log_len"])

        self._model["self.openai_serving_chat"] = OpenAIServingChat(
            self.async_engine_client,
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
            self.async_engine_client,
            model_config,
            served_model_names=[vllm_engine_config["served_model_name"]],
            lora_modules=vllm_model_config["lora_modules"],
            prompt_adapters=vllm_model_config["prompt_adapters"],
            request_logger=request_logger,
            return_tokens_as_token_ids=vllm_model_config["return_tokens_as_token_ids"]
        )
        self._model["self.openai_serving_embedding"] = OpenAIServingEmbedding(
            self.async_engine_client,
            model_config,
            served_model_names=[vllm_engine_config["served_model_name"]],
            request_logger=request_logger
        )
        self._model["self.openai_serving_tokenization"] = OpenAIServingTokenization(
            self.async_engine_client,
            model_config,
            served_model_names=[vllm_engine_config["served_model_name"]],
            lora_modules=vllm_model_config["lora_modules"],
            request_logger=request_logger,
            chat_template=vllm_model_config["chat_template"]
        )
