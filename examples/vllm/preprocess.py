"""Hugginface preprocessing module for ClearML Serving."""
from typing import Any, Optional, List, Callable, Union


class Preprocess:
    """Processing class will be run by the ClearML inference services before and after each request."""

    def __init__(self):
        """Set internal state, this will be called only once. (i.e. not per request)."""
        self.model_endpoint = None

    def load(self, local_file_name: str) -> Optional[Any]:  # noqa

        # vllm_engine_config = {
        #     "model": local_file_name,
        #     "tokenizer": local_file_name,
        #     "disable_log_requests": True,
        #     "disable_log_stats": False,
        #     "gpu_memory_utilization": 0.9,
        #     "quantization": None,
        #     "enforce_eager": True,
        #     "served_model_name": "test_vllm",
        #     "dtype": "float16",
        #     "max_model_len": 8192
        # }
        vllm_model_config = {
            "lora_modules": None, # [LoRAModulePath(name=a, path=b)]
            "prompt_adapters": None, # [PromptAdapterPath(name=a, path=b)]
            "response_role": "assistant",
            "chat_template": None,
            "return_tokens_as_token_ids": False,
            "max_log_len": None
        }
        chat_settings = {
            "enable_reasoning": False,
            "reasoning_parser": None,
            "enable_auto_tools": False,
            "tool_parser": None,
            "enable_prompt_tokens_details": False,
            "chat_template_content_format": "auto"
        }
        # self._model = {}
        # engine_args = AsyncEngineArgs(**self.vllm_engine_config)
        # async_engine_client = AsyncLLMEngine.from_engine_args(self.engine_args, usage_context=UsageContext.OPENAI_API_SERVER)
        # model_config = async_engine_client.engine.get_model_config()
        # request_logger = RequestLogger(max_log_len=self.vllm_model_config["max_log_len"])
        # self._model["openai_serving_models"] = OpenAIServingModels(
        #     async_engine_client,
        #     self.model_config,
        #     [BaseModelPath(name=self.vllm_engine_config["served_model_name"], model_path=self.vllm_engine_config["model"])],
        #     lora_modules=self.vllm_model_config["lora_modules"],
        #     prompt_adapters=self.vllm_model_config["prompt_adapters"],
        # )
        # self._model["openai_serving"] = OpenAIServing(
        #     async_engine_client,
        #     self.model_config,
        #     self._model["openai_serving_models"],
        #     request_logger=request_logger,
        #     return_tokens_as_token_ids=self.vllm_model_config["return_tokens_as_token_ids"]
        # )
        # self._model["openai_serving_chat"] = OpenAIServingChat(
        #     async_engine_client,
        #     self.model_config,
        #     self._model["openai_serving_models"],
        #     response_role=self.vllm_model_config["response_role"],
        #     request_logger=request_logger,
        #     chat_template=self.vllm_model_config["chat_template"],
        #     chat_template_content_format=self.chat_settings["chat_template_content_format"],
        #     return_tokens_as_token_ids=self.vllm_model_config["return_tokens_as_token_ids"],
        #     enable_reasoning=self.chat_settings["enable_reasoning"],
        #     reasoning_parser=self.chat_settings["reasoning_parser"],
        #     enable_auto_tools=self.chat_settings["enable_auto_tools"],
        #     tool_parser=self.chat_settings["tool_parser"],
        #     enable_prompt_tokens_details=self.chat_settings["enable_prompt_tokens_details"]
        # ) if self.model_config.runner_type == "generate" else None
        # self._model["openai_serving_completion"] = OpenAIServingCompletion(
        #     async_engine_client,
        #     self.model_config,
        #     self._model["openai_serving_models"],
        #     request_logger=request_logger,
        #     return_tokens_as_token_ids=self.vllm_model_config["return_tokens_as_token_ids"]
        # ) if self.model_config.runner_type == "generate" else None
        # self._model["openai_serving_embedding"] = OpenAIServingEmbedding(
        #     async_engine_client,
        #     self.model_config,
        #     self._model["openai_serving_models"],
        #     request_logger=request_logger,
        #     chat_template=self.vllm_model_config["chat_template"],
        #     chat_template_content_format=self.chat_settings["chat_template_content_format"]
        # ) if self.model_config.task == "embed" else None
        # self._model["openai_serving_tokenization"] = OpenAIServingTokenization(
        #     async_engine_client,
        #     self.model_config,
        #     self._model["openai_serving_models"],
        #     request_logger=request_logger,
        #     chat_template=self.vllm_model_config["chat_template"],
        #     chat_template_content_format=self.chat_settings["chat_template_content_format"]
        # )
        # return self._model
        return {
            # "vllm_engine_config": vllm_engine_config,
            "vllm_model_config": vllm_model_config,
            "chat_settings": chat_settings
        }

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

    async def preprocess(
        self,
        body: Union[bytes, dict],
        state: dict,
        collect_custom_statistics_fn: Optional[Callable[[dict], None]],
    ) -> Any:  # noqa
        if "messages" in body["request"]:
            body["request"]["messages"] = self.remove_extra_system_prompts(body["request"]["messages"])
        return body
