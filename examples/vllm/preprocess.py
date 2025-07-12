"""Hugginface preprocessing module for ClearML Serving."""
from typing import Any, Optional, List, Callable, Union


class Preprocess:
    """Processing class will be run by the ClearML inference services before and after each request."""

    def __init__(self):
        """Set internal state, this will be called only once. (i.e. not per request)."""
        self.model_endpoint = None

    def load(self, local_file_name: str) -> Optional[Any]:  # noqa
        vllm_model_config = {
            "lora_modules": None, # [LoRAModulePath(name=a, path=b)]
            "prompt_adapters": None, # [PromptAdapterPath(name=a, path=b)]
            "response_role": "assistant",
            "chat_template": None,
            "return_tokens_as_token_ids": False,
            "max_log_len": None
        }
        chat_settings = {
            "reasoning_parser": None,
            "enable_auto_tools": False,
            "expand_tools_even_if_tool_choice_none": False,
            "tool_parser": None,
            "enable_prompt_tokens_details": False,
            "enable_force_include_usage": False,
            "chat_template_content_format": "auto"
        }
        return {
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
