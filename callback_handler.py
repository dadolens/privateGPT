from datetime import datetime
from typing import Any, Dict, List
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import LLMResult


class CustomCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self) -> None:
        self.clear_timer()

    def clear_timer(self):
        self.start_time = None
        self.end_time = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        self.start_time = datetime.now()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if self.end_time is None:
            self.end_time = datetime.now()

    def get_requested_time(self):
        if self.start_time is None or self.start_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()
