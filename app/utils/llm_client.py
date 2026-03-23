import json


class LLMClient:
    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Заглушка.
        Здесь потом будет вызов реальной LLM.
        Пока можно возвращать тестовый JSON вручную.
        """
        raise NotImplementedError("Connect real LLM API here.")

    def generate_structured_text(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError("Connect real LLM API here.")