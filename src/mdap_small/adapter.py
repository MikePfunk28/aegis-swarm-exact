import httpx

from mdap_small.models import ModelSpec


class OllamaAdapter:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        openai_compat_url: str = "http://127.0.0.1:1234/v1",
        openai_compat_api_key: str = "not-needed",
    ):
        self.base_url = base_url.rstrip("/")
        self.openai_compat_url = openai_compat_url.rstrip("/")
        self.openai_compat_api_key = openai_compat_api_key

    async def generate(self, model: ModelSpec, prompt: str, temperature: float) -> str:
        if model.provider == "openai_compat":
            return await self._generate_openai_compat(model, prompt, temperature)
        return await self._generate_ollama(model, prompt, temperature)

    async def health(self, model: ModelSpec) -> dict:
        if model.provider == "openai_compat":
            return await self._health_openai_compat(model)
        return await self._health_ollama(model)

    async def _generate_ollama(
        self,
        model: ModelSpec,
        prompt: str,
        temperature: float,
    ) -> str:
        payload = {
            "model": model.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": model.max_tokens,
            },
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            if resp.status_code >= 400:
                detail = ""
                try:
                    data = resp.json()
                    if isinstance(data, dict):
                        detail = str(data.get("error") or data)
                except Exception:
                    detail = resp.text.strip()
                raise RuntimeError(
                    f"ollama_generate_failed status={resp.status_code} detail={detail}"
                )
            data = resp.json()
            return data.get("response", "")

    async def _generate_openai_compat(
        self,
        model: ModelSpec,
        prompt: str,
        temperature: float,
    ) -> str:
        payload = {
            "model": model.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": model.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.openai_compat_api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.openai_compat_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"openai_compat_generate_failed status={resp.status_code} detail={resp.text.strip()}"
                )
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return ""
            return choices[0].get("message", {}).get("content", "")

    async def _health_ollama(self, model: ModelSpec) -> dict:
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                tags = await client.get(f"{self.base_url}/api/tags")
                tags.raise_for_status()
                payload = tags.json()
                names = {
                    item.get("name", "")
                    for item in payload.get("models", [])
                    if isinstance(item, dict)
                }
                present = any(
                    model.model_id == name or model.model_id in name for name in names
                )
                return {
                    "ok": True,
                    "provider": "ollama",
                    "model": model.model_id,
                    "loaded": present,
                }
        except Exception as exc:
            return {
                "ok": False,
                "provider": "ollama",
                "model": model.model_id,
                "error": str(exc),
            }

    async def _health_openai_compat(self, model: ModelSpec) -> dict:
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_compat_api_key}",
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient(timeout=20) as client:
                res = await client.get(
                    f"{self.openai_compat_url}/models", headers=headers
                )
                res.raise_for_status()
                payload = res.json()
                ids = {
                    item.get("id", "")
                    for item in payload.get("data", [])
                    if isinstance(item, dict)
                }
                present = model.model_id in ids
                return {
                    "ok": True,
                    "provider": "openai_compat",
                    "model": model.model_id,
                    "loaded": present,
                }
        except Exception as exc:
            return {
                "ok": False,
                "provider": "openai_compat",
                "model": model.model_id,
                "error": str(exc),
            }
