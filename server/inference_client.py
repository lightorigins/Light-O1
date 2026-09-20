"""HTTP-only GPU connection: no model files or inference engine on the Control host."""

import httpx


class InferenceClient:
    def __init__(self, url: str, *, transport: httpx.AsyncBaseTransport | None = None):
        self.client = httpx.AsyncClient(
            base_url=url, timeout=httpx.Timeout(600, connect=5), transport=transport, trust_env=False
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        await self.client.aclose()

    async def _request(self, method, path, **kwargs):
        try:
            response = await self.client.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise RuntimeError("Inference API unavailable; check its address and GPU service logs") from error
        if response.is_error:
            if response.status_code == 409:
                raise RuntimeError("Inference model is busy; wait for the active generation to finish")
            if response.status_code == 422:
                raise RuntimeError("Inference request rejected; check context length and generation budgets")
            raise RuntimeError(f"Inference API returned HTTP {response.status_code}; check GPU service logs")
        try:
            result = response.json()
        except ValueError as error:
            raise RuntimeError("Inference API returned invalid JSON") from error
        if not isinstance(result, dict):
            raise RuntimeError("Inference API returned an invalid response")
        return result

    async def health(self):
        result = await self._request("GET", "/health", timeout=3)
        if type(result.get("ready")) is not bool or not isinstance(result.get("model"), str) or not result["model"]:
            raise RuntimeError("Inference API returned invalid health fields")
        if (result.get("representation"), result.get("action_dim"), result.get("fps")) != (
            "human_action_138_v1",
            138,
            20,
        ):
            raise RuntimeError("Inference API does not support human_action_138_v1 at 20 FPS")
        return result

    async def generate(self, request):
        return await self._request("POST", "/api/generate", json=request)
