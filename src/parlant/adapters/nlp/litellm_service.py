# Copyright 2026 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import time
from typing import Any, Mapping
from typing_extensions import override
import json
import jsonfinder  # type: ignore
import os

from pydantic import ValidationError
import tiktoken

import litellm

# Allow LiteLLM to silently drop unsupported params (e.g. temperature for GPT-5)
# rather than raising UnsupportedParamsError.
litellm.drop_params = True

# Some models have deprecated the `temperature` parameter outright and return a
# 400 if it's sent (e.g. newer Claude like Opus 4.7). litellm.drop_params only
# strips params it *knows* are unsupported, which lags behind new deprecations.
# We learn these at runtime: on a temperature-related 400 we drop temperature,
# retry, and remember the model so subsequent requests skip it. This keeps the
# (low) temperature where the model honours it, and stays compatible where it
# doesn't — no hardcoded model list to maintain.
#
# The learned set is persisted to a gitignored JSON so a model is only ever
# learned once (the first request after a fresh deploy), not once per restart.
_QUIRKS_PATH = os.environ.get(
    "LITELLM_TEMPERATURE_QUIRKS_PATH", ".litellm_temperature_quirks.json"
)


def _load_temperature_rejecting_models() -> set[str]:
    try:
        with open(_QUIRKS_PATH) as f:
            data = json.load(f)
        if isinstance(data, list):
            return {str(m) for m in data}
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        pass
    return set()


_MODELS_REJECTING_TEMPERATURE: set[str] = _load_temperature_rejecting_models()


def _remember_temperature_rejecting_model(model_name: str) -> None:
    if model_name in _MODELS_REJECTING_TEMPERATURE:
        return
    _MODELS_REJECTING_TEMPERATURE.add(model_name)
    # Best-effort atomic persist; the in-memory set still works if this fails.
    try:
        tmp_path = f"{_QUIRKS_PATH}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(sorted(_MODELS_REJECTING_TEMPERATURE), f)
        os.replace(tmp_path, _QUIRKS_PATH)
    except OSError:
        pass

from parlant.adapters.nlp.common import normalize_json_output, record_llm_metrics
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger
from parlant.core.tracer import Tracer
from parlant.core.meter import Meter
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.service import (
    EmbedderHints,
    NLPService,
    SchematicGeneratorHints,
    StreamingTextGeneratorHints,
)
from parlant.core.nlp.embedding import BaseEmbedder, Embedder, EmbeddingResult, NullEmbedder
from parlant.core.nlp.generation import (
    T,
    BaseSchematicGenerator,
    SchematicGenerationResult,
    StreamingTextGenerator,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.moderation import (
    ModerationService,
    NoModeration,
)

RATE_LIMIT_ERROR_MESSAGE = (
    "LiteLLM to provider API rate limit exceeded. Possible reasons:\n"
    "1. Your account may have insufficient API credits.\n"
    "2. You may be using a free-tier account with limited request capacity.\n"
    "3. You might have exceeded the requests-per-minute limit for your account.\n\n"
    "Recommended actions:\n"
    "- Check your LLM Provider account balance and billing status.\n"
    "- Review your API usage limits in Provider's dashboard.\n"
    "- For more details on rate limits and usage tiers, visit:\n"
    "  Your Provider's API documentation."
)


class LiteLLMEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens)


class LiteLLMSchematicGenerator(BaseSchematicGenerator[T]):
    supported_litellm_params = [
        "temperature",
        "max_tokens",
        "logit_bias",
        "adapter_id",
        "adapter_source",
    ]
    supported_hints = supported_litellm_params + ["strict", "model_name"]

    def __init__(
        self,
        base_url: str | None,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
    ) -> None:
        super().__init__(logger=logger, tracer=tracer, meter=meter, model_name=model_name)

        self.base_url = base_url
        self._client = litellm

        self._tokenizer = LiteLLMEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"litellm/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> LiteLLMEstimatingTokenizer:
        return self._tokenizer

    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        litellm_api_arguments = {
            k: v for k, v in hints.items() if k in self.supported_litellm_params
        }

        # Only pass api_key if explicitly set; otherwise let LiteLLM auto-detect
        # provider-specific keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
        api_key = os.environ.get("LITELLM_PROVIDER_API_KEY")

        # Use hint model_name if provided, otherwise fall back to default
        model_name = hints.get("model_name") or self.model_name

        call_kwargs: dict[str, Any] = {
            "base_url": self.base_url,
            "api_key": api_key,
            "messages": [{"role": "user", "content": prompt}],
            "model": model_name,
            "max_tokens": 5000,
            "response_format": {"type": "json_object"},
            **litellm_api_arguments,
        }

        # Skip temperature for models already known to reject it.
        if model_name in _MODELS_REJECTING_TEMPERATURE:
            call_kwargs.pop("temperature", None)

        # GPT-5.6 models 400 on our /v1/chat/completions calls: "Function tools
        # with reasoning_effort are not supported ... use /v1/responses or set
        # reasoning_effort to 'none'". Disable reasoning explicitly (pre-5.6
        # models defaulted to none).
        if model_name.split("/")[-1].startswith("gpt-5.6"):
            call_kwargs.setdefault("reasoning_effort", "none")

        t_start = time.time()

        try:
            response = await self._client.acompletion(**call_kwargs)
        except litellm.BadRequestError as e:
            if "temperature" in str(e).lower() and "temperature" in call_kwargs:
                self.logger.warning(
                    f"Model {model_name} rejected `temperature`; retrying without it "
                    "and skipping it for this model from now on."
                )
                _remember_temperature_rejecting_model(model_name)
                call_kwargs.pop("temperature", None)
                response = await self._client.acompletion(**call_kwargs)
            else:
                raise

        t_end = time.time()

        if response.usage:
            self.logger.trace(response.usage.model_dump_json(indent=2))

        raw_content = response.choices[0].message.content or "{}"

        try:
            json_content = json.loads(normalize_json_output(raw_content))
        except json.JSONDecodeError:
            self.logger.warning(
                f"Invalid JSON returned by litellm/{model_name}:\n{raw_content})"
            )
            json_content = jsonfinder.only_json(raw_content)[2]
            self.logger.warning("Found JSON content within model response; continuing...")

        try:
            content = self.schema.model_validate(json_content)
            assert response.usage

            await record_llm_metrics(
                self.meter,
                model_name,
                schema_name=self.schema.__name__,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cached_input_tokens=getattr(
                    response,
                    "usage.prompt_cache_hit_tokens",
                    0,
                ),
            )

            return SchematicGenerationResult(
                content=content,
                info=GenerationInfo(
                    schema_name=self.schema.__name__,
                    model=f"litellm/{model_name}",
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        extra={
                            "cached_input_tokens": getattr(
                                response,
                                "usage.prompt_cache_hit_tokens",
                                0,
                            )
                        },
                    ),
                ),
            )
        except ValidationError:
            self.logger.error(
                f"JSON content returned by litellm/{model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class LiteLLM_Default(LiteLLMSchematicGenerator[T]):
    def __init__(
        self, logger: Logger, tracer: Tracer, meter: Meter, base_url: str | None, model_name: str
    ) -> None:
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            logger=logger,
            tracer=tracer,
            meter=meter,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 5000

    # 8192 16381


class LiteLLMEmbedder(BaseEmbedder):
    """Embedder that uses LiteLLM to access various embedding providers."""

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        base_url: str | None = None,
    ) -> None:
        super().__init__(logger, tracer, meter, model_name)
        self._base_url = base_url
        self._client = litellm
        self._tokenizer = LiteLLMEstimatingTokenizer(model_name=model_name)

    @property
    @override
    def id(self) -> str:
        return f"litellm/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> LiteLLMEstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        return int(os.environ.get("LITELLM_EMBEDDING_MAX_TOKENS", 8192))

    @property
    @override
    def dimensions(self) -> int:
        return int(os.environ.get("LITELLM_EMBEDDING_DIMENSIONS", 1536))

    @override
    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        api_key = os.environ.get("LITELLM_PROVIDER_API_KEY")

        response = await self._client.aembedding(
            model=self.model_name,
            input=texts,
            api_key=api_key,
            api_base=self._base_url,
        )

        vectors = [data["embedding"] for data in response.data]
        return EmbeddingResult(vectors=vectors)


class LiteLLMService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("LITELLM_PROVIDER_MODEL_NAME"):
            return """\
You're using the LITELLM NLP service, but LITELLM_PROVIDER_MODEL_NAME is not set.
Please set LITELLM_PROVIDER_MODEL_NAME in your environment before running Parlant.
"""
        # Note: LITELLM_PROVIDER_API_KEY is optional. If not set, LiteLLM will
        # auto-detect provider-specific keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)

        return None

    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        self._base_url = os.environ.get("LITELLM_PROVIDER_BASE_URL")
        self._model_name = os.environ["LITELLM_PROVIDER_MODEL_NAME"]
        self._embedding_model_name = os.environ.get("LITELLM_EMBEDDING_MODEL_NAME")
        self.logger = logger
        self._tracer = tracer
        self._meter = meter

        log_msg = f"Initialized LiteLLMService with {self._model_name}"
        if self._embedding_model_name:
            log_msg += f" (embeddings: {self._embedding_model_name})"
        if self._base_url:
            log_msg += f" at {self._base_url}"
        self.logger.info(log_msg)

    @property
    @override
    def supports_streaming(self) -> bool:
        return False

    @override
    async def get_streaming_text_generator(
        self, hints: StreamingTextGeneratorHints = {}
    ) -> StreamingTextGenerator:
        raise NotImplementedError("Streaming is not supported. Check supports_streaming first.")

    @override
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> LiteLLMSchematicGenerator[T]:
        return LiteLLM_Default[t](  # type: ignore
            self.logger, self._tracer, self._meter, self._base_url, self._model_name
        )

    @override
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        if os.environ.get("LITELLM_DISABLE_EMBEDDER"):
            return NullEmbedder()
        if self._embedding_model_name:
            return LiteLLMEmbedder(
                model_name=self._embedding_model_name,
                logger=self.logger,
                tracer=self._tracer,
                meter=self._meter,
                base_url=self._base_url,
            )
        # Imported lazily: this pulls in torch + transformers (~4.6 GB resident
        # once the Jina model loads), which stateless deployments never need.
        from parlant.adapters.nlp.hugging_face import JinaAIEmbedder

        return JinaAIEmbedder(self.logger, self._tracer, self._meter)

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
