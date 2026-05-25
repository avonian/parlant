"""
Request-scoped inline guideline resolution for the stateless engine.

In stateless mode the container's GuidelineStore is empty — all guidelines arrive
inline per /v2/process request. Most of the engine reads guidelines from the inline
data via InMemoryEntityQueries, but a few code paths (notably journey node selection,
which resolves a journey's `conditions` via GuidelineStore.read_guideline) reach for
the container's GuidelineStore directly.

`InlineAwareGuidelineStore` wraps the container store and answers read_guideline /
list_guidelines from the current request's inline guidelines first (set via
set_inline_guidelines, propagated to gathered tasks through a ContextVar so it is
concurrency-safe), delegating everything else to the wrapped store unchanged.
"""

from contextvars import ContextVar
from typing import Mapping, Sequence

from parlant.core.guidelines import Guideline, GuidelineId

_inline_guidelines: ContextVar[Mapping[GuidelineId, Guideline]] = ContextVar(
    "inline_guidelines", default={}
)


def set_inline_guidelines(guidelines: Sequence[Guideline]) -> None:
    """Set the inline guidelines for the current request context."""
    _inline_guidelines.set({g.id: g for g in guidelines})


class InlineAwareGuidelineStore:
    """Delegates to the wrapped GuidelineStore, but resolves read_guideline /
    list_guidelines from the request's inline guidelines first. Duck-typed against
    the GuidelineStore ABC (the engine never type-checks the instance)."""

    def __init__(self, inner: object) -> None:
        self._inner = inner

    async def read_guideline(self, guideline_id: GuidelineId) -> Guideline:
        inline = _inline_guidelines.get()
        guideline = inline.get(guideline_id)
        if guideline is not None:
            return guideline
        return await self._inner.read_guideline(guideline_id)  # type: ignore[attr-defined]

    async def list_guidelines(self, *args: object, **kwargs: object) -> Sequence[Guideline]:
        inline = _inline_guidelines.get()
        if inline:
            return list(inline.values())
        return await self._inner.list_guidelines(*args, **kwargs)  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> object:
        # Delegate every other attribute/method to the wrapped store unchanged.
        return getattr(self._inner, name)
