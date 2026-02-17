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

import asyncio
from contextvars import ContextVar
import os
import traceback
from typing import Any, Awaitable, Callable, Mapping, TypeAlias

import mimetypes

from fastapi import APIRouter, FastAPI, Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from starlette.types import Receive, Scope, Send
from starlette.routing import Match


from lagom import Container

from parlant.adapters.loggers.websocket import WebSocketLogger
from parlant.api import agents, capabilities
from parlant.api import agent_tool_associations
from parlant.api import evaluations
from parlant.api import journeys
from parlant.api import playbooks
from parlant.api import relationships
from parlant.api import sessions
from parlant.api import glossary
from parlant.api import guidelines
from parlant.api import context_variables as variables
from parlant.api import services
from parlant.api import tags
from parlant.api import customers
from parlant.api import logs
from parlant.api import canned_responses
from parlant.api import test_suites
from parlant.api import static_playbooks
from parlant.api import playbook_resolve
from parlant.api.authorization import (
    AuthorizationException,
    AuthorizationPolicy,
    Operation,
    RateLimitExceededException,
)
from parlant.core.version import VERSION
from parlant.core.meter import Meter
from parlant.core.tracer import Tracer
from parlant.core.common import ItemNotFoundError, generate_id
from parlant.core.loggers import Logger
from parlant.core.agent_tool_associations import AgentToolAssociationStore
from parlant.core.canned_responses import CannedResponseStore
from parlant.core.context_variables import ContextVariableStore
from parlant.core.glossary import GlossaryStore
from parlant.core.guidelines import GuidelineStore
from parlant.core.guideline_tool_associations import GuidelineToolAssociationStore
from parlant.core.journey_guideline_projection import JourneyGuidelineProjection
from parlant.core.journeys import JourneyStore
from parlant.core.playbooks import PlaybookStore
from parlant.core.relationships import RelationshipStore
from parlant.core.static_playbooks import StaticPlaybookStore
from parlant.core.tags import TagStore
from parlant.core.application import Application


mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("image/svg+xml", ".svg")


ASGIApplication: TypeAlias = Callable[
    [
        Scope,
        Receive,
        Send,
    ],
    Awaitable[None],
]


class AppWrapper:
    def __init__(self, app: FastAPI) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """FastAPI's built-in exception handling doesn't catch BaseExceptions
        such as asyncio.CancelledError. This causes the server process to terminate
        with an ugly traceback. This wrapper addresses that by specifically allowing
        asyncio.CancelledError to gracefully exit.
        """
        try:
            return await self.app(scope, receive, send)
        except asyncio.CancelledError:
            pass


RECORDED_FLAG = "_otel_metrics_recorded"


def _resolve_operation_id(request: Request) -> str | None:
    route = request.scope.get("route")
    if isinstance(route, APIRoute):
        return route.operation_id

    # If scope['route'] not set (404/early errors/etc.), try to match manually
    for r in getattr(request.app.router, "routes", []):
        if isinstance(r, APIRoute) and r.matches(request.scope)[0] == Match.FULL:
            return r.operation_id
    return None


async def create_api_app(
    container: Container,
    configure: Callable[[FastAPI], Awaitable[None]] | None = None,
    contextvar_propagation: Mapping[ContextVar[Any], Any] = {},
) -> ASGIApplication:
    logger = container[Logger]
    websocket_logger = container[WebSocketLogger]
    tracer = container[Tracer]
    authorization_policy = container[AuthorizationPolicy]
    application = container[Application]
    agent_tool_association_store = container[AgentToolAssociationStore]
    static_playbook_store = container[StaticPlaybookStore]
    playbook_store = container[PlaybookStore]
    tag_store = container[TagStore]
    guideline_store = container[GuidelineStore]
    guideline_tool_assoc_store = container[GuidelineToolAssociationStore]
    relationship_store = container[RelationshipStore]
    glossary_store = container[GlossaryStore]
    canned_response_store = container[CannedResponseStore]
    context_variable_store = container[ContextVariableStore]
    journey_store = container[JourneyStore]
    journey_guideline_projection = container[JourneyGuidelineProjection]

    meter = container[Meter]
    _hist_http_request_duration = meter.create_duration_histogram(
        name="httpreq",
        description="HTTP Request Duration",
    )

    api_app = FastAPI(
        title="Parlant API",
        description="API documentation for the Parlant server.",
        version=VERSION,
    )

    api_app = await authorization_policy.configure_app(api_app)

    @api_app.middleware("http")
    async def propagate_contextvars_into_request_task(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        for var, value in contextvar_propagation.items():
            var.set(value)
        return await call_next(request)

    @api_app.middleware("http")
    async def handle_cancellation(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        try:
            return await call_next(request)
        except asyncio.CancelledError:
            return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

    @api_app.middleware("http")
    async def add_trace_id(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if (
            request.url.path.startswith("/docs")
            or request.url.path.startswith("/redoc")
            or request.url.path.startswith("/openapi.json")
        ):
            await authorization_policy.authorize(
                request=request,
                operation=Operation.ACCESS_API_DOCS,
            )
            return await call_next(request)

        if request.url.path.startswith("/chat/"):
            await authorization_policy.authorize(
                request=request,
                operation=Operation.ACCESS_INTEGRATED_UI,
            )

            return await call_next(request)

        operation_id = _resolve_operation_id(request)

        if operation_id is None:
            return await call_next(request)

        request_id = generate_id()
        with tracer.span(
            "http.request",
            {
                "http.request.id": request_id,
                "http.request.operation": operation_id,
                "http.request.method": request.method,
                **request.path_params,
            },
        ):
            async with _hist_http_request_duration.measure(
                {
                    "http.request.operation": operation_id,
                    "http.request.method": request.method,
                },
            ):
                return await call_next(request)

    def _cors_headers(request: Request) -> dict[str, str]:
        """
        Build CORS headers for error responses.

        CORSMiddleware doesn't always add headers to exception handler responses,
        so we add them explicitly to ensure cross-origin errors are readable.
        """
        origin = request.headers.get("origin")
        if origin:
            return {
                "access-control-allow-origin": origin,
                "access-control-allow-credentials": "true",
            }
        return {}

    @api_app.exception_handler(RateLimitExceededException)
    async def rate_limit_exceeded_handler(
        request: Request, exc: RateLimitExceededException
    ) -> JSONResponse:
        logger.trace(f"Rate limit exceeded: {exc}")

        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": str(exc)},
            headers=_cors_headers(request),
        )

    @api_app.exception_handler(AuthorizationException)
    async def authorization_error_handler(
        request: Request, exc: AuthorizationException
    ) -> JSONResponse:
        logger.trace(f"Authorization error: {exc}")

        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": str(exc)},
            headers=_cors_headers(request),
        )

    @api_app.exception_handler(ItemNotFoundError)
    async def item_not_found_error_handler(
        request: Request, exc: ItemNotFoundError
    ) -> JSONResponse:
        logger.info(str(exc))

        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc)},
            headers=_cors_headers(request),
        )

    @api_app.exception_handler(Exception)
    async def server_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(str(exc))
        logger.error(str(traceback.format_exception(exc)))

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(exc)},
            headers=_cors_headers(request),
        )

    static_dir = os.path.join(os.path.dirname(__file__), "chat/dist")
    api_app.mount("/chat", StaticFiles(directory=static_dir, html=True), name="static")

    @api_app.get("/", include_in_schema=False)
    async def root() -> Response:
        return RedirectResponse("/chat")

    @api_app.get("/healthz")
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    agent_router = APIRouter(prefix="/agents")

    api_app.include_router(
        router=agents.create_router(
            policy=authorization_policy,
            app=application,
        ),
        prefix="/agents",
    )

    api_app.include_router(
        router=agent_router,
    )

    api_app.include_router(
        prefix="/agents",
        router=agent_tool_associations.create_router(
            authorization_policy=authorization_policy,
            association_store=agent_tool_association_store,
        ),
    )

    api_app.include_router(
        prefix="/sessions",
        router=sessions.create_router(
            authorization_policy=authorization_policy,
            app=application,
        ),
    )

    api_app.include_router(
        prefix="/services",
        router=services.create_router(
            authorization_policy=authorization_policy,
            app=application,
        ),
    )

    api_app.include_router(
        prefix="/tags",
        router=tags.create_router(
            authorization_policy=authorization_policy,
            app=application,
        ),
    )

    api_app.include_router(
        prefix="/terms",
        router=glossary.create_router(
            authorization_policy=authorization_policy,
            app=application,
        ),
    )

    api_app.include_router(
        prefix="/customers",
        router=customers.create_router(
            authorization_policy=authorization_policy,
            app=application,
        ),
    )

    api_app.include_router(
        prefix="/canned_responses",
        router=canned_responses.create_router(
            authorization_policy=authorization_policy,
            app=application,
        ),
    )

    api_app.include_router(
        prefix="/context-variables",
        router=variables.create_router(
            authorization_policy=authorization_policy,
            app=application,
        ),
    )

    api_app.include_router(
        prefix="/guidelines",
        router=guidelines.create_router(
            authorization_policy=authorization_policy,
            app=application,
        ),
    )

    api_app.include_router(
        prefix="/relationships",
        router=relationships.create_router(
            authorization_policy=authorization_policy,
            app=application,
        ),
    )

    api_app.include_router(
        prefix="/journeys",
        router=journeys.create_router(
            authorization_policy=authorization_policy,
            app=application,
        ),
    )

    api_app.include_router(
        prefix="/evaluations",
        router=evaluations.create_router(
            authorization_policy=authorization_policy,
            app=application,
        ),
    )

    api_app.include_router(
        prefix="/capabilities",
        router=capabilities.create_router(
            authorization_policy=authorization_policy,
            app=application,
        ),
    )

    api_app.include_router(
        prefix="/test-suites",
        router=test_suites.create_router(
            authorization_policy=authorization_policy,
            app=application,
        ),
    )

    api_app.include_router(
        prefix="/playbooks",
        router=playbooks.create_router(
            authorization_policy=authorization_policy,
            app=application,
        ),
    )

    api_app.include_router(
        prefix="/static-playbooks",
        router=static_playbooks.create_router(
            static_playbook_store=static_playbook_store,
        ),
    )

    api_app.include_router(
        prefix="/playbooks",
        router=playbook_resolve.create_router(
            playbook_store=playbook_store,
            tag_store=tag_store,
            guideline_store=guideline_store,
            guideline_tool_association_store=guideline_tool_assoc_store,
            relationship_store=relationship_store,
            glossary_store=glossary_store,
            canned_response_store=canned_response_store,
            context_variable_store=context_variable_store,
            journey_store=journey_store,
            journey_guideline_projection=journey_guideline_projection,
        ),
    )

    api_app.include_router(
        router=logs.create_router(
            websocket_logger,
        )
    )

    # Call configure_api hook if provided
    if configure:
        await configure(api_app)

    # Store FastAPI app in container for access via Server.api property
    container[FastAPI] = api_app

    return AppWrapper(api_app)
