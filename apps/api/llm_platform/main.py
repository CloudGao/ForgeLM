from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import get_settings
from .database import SessionLocal, init_database
from .models import WorkerNode
from .routers import (
    catalog,
    code,
    data,
    data_conversion,
    data_generation,
    data_processing,
    dataset_materialization,
    evaluations,
    execution_environments,
    imports,
    labels,
    model_specifications,
    model_weights,
    models,
    pipelines,
    playground,
    plugins,
    post_training,
    projects,
    releases,
    storage,
    suite_packages,
    suite_services,
    system,
    tasks,
    tokenized_corpora,
    tokenized_data,
    tokenizers,
    training,
    training_data_views,
    training_instances,
)
from .services.global_data import ensure_global_data_project
from .services.local_environments import (
    local_native_worker_supervisor,
    local_wsl_lifecycle_supervisor,
    managed_local_wsl_distros,
)
from .services.plugins import register_builtin_plugins
from .services.project_storage import ProjectStorageError
from .services.ssh_tunnels import tunnel_manager
from .services.tasks import mark_stale_workers_offline, reap_expired_leases


@asynccontextmanager
async def lifespan(_app: FastAPI):
    import os
    if get_settings().storage_root:
        os.environ["LLM_PLATFORM_STORAGE_ROOT"] = get_settings().storage_root
    init_database()
    with SessionLocal() as session:
        ensure_global_data_project(session)
        register_builtin_plugins(session)
        from .services.training_history import ensure_history_instances
        ensure_history_instances(session)
        session.commit()
    stop_reaper = asyncio.Event()
    from .services.remote_workers import recover_interrupted_remote_operations
    recover_interrupted_remote_operations()
    from .services.model_deletion import recover_interrupted_plans
    recover_interrupted_plans()
    from .services.data_cleanup import recover_interrupted
    recover_interrupted()
    from .services.storage_relocation import recover_interrupted as recover_relocations
    recover_relocations()
    tunnel_manager.app = _app

    async def remote_tunnel_supervisor() -> None:
        while not stop_reaper.is_set():
            await asyncio.to_thread(tunnel_manager.reconcile)
            try:
                await asyncio.wait_for(stop_reaper.wait(), timeout=3)
            except TimeoutError:
                continue

    async def lease_reaper() -> None:
        while not stop_reaper.is_set():
            with SessionLocal() as session:
                reap_expired_leases(session)
                mark_stale_workers_offline(session)
                from .services.training_instances import reconcile_training_attempts
                reconcile_training_attempts(session)
                from .services.storage_lifecycle import reconcile_storage_maintenance
                try:
                    reconcile_storage_maintenance(session)
                except Exception:
                    import logging
                    session.rollback()
                    logging.getLogger(__name__).exception("Storage maintenance reconciliation failed")
                ensure_history_instances(session)
                session.commit()
            try:
                await asyncio.wait_for(stop_reaper.wait(), timeout=15)
            except TimeoutError:
                continue

    async def local_wsl_supervisor() -> None:
        while not stop_reaper.is_set():
            with SessionLocal() as session:
                distros = managed_local_wsl_distros(
                    session.scalars(select(WorkerNode)).all()
                )
            await asyncio.to_thread(local_wsl_lifecycle_supervisor.reconcile, distros)
            try:
                await asyncio.wait_for(stop_reaper.wait(), timeout=5)
            except TimeoutError:
                continue

    reaper_task = asyncio.create_task(lease_reaper())
    wsl_supervisor_task = asyncio.create_task(local_wsl_supervisor())
    tunnel_task = asyncio.create_task(remote_tunnel_supervisor())
    try:
        yield
    finally:
        stop_reaper.set()
        await asyncio.gather(reaper_task, wsl_supervisor_task, tunnel_task)
        await asyncio.gather(
            asyncio.to_thread(local_wsl_lifecycle_supervisor.shutdown),
            asyncio.to_thread(local_native_worker_supervisor.shutdown),
            asyncio.to_thread(tunnel_manager.shutdown),
        )


app = FastAPI(
    title="Local LLM Platform API",
    version="0.1.0",
    description="Local-first control plane for the complete LLM lifecycle.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def safe_validation_error(request: Request, exc: RequestValidationError):
    if request.url.path.startswith("/api/execution-environments/remote"):
        # Pydantic model validators include the entire input in errors by default.
        return JSONResponse(status_code=422, content={"detail": [
            {"type": error["type"], "loc": error["loc"], "msg": error["msg"]}
            for error in exc.errors()
        ]})
    return await request_validation_exception_handler(request, exc)


@app.middleware("http")
async def add_utf8_charset_to_json(request: Request, call_next):
    response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    media_type = content_type.partition(";")[0].strip().lower()
    if (
        (media_type == "application/json" or media_type.endswith("+json"))
        and "charset=" not in content_type.lower()
    ):
        response.headers["content-type"] = f"{content_type}; charset=utf-8"
    return response


@app.exception_handler(ProjectStorageError)
async def project_storage_error_handler(_request, exc: ProjectStorageError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": "project_storage_unavailable",
                "message": str(exc),
            }
        },
    )


@app.exception_handler(StarletteHTTPException)
async def api_not_found_handler(request: Request, exc: StarletteHTTPException):
    request_path = request.url.path
    if exc.status_code == 404 and (
        request_path == "/api" or request_path.startswith("/api/")
    ):
        return JSONResponse(
            status_code=404,
            content={
                "detail": {
                    "code": "api_route_not_found",
                    "message": f"API 路由不存在：{request_path}",
                }
            },
        )
    return await http_exception_handler(request, exc)


for router in (
    system.router,
    execution_environments.router,
    projects.router,
    pipelines.router,
    catalog.router,
    labels.router,
    code.router,
    imports.router,
    data.router,
    dataset_materialization.router,
    data_conversion.router,
    data_processing.router,
    data_generation.router,
    tokenizers.router,
    tokenized_corpora.router,
    tokenized_data.router,
    training_data_views.router,
    post_training.router,
    training.router,
    training_instances.router,
    model_specifications.router,
    model_weights.router,
    models.router,
    evaluations.router,
    plugins.router,
    suite_packages.router,
    suite_services.router,
    releases.router,
    playground.router,
    storage.router,
    tasks.router,
):
    app.include_router(router, prefix="/api")


@app.get("/api")
def api_root() -> dict:
    return {"name": "Local LLM Platform", "version": app.version, "docs": "/docs"}


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        request_path = str(scope.get("path") or "")
        if request_path == "/api" or request_path.startswith("/api/"):
            return JSONResponse(
                status_code=404,
                content={
                    "detail": {
                        "code": "api_route_not_found",
                        "message": f"API 路由不存在：{request_path}",
                    }
                },
            )
        try:
            response = await super().get_response(path, scope)
            if response.status_code != 404:
                return response
        except StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise
        return FileResponse(Path(self.directory) / "index.html")


web_dist = Path(__file__).resolve().parents[2] / "web" / "dist"
if web_dist.exists():
    app.mount("/", SPAStaticFiles(directory=web_dist, html=True), name="web")


def run() -> None:
    settings = get_settings()
    server = uvicorn.Server(uvicorn.Config(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        workers=1,
    ))
    app.state.control_plane_server = server
    try:
        server.run()
    finally:
        if getattr(app.state, "control_plane_server", None) is server:
            del app.state.control_plane_server


if __name__ == "__main__":
    run()
