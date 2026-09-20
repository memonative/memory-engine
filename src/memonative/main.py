import sys
from contextlib import asynccontextmanager
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from memonative.api.routes import router, health_router
from memonative.auth.bootstrap import ensure_default_tenant_and_master_key
from memonative.errors import EngineError
from memonative.observability import setup_logging, set_request_id

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_default_tenant_and_master_key()
    yield


app = FastAPI(title="Memonative Memory Engine", lifespan=lifespan)


@app.exception_handler(EngineError)
async def engine_error_handler(request: Request, exc: EngineError):
    """Translate engine errors back into the HTTP responses they used to be.

    The engine no longer raises `HTTPException` — it carries the status code
    on the exception instead. This body matches FastAPI's own, so clients see
    no difference.
    """
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = set_request_id(request.headers.get("X-Request-ID"))
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


app.include_router(router)
app.include_router(health_router)

# Admin API — master key guards access
from memonative.api.admin import router as admin_router
app.include_router(admin_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("memonative.main:app", host="0.0.0.0", port=8000, reload=True)
