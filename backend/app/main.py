from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .api.transcripts import router as transcripts_router
from .api.runs import router as runs_router
from .config import settings
from .database import SchemaVerificationError, check_database_connection, init_db
from .logging_config import configure_logging, get_logger


configure_logging()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        init_db()
    except SchemaVerificationError as error:
        logger.error("Database schema verification failed: %s", error)
        raise RuntimeError(str(error)) from None
    except Exception:
        logger.error("Database migration or schema verification failed")
        raise RuntimeError(
            "Database migration or schema verification failed; review database health and migration state"
        ) from None
    logger.info("Database migrations applied and schema verification passed")
    yield


app = FastAPI(title="SignalBridge API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transcripts_router)
app.include_router(runs_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "app": "SignalBridge API",
        "environment": settings.app_env,
    }


@app.get("/health/db")
def database_health() -> dict[str, str]:
    try:
        check_database_connection()
    except Exception:
        # Keep exception details out of the response and logs because connection
        # errors can contain database credentials or other sensitive metadata.
        logger.warning("Database health check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection unavailable",
        ) from None

    return {"status": "ok", "database": "connected"}

