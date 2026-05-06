from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import sqlite3
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .app_config import AppConfig, load_app_config
from .app_security import PasswordHasher, PayloadCipher, create_session_token
from .app_service import PostureAnalyzer
from .app_storage import AppStorage


class RegisterRequest(BaseModel):
    username: str = Field(min_length=2, max_length=80)
    password: str = Field(min_length=4)
    display_name: str = Field(min_length=1, max_length=120)


class LoginRequest(BaseModel):
    username: str = Field(min_length=2, max_length=80)
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    display_name: str
    role: str
    created_at: str


def _get_config() -> AppConfig:
    return load_app_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = _get_config()
    cipher = PayloadCipher(config.secret_key)
    storage = AppStorage(config.database_path, cipher)
    storage.init_schema()
    password_hasher = PasswordHasher()
    if config.seed_default_users:
        storage.upsert_user(
            username="admin",
            display_name="Administrador",
            password_hash=password_hasher.hash_password("admin"),
            role="dev",
        )
        storage.upsert_user(
            username="Pablo",
            display_name="Pablo",
            password_hash=password_hasher.hash_password("1234"),
            role="user",
        )
    app.state.config = config
    app.state.password_hasher = password_hasher
    app.state.storage = storage
    app.state.analyzer = PostureAnalyzer(
        yolo_device=config.yolo_device,
    )
    yield
    app.state.analyzer.close()


app = FastAPI(
    title="Ergonomics Posture API",
    version="0.1.0",
    lifespan=lifespan,
)

# Agregar CORS middleware antes de que la aplicación inicie
config = _get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _storage() -> AppStorage:
    return app.state.storage


def _password_hasher() -> PasswordHasher:
    return app.state.password_hasher


def _user_response(user: dict | sqlite3.Row) -> dict:
    return {
        "id": user["id"],
        "username": user["email"],  # Nota: columna 'email' almacena el username en la BD
        "display_name": user["display_name"],
        "role": user["role"],
        "created_at": user["created_at"],
    }


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


async def optional_user(authorization: Annotated[str | None, Header()] = None) -> dict | None:
    token = _extract_bearer_token(authorization)
    if token is None:
        return None
    user = _storage().get_user_by_session_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Sesión inválida o caducada.")
    return user


async def current_user(user: Annotated[dict | None, Depends(optional_user)]) -> dict:
    if user is None:
        raise HTTPException(status_code=401, detail="Autenticación requerida.")
    return user


def _validate_upload(file: UploadFile) -> None:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="El archivo debe ser una imagen.")


async def _save_upload(file: UploadFile) -> Path:
    _validate_upload(file)
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        return Path(tmp.name)


async def _analyze_upload(file: UploadFile, view: str, user: dict | None = None) -> dict:
    image_path = await _save_upload(file)
    try:
        analyzer: PostureAnalyzer = app.state.analyzer
        if view == "front":
            result = analyzer.analyze_front_image(image_path)
        elif view == "lateral":
            result = analyzer.analyze_lateral_image(image_path)
        else:
            raise HTTPException(status_code=400, detail=f"Vista desconocida: {view}")
        if user is not None:
            saved = _storage().save_analysis(user_id=user["id"], result=result)
            result = {**result, "id": saved["id"], "created_at": saved["created_at"]}
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            image_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.get("/api/health")
def health() -> dict:
    config: AppConfig = app.state.config
    return {
        "ok": True,
        "service": "ergonomics-posture-api",
        "front_model": "MediaPipe Pose",
        "lateral_model": "YOLO Pose",
        "database": str(config.database_path),
        "auth_required": config.require_auth,
        "mode": "local-first",
        "default_users_seeded": config.seed_default_users,
    }


@app.post("/api/analyze/front")
async def analyze_front(
    user: Annotated[dict | None, Depends(optional_user)],
    file: UploadFile = File(...),
) -> dict:
    if app.state.config.require_auth and user is None:
        raise HTTPException(status_code=401, detail="Autenticación requerida.")
    return await _analyze_upload(file, "front", user)


@app.post("/api/analyze/lateral")
async def analyze_lateral(
    user: Annotated[dict | None, Depends(optional_user)],
    file: UploadFile = File(...),
) -> dict:
    if app.state.config.require_auth and user is None:
        raise HTTPException(status_code=401, detail="Autenticación requerida.")
    return await _analyze_upload(file, "lateral", user)


@app.post("/api/auth/register")
def register(payload: RegisterRequest) -> dict:
    hasher = _password_hasher()
    try:
        user = _storage().create_user(
            username=payload.username,
            display_name=payload.display_name,
            password_hash=hasher.hash_password(payload.password),
            role="user",
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Ya existe un usuario con ese nombre.") from exc
    token = create_session_token()
    session = _storage().create_session(user_id=int(user["id"]), token=token)
    return {"user": _user_response(user), **session}


@app.post("/api/auth/login")
def login(payload: LoginRequest) -> dict:
    user = _storage().get_user_by_username(payload.username)
    if user is None or not _password_hasher().verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Credenciales incorrectas.")
    token = create_session_token()
    session = _storage().create_session(user_id=int(user["id"]), token=token)
    return {"user": _user_response(user), **session}


@app.get("/api/auth/me", response_model=UserResponse)
def me(user: Annotated[dict, Depends(current_user)]) -> dict:
    return _user_response(user)


@app.post("/api/auth/logout")
def logout(authorization: Annotated[str | None, Header()] = None) -> dict:
    token = _extract_bearer_token(authorization)
    if token is not None:
        _storage().delete_session(token)
    return {"ok": True}


@app.get("/api/analyses")
def list_analyses(user: Annotated[dict, Depends(current_user)], limit: int = 50) -> dict:
    bounded_limit = min(max(limit, 1), 200)
    return {"items": _storage().list_analyses(user_id=user["id"], limit=bounded_limit)}


@app.delete("/api/analyses")
def delete_analyses(user: Annotated[dict, Depends(current_user)]) -> dict:
    deleted = _storage().delete_analyses(user_id=user["id"])
    return {"ok": True, "deleted": deleted}


@app.get("/api/summary")
def summary(user: Annotated[dict, Depends(current_user)]) -> dict:
    return _storage().analysis_summary(user_id=user["id"])
