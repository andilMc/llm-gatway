"""Admin dashboard routes for LLM Gateway."""

from fastapi import (
    APIRouter,
    Request,
    WebSocket,
    WebSocketDisconnect,
    Form,
    HTTPException,
)
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from typing import Dict, Any, List, Optional
import json
import logging
import asyncio
from datetime import datetime

from gateway.database import get_db
from gateway.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])
templates = Jinja2Templates(directory="gateway/templates")


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)

        for conn in disconnected:
            self.active_connections.remove(conn)


manager = ConnectionManager()


@router.get("/", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Main admin dashboard page."""
    db = get_db()

    # Get stats
    stats = db.get_requests_stats(hours=24)
    providers = db.get_all_providers()
    total_keys = len(db.get_all_api_keys())

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "stats": stats,
            "providers_count": len(providers),
            "total_keys": total_keys,
        },
    )


@router.get("/providers", response_class=HTMLResponse)
async def list_providers(request: Request):
    """List all providers."""
    db = get_db()
    providers = db.get_all_providers()
    health_status = db.get_health_status()

    # Add health info to providers
    health_map = {h["id"]: h for h in health_status}
    for provider in providers:
        health = health_map.get(provider["id"], {})
        # Calculate is_healthy based on circuit state and available keys
        circuit_state = health.get("circuit_state", "closed")
        available_keys = health.get("available_keys", 0)
        health["is_healthy"] = circuit_state == "closed" and available_keys > 0
        provider["health"] = health
        provider["api_keys"] = db.get_api_keys_by_provider(provider["id"])

    return templates.TemplateResponse(
        "providers.html",
        {
            "request": request,
            "providers": providers,
        },
    )


@router.get("/providers/new", response_class=HTMLResponse)
async def new_provider_form(request: Request):
    """Form to create new provider."""
    return templates.TemplateResponse(
        "provider_form.html",
        {
            "request": request,
            "provider": None,
        },
    )


@router.post("/providers")
async def create_provider(
    name: str = Form(...),
    provider_type: str = Form(...),
    base_url: str = Form(...),
    strategy: str = Form("sequential"),
    timeout: float = Form(120.0),
    max_retries: int = Form(3),
):
    """Create a new provider."""
    db = get_db()

    try:
        provider_id = db.create_provider(
            name=name,
            provider_type=provider_type,
            base_url=base_url,
            strategy=strategy,
            timeout=timeout,
            max_retries=max_retries,
        )
        return JSONResponse({"success": True, "provider_id": provider_id})
    except Exception as e:
        logger.error(f"Failed to create provider: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/providers/{provider_id}/edit", response_class=HTMLResponse)
async def edit_provider_form(request: Request, provider_id: int):
    """Form to edit provider."""
    db = get_db()
    provider = db.get_provider(provider_id)

    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    provider["keys"] = db.get_api_keys_by_provider(provider_id)

    return templates.TemplateResponse(
        "provider_form.html",
        {
            "request": request,
            "provider": provider,
        },
    )


@router.post("/providers/{provider_id}")
async def update_provider(
    provider_id: int,
    name: str = Form(...),
    base_url: str = Form(...),
    strategy: str = Form("sequential"),
    timeout: float = Form(120.0),
    max_retries: int = Form(3),
):
    """Update a provider."""
    db = get_db()

    try:
        db.update_provider(
            provider_id=provider_id,
            name=name,
            base_url=base_url,
            strategy=strategy,
            timeout=timeout,
            max_retries=max_retries,
        )
        return JSONResponse({"success": True})
    except Exception as e:
        logger.error(f"Failed to update provider: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/providers/{provider_id}")
async def delete_provider(provider_id: int):
    """Delete a provider."""
    db = get_db()

    try:
        db.delete_provider(provider_id)
        return JSONResponse({"success": True})
    except Exception as e:
        logger.error(f"Failed to delete provider: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api-keys", response_class=HTMLResponse)
async def list_api_keys(request: Request):
    """List all API keys."""
    db = get_db()
    keys = db.get_all_api_keys()
    providers = db.get_all_providers()

    provider_map = {p["id"]: p["name"] for p in providers}

    return templates.TemplateResponse(
        "api_keys.html",
        {
            "request": request,
            "keys": keys,
            "providers": providers,
            "provider_map": provider_map,
        },
    )


@router.post("/api-keys")
async def create_api_key(
    provider_id: int = Form(...),
    api_key: str = Form(...),
    models: str = Form(""),
):
    """Create a new API key."""
    db = get_db()

    try:
        models_list = [m.strip() for m in models.split(",") if m.strip()]
        key_id = db.create_api_key(
            provider_id=provider_id,
            api_key=api_key,
            models=models_list,
        )
        return JSONResponse({"success": True, "key_id": key_id})
    except Exception as e:
        logger.error(f"Failed to create API key: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/api-keys/{key_id}")
async def delete_api_key(key_id: int):
    """Delete an API key."""
    db = get_db()

    try:
        db.delete_api_key(key_id)
        return JSONResponse({"success": True})
    except Exception as e:
        logger.error(f"Failed to delete API key: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/models", response_class=HTMLResponse)
async def list_models(request: Request):
    """List all models."""
    db = get_db()
    models = db.get_all_models()
    providers = db.get_all_providers()

    provider_map = {p["id"]: p["name"] for p in providers}

    return templates.TemplateResponse(
        "models.html",
        {
            "request": request,
            "models": models,
            "providers": providers,
            "provider_map": provider_map,
        },
    )


@router.post("/models")
async def create_model(
    alias: str = Form(...),
    technical_name: str = Form(...),
    provider_id: int = Form(...),
    description: str = Form(None),
):
    """Create a new model."""
    db = get_db()

    try:
        model_id = db.create_model(
            alias=alias,
            technical_name=technical_name,
            provider_id=provider_id,
            description=description,
        )
        return JSONResponse({"success": True, "model_id": model_id})
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/analytics", response_class=HTMLResponse)
async def analytics_dashboard(request: Request):
    """Analytics dashboard."""
    db = get_db()

    # Get stats for different time periods
    stats_24h = db.get_requests_stats(hours=24)
    stats_7d = db.get_requests_stats(hours=168)

    # Get recent requests
    recent_requests = db.get_recent_requests(limit=100)

    return templates.TemplateResponse(
        "analytics.html",
        {
            "request": request,
            "stats_24h": stats_24h,
            "stats_7d": stats_7d,
            "recent_requests": recent_requests,
        },
    )


@router.get("/api/analytics/data")
async def get_analytics_data(hours: int = 24):
    """Get analytics data for charts."""
    db = get_db()
    stats = db.get_requests_stats(hours=hours)

    return JSONResponse(
        {
            "success": True,
            "data": stats,
        }
    )


@router.get("/logs", response_class=HTMLResponse)
async def logs_viewer(request: Request):
    """Real-time logs viewer."""
    return templates.TemplateResponse(
        "logs.html",
        {
            "request": request,
        },
    )


@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """WebSocket for real-time logs."""
    await manager.connect(websocket)

    try:
        db = get_db()

        # Send recent logs
        recent_logs = db.get_recent_logs(limit=50)
        for log in reversed(recent_logs):
            await websocket.send_json(
                {
                    "type": "log",
                    "level": log["level"],
                    "logger": log["logger_name"],
                    "message": log["message"],
                    "timestamp": log["timestamp"],
                }
            )

        # Keep connection alive
        while True:
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.get("/opencode/config", response_class=PlainTextResponse)
async def get_opencode_config():
    """Generate OpenCode configuration."""
    db = get_db()
    models = db.get_all_models()

    # Build OpenCode config
    config = {
        "version": "1.0",
        "gateway_url": "http://localhost:48001/v1",
        "models": [],
    }

    for model in models:
        config["models"].append(
            {
                "id": model["alias"],
                "name": model["technical_name"],
                "description": model.get("description", ""),
            }
        )

    return json.dumps(config, indent=2)


@router.get("/opencode", response_class=HTMLResponse)
async def opencode_config_page(request: Request):
    """OpenCode configuration page."""
    import platform

    db = get_db()
    models = db.get_all_models()

    # Get saved config from database
    gateway_url = db.get_config("opencode_gateway_url", "http://localhost:48001/v1")
    api_key = db.get_config("opencode_api_key", "any-key-works")
    default_model = db.get_config("opencode_default_model", "")
    enabled_models_str = db.get_config("opencode_enabled_models", "")
    enabled_models = enabled_models_str.split(",") if enabled_models_str else []

    # If no saved config, enable all models
    if not enabled_models:
        enabled_models = [m["alias"] for m in models]

    # Build default model list
    model_list = []
    for model in models:
        model_list.append(
            {
                "alias": model["alias"],
                "technical_name": model["technical_name"],
                "is_enabled": model["alias"] in enabled_models,
            }
        )

    # Detect OS
    system = platform.system()
    if system == "Darwin":
        config_path = "~/.config/opencode/config.json"
    elif system == "Linux":
        config_path = "~/.config/opencode/config.json"
    else:
        config_path = "%APPDATA%\\opencode\\config.json"

    # Build initial JSON config
    opencode_config = {
        "version": "1.0",
        "llm": {
            "provider": "openai",
            "base_url": gateway_url,
            "api_key": api_key,
        },
        "models": {},
    }

    for m in model_list:
        if m["is_enabled"]:
            opencode_config["models"][m["alias"]] = {
                "provider": "openai",
                "model": m["alias"],
            }

    if default_model and default_model in enabled_models:
        opencode_config["default_model"] = default_model

    config_json = json.dumps(opencode_config, indent=2)

    return templates.TemplateResponse(
        "opencode.html",
        {
            "request": request,
            "models": model_list,
            "config": {
                "gateway_url": gateway_url,
                "api_key": api_key,
                "default_model": default_model,
                "enabled_models": enabled_models,
                "json": config_json,
            },
            "config_path": config_path,
        },
    )


@router.post("/opencode/save")
async def save_opencode_config(
    gateway_url: str = Form(...),
    api_key: str = Form(...),
    default_model: str = Form(""),
    enabled_models: str = Form(""),
):
    """Save OpenCode configuration to database."""
    db = get_db()

    db.set_config("opencode_gateway_url", gateway_url)
    db.set_config("opencode_api_key", api_key)
    db.set_config("opencode_default_model", default_model)
    db.set_config("opencode_enabled_models", enabled_models)

    return JSONResponse({"success": True, "message": "Configuration sauvegardée !"})


@router.get("/api/stats")
async def get_stats():
    """Get current stats for dashboard."""
    db = get_db()

    stats = db.get_requests_stats(hours=24)
    health = db.get_health_status()

    return JSONResponse(
        {
            "success": True,
            "stats": stats,
            "health": health,
        }
    )


@router.websocket("/ws/stats")
async def websocket_stats(websocket: WebSocket):
    """WebSocket for real-time stats updates."""
    await manager.connect(websocket)

    try:
        while True:
            db = get_db()
            stats = db.get_requests_stats(hours=1)  # Last hour
            health = db.get_health_status()

            await websocket.send_json(
                {
                    "type": "stats",
                    "data": {
                        "stats": stats,
                        "health": health,
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            )

            await asyncio.sleep(5)  # Update every 5 seconds

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
