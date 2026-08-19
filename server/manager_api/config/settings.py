import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ServerConfig:
    deployment: str
    host: str
    control_port: int
    webrtc_port: int
    manager_port: int
    manager_web_port: int
    stun_urls: tuple[str, ...]
    turn_urls: tuple[str, ...]
    turn_static_auth_secret: str
    turn_ttl_seconds: int
    database_path: Path
    heartbeat_timeout_seconds: int
    xiaozhi_ws_url: str
    xiaozhi_mcp_port: int
    xiaozhi_mcp_token: str


def load_config() -> ServerConfig:
    deployment = os.getenv("CRUISECAR_DEPLOYMENT", os.getenv("CRUISECAR_MODE", "full")).lower()
    if deployment != "full":
        raise ValueError("CRUISECAR_DEPLOYMENT must be full")
    return ServerConfig(
        deployment=deployment,
        host=os.getenv("CRUISECAR_HOST", "0.0.0.0"),
        control_port=int(os.getenv("CRUISECAR_CONTROL_PORT", "42110")),
        webrtc_port=int(os.getenv("CRUISECAR_WEBRTC_PORT", "42112")),
        manager_port=int(os.getenv("CRUISECAR_MANAGER_PORT", "8088")),
        manager_web_port=int(os.getenv("CRUISECAR_MANAGER_WEB_PORT", "8089")),
        stun_urls=_csv_env("CRUISECAR_STUN_URLS", "stun:stun.l.google.com:19302"),
        turn_urls=_csv_env("CRUISECAR_TURN_URLS", ""),
        turn_static_auth_secret=os.getenv("CRUISECAR_TURN_STATIC_AUTH_SECRET", ""),
        turn_ttl_seconds=int(os.getenv("CRUISECAR_TURN_TTL_SECONDS", "3600")),
        database_path=Path(
            os.getenv("CRUISECAR_DB", str(ROOT_DIR / "cruisecar.db"))
        ),
        heartbeat_timeout_seconds=int(os.getenv("CRUISECAR_HEARTBEAT_TIMEOUT", "20")),
        xiaozhi_ws_url=os.getenv("CRUISECAR_XIAOZHI_WS_URL", "ws://127.0.0.1:8000"),
        xiaozhi_mcp_port=int(os.getenv("CRUISECAR_XIAOZHI_MCP_PORT", "8090")),
        xiaozhi_mcp_token=os.getenv("CRUISECAR_XIAOZHI_MCP_TOKEN", ""),
    )


def _csv_env(key: str, default: str) -> tuple[str, ...]:
    value = os.getenv(key, default)
    return tuple(item.strip() for item in value.split(",") if item.strip())
