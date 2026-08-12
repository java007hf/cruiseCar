import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ServerConfig:
    deployment: str
    host: str
    control_port: int
    webrtc_port: int
    manager_port: int
    database_path: Path
    auth_token: str
    heartbeat_timeout_seconds: int


def load_config() -> ServerConfig:
    deployment = os.getenv("CRUISECAR_DEPLOYMENT", os.getenv("CRUISECAR_MODE", "light")).lower()
    if deployment not in {"light", "full"}:
        raise ValueError("CRUISECAR_DEPLOYMENT must be light or full")
    return ServerConfig(
        deployment=deployment,
        host=os.getenv("CRUISECAR_HOST", "0.0.0.0"),
        control_port=int(os.getenv("CRUISECAR_CONTROL_PORT", "42110")),
        webrtc_port=int(os.getenv("CRUISECAR_WEBRTC_PORT", "42112")),
        manager_port=int(os.getenv("CRUISECAR_MANAGER_PORT", "8088")),
        database_path=Path(
            os.getenv("CRUISECAR_DB", str(ROOT_DIR / "cruisecar.db"))
        ),
        auth_token=os.getenv("CRUISECAR_AUTH_TOKEN", ""),
        heartbeat_timeout_seconds=int(os.getenv("CRUISECAR_HEARTBEAT_TIMEOUT", "20")),
    )
