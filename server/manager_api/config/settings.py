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
    database_path: Path
    heartbeat_timeout_seconds: int


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
        database_path=Path(
            os.getenv("CRUISECAR_DB", str(ROOT_DIR / "cruisecar.db"))
        ),
        heartbeat_timeout_seconds=int(os.getenv("CRUISECAR_HEARTBEAT_TIMEOUT", "20")),
    )
