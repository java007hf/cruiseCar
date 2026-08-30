from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

from manager_api.config.settings import ServerConfig, load_config
from manager_api.protocol.control_protocol import (
    TRACE_STAGE_SENDER_CREATED,
    packet_from_command,
    packet_to_hex,
    parse_packet,
    trace_mark,
)
from manager_api.storage.store import Store

if TYPE_CHECKING:
    from control_server.server import ConnectionHub


logger = logging.getLogger(__name__)


class ManagerApi:
    def __init__(
        self,
        config: ServerConfig | None = None,
        store: Store | None = None,
        hub: ConnectionHub | None = None,
        event_loop: asyncio.AbstractEventLoop | None = None,
        bridge: Any = None,
    ):
        self.config = config or load_config()
        self.store = store or Store(self.config.database_path)
        self.hub = hub
        self.event_loop = event_loop
        self.bridge = bridge
        self.httpd: ThreadingHTTPServer | None = None

    def start(self) -> None:
        handler = self._handler_class()
        self.httpd = ThreadingHTTPServer((self.config.host, self.config.manager_port), handler)
        logger.info("manager-api listening on %s:%s", self.config.host, self.config.manager_port)
        self.httpd.serve_forever()

    def start_in_thread(self) -> threading.Thread:
        thread = threading.Thread(target=self.start, name="manager-api", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        api = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "CruiseCarManagerApi/0.1"

            def do_OPTIONS(self) -> None:
                self._send_json({"ok": True})

            def do_DELETE(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path.rstrip("/") or "/"
                if not self._authorized():
                    self._send_json({"ok": False, "error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
                    return
                try:
                    if path.startswith("/api/receivers/"):
                        device_id = path.split("/")[-1]
                        user = self._current_user()
                        receiver = api.store.get_receiver(device_id)
                        if not receiver or not self._can_access(receiver):
                            self._send_json({"ok": False, "error": "receiver not found"}, HTTPStatus.NOT_FOUND)
                            return
                        if api.hub and self._is_receiver_online(device_id):
                            self._send_json({"ok": False, "error": "receiver is online, please stop it before deleting"}, HTTPStatus.CONFLICT)
                            return
                        deleted = api.store.delete_receiver(device_id, user["username"] if user else "")
                        self._send_json({"ok": True, "data": {"deleted": deleted, "device_id": device_id}})
                    elif path.startswith("/api/agents/templates/"):
                        template_id = path.split("/")[-1]
                        user = self._current_user()
                        deleted = api.store.delete_agent_template(template_id, user["username"] if user else "")
                        self._send_json({"ok": True, "data": {"deleted": deleted, "template_id": template_id}})
                    else:
                        self._send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
                except PermissionError as exc:
                    self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.FORBIDDEN)
                except Exception as exc:
                    logger.exception("manager api delete failed")
                    self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path.rstrip("/") or "/"
                query = parse_qs(parsed.query)

                if path == "/health":
                    self._send_json({"ok": True, "service": "manager-api"})
                elif path == "/":
                    self._send_json({"ok": True, "service": "manager-api", "manager_web_url": api.manager_web_url()})
                elif path == "/api/receivers":
                    user = self._current_user()
                    data = api.store.list_receivers_for_user(user["username"]) if user else api.store.list_receivers()
                    self._send_json({"ok": True, "data": data})
                elif path.startswith("/api/receivers/"):
                    device_id = path.split("/")[-1]
                    receiver = api.store.get_receiver(device_id)
                    if receiver and self._can_access(receiver):
                        self._send_json({"ok": True, "data": receiver})
                    else:
                        self._send_json({"ok": False, "error": "receiver not found"}, HTTPStatus.NOT_FOUND)
                elif path == "/api/senders":
                    user = self._current_user()
                    data = api.store.list_senders_for_user(user["username"]) if user else api.store.list_senders()
                    self._send_json({"ok": True, "data": data})
                elif path == "/api/webrtc/ice-servers":
                    user = self._current_user()
                    if not user:
                        self._send_json({"ok": False, "error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
                        return
                    self._send_json({"ok": True, "data": api.issue_ice_servers(user["username"])})
                elif path == "/api/events":
                    limit = int((query.get("limit") or [100])[0])
                    device_id = (query.get("device_id") or [""])[0]
                    self._send_json({"ok": True, "data": api.store.list_events(limit=limit, device_id=device_id)})
                elif path == "/api/agents/templates":
                    user = self._current_user()
                    data = api.store.list_agent_templates(user["username"] if user else "")
                    self._send_json({"ok": True, "data": data})
                elif path.startswith("/api/agents/car/"):
                    device_id = path.split("/")[-1]
                    data = api.store.get_agent_config_for_car(device_id)
                    if data:
                        self._send_json({"ok": True, "data": data})
                    else:
                        self._send_json({"ok": False, "error": "no agent binding for this car"}, HTTPStatus.NOT_FOUND)
                elif path == "/api/bridge/status":
                    data = api._bridge_status()
                    self._send_json({"ok": True, "data": data})
                elif path == "/api/bridge/events":
                    device_id = (query.get("device_id") or [""])[0]
                    after_seq = int((query.get("after_seq") or ["0"])[0])
                    limit = int((query.get("limit") or ["100"])[0])
                    data = api._bridge_events(device_id, after_seq, limit)
                    self._send_json({"ok": True, "data": data})
                else:
                    self._send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path.rstrip("/") or "/"
                body = self._read_json()
                if path not in {"/api/auth/login", "/api/auth/register"} and not self._authorized():
                    self._send_json({"ok": False, "error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
                    return

                try:
                    if path in {"/api/auth/login", "/api/auth/register"}:
                        data = api.store.create_or_login_user(
                            self._required(body, "username"), self._required(body, "password")
                        )
                        self._send_json({"ok": True, "data": data})
                    elif path == "/api/receivers":
                        device_id = self._required(body, "device_id")
                        user = self._current_user()
                        data = api.store.upsert_receiver(
                            device_id,
                            str(body.get("name", "")),
                            str(body.get("token", "")),
                            user["username"] if user else str(body.get("owner_username", "")),
                        )
                        self._send_json({"ok": True, "data": data})
                    elif path == "/api/senders":
                        sender_id = self._required(body, "sender_id")
                        user = self._current_user()
                        data = api.store.upsert_sender(
                            sender_id,
                            str(body.get("name", "")),
                            str(body.get("token", "")),
                            str(body.get("target_device_id", "")),
                            user["username"] if user else str(body.get("owner_username", "")),
                        )
                        self._send_json({"ok": True, "data": data})
                    elif path.startswith("/api/receivers/") and path.endswith("/commands"):
                        device_id = path.split("/")[-2]
                        result = api.dispatch_command(device_id, body)
                        self._send_json({"ok": True, "data": result})
                    elif path == "/api/agents/templates":
                        user = self._current_user()
                        template_id = self._required(body, "id")
                        data = api.store.upsert_agent_template(
                            template_id,
                            self._required(body, "name"),
                            str(body.get("prompt", "")),
                            json.dumps(body.get("skills", []), ensure_ascii=False),
                            user["username"] if user else "",
                            bool(body.get("is_builtin", False)),
                        )
                        self._send_json({"ok": True, "data": data})
                    elif path == "/api/agents/bind":
                        device_id = self._required(body, "device_id")
                        template_id = self._required(body, "template_id")
                        data = api.store.bind_car_agent(
                            device_id,
                            template_id,
                            str(body.get("custom_prompt", "")),
                        )
                        self._send_json({"ok": True, "data": data})
                    elif path == "/api/bridge/connect":
                        device_id = self._required(body, "device_id")
                        device_name = str(body.get("device_name", ""))
                        result = api._bridge_connect(device_id, device_name)
                        self._send_json({"ok": True, "data": result})
                    elif path == "/api/bridge/disconnect":
                        device_id = self._required(body, "device_id")
                        result = api._bridge_disconnect(device_id)
                        self._send_json({"ok": True, "data": result})
                    elif path == "/api/bridge/send":
                        device_id = self._required(body, "device_id")
                        text = self._required(body, "text")
                        result = api._bridge_send_text(device_id, text)
                        self._send_json({"ok": True, "data": result})
                    elif path == "/api/bridge/audio/start":
                        device_id = self._required(body, "device_id")
                        result = api._bridge_audio_start(device_id)
                        self._send_json({"ok": True, "data": result})
                    elif path == "/api/bridge/audio/frame":
                        device_id = self._required(body, "device_id")
                        audio_b64 = self._required(body, "audio_b64")
                        result = api._bridge_audio_frame(device_id, audio_b64)
                        self._send_json({"ok": True, "data": result})
                    elif path == "/api/bridge/audio/stop":
                        device_id = self._required(body, "device_id")
                        result = api._bridge_audio_stop(device_id)
                        self._send_json({"ok": True, "data": result})
                    else:
                        self._send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
                except ValueError as exc:
                    self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                except PermissionError as exc:
                    self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.FORBIDDEN)
                except Exception as exc:
                    logger.exception("manager api request failed")
                    self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

            def log_message(self, fmt: str, *args: Any) -> None:
                logger.info("%s - %s", self.address_string(), fmt % args)

            def _authorized(self) -> bool:
                header = self.headers.get("Authorization", "")
                token = header[7:] if header.startswith("Bearer ") else header
                return bool(api.store.user_by_token(token))

            def _current_user(self) -> dict[str, Any] | None:
                header = self.headers.get("Authorization", "")
                token = header[7:] if header.startswith("Bearer ") else header
                return api.store.user_by_token(token)

            def _can_access(self, row: dict[str, Any]) -> bool:
                user = self._current_user()
                return not user or not row.get("owner_username") or row.get("owner_username") == user.get("username")

            def _is_receiver_online(self, device_id: str) -> bool:
                return bool(api.hub and device_id in api.hub.receivers)

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length == 0:
                    return {}
                return json.loads(self.rfile.read(length).decode("utf-8"))

            def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status.value)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                self.end_headers()
                self.wfile.write(data)

            @staticmethod
            def _required(body: dict[str, Any], key: str) -> str:
                value = str(body.get(key, "")).strip()
                if not value:
                    raise ValueError(f"missing {key}")
                return value

        return Handler

    def manager_web_url(self) -> str:
        return f"http://{self.config.host}:{self.config.manager_web_port}/"

    def issue_ice_servers(self, username: str) -> dict[str, Any]:
        servers: list[dict[str, Any]] = [{"urls": list(self.config.stun_urls)}] if self.config.stun_urls else []
        expires_at = int(time.time()) + max(60, self.config.turn_ttl_seconds)
        if self.config.turn_urls and self.config.turn_static_auth_secret:
            turn_username = f"{expires_at}:{username}"
            digest = hmac.new(
                self.config.turn_static_auth_secret.encode("utf-8"),
                turn_username.encode("utf-8"),
                hashlib.sha1,
            ).digest()
            servers.append(
                {
                    "urls": list(self.config.turn_urls),
                    "username": turn_username,
                    "credential": base64.b64encode(digest).decode("ascii"),
                    "credentialType": "password",
                }
            )
        return {"ice_servers": servers, "expires_at": expires_at}

    def dispatch_command(self, device_id: str, command: dict[str, Any]) -> dict[str, Any]:
        if not self.store.get_receiver(device_id):
            raise ValueError("receiver not found")
        packet = packet_from_command(command)
        transport = (
            trace_mark(packet, TRACE_STAGE_SENDER_CREATED, command.get("client_sent_at_ms"))
            if command.get("debug_trace")
            else packet
        )
        packet_hex = packet_to_hex(packet)
        parsed = parse_packet(packet)

        delivered = False
        if self.hub and self.event_loop and self.event_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self.hub.send_to_receiver(device_id, transport), self.event_loop)
            delivered = bool(future.result(timeout=3))

        queue_id = None
        if not delivered:
            queue_id = self.store.enqueue_command(device_id, transport, packet_to_hex(transport))

        self.store.add_event(
            "manager_to_receiver" if delivered else "manager_to_queue",
            packet_hex,
            device_id=device_id,
            frame_type=parsed.frame_type.name.lower() if parsed else "",
            payload=parsed.payload if parsed else {},
        )
        return {"delivered": delivered, "queue_id": queue_id, "packet_hex": packet_hex}

    def _bridge_connect(self, device_id: str, device_name: str = "") -> dict[str, Any]:
        if not self.bridge:
            raise ValueError("xiaozhi bridge not available")
        if not self.event_loop or not self.event_loop.is_running():
            raise ValueError("event loop not available")
        future = asyncio.run_coroutine_threadsafe(self.bridge.connect(device_id, device_name), self.event_loop)
        session = future.result(timeout=10)
        return {"device_id": device_id, "session_id": session.session_id, "connected": session.connected}

    def _bridge_disconnect(self, device_id: str) -> dict[str, Any]:
        if not self.bridge:
            raise ValueError("xiaozhi bridge not available")
        if not self.event_loop or not self.event_loop.is_running():
            raise ValueError("event loop not available")
        future = asyncio.run_coroutine_threadsafe(self.bridge.disconnect(device_id), self.event_loop)
        future.result(timeout=5)
        return {"device_id": device_id, "connected": False}

    def _bridge_send_text(self, device_id: str, text: str) -> dict[str, Any]:
        if not self.bridge:
            raise ValueError("xiaozhi bridge not available")
        if not self.event_loop or not self.event_loop.is_running():
            raise ValueError("event loop not available")
        future = asyncio.run_coroutine_threadsafe(self.bridge.send_text(device_id, text), self.event_loop)
        sent = bool(future.result(timeout=5))
        return {"device_id": device_id, "sent": sent}

    def _bridge_audio_start(self, device_id: str) -> dict[str, Any]:
        if not self.bridge:
            raise ValueError("xiaozhi bridge not available")
        if not self.event_loop or not self.event_loop.is_running():
            raise ValueError("event loop not available")
        future = asyncio.run_coroutine_threadsafe(self.bridge.send_audio_start(device_id), self.event_loop)
        sent = bool(future.result(timeout=5))
        logger.info("bridge audio start: device=%s sent=%s", device_id, sent)
        return {"device_id": device_id, "sent": sent}

    def _bridge_audio_frame(self, device_id: str, audio_b64: str) -> dict[str, Any]:
        if not self.bridge:
            raise ValueError("xiaozhi bridge not available")
        if not self.event_loop or not self.event_loop.is_running():
            raise ValueError("event loop not available")
        try:
            opus_data = base64.b64decode(audio_b64, validate=True)
        except Exception as exc:
            raise ValueError("invalid audio_b64") from exc
        if len(opus_data) > 64 * 1024:
            raise ValueError("audio frame too large")
        future = asyncio.run_coroutine_threadsafe(self.bridge.send_audio_frame(device_id, opus_data), self.event_loop)
        sent = bool(future.result(timeout=5))
        logger.info("bridge audio frame: device=%s bytes=%s sent=%s", device_id, len(opus_data), sent)
        return {"device_id": device_id, "sent": sent, "bytes": len(opus_data)}

    def _bridge_audio_stop(self, device_id: str) -> dict[str, Any]:
        if not self.bridge:
            raise ValueError("xiaozhi bridge not available")
        if not self.event_loop or not self.event_loop.is_running():
            raise ValueError("event loop not available")
        future = asyncio.run_coroutine_threadsafe(self.bridge.send_audio_stop(device_id), self.event_loop)
        sent = bool(future.result(timeout=5))
        logger.info("bridge audio stop: device=%s sent=%s", device_id, sent)
        return {"device_id": device_id, "sent": sent}

    def _bridge_events(self, device_id: str, after_seq: int, limit: int) -> list[dict[str, Any]]:
        if not device_id:
            raise ValueError("missing device_id")
        if not self.bridge:
            return []
        return self.bridge.get_events(device_id, after_seq, limit)

    def _bridge_status(self) -> dict[str, Any]:
        if not self.bridge:
            return {"available": False, "sessions": []}
        sessions = []
        for device_id, session in self.bridge._sessions.items():
            sessions.append({
                "device_id": device_id,
                "session_id": session.session_id,
                "connected": session.connected,
                "connected_at": session.connected_at,
                "last_activity": session.last_activity,
            })
        return {"available": True, "sessions": sessions}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ManagerApi().start()


if __name__ == "__main__":
    main()
