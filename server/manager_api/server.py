from __future__ import annotations

import asyncio
import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from server.common.config import ServerConfig, load_config
from server.common.protocol import packet_from_command, packet_to_hex, parse_packet
from server.common.store import Store
from server.control_server.server import ConnectionHub


logger = logging.getLogger(__name__)


class ManagerApi:
    def __init__(
        self,
        config: ServerConfig | None = None,
        store: Store | None = None,
        hub: ConnectionHub | None = None,
        event_loop: asyncio.AbstractEventLoop | None = None,
    ):
        self.config = config or load_config()
        self.store = store or Store(self.config.database_path)
        self.hub = hub
        self.event_loop = event_loop
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

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path.rstrip("/") or "/"
                query = parse_qs(parsed.query)

                if path == "/health":
                    self._send_json({"ok": True, "service": "manager-api"})
                elif path == "/":
                    self._send_html(api.manager_web_html())
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
                elif path == "/api/events":
                    limit = int((query.get("limit") or [100])[0])
                    device_id = (query.get("device_id") or [""])[0]
                    self._send_json({"ok": True, "data": api.store.list_events(limit=limit, device_id=device_id)})
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
                    else:
                        self._send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
                except ValueError as exc:
                    self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                except Exception as exc:
                    logger.exception("manager api request failed")
                    self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

            def log_message(self, fmt: str, *args: Any) -> None:
                logger.info("%s - %s", self.address_string(), fmt % args)

            def _authorized(self) -> bool:
                header = self.headers.get("Authorization", "")
                token = header[7:] if header.startswith("Bearer ") else header
                return bool((api.config.auth_token and token == api.config.auth_token) or api.store.user_by_token(token))

            def _current_user(self) -> dict[str, Any] | None:
                header = self.headers.get("Authorization", "")
                token = header[7:] if header.startswith("Bearer ") else header
                return api.store.user_by_token(token)

            def _can_access(self, row: dict[str, Any]) -> bool:
                user = self._current_user()
                return not user or not row.get("owner_username") or row.get("owner_username") == user.get("username")

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
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                self.end_headers()
                self.wfile.write(data)

            def _send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
                data = html.encode("utf-8")
                self.send_response(status.value)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            @staticmethod
            def _required(body: dict[str, Any], key: str) -> str:
                value = str(body.get(key, "")).strip()
                if not value:
                    raise ValueError(f"missing {key}")
                return value

        return Handler

    @staticmethod
    def manager_web_html() -> str:
        return """<!doctype html><html><head><meta charset='utf-8'><title>CruiseCar Manager</title>
<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:28px;background:#f7f7f7}input,button{font-size:15px;padding:8px;margin:4px}table{border-collapse:collapse;background:white;margin-top:12px}td,th{border:1px solid #ddd;padding:8px 12px}pre{background:#111;color:#0f0;padding:12px;white-space:pre-wrap}</style></head>
<body><h2>CruiseCar Manager</h2>
<div>账号 <input id='u' placeholder='username'> 密码 <input id='p' type='password' placeholder='password'><button onclick='login()'>登录/注册</button></div>
<div>Token <input id='t' style='width:520px' placeholder='login token'></div>
<h3>加入接收端</h3><input id='did' placeholder='device_id'><input id='dn' placeholder='name'><button onclick='addReceiver()'>加入</button>
<h3>接收端列表</h3><button onclick='loadReceivers()'>刷新</button><div id='devices'></div><h3>事件</h3><button onclick='loadEvents()'>刷新事件</button><pre id='log'></pre>
<script>
async function api(path,opt={}){opt.headers=Object.assign({'Content-Type':'application/json'},opt.headers||{});let tok=document.getElementById('t').value;if(tok)opt.headers.Authorization='Bearer '+tok;let r=await fetch(path,opt);let j=await r.json();if(!j.ok)throw new Error(j.error||'failed');return j.data||j}
async function login(){let d=await api('/api/auth/login',{method:'POST',body:JSON.stringify({username:u.value,password:p.value})});t.value=d.token;loadReceivers()}
async function addReceiver(){await api('/api/receivers',{method:'POST',body:JSON.stringify({device_id:did.value,name:dn.value})});loadReceivers()}
async function loadReceivers(){let rows=await api('/api/receivers');devices.innerHTML='<table><tr><th>ID</th><th>名称</th><th>在线</th><th>ESP32</th><th>模式</th><th>地址</th></tr>'+rows.map(x=>`<tr><td>${x.device_id}</td><td>${x.name||''}</td><td>${x.online}</td><td>${x.esp_connected}</td><td>${x.mode}</td><td>${x.remote_addr||''}</td></tr>`).join('')+'</table>'}
async function loadEvents(){let rows=await api('/api/events?limit=50');log.textContent=JSON.stringify(rows,null,2)}
</script></body></html>"""

    def dispatch_command(self, device_id: str, command: dict[str, Any]) -> dict[str, Any]:
        if not self.store.get_receiver(device_id):
            raise ValueError("receiver not found")
        packet = packet_from_command(command)
        packet_hex = packet_to_hex(packet)
        parsed = parse_packet(packet)

        delivered = False
        if self.hub and self.event_loop and self.event_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self.hub.send_to_receiver(device_id, packet), self.event_loop)
            delivered = bool(future.result(timeout=3))

        queue_id = None
        if not delivered:
            queue_id = self.store.enqueue_command(device_id, packet, packet_hex)

        self.store.add_event(
            "manager_to_receiver" if delivered else "manager_to_queue",
            packet_hex,
            device_id=device_id,
            frame_type=parsed.frame_type.name.lower() if parsed else "",
            payload=parsed.payload if parsed else {},
        )
        return {"delivered": delivered, "queue_id": queue_id, "packet_hex": packet_hex}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ManagerApi().start()


if __name__ == "__main__":
    main()
