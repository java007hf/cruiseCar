from __future__ import annotations

import asyncio
import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

from manager_api.config.settings import ServerConfig, load_config
from manager_api.mcp_tools import TOOL_DEFINITIONS, handle_tool_call

if TYPE_CHECKING:
    from control_server.server import ConnectionHub
    from manager_api.storage.store import Store

logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2025-03-26"
SERVER_NAME = "cruiseCar-mcp"
SERVER_VERSION = "1.0.0"


class McpServer:
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
        self._device_sessions: dict[str, str] = {}

    def start(self) -> None:
        handler = self._handler_class()
        self.httpd = ThreadingHTTPServer((self.config.host, self.config.xiaozhi_mcp_port), handler)
        logger.info("mcp-server listening on %s:%s", self.config.host, self.config.xiaozhi_mcp_port)
        self.httpd.serve_forever()

    def start_in_thread(self) -> threading.Thread:
        thread = threading.Thread(target=self.start, name="mcp-server", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()

    def set_device_session(self, session_id: str, device_id: str) -> None:
        self._device_sessions[session_id] = device_id

    def get_device_id(self, session_id: str) -> str | None:
        return self._device_sessions.get(session_id)

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        mcp = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "CruiseCarMCP/1.0"

            def do_POST(self) -> None:
                body = self._read_json()
                if body is None:
                    self._send_json({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None}, HTTPStatus.BAD_REQUEST)
                    return

                if isinstance(body, list):
                    self._send_json({"jsonrpc": "2.0", "error": {"code": -32600, "message": "Batch not supported"}, "id": None}, HTTPStatus.BAD_REQUEST)
                    return

                if not self._validate_mcp_auth():
                    self._send_json({"jsonrpc": "2.0", "error": {"code": -32001, "message": "Unauthorized"}, "id": body.get("id")}, HTTPStatus.UNAUTHORIZED)
                    return

                response = mcp._handle_request(body, self.headers.get("Mcp-Session-Id", ""))
                self._send_json(response)

            def do_GET(self) -> None:
                if self.path.rstrip("/") == "/mcp":
                    self._send_json({"name": SERVER_NAME, "version": SERVER_VERSION, "protocol": MCP_PROTOCOL_VERSION})
                else:
                    self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

            def do_OPTIONS(self) -> None:
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, Mcp-Session-Id")
                self.end_headers()

            def log_message(self, fmt: str, *args: Any) -> None:
                logger.info("%s - %s", self.address_string(), fmt % args)

            def _read_json(self) -> dict[str, Any] | None:
                try:
                    length = int(self.headers.get("Content-Length", "0") or "0")
                    if length == 0:
                        return None
                    return json.loads(self.rfile.read(length).decode("utf-8"))
                except (json.JSONDecodeError, ValueError):
                    return None

            def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status.value)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, Mcp-Session-Id")
                self.end_headers()
                self.wfile.write(data)

            def _validate_mcp_auth(self) -> bool:
                token = mcp.config.xiaozhi_mcp_token
                if not token:
                    return True
                header = self.headers.get("Authorization", "")
                bearer = header[7:] if header.startswith("Bearer ") else header
                return bearer == token

        return Handler

    def _handle_request(self, request: dict[str, Any], session_id: str) -> dict[str, Any]:
        jsonrpc = request.get("jsonrpc")
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        if jsonrpc != "2.0":
            return self._error(req_id, -32600, "Invalid Request: jsonrpc must be 2.0")

        if method == "initialize":
            return self._handle_initialize(req_id, params, session_id)
        if method == "notifications/initialized":
            return {}
        if method == "tools/list":
            return self._handle_tools_list(req_id)
        if method == "tools/call":
            return self._handle_tools_call(req_id, params, session_id)
        if method == "ping":
            return {"jsonrpc": "2.0", "result": {}, "id": req_id}

        return self._error(req_id, -32601, f"Method not found: {method}")

    def _handle_initialize(self, req_id: Any, params: dict[str, Any], session_id: str) -> dict[str, Any]:
        new_session = f"mcp-{id(params):x}"
        device_id = ""
        meta = params.get("_meta", {})
        if isinstance(meta, dict):
            device_id = str(meta.get("device_id", ""))
        if not device_id:
            device_id = str(params.get("device_id", ""))
        if device_id:
            self._device_sessions[new_session] = device_id
            logger.info("MCP session initialized: session=%s device=%s", new_session, device_id)

        return {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "instructions": "cruiseCar smart RC car control. Use tools to move, set mode, adjust servo, connect ESP32, or get status.",
            },
            "id": req_id,
        }

    def _handle_tools_list(self, req_id: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "result": {"tools": TOOL_DEFINITIONS}, "id": req_id}

    def _handle_tools_call(self, req_id: Any, params: dict[str, Any], session_id: str) -> dict[str, Any]:
        tool_name = str(params.get("name", ""))
        arguments = params.get("arguments", {})
        device_id = self._device_sessions.get(session_id, "")

        if not device_id:
            meta = params.get("_meta", {})
            if isinstance(meta, dict):
                device_id = str(meta.get("device_id", ""))

        if not device_id:
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [{"type": "text", "text": "No device_id associated with this session. Pass device_id in _meta or initialize with device_id."}],
                    "isError": True,
                },
                "id": req_id,
            }

        logger.info("MCP tool call: tool=%s device=%s args=%s", tool_name, device_id, arguments)
        result = handle_tool_call(device_id, tool_name, arguments, self.store, self.hub, self.event_loop)
        return {"jsonrpc": "2.0", "result": result, "id": req_id}

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "error": {"code": code, "message": message}, "id": req_id}
