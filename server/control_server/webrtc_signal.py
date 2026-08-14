from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import struct
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from manager_api.config.settings import ServerConfig, load_config
from manager_api.storage.store import Store


logger = logging.getLogger(__name__)
MAX_SIGNAL_BYTES = 1024 * 1024


@dataclass
class WebRtcPeer:
    room_id: str
    role: str
    writer: asyncio.StreamWriter
    remote_addr: str
    connected_at: float
    send: Callable[[bytes], Awaitable[None]]


class WebRtcSignalHub:
    """WebRTC 信令转发器。

    客户端先发送一行 JSON 握手：
      {"room_id":"car-001", "role":"caller"}
      {"room_id":"car-001", "role":"answerer"}

    握手后继续使用 Android 当前 SignalChannel 的 4 字节大端长度 + JSON 消息格式。
    server 不理解 SDP/ICE 内容，只按 room 在 caller/answerer 之间透明转发。
    """

    def __init__(self, store: Store):
        self.store = store
        self.rooms: dict[str, dict[str, WebRtcPeer]] = {}
        self._lock = asyncio.Lock()

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        room_id = ""
        role = ""
        try:
            hello = await asyncio.wait_for(reader.readline(), timeout=5)
            if hello.startswith(b"GET "):
                await self._handle_websocket_client(hello, reader, writer)
                return
            meta = json.loads(hello.decode("utf-8"))
            room_id = str(meta.get("room_id") or meta.get("device_id") or meta.get("device-id") or "").strip()
            role = str(meta.get("role", "")).lower().strip()
            token = str(meta.get("token", ""))
            if not room_id:
                raise PermissionError("missing room_id")
            if role not in {"caller", "answerer"}:
                raise PermissionError("role must be caller or answerer")
            if not self.store.user_by_token(token):
                raise PermissionError("account login required")

            await self._join(room_id, role, writer, lambda payload: self._send_tcp_payload(writer, payload))
            await self._write_json_line(writer, {"ok": True, "room_id": room_id, "role": role})
            await self._relay_loop_tcp(room_id, role, reader)
        except (json.JSONDecodeError, UnicodeDecodeError):
            await self._write_json_line(writer, {"ok": False, "error": "first line must be json handshake"})
        except PermissionError as exc:
            await self._write_json_line(writer, {"ok": False, "error": str(exc)})
        except asyncio.IncompleteReadError:
            logger.info("webrtc peer disconnected room=%s role=%s", room_id, role)
        except Exception:
            logger.exception("webrtc signal client error room=%s role=%s", room_id, role)
        finally:
            await self._leave(room_id, role, writer)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _handle_websocket_client(
        self,
        first_line: bytes,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        room_id = ""
        role = ""
        try:
            await self._accept_websocket(first_line, reader, writer)
            hello_payload = await self._read_ws_payload(reader, writer)
            if hello_payload is None:
                raise PermissionError("missing websocket handshake payload")
            meta = json.loads(hello_payload.decode("utf-8"))
            room_id = str(meta.get("room_id") or meta.get("device_id") or meta.get("device-id") or "").strip()
            role = str(meta.get("role", "")).lower().strip()
            token = str(meta.get("token", ""))
            if not room_id:
                raise PermissionError("missing room_id")
            if role not in {"caller", "answerer"}:
                raise PermissionError("role must be caller or answerer")
            if not self.store.user_by_token(token):
                raise PermissionError("account login required")

            await self._join(room_id, role, writer, lambda payload: self._send_ws_payload(writer, payload))
            await self._send_ws_payload(writer, json.dumps({"ok": True, "room_id": room_id, "role": role}).encode("utf-8"))
            await self._relay_loop_ws(room_id, role, reader, writer)
        except (json.JSONDecodeError, UnicodeDecodeError):
            await self._send_ws_payload(writer, json.dumps({"ok": False, "error": "first frame must be json handshake"}).encode("utf-8"))
        except PermissionError as exc:
            await self._send_ws_payload(writer, json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"))
        except asyncio.IncompleteReadError:
            logger.info("webrtc websocket peer disconnected room=%s role=%s", room_id, role)
        except Exception:
            logger.exception("webrtc websocket client error room=%s role=%s", room_id, role)
        finally:
            await self._leave(room_id, role, writer)

    async def _join(
        self,
        room_id: str,
        role: str,
        writer: asyncio.StreamWriter,
        send: Callable[[bytes], Awaitable[None]],
    ) -> None:
        async with self._lock:
            room = self.rooms.setdefault(room_id, {})
            old = room.get(role)
            if old and old.writer is not writer:
                old.writer.close()
            room[role] = WebRtcPeer(room_id, role, writer, self._remote_addr(writer), time.time(), send)
            logger.info("webrtc peer online room=%s role=%s addr=%s", room_id, role, self._remote_addr(writer))

    async def _leave(self, room_id: str, role: str, writer: asyncio.StreamWriter) -> None:
        if not room_id or not role:
            return
        async with self._lock:
            room = self.rooms.get(room_id)
            if not room:
                return
            peer = room.get(role)
            if peer and peer.writer is writer:
                room.pop(role, None)
                logger.info("webrtc peer offline room=%s role=%s", room_id, role)
            if not room:
                self.rooms.pop(room_id, None)

    async def _relay_loop_tcp(self, room_id: str, role: str, reader: asyncio.StreamReader) -> None:
        while True:
            header = await reader.readexactly(4)
            length = struct.unpack(">I", header)[0]
            if length <= 0 or length > MAX_SIGNAL_BYTES:
                raise ValueError(f"invalid signal length: {length}")
            payload = await reader.readexactly(length)
            peer = await self._opposite(room_id, role)
            if peer is None:
                logger.info("webrtc signal queued nowhere room=%s from=%s bytes=%s", room_id, role, length)
                continue
            try:
                await peer.send(payload)
            except Exception:
                await self._leave(room_id, peer.role, peer.writer)

    async def _relay_loop_ws(
        self,
        room_id: str,
        role: str,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        while True:
            payload = await self._read_ws_payload(reader, writer)
            if payload is None:
                break
            if len(payload) <= 0 or len(payload) > MAX_SIGNAL_BYTES:
                raise ValueError(f"invalid websocket signal length: {len(payload)}")
            peer = await self._opposite(room_id, role)
            if peer is None:
                logger.info("webrtc websocket signal queued nowhere room=%s from=%s bytes=%s", room_id, role, len(payload))
                continue
            try:
                await peer.send(payload)
            except Exception:
                await self._leave(room_id, peer.role, peer.writer)

    @staticmethod
    async def _send_tcp_payload(writer: asyncio.StreamWriter, payload: bytes) -> None:
        writer.write(struct.pack(">I", len(payload)) + payload)
        await writer.drain()

    async def _accept_websocket(
        self,
        first_line: bytes,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        headers: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if line in {b"\r\n", b"\n", b""}:
                break
            text = line.decode("latin1").strip()
            if ":" in text:
                key, value = text.split(":", 1)
                headers[key.lower()] = value.strip()
        ws_key = headers.get("sec-websocket-key")
        if not ws_key:
            raise PermissionError("missing Sec-WebSocket-Key")
        accept = base64.b64encode(
            hashlib.sha1((ws_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        ).decode("ascii")
        writer.write(
            (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n"
                "\r\n"
            ).encode("ascii")
        )
        await writer.drain()

    async def _read_ws_payload(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> bytes | None:
        while True:
            head = await reader.readexactly(2)
            opcode = head[0] & 0x0F
            masked = (head[1] & 0x80) != 0
            length = head[1] & 0x7F
            if length == 126:
                length = struct.unpack(">H", await reader.readexactly(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", await reader.readexactly(8))[0]
            if length > MAX_SIGNAL_BYTES:
                raise ValueError(f"invalid websocket frame length: {length}")
            mask = await reader.readexactly(4) if masked else b""
            payload = await reader.readexactly(length) if length else b""
            if masked:
                payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
            if opcode == 0x8:
                return None
            if opcode == 0x9:
                await self._send_ws_payload(writer, payload, opcode=0xA)
                continue
            if opcode in {0x1, 0x2}:
                return payload

    @staticmethod
    async def _send_ws_payload(writer: asyncio.StreamWriter, payload: bytes, opcode: int = 0x1) -> None:
        header = bytearray([0x80 | opcode])
        length = len(payload)
        if length < 126:
            header.append(length)
        elif length <= 0xFFFF:
            header.append(126)
            header.extend(struct.pack(">H", length))
        else:
            header.append(127)
            header.extend(struct.pack(">Q", length))
        writer.write(bytes(header) + payload)
        await writer.drain()

    async def _opposite(self, room_id: str, role: str) -> WebRtcPeer | None:
        opposite_role = "answerer" if role == "caller" else "caller"
        async with self._lock:
            return self.rooms.get(room_id, {}).get(opposite_role)

    @staticmethod
    async def _write_json_line(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
        writer.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        await writer.drain()

    @staticmethod
    def _remote_addr(writer: asyncio.StreamWriter) -> str:
        peer = writer.get_extra_info("peername")
        if isinstance(peer, tuple) and len(peer) >= 2:
            return f"{peer[0]}:{peer[1]}"
        return str(peer or "")


class WebRtcSignalServer:
    def __init__(self, config: ServerConfig | None = None, store: Store | None = None):
        self.config = config or load_config()
        self.store = store or Store(self.config.database_path)
        self.hub = WebRtcSignalHub(store=self.store)

    async def start(self) -> None:
        server = await asyncio.start_server(self.hub.handle_client, self.config.host, self.config.webrtc_port)
        sockets = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
        logger.info("webrtc_signal listening on %s", sockets)
        async with server:
            await server.serve_forever()


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    await WebRtcSignalServer().start()


if __name__ == "__main__":
    asyncio.run(main())
