from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from dataclasses import dataclass
from typing import Any

from manager_api.config.settings import ServerConfig, load_config


logger = logging.getLogger(__name__)
MAX_SIGNAL_BYTES = 1024 * 1024


@dataclass
class WebRtcPeer:
    room_id: str
    role: str
    writer: asyncio.StreamWriter
    remote_addr: str
    connected_at: float


class WebRtcSignalHub:
    """WebRTC 信令转发器。

    客户端先发送一行 JSON 握手：
      {"room_id":"car-001", "role":"caller"}
      {"room_id":"car-001", "role":"answerer"}

    握手后继续使用 Android 当前 SignalChannel 的 4 字节大端长度 + JSON 消息格式。
    server 不理解 SDP/ICE 内容，只按 room 在 caller/answerer 之间透明转发。
    """

    def __init__(self, auth_token: str = ""):
        self.auth_token = auth_token
        self.rooms: dict[str, dict[str, WebRtcPeer]] = {}
        self._lock = asyncio.Lock()

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        room_id = ""
        role = ""
        try:
            hello = await asyncio.wait_for(reader.readline(), timeout=5)
            meta = json.loads(hello.decode("utf-8"))
            room_id = str(meta.get("room_id") or meta.get("device_id") or meta.get("device-id") or "").strip()
            role = str(meta.get("role", "")).lower().strip()
            token = str(meta.get("token", ""))
            if not room_id:
                raise PermissionError("missing room_id")
            if role not in {"caller", "answerer"}:
                raise PermissionError("role must be caller or answerer")
            if self.auth_token and token != self.auth_token:
                raise PermissionError("invalid token")

            await self._join(room_id, role, writer)
            await self._write_json_line(writer, {"ok": True, "room_id": room_id, "role": role})
            await self._relay_loop(room_id, role, reader)
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

    async def _join(self, room_id: str, role: str, writer: asyncio.StreamWriter) -> None:
        async with self._lock:
            room = self.rooms.setdefault(room_id, {})
            old = room.get(role)
            if old and old.writer is not writer:
                old.writer.close()
            room[role] = WebRtcPeer(room_id, role, writer, self._remote_addr(writer), time.time())
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

    async def _relay_loop(self, room_id: str, role: str, reader: asyncio.StreamReader) -> None:
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
                peer.writer.write(header + payload)
                await peer.writer.drain()
            except Exception:
                await self._leave(room_id, peer.role, peer.writer)

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
    def __init__(self, config: ServerConfig | None = None):
        self.config = config or load_config()
        self.hub = WebRtcSignalHub(auth_token=self.config.auth_token)

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
