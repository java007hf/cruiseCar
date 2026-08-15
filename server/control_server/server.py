from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
from dataclasses import dataclass
from typing import Any

from manager_api.config.settings import ServerConfig, load_config
from manager_api.protocol.control_protocol import (
    PACKET_SIZE,
    TRACE_TYPE,
    FrameType,
    control_packet_from_transport,
    packet_to_hex,
    parse_packet,
    trace_mark,
    TRACE_STAGE_SERVER_FORWARD,
    TRACE_STAGE_SERVER_RECEIVED,
)
from manager_api.storage.store import Store


logger = logging.getLogger(__name__)


@dataclass
class ReceiverSession:
    device_id: str
    writer: asyncio.StreamWriter
    remote_addr: str
    connected_at: float


@dataclass
class SenderSession:
    sender_id: str
    target_device_id: str
    writer: asyncio.StreamWriter
    remote_addr: str
    connected_at: float


class ConnectionHub:
    def __init__(self, store: Store, config: ServerConfig):
        self.store = store
        self.config = config
        self.receivers: dict[str, ReceiverSession] = {}
        self.senders: dict[str, SenderSession] = {}
        self._lock = asyncio.Lock()

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        remote_addr = self._remote_addr(writer)
        role = "unknown"
        identity = ""
        try:
            hello = await asyncio.wait_for(reader.readline(), timeout=5)
            if not hello:
                return
            meta = json.loads(hello.decode("utf-8"))
            role = str(meta.get("role", "")).lower()

            if role == "receiver":
                identity = await self._register_receiver(meta, writer, remote_addr)
                await self._receiver_loop(identity, reader)
            elif role == "sender":
                identity = await self._register_sender(meta, writer, remote_addr)
                await self._sender_loop(identity, reader)
            else:
                await self._write_json(writer, {"ok": False, "error": "role must be receiver or sender"})
        except (json.JSONDecodeError, UnicodeDecodeError):
            await self._write_json(writer, {"ok": False, "error": "first line must be json handshake"})
        except PermissionError as exc:
            await self._write_json(writer, {"ok": False, "error": str(exc)})
        except asyncio.IncompleteReadError:
            logger.info("%s %s disconnected", role, identity)
        except Exception:
            logger.exception("client error role=%s identity=%s", role, identity)
        finally:
            await self._disconnect(role, identity)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def send_to_receiver(self, device_id: str, packet: bytes) -> bool:
        session = self.receivers.get(device_id)
        if not session:
            return False
        try:
            transport = trace_mark(packet, TRACE_STAGE_SERVER_FORWARD) if len(packet) != PACKET_SIZE else packet
            session.writer.write(transport)
            await session.writer.drain()
            return True
        except Exception:
            await self._disconnect("receiver", device_id)
            return False

    async def flush_pending(self, device_id: str) -> int:
        sent = 0
        for command in self.store.take_pending_commands(device_id):
            if await self.send_to_receiver(device_id, command["packet"]):
                sent += 1
                self.store.add_event(
                    direction="manager_to_receiver",
                    packet_hex=command["packet_hex"],
                    device_id=device_id,
                )
            else:
                break
        return sent

    async def _register_receiver(self, meta: dict[str, Any], writer: asyncio.StreamWriter, remote_addr: str) -> str:
        device_id = str(meta.get("device_id") or meta.get("device-id") or "").strip()
        if not device_id:
            raise PermissionError("missing device_id")
        user = self._require_user_token(meta.get("token"))
        self.store.upsert_receiver(device_id, str(meta.get("name", "")), str(meta.get("token", "")), user["username"])

        async with self._lock:
            old = self.receivers.get(device_id)
            if old and old.writer is not writer:
                old.writer.close()
            self.receivers[device_id] = ReceiverSession(device_id, writer, remote_addr, time.time())
        self.store.set_receiver_online(device_id, True, remote_addr)
        await self._write_json(writer, {"ok": True, "role": "receiver", "device_id": device_id})
        await self.flush_pending(device_id)
        logger.info("receiver online device_id=%s addr=%s", device_id, remote_addr)
        return device_id

    async def _register_sender(self, meta: dict[str, Any], writer: asyncio.StreamWriter, remote_addr: str) -> str:
        sender_id = str(meta.get("sender_id") or meta.get("client_id") or meta.get("client-id") or "").strip()
        target_device_id = str(meta.get("target_device_id") or meta.get("device_id") or meta.get("device-id") or "").strip()
        if not sender_id:
            raise PermissionError("missing sender_id")
        if not target_device_id:
            raise PermissionError("missing target_device_id")
        user = self._require_user_token(meta.get("token"))
        receiver = self.store.get_receiver(target_device_id)
        if not receiver or receiver.get("owner_username") != user["username"]:
            raise PermissionError("target receiver is not in this account")
        self.store.upsert_sender(
            sender_id,
            str(meta.get("name", "")),
            str(meta.get("token", "")),
            target_device_id,
            user["username"],
        )

        async with self._lock:
            self.senders[sender_id] = SenderSession(sender_id, target_device_id, writer, remote_addr, time.time())
        self.store.set_sender_online(sender_id, True, remote_addr)
        await self._write_json(
            writer,
            {"ok": True, "role": "sender", "sender_id": sender_id, "target_device_id": target_device_id},
        )
        logger.info("sender online sender_id=%s target=%s addr=%s", sender_id, target_device_id, remote_addr)
        return sender_id

    async def _receiver_loop(self, device_id: str, reader: asyncio.StreamReader) -> None:
        while True:
            transport = await read_control_transport(reader)
            if len(transport) != PACKET_SIZE:
                transport = trace_mark(transport, TRACE_STAGE_SERVER_RECEIVED)
            packet = control_packet_from_transport(transport)
            parsed = parse_packet(packet)
            packet_hex = packet_to_hex(packet)
            if parsed is None:
                self.store.add_event("receiver_invalid", packet_hex, device_id=device_id)
                continue

            if parsed.frame_type == FrameType.STATUS:
                self.store.update_receiver_status(
                    device_id,
                    bool(parsed.payload.get("esp_connected")),
                    str(parsed.payload.get("mode", "manual")),
                )
                await self._broadcast_status(device_id, packet)

            self.store.add_event(
                "receiver_to_server",
                packet_hex,
                device_id=device_id,
                frame_type=parsed.frame_type.name.lower(),
                payload=parsed.payload,
            )
            await self.flush_pending(device_id)

    async def _sender_loop(self, sender_id: str, reader: asyncio.StreamReader) -> None:
        while True:
            transport = await read_control_transport(reader)
            if len(transport) != PACKET_SIZE:
                transport = trace_mark(transport, TRACE_STAGE_SERVER_RECEIVED)
            packet = control_packet_from_transport(transport)
            parsed = parse_packet(packet)
            packet_hex = packet_to_hex(packet)
            session = self.senders.get(sender_id)
            target_device_id = session.target_device_id if session else ""
            if parsed is None:
                self.store.add_event("sender_invalid", packet_hex, device_id=target_device_id, sender_id=sender_id)
                continue

            delivered = await self.send_to_receiver(target_device_id, transport)
            if not delivered:
                self.store.enqueue_command(target_device_id, transport, packet_to_hex(transport))
            self.store.add_event(
                "sender_to_receiver" if delivered else "sender_to_queue",
                packet_hex,
                device_id=target_device_id,
                sender_id=sender_id,
                frame_type=parsed.frame_type.name.lower(),
                payload=parsed.payload,
            )

    async def _broadcast_status(self, device_id: str, packet: bytes) -> None:
        stale: list[str] = []
        for sender_id, sender in self.senders.items():
            if sender.target_device_id != device_id:
                continue
            try:
                sender.writer.write(packet)
                await sender.writer.drain()
            except Exception:
                stale.append(sender_id)
        for sender_id in stale:
            await self._disconnect("sender", sender_id)

    async def _disconnect(self, role: str, identity: str) -> None:
        if not identity:
            return
        async with self._lock:
            if role == "receiver" and self.receivers.pop(identity, None):
                self.store.set_receiver_online(identity, False)
                logger.info("receiver offline device_id=%s", identity)
            elif role == "sender" and self.senders.pop(identity, None):
                self.store.set_sender_online(identity, False)
                logger.info("sender offline sender_id=%s", identity)

    def _require_user_token(self, token: Any) -> dict[str, Any]:
        user = self.store.user_by_token(str(token or ""))
        if not user:
            raise PermissionError("account login required")
        return user

    @staticmethod
    async def _write_json(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
        writer.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        await writer.drain()

    @staticmethod
    def _remote_addr(writer: asyncio.StreamWriter) -> str:
        peer = writer.get_extra_info("peername")
        if isinstance(peer, tuple) and len(peer) >= 2:
            return f"{peer[0]}:{peer[1]}"
        return str(peer or "")


async def read_control_transport(reader: asyncio.StreamReader) -> bytes:
    while True:
        first = await reader.readexactly(1)
        if first[0] != 0xAA:
            continue
        second = await reader.readexactly(1)
        if second[0] != 0x55:
            continue
        frame_type = await reader.readexactly(1)
        if frame_type[0] == TRACE_TYPE:
            payload_len = await reader.readexactly(1)
            rest = await reader.readexactly(payload_len[0] + 1)
            return first + second + frame_type + payload_len + rest
        rest = await reader.readexactly(PACKET_SIZE - 3)
        return first + second + frame_type + rest


class ControlServer:
    def __init__(self, config: ServerConfig | None = None, store: Store | None = None):
        self.config = config or load_config()
        self.store = store or Store(self.config.database_path)
        self.hub = ConnectionHub(self.store, self.config)

    async def start(self) -> None:
        server = await asyncio.start_server(self.hub.handle_client, self.config.host, self.config.control_port)
        sockets = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
        logger.info("control_server listening on %s", sockets)
        async with server:
            await server.serve_forever()


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from control_server.webrtc_signal import WebRtcSignalServer
    from manager_api.server import ManagerApi
    from manager_web.server import ManagerWeb

    config = load_config()
    store = Store(config.database_path)
    control_server = ControlServer(config=config, store=store)
    webrtc_server = WebRtcSignalServer(config=config, store=store)

    loop = asyncio.get_running_loop()
    manager_api = ManagerApi(config=config, store=store, hub=control_server.hub, event_loop=loop)
    manager_api.start_in_thread()
    manager_web = ManagerWeb(config=config)
    manager_web.start_in_thread()
    logging.info("deployment=full: control_server + webrtc_signal + manager-api + manager-web")

    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    control_task = asyncio.create_task(control_server.start())
    webrtc_task = asyncio.create_task(webrtc_server.start())
    stop_task = asyncio.create_task(stop_event.wait())
    done, pending = await asyncio.wait({control_task, webrtc_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    if manager_api:
        manager_api.stop()
    if manager_web:
        manager_web.stop()
    for task in done:
        if task in {control_task, webrtc_task}:
            task.result()


if __name__ == "__main__":
    asyncio.run(main())
