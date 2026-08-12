from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any


class Store:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS receivers (
                    device_id TEXT PRIMARY KEY,
                    owner_username TEXT NOT NULL DEFAULT '',
                    name TEXT NOT NULL DEFAULT '',
                    token TEXT NOT NULL DEFAULT '',
                    online INTEGER NOT NULL DEFAULT 0,
                    esp_connected INTEGER NOT NULL DEFAULT 0,
                    mode TEXT NOT NULL DEFAULT 'manual',
                    remote_addr TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS senders (
                    sender_id TEXT PRIMARY KEY,
                    owner_username TEXT NOT NULL DEFAULT '',
                    name TEXT NOT NULL DEFAULT '',
                    token TEXT NOT NULL DEFAULT '',
                    target_device_id TEXT NOT NULL DEFAULT '',
                    online INTEGER NOT NULL DEFAULT 0,
                    remote_addr TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL DEFAULT '',
                    sender_id TEXT NOT NULL DEFAULT '',
                    direction TEXT NOT NULL,
                    frame_type TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    packet_hex TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS command_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    packet BLOB NOT NULL,
                    packet_hex TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    sent_at REAL
                );

                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    token TEXT NOT NULL UNIQUE,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            self._ensure_column(conn, "receivers", "owner_username", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "senders", "owner_username", "TEXT NOT NULL DEFAULT ''")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        if column not in {row["name"] for row in rows}:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def create_or_login_user(self, username: str, password: str) -> dict[str, Any]:
        username = username.strip()
        if not username:
            raise ValueError("username is required")
        if not password:
            raise ValueError("password is required")
        password_hash = self._password_hash(password)
        now = time.time()
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
            if row:
                if row["password_hash"] != password_hash:
                    raise PermissionError("invalid username or password")
                token = row["token"]
                conn.execute("UPDATE users SET updated_at=? WHERE username=?", (now, username))
            else:
                token = secrets.token_urlsafe(32)
                conn.execute(
                    "INSERT INTO users(username, password_hash, token, created_at, updated_at) VALUES(?, ?, ?, ?, ?)",
                    (username, password_hash, token, now, now),
                )
        return {"username": username, "token": token}

    def user_by_token(self, token: str) -> dict[str, Any] | None:
        if not token:
            return None
        row = self._get_one("SELECT username, token, created_at, updated_at FROM users WHERE token=?", (token,))
        return row

    def upsert_receiver(self, device_id: str, name: str = "", token: str = "", owner_username: str = "") -> dict[str, Any]:
        now = time.time()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO receivers(device_id, owner_username, name, token, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(device_id) DO UPDATE SET
                    owner_username=COALESCE(NULLIF(excluded.owner_username, ''), receivers.owner_username),
                    name=COALESCE(NULLIF(excluded.name, ''), receivers.name),
                    token=COALESCE(NULLIF(excluded.token, ''), receivers.token),
                    updated_at=excluded.updated_at
                """,
                (device_id, owner_username, name, token, now, now),
            )
        return self.get_receiver(device_id) or {}

    def upsert_sender(self, sender_id: str, name: str = "", token: str = "", target_device_id: str = "", owner_username: str = "") -> dict[str, Any]:
        now = time.time()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO senders(sender_id, owner_username, name, token, target_device_id, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sender_id) DO UPDATE SET
                    owner_username=COALESCE(NULLIF(excluded.owner_username, ''), senders.owner_username),
                    name=COALESCE(NULLIF(excluded.name, ''), senders.name),
                    token=COALESCE(NULLIF(excluded.token, ''), senders.token),
                    target_device_id=COALESCE(NULLIF(excluded.target_device_id, ''), senders.target_device_id),
                    updated_at=excluded.updated_at
                """,
                (sender_id, owner_username, name, token, target_device_id, now, now),
            )
        return self.get_sender(sender_id) or {}

    def set_receiver_online(self, device_id: str, online: bool, remote_addr: str = "") -> None:
        now = time.time()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE receivers
                SET online=?, remote_addr=?, updated_at=?, last_seen_at=?
                WHERE device_id=?
                """,
                (1 if online else 0, remote_addr, now, now, device_id),
            )

    def set_sender_online(self, sender_id: str, online: bool, remote_addr: str = "") -> None:
        now = time.time()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE senders
                SET online=?, remote_addr=?, updated_at=?, last_seen_at=?
                WHERE sender_id=?
                """,
                (1 if online else 0, remote_addr, now, now, sender_id),
            )

    def update_receiver_status(self, device_id: str, esp_connected: bool, mode: str) -> None:
        now = time.time()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE receivers
                SET esp_connected=?, mode=?, updated_at=?, last_seen_at=?
                WHERE device_id=?
                """,
                (1 if esp_connected else 0, mode, now, now, device_id),
            )

    def get_receiver(self, device_id: str) -> dict[str, Any] | None:
        return self._get_one("SELECT * FROM receivers WHERE device_id=?", (device_id,))

    def get_sender(self, sender_id: str) -> dict[str, Any] | None:
        return self._get_one("SELECT * FROM senders WHERE sender_id=?", (sender_id,))

    def list_receivers(self) -> list[dict[str, Any]]:
        return self._get_all("SELECT * FROM receivers ORDER BY updated_at DESC")

    def list_receivers_for_user(self, username: str) -> list[dict[str, Any]]:
        return self._get_all(
            "SELECT * FROM receivers WHERE owner_username=? ORDER BY updated_at DESC", (username,)
        )

    def list_senders(self) -> list[dict[str, Any]]:
        return self._get_all("SELECT * FROM senders ORDER BY updated_at DESC")

    def list_senders_for_user(self, username: str) -> list[dict[str, Any]]:
        return self._get_all(
            "SELECT * FROM senders WHERE owner_username=? ORDER BY updated_at DESC", (username,)
        )

    def add_event(
        self,
        direction: str,
        packet_hex: str,
        device_id: str = "",
        sender_id: str = "",
        frame_type: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO events(device_id, sender_id, direction, frame_type, payload_json, packet_hex, created_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (device_id, sender_id, direction, frame_type, json.dumps(payload or {}, ensure_ascii=False), packet_hex, time.time()),
            )

    def list_events(self, limit: int = 100, device_id: str = "") -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        if device_id:
            rows = self._get_all(
                "SELECT * FROM events WHERE device_id=? ORDER BY id DESC LIMIT ?", (device_id, limit)
            )
        else:
            rows = self._get_all("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))
        for row in rows:
            row["payload"] = json.loads(row.pop("payload_json") or "{}")
        return rows

    def enqueue_command(self, device_id: str, packet: bytes, packet_hex: str) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO command_queue(device_id, packet, packet_hex, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (device_id, packet, packet_hex, time.time()),
            )
            return int(cur.lastrowid)

    def take_pending_commands(self, device_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM command_queue
                WHERE device_id=? AND status='pending'
                ORDER BY id ASC
                LIMIT ?
                """,
                (device_id, limit),
            ).fetchall()
            ids = [str(row["id"]) for row in rows]
            if ids:
                now = time.time()
                conn.execute(
                    f"UPDATE command_queue SET status='sent', sent_at=? WHERE id IN ({','.join('?' for _ in ids)})",
                    (now, *ids),
                )
        return [self._row_to_dict(row) for row in rows]

    def _get_one(self, sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return self._row_to_dict(row) if row else None

    def _get_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        for key in ("online", "esp_connected"):
            if key in data:
                data[key] = bool(data[key])
        return data

    @staticmethod
    def _password_hash(password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()
