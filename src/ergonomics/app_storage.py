from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from .app_security import PayloadCipher, hash_session_token


SESSION_TTL_DAYS = 14


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


class AppStorage:
    def __init__(self, database_path: Path, cipher: PayloadCipher) -> None:
        self.database_path = database_path
        self.cipher = cipher

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        finally:
            connection.close()

    def init_schema(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    view TEXT NOT NULL,
                    status TEXT NOT NULL,
                    status_label TEXT NOT NULL,
                    model TEXT NOT NULL,
                    pose_detected INTEGER NOT NULL,
                    visible_landmarks_count INTEGER,
                    encrypted_payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_analyses_user_created
                    ON analyses(user_id, created_at DESC);
                """
            )
            self._ensure_column(connection, "users", "role", "TEXT NOT NULL DEFAULT 'user'")

    def _ensure_column(self, connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def create_user(self, *, username: str, display_name: str, password_hash: str, role: str = "user") -> dict[str, Any]:
        now = _to_iso(_utc_now())
        normalized_username = username.strip()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (email, display_name, password_hash, role, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (normalized_username, display_name.strip(), password_hash, role, now),
            )
            return {
                "id": cursor.lastrowid,
                "username": normalized_username,
                "email": normalized_username,
                "display_name": display_name.strip(),
                "role": role,
                "created_at": now,
            }

    def upsert_user(self, *, username: str, display_name: str, password_hash: str, role: str) -> dict[str, Any]:
        now = _to_iso(_utc_now())
        normalized_username = username.strip()
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT * FROM users WHERE email = ? COLLATE NOCASE",
                (normalized_username,),
            ).fetchone()
            if existing is None:
                cursor = connection.execute(
                    """
                    INSERT INTO users (email, display_name, password_hash, role, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (normalized_username, display_name.strip(), password_hash, role, now),
                )
                return {
                    "id": cursor.lastrowid,
                    "username": normalized_username,
                    "email": normalized_username,
                    "display_name": display_name.strip(),
                    "role": role,
                    "created_at": now,
                }
            connection.execute(
                """
                UPDATE users
                SET display_name = ?, password_hash = ?, role = ?
                WHERE id = ?
                """,
                (display_name.strip(), password_hash, role, existing["id"]),
            )
            return {
                "id": existing["id"],
                "username": existing["email"],
                "email": existing["email"],
                "display_name": display_name.strip(),
                "role": role,
                "created_at": existing["created_at"],
            }

    def get_user_by_username(self, username: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM users WHERE email = ? COLLATE NOCASE",
                (username.strip(),),
            ).fetchone()

    def get_user_by_id(self, user_id: int) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()

    def create_session(self, *, user_id: int, token: str) -> dict[str, str]:
        now = _utc_now()
        expires_at = now + timedelta(days=SESSION_TTL_DAYS)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (user_id, token_hash, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, hash_session_token(token), _to_iso(now), _to_iso(expires_at)),
            )
        return {"access_token": token, "token_type": "bearer", "expires_at": _to_iso(expires_at)}

    def get_user_by_session_token(self, token: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT users.id, users.email, users.display_name, users.role, users.created_at, sessions.expires_at
                FROM sessions
                JOIN users ON users.id = sessions.user_id
                WHERE sessions.token_hash = ?
                """,
                (hash_session_token(token),),
            ).fetchone()
        if row is None or _parse_iso(row["expires_at"]) <= _utc_now():
            return None
        return {
            "id": row["id"],
            "username": row["email"],
            "email": row["email"],
            "display_name": row["display_name"],
            "role": row["role"],
            "created_at": row["created_at"],
        }

    def delete_session(self, token: str) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM sessions WHERE token_hash = ?", (hash_session_token(token),))
    
    def cleanup_expired_sessions(self) -> int:
        """Eliminar sesiones caducadas. Retorna número de filas eliminadas."""
        now = _to_iso(_utc_now())
        with self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM sessions WHERE expires_at < ?",
                (now,),
            )
            return cursor.rowcount

    def save_analysis(self, *, user_id: int, result: dict[str, Any], created_at: datetime | None = None) -> dict[str, Any]:
        now = _to_iso(created_at or _utc_now())
        encrypted_payload = self.cipher.encrypt_json(result)
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO analyses (
                    user_id, view, status, status_label, model, pose_detected,
                    visible_landmarks_count, encrypted_payload, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    result.get("view"),
                    result.get("status"),
                    result.get("status_label"),
                    result.get("model"),
                    1 if result.get("pose_detected") else 0,
                    result.get("visible_landmarks_count"),
                    encrypted_payload,
                    now,
                ),
            )
        return {"id": cursor.lastrowid, "created_at": now}

    def list_analyses(self, *, user_id: int, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, view, status, status_label, model, pose_detected,
                       visible_landmarks_count, encrypted_payload, created_at
                FROM analyses
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            payload = self.cipher.decrypt_json(row["encrypted_payload"])
            payload["id"] = row["id"]
            payload["created_at"] = row["created_at"]
            items.append(payload)
        return items

    def delete_analyses(self, *, user_id: int) -> int:
        with self.connect() as connection:
            cursor = connection.execute("DELETE FROM analyses WHERE user_id = ?", (user_id,))
            return cursor.rowcount

    def analysis_summary(self, *, user_id: int) -> dict[str, Any]:
        now = _utc_now()
        cutoff_7 = _to_iso(now - timedelta(days=7))
        cutoff_30 = _to_iso(now - timedelta(days=30))
        
        with self.connect() as connection:
            # Conteos generales (sin desencriptar)
            total = connection.execute(
                "SELECT COUNT(*) AS count FROM analyses WHERE user_id = ?",
                (user_id,),
            ).fetchone()["count"]
            
            by_status = connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM analyses
                WHERE user_id = ?
                GROUP BY status
                ORDER BY count DESC
                """,
                (user_id,),
            ).fetchall()
            
            by_view = connection.execute(
                """
                SELECT view, COUNT(*) AS count
                FROM analyses
                WHERE user_id = ?
                GROUP BY view
                ORDER BY count DESC
                """,
                (user_id,),
            ).fetchall()
            
            latest = connection.execute(
                """
                SELECT created_at
                FROM analyses
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            
            # Filtrar por fechas con SQL (evita desencriptar)
            rows_7_days = connection.execute(
                """
                SELECT id, view, status, status_label, model, pose_detected,
                       visible_landmarks_count, encrypted_payload, created_at
                FROM analyses
                WHERE user_id = ? AND created_at >= ?
                ORDER BY created_at DESC
                LIMIT 100
                """,
                (user_id, cutoff_7),
            ).fetchall()
            
            rows_30_days = connection.execute(
                """
                SELECT id, view, status, status_label, model, pose_detected,
                       visible_landmarks_count, encrypted_payload, created_at
                FROM analyses
                WHERE user_id = ? AND created_at >= ?
                ORDER BY created_at DESC
                LIMIT 300
                """,
                (user_id, cutoff_30),
            ).fetchall()
            
            # Todos (para recomendaciones, máx 100)
            all_rows = connection.execute(
                """
                SELECT id, view, status, status_label, model, pose_detected,
                       visible_landmarks_count, encrypted_payload, created_at
                FROM analyses
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 100
                """,
                (user_id,),
            ).fetchall()
        
        # Desencriptar solo lo necesario
        def decrypt_rows(rows):
            items = []
            for row in rows:
                payload = self.cipher.decrypt_json(row["encrypted_payload"])
                payload["id"] = row["id"]
                payload["created_at"] = row["created_at"]
                items.append(payload)
            return items
        
        last_7_days = decrypt_rows(rows_7_days)
        last_30_days = decrypt_rows(rows_30_days)
        analyses = decrypt_rows(all_rows)
        
        return {
            "total": total,
            "by_status": {row["status"]: row["count"] for row in by_status},
            "by_view": {row["view"]: row["count"] for row in by_view},
            "latest_at": latest["created_at"] if latest else None,
            "periods": {
                "last_7_days": self._period_summary(last_7_days),
                "last_30_days": self._period_summary(last_30_days),
                "all_time": self._period_summary(analyses),
            },
            "timeline": self._build_weekly_timeline(last_30_days),
            "recommendations": self._build_recommendations(analyses),
        }

    def _period_summary(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"total": 0, "adequate_ratio": 0, "risk_count": 0, "improvable_count": 0}
        adequate = sum(1 for item in items if item.get("status") == "adequate")
        risk = sum(1 for item in items if item.get("status") == "risk")
        improvable = sum(1 for item in items if item.get("status") == "improvable")
        return {
            "total": len(items),
            "adequate_ratio": round(adequate / len(items), 3),
            "risk_count": risk,
            "improvable_count": improvable,
        }

    def _build_weekly_timeline(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            day = _parse_iso(item["created_at"]).date().isoformat()
            buckets.setdefault(day, []).append(item)
        return [
            {
                "date": day,
                "total": len(day_items),
                "adequate": sum(1 for item in day_items if item.get("status") == "adequate"),
                "improvable": sum(1 for item in day_items if item.get("status") == "improvable"),
                "risk": sum(1 for item in day_items if item.get("status") == "risk"),
            }
            for day, day_items in sorted(buckets.items())
        ]

    def _build_recommendations(self, items: list[dict[str, Any]]) -> list[str]:
        if not items:
            return ["Aun no hay datos suficientes. Realiza varias capturas durante la jornada para generar tendencias."]
        recent = items[: min(len(items), 30)]
        recommendations: list[str] = []
        risk_count = sum(1 for item in recent if item.get("status") == "risk")
        improvable_count = sum(1 for item in recent if item.get("status") == "improvable")
        if risk_count:
            recommendations.append(f"En las ultimas revisiones aparecen {risk_count} casos de riesgo. Conviene revisar altura de pantalla, apoyo lumbar y distancia al teclado.")
        elif improvable_count:
            recommendations.append("La mayoria de alertas recientes son mejorables, no criticas. Pequenos ajustes de encuadre y postura pueden estabilizar la tendencia.")
        else:
            recommendations.append("La tendencia reciente es estable. Mantén pausas breves y evita permanecer demasiado tiempo en la misma posicion.")

        metric_values: dict[str, list[float]] = {}
        for item in recent:
            for key, value in (item.get("metrics") or {}).items():
                if isinstance(value, (int, float)):
                    metric_values.setdefault(key, []).append(float(value))
        if metric_values.get("shoulder_tilt_deg"):
            average = sum(metric_values["shoulder_tilt_deg"]) / len(metric_values["shoulder_tilt_deg"])
            if average >= 4:
                recommendations.append("Se repite inclinacion de hombros. Comprueba si apoyas mas peso en un brazo o si la mesa queda descompensada.")
        if metric_values.get("neck_tilt_deg"):
            average = sum(metric_values["neck_tilt_deg"]) / len(metric_values["neck_tilt_deg"])
            if average >= 8:
                recommendations.append("Hay senal recurrente de inclinacion cervical. Ajusta la pantalla al centro y evita trabajar mirando de lado.")
        if metric_values.get("head_lateral_offset_ratio"):
            average = sum(metric_values["head_lateral_offset_ratio"]) / len(metric_values["head_lateral_offset_ratio"])
            if average >= 0.08:
                recommendations.append("La cabeza aparece desplazada respecto al eje corporal. Revisa que camara, pantalla y silla esten alineadas.")
        return recommendations[:4]
