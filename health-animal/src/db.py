"""Database access: settings from .env, read-only UTC SQLAlchemy engine, optional SSH tunnel.

The Dev DB must only ever be read. Every connection is a ReadOnlyConnection (src/readonly.py), which rejects
any non-read statement before sending it, and the session is set to READ ONLY and verified on connect.
"""

import os
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

from src.readonly import ReadOnlyConnection, ReadOnlyViolation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _make_engine(host: str, port: int) -> Engine:
    connect_kwargs = {
        "host": host,
        "port": port,
        "user": env("DB_USER"),
        "password": env("DB_PASSWORD"),
        "database": env("DB_NAME"),
        "charset": env("DB_CHARSET", "utf8mb4"),
        "connect_timeout": 15,
        "local_infile": False,  # no LOAD DATA LOCAL
        "client_flag": 0,       # in particular no CLIENT.MULTI_STATEMENTS
    }
    if env("DB_SSL_CA"):
        connect_kwargs["ssl"] = {"ca": os.path.expanduser(env("DB_SSL_CA"))}

    # creator= makes SQLAlchemy use our guarded connection class for every pooled connection
    engine = create_engine(
        "mysql+pymysql://", creator=lambda: ReadOnlyConnection(**connect_kwargs), pool_pre_ping=True
    )

    @event.listens_for(engine, "connect")
    def _session_setup(dbapi_conn, _record):
        # Antz stores server time in UTC; read it as-is. Server-side read-only as a second layer.
        dbapi_conn._setup_query("SET time_zone = '+00:00'")
        dbapi_conn._setup_query("SET SESSION TRANSACTION READ ONLY")
        with dbapi_conn.cursor() as cur:
            cur.execute("SELECT @@SESSION.transaction_read_only, @@SESSION.sql_mode")
            read_only, sql_mode = cur.fetchone()
        if read_only != 1:
            dbapi_conn.close()
            raise ReadOnlyViolation("Server did not accept SESSION TRANSACTION READ ONLY; refusing to continue")
        if "NO_BACKSLASH_ESCAPES" in (sql_mode or "").upper():
            # The guard parses strings with backslash escapes; under this mode it could misread them.
            dbapi_conn.close()
            raise ReadOnlyViolation("sql_mode NO_BACKSLASH_ESCAPES is set; the read-only guard can't parse safely")

    return engine


@contextmanager
def get_engine():
    """Yield a read-only engine, tunnelling over SSH when SSH_HOST is set."""
    missing = [k for k in ("DB_HOST", "DB_NAME", "DB_USER") if not env(k)]
    if missing:
        raise RuntimeError(f"Missing in .env: {', '.join(missing)}")

    if not env("SSH_HOST"):
        engine = _make_engine(env("DB_HOST"), int(env("DB_PORT", "3306")))
        try:
            yield engine
        finally:
            engine.dispose()
        return

    from sshtunnel import SSHTunnelForwarder

    with SSHTunnelForwarder(
        (env("SSH_HOST"), int(env("SSH_PORT", "22"))),
        ssh_username=env("SSH_USER"),
        ssh_pkey=os.path.expanduser(env("SSH_KEY_PATH", "~/.ssh/id_rsa")),
        remote_bind_address=(env("DB_HOST"), int(env("DB_PORT", "3306"))),
        local_bind_address=("127.0.0.1", int(env("SSH_LOCAL_PORT", "13306"))),
    ) as tunnel:
        engine = _make_engine("127.0.0.1", tunnel.local_bind_port)
        try:
            yield engine
        finally:
            engine.dispose()
