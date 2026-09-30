"""Database access: settings from .env, read-only UTC SQLAlchemy engine, optional SSH tunnel."""

import os
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.engine import URL, Engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _make_engine(host: str, port: int) -> Engine:
    url = URL.create(
        "mysql+pymysql",
        username=env("DB_USER"),
        password=env("DB_PASSWORD"),
        host=host,
        port=port,
        database=env("DB_NAME"),
        query={"charset": env("DB_CHARSET", "utf8mb4")},
    )
    connect_args = {"connect_timeout": 15}
    if env("DB_SSL_CA"):
        connect_args["ssl"] = {"ca": os.path.expanduser(env("DB_SSL_CA"))}

    engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)

    @event.listens_for(engine, "connect")
    def _session_setup(dbapi_conn, _record):
        # Antz stores server time in UTC; read it as-is. Refuse writes even if the user has grants.
        with dbapi_conn.cursor() as cur:
            cur.execute("SET time_zone = '+00:00'")
            cur.execute("SET SESSION TRANSACTION READ ONLY")

    return engine


@contextmanager
def get_engine():
    """Yield an engine, tunnelling over SSH when SSH_HOST is set."""
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
