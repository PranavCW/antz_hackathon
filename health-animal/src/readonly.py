"""Read-only guard for the Dev DB: every SQL statement is checked before it reaches the server.

Three layers, any one of which blocks a write:
  1. This guard — only SELECT / SHOW / DESCRIBE / EXPLAIN / WITH pass; write keywords, locking reads,
     SELECT … INTO (incl. OUTFILE, which READ ONLY sessions do NOT block), executable /*! */ comments and
     multi-statements are rejected before sending.
  2. The session is set to READ ONLY on connect and verified (see db.py).
  3. The DB user's own grants (ask for a read-only user; the current one can write).
"""

import re

import pymysql


class ReadOnlyViolation(RuntimeError):
    """Raised when code tries to send a statement that could change the database."""


ALLOWED_FIRST_WORDS = {"SELECT", "SHOW", "DESCRIBE", "DESC", "EXPLAIN", "WITH"}

# Session statements SQLAlchemy issues itself on connect; matched exactly, no other SET is allowed.
ALLOWED_EXACT = [re.compile(r"SET\s+NAMES\s+\w+(\s+COLLATE\s+\w+)?", re.IGNORECASE)]

# Checked anywhere in SELECT / WITH / EXPLAIN (after strings and quoted names are masked).
# SHOW is exempt: it can't modify anything, and "SHOW CREATE TABLE" / "SHOW GRANTS" contain these words.
FORBIDDEN_ANYWHERE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|REPLACE\s+INTO|MERGE|DROP|ALTER|CREATE|TRUNCATE|RENAME|GRANT|REVOKE|"
    r"CALL|LOAD|HANDLER|LOCK|UNLOCK|SET|DO|IMPORT|INSTALL|UNINSTALL|FLUSH|KILL|RESET|PURGE)\b"
    r"|\bINTO\s+(OUTFILE|DUMPFILE|@)"
    r"|\bFOR\s+(UPDATE|SHARE)\b"
    r"|\bLOCK\s+IN\s+SHARE\s+MODE\b"
    r"|\b(GET_LOCK|RELEASE_LOCK|RELEASE_ALL_LOCKS|SLEEP|BENCHMARK)\s*\(",
    re.IGNORECASE,
)

# One alternation scanned left to right, so whichever of comment / string starts first wins: an apostrophe
# inside a comment can't open a fake string, and "--" inside a string can't open a fake comment.
# MySQL rules: "--" is a comment only when followed by whitespace/control char; /*! … */ and /*M! … */ are
# *executed* (handled in check()); optimizer hints /*+ … */ are plain comments for our purposes.
_TOKENS = re.compile(
    r"(?P<exec>/\*M?!)"
    r"|(?P<comment>/\*.*?\*/|--[\x00-\x20][^\n]*|#[^\n]*)"
    r"|(?P<quoted>'(?:[^'\\]|\\.|'')*'|\"(?:[^\"\\]|\\.|\"\")*\"|`(?:[^`]|``)*`)",
    re.DOTALL,
)


class _ExecutableComment(Exception):
    pass


def _mask(sql: str) -> str:
    """Blank out comments, string literals and quoted identifiers in a single pass.

    Unterminated strings/comments are left as-is, so their contents are still checked as SQL.
    """
    def repl(m):
        if m.group("exec"):
            raise _ExecutableComment
        return " " if m.group("comment") else "''"

    return _TOKENS.sub(repl, sql)


def check(sql) -> None:
    """Raise ReadOnlyViolation unless `sql` is a single read-only statement."""
    if isinstance(sql, (bytes, bytearray)):
        sql = sql.decode("utf-8", errors="replace")
    try:
        masked = _mask(sql).strip().rstrip(";").strip()
    except _ExecutableComment:
        raise ReadOnlyViolation(f"MySQL executable comments (/*! … */) are not allowed: {sql[:120]!r}") from None

    if any(p.fullmatch(masked) for p in ALLOWED_EXACT):
        return

    if ";" in masked:
        raise ReadOnlyViolation(f"Multiple statements are not allowed: {sql[:120]!r}")

    first = re.match(r"[\s(]*([A-Za-z]+)", masked)
    first_word = first.group(1).upper() if first else ""
    if first_word not in ALLOWED_FIRST_WORDS:
        raise ReadOnlyViolation(f"Only read queries are allowed, got {first_word or 'empty'}: {sql[:120]!r}")

    if first_word != "SHOW":
        hit = FORBIDDEN_ANYWHERE.search(masked)
        if hit:
            raise ReadOnlyViolation(f"Forbidden in read-only mode ({hit.group(0).strip()}): {sql[:120]!r}")


class ReadOnlyConnection(pymysql.connections.Connection):
    """pymysql connection whose query() — the single path every cursor uses — enforces check()."""

    def query(self, sql, unbuffered=False):
        check(sql)
        return super().query(sql, unbuffered)

    def _setup_query(self, sql):
        """Session setup statements issued by db.py itself (SET time_zone, SET … READ ONLY)."""
        return super().query(sql)
