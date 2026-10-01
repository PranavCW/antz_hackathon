"""The read-only guard must let every read the pipeline needs through and block everything else."""

import pytest

from src.readonly import ReadOnlyViolation, check

ALLOWED = [
    "SELECT * FROM antz_animals WHERE zoo_id = 11",
    "  select animal_id from antz_animals",
    "(SELECT 1) UNION (SELECT 2)",
    "SELECT 1;",
    "/* audit */ SELECT 1",
    "WITH w AS (SELECT * FROM antz_animal_assessments) SELECT * FROM w",
    "SHOW CREATE TABLE antz_animals",
    "SHOW GRANTS",
    "SHOW TABLES",
    "DESCRIBE antz_food_wastage",
    "EXPLAIN SELECT * FROM antz_observation",
    # write words inside strings / quoted names / other identifiers are harmless
    "SELECT * FROM antz_observation WHERE observation_name LIKE '%delete this; drop table%'",
    "SELECT `update`, `set` FROM t",
    "SELECT created_at, updated_at, UPDATE_TIME, CREATE_TIME FROM information_schema.TABLES",
    "SELECT REPLACE(assessment_value, ',', '.') FROM antz_animal_assessments",
    "SELECT * FROM t ORDER BY created_at DESC",
    "SELECT 'it''s fine', 'back\\'slash' FROM t",
    b"SELECT 1",
    "SET NAMES utf8mb4",  # issued by SQLAlchemy on connect
    "SELECT 1 /* don't worry */ FROM t",
    "SELECT 1 -- it's a comment\nFROM t",
    "SELECT 5--1 FROM t",  # "--" without whitespace is minus-minus, not a comment
    "SELECT '/*! not a comment */', '-- nor this' FROM t",
    "SELECT /*+ MAX_EXECUTION_TIME(1000) */ * FROM t",
]

BLOCKED = [
    "UPDATE antz_animals SET is_alive = 0",
    "DELETE FROM antz_animals",
    "INSERT INTO antz_animals (animal_id) VALUES (1)",
    "REPLACE INTO t VALUES (1)",
    "DROP TABLE antz_animals",
    "TRUNCATE antz_food_wastage",
    "ALTER TABLE t ADD c INT",
    "CREATE TABLE t (id INT)",
    "SET SESSION TRANSACTION READ WRITE",
    "SET time_zone = '+05:30'",
    "SET NAMES utf8mb4; DELETE FROM t",
    "SET NAMES utf8mb4, SESSION TRANSACTION READ WRITE",
    "CALL some_proc()",
    "LOAD DATA INFILE 'x' INTO TABLE t",
    "LOCK TABLES t READ",
    "GRANT ALL ON *.* TO x",
    "START TRANSACTION",
    "COMMIT",
    "SELECT 1; DELETE FROM antz_animals",
    "SELECT * FROM t FOR UPDATE",
    "SELECT * FROM t FOR SHARE",
    "SELECT * FROM t LOCK IN SHARE MODE",
    "SELECT * INTO OUTFILE '/tmp/x' FROM t",
    "SELECT 1 INTO @v",
    "WITH d AS (SELECT 1) DELETE FROM t",
    "EXPLAIN ANALYZE UPDATE t SET a = 1",
    "SELECT GET_LOCK('x', 10)",
    "SELECT SLEEP(100)",
    "/* SELECT */ DELETE FROM t",
    "-- SELECT\nDELETE FROM t",
    "",
    # comment/string confusion (review findings): apostrophes in comments must not open a fake string
    "SELECT 1 /* don't */ INTO OUTFILE '/tmp/x'",
    "SELECT 1 -- it's\n INTO OUTFILE '/tmp/x'",
    "SELECT 1 # it's\n FOR UPDATE",
    "SELECT 1--1 FOR UPDATE",
    "SELECT 1 /* ' */ ; DELETE FROM t -- '",
    # MySQL executes the body of /*! … */ and /*M! … */
    "SELECT 1 /*! INTO OUTFILE '/tmp/x' */",
    "SELECT 1 /*!50000 FOR UPDATE */",
    "SELECT 1 /*M! FOR UPDATE */",
    # unterminated string/comment: contents still checked
    "SELECT 1 FOR UPDATE /* unterminated",
    "SELECT ' FROM t; DELETE FROM t",
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_reads_allowed(sql):
    check(sql)


@pytest.mark.parametrize("sql", BLOCKED)
def test_writes_blocked(sql):
    with pytest.raises(ReadOnlyViolation):
        check(sql)
