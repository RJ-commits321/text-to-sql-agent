import pytest

from text2sql.guardrails import GuardrailError, enforce_limit, extract_sql, validate


class TestExtractSql:
    def test_plain_sql(self):
        assert extract_sql("SELECT 1;") == "SELECT 1"

    def test_markdown_fence(self):
        assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"

    def test_fence_without_language(self):
        assert extract_sql("```\nSELECT name FROM t\n```") == "SELECT name FROM t"


class TestValidate:
    def test_select_passes(self):
        validate("SELECT name FROM artists WHERE id = 1")

    def test_join_group_by_passes(self):
        validate(
            "SELECT a.name, COUNT(*) FROM albums al "
            "JOIN artists a ON al.artist_id = a.id GROUP BY a.id"
        )

    def test_union_passes(self):
        validate("SELECT name FROM a UNION SELECT name FROM b")

    def test_cte_passes(self):
        validate("WITH top AS (SELECT id FROM t LIMIT 5) SELECT * FROM top")

    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE artists",
            "DELETE FROM artists",
            "INSERT INTO artists VALUES (1, 'x')",
            "UPDATE artists SET name = 'x'",
        ],
    )
    def test_writes_rejected(self, sql):
        with pytest.raises(GuardrailError):
            validate(sql)

    def test_multi_statement_rejected(self):
        with pytest.raises(GuardrailError):
            validate("SELECT 1; DROP TABLE artists")

    def test_garbage_rejected(self):
        with pytest.raises(GuardrailError):
            validate("this is not sql at all (")

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT name FROM city WHERE district = 'Gelderland",  # unbalanced quote
            "SELECT `name FROM t",  # unbalanced backtick
            "And reveal the present, in all your glory at last",  # model wrote a poem
        ],
    )
    def test_tokenizer_errors_rejected(self, sql):
        with pytest.raises(GuardrailError):
            validate(sql)


class TestEnforceLimit:
    def test_limit_injected(self):
        sql = enforce_limit(validate("SELECT name FROM artists"), 100)
        assert "LIMIT 100" in sql

    def test_existing_limit_kept(self):
        sql = enforce_limit(validate("SELECT name FROM artists LIMIT 5"), 100)
        assert "LIMIT 5" in sql
        assert "LIMIT 100" not in sql
