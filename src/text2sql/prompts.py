"""Prompt templates. Bump PROMPT_VERSION whenever these change so MLflow runs
stay comparable."""

PROMPT_VERSION = "v3"
CANNOT_ANSWER = "CANNOT_ANSWER"

# Each rule below was added to fix a specific failure mode found in evaluation.
# Bump PROMPT_VERSION on any change here so MLflow runs stay comparable.
SYSTEM_TEMPLATE = """You are an expert SQLite data analyst. Given a database schema and a \
question, write one SQLite SELECT query that answers the question.

Rules:
- Output ONLY the SQL query. No explanation, no markdown fences.
- Use ONLY table and column names copied exactly from the CREATE TABLE statements below. \
Before writing the query, check which table each column belongs to.
- SELECT exactly the columns the question asks for — no extra columns.
- When filtering on a string, copy the value EXACTLY as it appears in the sample rows or \
"exact stored values" hints, including letter case — never guess capitalization.
- Write a single SELECT statement (JOINs, subqueries, GROUP BY, UNION are fine).
- Never write INSERT, UPDATE, DELETE, DROP or anything that modifies data.
- Output CANNOT_ANSWER only when the question is about data clearly outside this schema \
(weather, news, or entities with no matching table). If the question relates to any table \
in the schema, always attempt a query.

{examples}
Now the real schema:

{schema}"""

# Fixed few-shot examples (a toy schema) used when dynamic few-shot is off.
DEFAULT_EXAMPLES = """Examples (these use a different toy schema):

Schema:
CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, salary REAL, \
department_id INTEGER REFERENCES departments(id));

Question: How many employees are there?
SQL: SELECT COUNT(*) FROM employees;

Question: What is the name of the department with the highest average salary?
SQL: SELECT d.name FROM departments d JOIN employees e ON e.department_id = d.id \
GROUP BY d.id ORDER BY AVG(e.salary) DESC LIMIT 1;

Question: What is the name of the oldest employee?
SQL: SELECT name FROM employees ORDER BY birth_date ASC LIMIT 1;

Question: What is the weather today?
SQL: CANNOT_ANSWER
"""


def format_examples(pairs: list[tuple[str, str]]) -> str:
    """Format retrieved (question, SQL) pairs as few-shot examples. These come from
    OTHER databases, so the model is told to copy structure, not exact names."""
    lines = [
        "Examples of similar questions solved on OTHER databases "
        "(use them for query structure, not exact table/column names):",
        "",
    ]
    for q, sql in pairs:
        lines.append(f"Question: {q}")
        lines.append(f"SQL: {sql}")
        lines.append("")
    return "\n".join(lines)


def build_messages(
    question: str, schema: str, examples: list[tuple[str, str]] | None = None
) -> list[dict]:
    ex_text = format_examples(examples) if examples else DEFAULT_EXAMPLES
    return [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(examples=ex_text, schema=schema)},
        {"role": "user", "content": f"Question: {question}\nSQL:"},
    ]


def error_feedback(error: str, catalog: str | None = None) -> str:
    hint = f"\nThe only valid tables and columns are:\n{catalog}\n" if catalog else "\n"
    return (
        f"That query failed with this error:\n{error}\n{hint}"
        "Write a corrected SQLite SELECT query for the original question. Output only the SQL."
    )
