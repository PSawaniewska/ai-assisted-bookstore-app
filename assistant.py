import json
import re
import time

import duckdb
from openai import BadRequestError, OpenAI, RateLimitError

BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "openai/gpt-oss-120b"
MAX_ROWS = 200
MAX_CHARS = 4000
MAX_STEPS = 8
MAX_HISTORY = 6
MAX_CLASSIFY_ROWS = 800
CLASSIFY_BATCH = 100

SYSTEM_PROMPT = """You are the data assistant of the Quill & Page bookstore dashboard. The current branch is {store}.
Never guess numbers: get every fact from the data with the run_sql tool (DuckDB SQL, SELECT only).
Dashboard facts, computed by the app and correct (your answers must agree with them): {facts}.
The SQL tables hold only the current branch. Compare branches with the all-branches summary above;
for other details about another branch, offer to switch to it.

Tables:
- books(id, title, authors, genre, publisher, year). authors is a comma-separated, alphabetically sorted list; one distinct value = one author set.
- users(id, person_id, name, address, phone, email). One real person can have several user ids (aliases, changed address, phone or e-mail); person_id groups them. Questions about users, customers or buyers are about real people: group by person_id (e.g. orders per person, spending per person). When you name a customer, give their name and all their user ids, e.g. "Cyril Hegmann (user ids: [48403, 48904, 50056])" (list(DISTINCT id ORDER BY id), list(DISTINCT name)); if the person used several names, mention all of them. Never mention person_id or "person N", it is internal.
- orders(id, user_id, book_id, quantity, unit_price, paid_price, timestamp, date). Prices are in USD, already converted and deduplicated. Revenue = SUM(paid_price).

For books with people's names in the title, call show_books_with_names: it finds and checks them and filters the Explorer itself.

For filters about meaning (animals, places, colours, female names, ...), call filter_by_meaning: it checks every row and filters the Explorer itself.
Never build such filters from LIKE keyword lists.
Filters about form (letters, digits, number of words, length) are exact, so use show_table with SQL, e.g.
len(regexp_extract_all(title, '[0-9]+')) >= 2 for at least two numbers, len(string_split(title, ' ')) = 1 for one-word titles.
If a request can be read in more than one way, use the most likely reading and say which one you used.

The dashboard has an Explorer table. For filters that SQL can express exactly (e-mail starts with s, genre is Fantasy, year before 2000) call show_table
with a query that returns rows of that table. If it is unclear whether to keep or remove the matching rows, keep them and say so.
After filtering, do not list the rows in your answer; say what the filter is, how many rows are shown, and always end with: "You can see them in the Explorer tab."
Call reset_table to show all rows again,
and select_store to switch branches ({stores}).

Always answer in the language of the user's last message (a question in Polish gets a Polish answer). Keep it brief. Dates as YYYY-MM-DD, money as $1,234.56. If a question cannot be answered from these tables, say so."""

SQL_TABLES = ("books", "users", "orders")
NAME_CANDIDATES = """
    WITH words AS (SELECT id, title, unnest(regexp_split_to_array(title, '[^\\p{L}]+')) AS word FROM books)
    SELECT DISTINCT id, title FROM words JOIN person_names ON name = word ORDER BY id
"""
NAME_CRITERION = (
    "the title names a specific person or character (e.g. Lawrence of Arabia, Donnie Darko, Max Payne, Life of Brian). "
    "Words that are names only by coincidence do not count: Will as a verb, Sun, Star, King, Green, Princess as a title."
)

CLASSIFY_PROMPT = """For each item decide whether it matches this criterion: {criterion}
Items are [id, text]. Check every item.
Reply only with the ids of the matching items, separated by commas, or with the word NONE if no item matches."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Run a DuckDB SELECT query. Returns the total row count and the first rows (long results are cut).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_table",
            "description": "Filter the Explorer table with an exact SQL condition. Returns the row count and a sample to check.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "enum": ["books", "users", "orders"]},
                    "query": {"type": "string", "description": "SELECT query returning rows of that table"},
                    "title": {"type": "string", "description": "Short description of the filter for the user"},
                },
                "required": ["table", "query", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by_meaning",
            "description": "Filter the Explorer to rows whose text matches a criterion that needs understanding, e.g. books with an animal in the title.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "enum": ["books", "users", "orders"]},
                    "column": {"type": "string", "description": "Text column to judge, e.g. title or name"},
                    "criterion": {"type": "string", "description": "What a matching value looks like"},
                    "title": {"type": "string", "description": "Short description of the filter for the user"},
                    "where": {"type": ["string", "null"], "description": f"SQL condition, used only when the table has more than {MAX_CLASSIFY_ROWS} rows"},
                },
                "required": ["table", "column", "criterion", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_books_with_names",
            "description": "Filter the Explorer to books whose titles contain people's names.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_table",
            "description": "Remove the filter from the Explorer table.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_store",
            "description": "Switch the dashboard to another branch.",
            "parameters": {
                "type": "object",
                "properties": {"store": {"type": "string"}},
                "required": ["store"],
            },
        },
    },
]


def run_sql(tables, query, names=SQL_TABLES):
    query = query.strip().rstrip(";")
    # one read-only statement; the data lives in memory, so nothing outside the app can be reached or changed
    if ";" in query or not re.match(r"(?is)(select|with)\b", query):
        raise ValueError("Only a single SELECT query is allowed.")

    con = duckdb.connect(config={"enable_external_access": False})
    for name in names:
        con.register(name, tables[name])
    return con.execute(query).df()


def rows_as_json(df):
    # the free plan has a small token budget: column names are sent once and long results are cut
    table = json.loads(df.head(MAX_ROWS).to_json(orient="split", index=False, date_format="iso"))
    rows = table["data"]
    while len(json.dumps(rows)) > MAX_CHARS:
        rows = rows[: len(rows) // 2]
    return {"row_count": len(df), "rows_returned": len(rows), "columns": table["columns"], "rows": rows}


def classify(client, rows, criterion):
    # a separate, short request per batch makes the model really check each item
    kept = []
    for start in range(0, len(rows), CLASSIFY_BATCH):
        batch = rows[start:start + CLASSIFY_BATCH]
        messages = [
            {"role": "system", "content": CLASSIFY_PROMPT.format(criterion=criterion)},
            {"role": "user", "content": json.dumps(batch, ensure_ascii=False)},
        ]
        for attempt in range(2):
            text = complete(client, messages, tools=None, effort="medium").choices[0].message.content or ""
            # an empty reply means the model failed, not that nothing matched, so it is asked again
            if text.strip():
                break
        if not text.strip():
            raise RuntimeError("The model returned no answer while checking the rows. Try again.")
        # every number in the reply is read, and only ids from this batch are kept, so invented ids cannot get through
        found = {int(number) for number in re.findall(r"\d+", text)}
        matches = [row[0] for row in batch if row[0] in found]
        print(f"Classify rows {start}-{start + len(batch)}: {len(matches)} matched, reply: {text[:120]!r}")
        kept += matches
    return kept


def show_matching(client, rows, criterion, table, title, actions):
    # the judging and the filter both happen in code, so the model cannot skip the check
    kept = classify(client, rows, criterion)
    query = f"SELECT * FROM {table} WHERE id IN ({', '.join(map(str, kept)) or 'NULL'})"
    actions.append({"view": {"table": table, "query": query, "title": title}})
    return {"rows_checked": len(rows), "shown_rows": len(kept)}


def call_tool(client, name, args, tables, stores, actions):
    if name == "run_sql":
        return rows_as_json(run_sql(tables, args["query"]))
    if name == "filter_by_meaning":
        table, column = args["table"], args["column"]
        if column not in tables[table].columns:
            raise ValueError(f"Unknown column. Columns of {table}: {', '.join(tables[table].columns)}")
        candidates = run_sql(tables, f'SELECT id, "{column}" FROM {table}')
        # the model tends to pre-filter with keywords and lose matches, so where is only used for big tables
        if len(candidates) > MAX_CLASSIFY_ROWS and args.get("where"):
            candidates = run_sql(tables, f'SELECT id, "{column}" FROM {table} WHERE {args["where"]}')
        if len(candidates) > MAX_CLASSIFY_ROWS:
            raise ValueError(f"Too many rows ({len(candidates)}). Add a where condition to keep at most {MAX_CLASSIFY_ROWS}.")
        rows = [[int(row[0]), str(row[1])] for row in candidates.itertuples(index=False)]
        return show_matching(client, rows, args["criterion"], table, args["title"], actions)
    if name == "show_table":
        table = args["table"]
        shown = run_sql(tables, args["query"])
        if list(shown.columns) != list(tables[table].columns):
            raise ValueError(f"The query must return whole rows: SELECT * FROM {table} WHERE ...")
        actions.append({"view": {"table": table, "query": args["query"], "title": args["title"]}})
        # a few rows go back to the model, so it can see whether the filter does what was asked
        return {"shown_rows": len(shown), "sample": rows_as_json(shown.head(5))["rows"]}
    if name == "show_books_with_names":
        # every step runs in code, so the model cannot skip the check and show raw candidates
        candidates = run_sql(tables, NAME_CANDIDATES, names=tables.keys())
        rows = [[int(row.id), row.title] for row in candidates.itertuples(index=False)]
        return show_matching(client, rows, NAME_CRITERION, "books", "Books with people's names in the title", actions)
    if name == "reset_table":
        actions.append({"view": None})
        return {"status": "ok"}
    if name == "select_store":
        if args["store"] not in stores:
            raise ValueError(f"Unknown branch. Available: {', '.join(stores)}")
        actions.append({"store": args["store"]})
        return {"status": "ok"}
    raise ValueError(f"Unknown tool {name}")


def complete(client, messages, tools=TOOLS, effort="low", json_mode=False):
    options = {"tools": tools} if tools else {}
    if json_mode:
        options["response_format"] = {"type": "json_object"}
    for attempt in range(5):
        try:
            return client.chat.completions.create(
                model=MODEL, messages=messages, reasoning_effort=effort, **options
            )
        except RateLimitError as error:
            # the per-minute limit clears quickly, so wait; the daily limit does not, so give up at once
            if "per day" in str(error) or attempt == 4:
                raise
            time.sleep(15)
        except BadRequestError as error:
            # the model sometimes writes a broken tool call; asking again usually fixes it
            if "tool call" not in str(error) or attempt == 4:
                raise


def ask(api_key, history, tables, store, stores, facts):
    # retries are handled in complete(), so the client does not add its own waits
    client = OpenAI(api_key=api_key, base_url=BASE_URL, max_retries=0)
    system = SYSTEM_PROMPT.format(store=store, stores=", ".join(stores), facts=facts)
    # only the last messages are sent, to stay within the free token budget
    messages = [{"role": "system", "content": system}] + history[-MAX_HISTORY:]
    actions = []

    for _ in range(MAX_STEPS):
        message = complete(client, messages).choices[0].message
        if not message.tool_calls:
            return message.content, actions

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [call.model_dump() for call in message.tool_calls],
        })
        for call in message.tool_calls:
            # printed in the terminal, so wrong answers can be traced back to the query behind them
            print(f"Tool {call.function.name}: {call.function.arguments}")
            try:
                result = call_tool(client, call.function.name, json.loads(call.function.arguments), tables, stores, actions)
            except Exception as error:
                # errors go back to the model, so it can fix its own query
                result = {"error": str(error)}
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)})

    return "Sorry, I could not finish this one. Try asking in a simpler way.", actions
