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
MAX_CLASSIFY_ROWS = 300
CLASSIFY_BATCH = 50

SYSTEM_PROMPT = """You are a data assistant for a bookstore dashboard. The current store is {store}.
Never guess numbers: get every fact from the data with the run_sql tool (DuckDB SQL, SELECT only).

Tables:
- books(id, title, authors, genre, publisher, year). authors is a comma-separated, alphabetically sorted list; one distinct value = one author set.
- users(id, person_id, name, address, phone, email). One real person can have several user ids (aliases, changed address, phone or e-mail); person_id groups them. Questions about users, customers or buyers are about real people: group by person_id (e.g. orders per person, spending per person). When you name a customer, give their name and all their user ids, e.g. "Cyril Hegmann (user ids: [48403, 48904, 50056])" (list(DISTINCT id ORDER BY id), list(DISTINCT name)); if the person used several names, mention all of them. Never mention person_id or "person N", it is internal.
- orders(id, user_id, book_id, quantity, unit_price, paid_price, timestamp, date). Prices are in USD, already converted and deduplicated. Revenue = SUM(paid_price).
- person_names(name): first and last names of all customers.

To find people's names in titles: get candidates with
WITH words AS (SELECT id, title, unnest(regexp_split_to_array(title, '[^\\p{{L}}]+')) AS word FROM books) SELECT DISTINCT id, title FROM words JOIN person_names ON name = word
and pass that query to classify_rows with the criterion: "the title names a specific person or character (e.g. Lawrence of Arabia, Donnie Darko, Max Payne, Life of Brian).
Words that are names only by coincidence do not count: Will as a verb, Sun, Star, King, Green, Princess as a title"

For any filter that needs understanding of text (names, animals, places, ...), call classify_rows with a query returning id and the text
(narrow it with SQL first when you can), then call show_table with WHERE id IN (...) using exactly the kept_ids it returns.

The dashboard has an Explorer table. When the user asks to show, filter, select or sort books, users or orders, call show_table
with a query that returns rows of that table. If it is unclear whether to keep or remove the matching rows, keep them and say so.
After show_table, do not list the rows in your answer (the user sees them in the Explorer); just say what the filter is, how many rows are shown and that they are in the Explorer tab.
Call reset_table to show all rows again,
and select_store to switch stores ({stores}).

Always answer in the language of the user's last message (a question in Polish gets a Polish answer). Keep it brief. Dates as YYYY-MM-DD, money as $1,234.56. If a question cannot be answered from these tables, say so."""

CLASSIFY_PROMPT = """For each item decide whether it matches this criterion: {criterion}
Items are [id, text]. Check every item. Reply only with JSON: {{"ids": [ids of the matching items]}}"""

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
            "description": "Show the result of a query in the dashboard Explorer table.",
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
            "name": "classify_rows",
            "description": "Check rows against a criterion that needs understanding of text. Returns the ids that match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SELECT returning id as the first column and the text to judge as the second"},
                    "criterion": {"type": "string"},
                },
                "required": ["query", "criterion"],
            },
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
            "description": "Switch the dashboard to another store.",
            "parameters": {
                "type": "object",
                "properties": {"store": {"type": "string"}},
                "required": ["store"],
            },
        },
    },
]


def run_sql(tables, query):
    query = query.strip().rstrip(";")
    # one read-only statement; the data lives in memory, so nothing outside the app can be reached or changed
    if ";" in query or not re.match(r"(?is)(select|with)\b", query):
        raise ValueError("Only a single SELECT query is allowed.")

    con = duckdb.connect(config={"enable_external_access": False})
    for name, df in tables.items():
        con.register(name, df)
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
        reply = complete(client, messages, tools=None, effort="medium", json_mode=True)
        ids = json.loads(reply.choices[0].message.content).get("ids", [])
        # only ids that were really in the batch are kept, so invented ids cannot get through
        valid = {row[0] for row in batch}
        kept += [item for item in ids if item in valid]
    return kept


def call_tool(client, name, args, tables, stores, actions):
    if name == "run_sql":
        return rows_as_json(run_sql(tables, args["query"]))
    if name == "classify_rows":
        df = run_sql(tables, args["query"])
        if df.shape[1] < 2:
            raise ValueError("The query must return id and the text to judge.")
        if len(df) > MAX_CLASSIFY_ROWS:
            raise ValueError(f"Too many rows ({len(df)}). Narrow the query to at most {MAX_CLASSIFY_ROWS}.")
        rows = [[int(row[0]), str(row[1])] for row in df.itertuples(index=False)]
        kept = classify(client, rows, args["criterion"])
        return {"checked": len(rows), "kept": len(kept), "kept_ids": kept}
    if name == "show_table":
        # run it first, so a broken query is reported back to the model instead of breaking the dashboard
        df = run_sql(tables, args["query"])
        actions.append({"view": {"table": args["table"], "query": args["query"], "title": args["title"]}})
        return {"shown_rows": len(df)}
    if name == "reset_table":
        actions.append({"view": None})
        return {"status": "ok"}
    if name == "select_store":
        if args["store"] not in stores:
            raise ValueError(f"Unknown store. Available: {', '.join(stores)}")
        actions.append({"store": args["store"]})
        return {"status": "ok"}
    raise ValueError(f"Unknown tool {name}")


def complete(client, messages, tools=TOOLS, effort="low", json_mode=False):
    options = {"tools": tools} if tools else {}
    if json_mode:
        options["response_format"] = {"type": "json_object"}
    for attempt in range(3):
        try:
            return client.chat.completions.create(
                model=MODEL, messages=messages, reasoning_effort=effort, **options
            )
        except RateLimitError:
            # the free plan allows only a few thousand tokens per minute, so wait instead of failing
            if attempt == 2:
                raise
            time.sleep(20)
        except BadRequestError as error:
            # the model sometimes writes a broken tool call; asking again usually fixes it
            if "tool call" not in str(error) or attempt == 2:
                raise


def ask(api_key, history, tables, store, stores):
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    system = SYSTEM_PROMPT.format(store=store, stores=", ".join(stores))
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
            try:
                result = call_tool(client, call.function.name, json.loads(call.function.arguments), tables, stores, actions)
            except Exception as error:
                # errors go back to the model, so it can fix its own query
                result = {"error": str(error)}
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)})

    return "Sorry, I could not finish this one. Try asking in a simpler way.", actions
