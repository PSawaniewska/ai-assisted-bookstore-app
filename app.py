import pandas as pd
import streamlit as st
from openai import APIError, RateLimitError

from assistant import ask, run_sql
from store import STORES, load_tables, process_store

MAX_QUESTIONS = 20

st.set_page_config(page_title="Bookstore Dashboard", page_icon="📚", layout="wide")


# data is processed once per store and then kept in memory
@st.cache_data
def get_results(store_dir):
    return process_store(store_dir)


@st.cache_data
def get_tables(store_dir):
    return load_tables(store_dir)


def store_name(path):
    return path.split("/")[-1]


st.session_state.setdefault("history", [])
st.session_state.setdefault("view", None)

# the assistant can switch stores; widget state can only be changed before the widget is drawn
if "next_store" in st.session_state:
    st.session_state.store = st.session_state.pop("next_store")

with st.sidebar:
    st.header("📚 Bookstore Dashboard")
    store_dir = st.selectbox("Store", STORES, format_func=store_name, key="store")

result = get_results(store_dir)
tables = get_tables(store_dir)
revenue = result["daily_revenue"]

dashboard, chat = st.columns([2, 1], gap="large")

with dashboard:
    st.title(store_name(store_dir))
    st.caption(
        f"{revenue.index.min():%Y-%m-%d} to {revenue.index.max():%Y-%m-%d}  ·  "
        f"{result['orders']:,} orders  ·  ${revenue.sum():,.2f} total revenue"
    )

    tab_days, tab_metrics, tab_chart, tab_explorer = st.tabs([
        ":material/calendar_month: Top 5 days",
        ":material/insights: Key metrics",
        ":material/show_chart: Daily revenue",
        ":material/table_view: Explorer",
    ])

    with tab_days:
        top_days = pd.DataFrame({
            "Date": result["top5_days"],
            "Revenue": [revenue[pd.Timestamp(day)] for day in result["top5_days"]],
        })
        st.dataframe(
            top_days,
            hide_index=True,
            column_config={
                "Revenue": st.column_config.ProgressColumn(
                    "Revenue (USD)", format="$%.2f", min_value=0, max_value=top_days["Revenue"].max()
                )
            },
        )

    with tab_metrics:
        col1, col2 = st.columns(2)
        col1.metric("Unique users", f"{result['unique_users']:,}", border=True)
        col2.metric("Unique author sets", result["author_sets"], border=True)

        col3, col4 = st.columns(2)
        with col3.container(border=True):
            st.caption("Most popular author(s)")
            for authors in result["top_authors"]:
                st.markdown(f"#### {', '.join(authors)}")
        with col4.container(border=True):
            st.caption("Best buyer (all ids of the same person)")
            st.markdown(f"#### {result['best_buyer']}")

    with tab_chart:
        chart = pd.DataFrame({
            "Daily revenue": revenue,
            "7-day average": revenue.rolling(7).mean(),
        })
        st.line_chart(chart, x_label="Date", y_label="Revenue (USD)", color=["#2E5F7A", "#4FC3F7"])

    with tab_explorer:
        view = st.session_state.view
        if view and view["store"] == store_dir:
            st.markdown(f"**{view['title']}**")
            if st.button("Show all rows", icon=":material/filter_alt_off:"):
                st.session_state.view = None
                st.rerun()
            st.dataframe(run_sql(tables, view["query"]), hide_index=True)
        else:
            table = st.segmented_control("Table", ["books", "users", "orders"], default="books")
            st.dataframe(tables[table or "books"], hide_index=True)
            st.caption('Ask the assistant to filter, e.g. "select users with e-mails starting with s or p".')

with chat:
    st.subheader(":material/forum: Ask about the data")
    messages = st.container(height=520)
    for message in st.session_state.history:
        messages.chat_message(message["role"]).markdown(message["content"])

    question = st.chat_input("e.g. Which genre sold the most copies?")
    if question:
        asked = sum(m["role"] == "user" for m in st.session_state.history)
        if asked >= MAX_QUESTIONS:
            st.warning("Question limit for this session reached. Refresh the page to start again.")
        else:
            st.session_state.history.append({"role": "user", "content": question})
            messages.chat_message("user").markdown(question)
            with messages.chat_message("assistant"), st.spinner("Checking the data..."):
                try:
                    answer, actions = ask(
                        st.secrets["GROQ_API_KEY"], st.session_state.history, tables,
                        store_name(store_dir), [store_name(s) for s in STORES],
                    )
                except RateLimitError:
                    answer, actions = "The free AI quota is busy right now. Please try again in a minute.", []
                except APIError as error:
                    # shown in the terminal, so failures can be checked later
                    print(f"AI error: {error!r}")
                    answer, actions = "The AI service could not handle this request. Try a shorter or simpler question.", []

            st.session_state.history.append({"role": "assistant", "content": answer})
            for action in actions:
                if "view" in action:
                    st.session_state.view = action["view"] and {**action["view"], "store": store_dir}
                if "store" in action:
                    st.session_state.next_store = f"data/{action['store']}"
                    st.session_state.view = None
            st.rerun()
