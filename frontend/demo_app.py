"""MemCache demo frontend. UI only — every side effect lives in demo_runtime.

    .venv/bin/python -m streamlit run frontend/demo_app.py
"""

from __future__ import annotations

import streamlit as st

from frontend import demo_runtime as rt
from frontend.demos import DEMOS

st.set_page_config(page_title="MemCache Demos", page_icon="🧠", layout="wide")

# ----------------------------------------------------------------- sidebar

st.sidebar.title("🧠 MemCache")
st.sidebar.caption("Memory infrastructure for LLM agents — live demos")

status = rt.stack_status()
stack_ok = all(ok for ok, _ in status.values())
with st.sidebar.expander("Stack status", expanded=not stack_ok):
    for name, (ok, detail) in status.items():
        st.write(("✅" if ok else "❌") + f" {name} — {detail}")
    if not stack_ok:
        st.code("docker compose up -d redis postgres neo4j\nollama pull qwen3:4b", language="bash")

agent_ok = status.get(f"ollama {rt.DEMO_AGENT_MODEL}", (False, ""))[0]
st.sidebar.caption(f"agents: `{rt.DEMO_AGENT_MODEL}` · internals: `qwen2.5:3b`")

titles = {d.title: d for d in DEMOS}
seeded_now = {d.key: rt.is_seeded(d) for d in DEMOS} if stack_ok else {}
choice = st.sidebar.radio(
    "Demo",
    list(titles),
    format_func=lambda t: f"{t} {'●' if seeded_now.get(titles[t].key) else '○'}",
    # Explicit key: without it the widget's auto-id hashes its args, so a
    # seeded badge flipping ○→● made Streamlit treat it as a NEW widget and
    # reset the selection to the first demo (seen in browser review).
    key="demo_choice",
)
demo = titles[choice]
st.sidebar.caption("● seeded — reruns are instant · ○ will seed on first run")

col_a, col_b = st.sidebar.columns(2)
if col_a.button("Reset this demo", width="stretch", disabled=not stack_ok):
    rt.reset(demo)
    st.session_state.pop(f"result-{demo.key}", None)
    st.rerun()
if col_b.button("Reset all", width="stretch", disabled=not stack_ok):
    rt.reset_all()
    for d in DEMOS:
        st.session_state.pop(f"result-{d.key}", None)
    st.rerun()

# -------------------------------------------------------------------- main

st.title(demo.title)
st.write(demo.blurb)

st.subheader("The scripted conversations")
for i, session in enumerate(demo.sessions):
    with st.expander(session.label, expanded=False):
        for m in session.messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])
        for tf in session.tool_failures:
            st.error(f"$ {tf['tool_name']} — recorded to L4: `{tf['error']}`")

run = st.button(
    "▶ Run demo",
    type="primary",
    disabled=not stack_ok,
    help=None if stack_ok else "Fix the stack status in the sidebar first",
)

if run:
    try:
        if not rt.is_seeded(demo):
            with st.status("Seeding memory through the real pipeline…", expanded=True) as box:

                def cb(i, n, label):
                    box.write(f"({i + 1}/{n}) {label}: Ollama summarize → embed → graph → claim")

                rt.seed(demo, progress_cb=cb)
                box.update(label="Seeded — L2, L3, and L4 written", state="complete")
        with st.spinner("Retrieving memory for a brand-new session…"):
            retrieved = rt.retrieve(demo)
        answers = {}
        if agent_ok:
            with st.spinner(f"Asking {rt.DEMO_AGENT_MODEL} twice — with and without memory…"):
                answers["with"] = rt.ask_agent(demo.agent_question, retrieved["context"])
                answers["without"] = rt.ask_agent(demo.agent_question, None)
        st.session_state[f"result-{demo.key}"] = {"retrieved": retrieved, "answers": answers}
        st.rerun()
    except Exception as exc:  # never a stack trace on the page
        st.error(f"Demo run failed: {exc}")

result = st.session_state.get(f"result-{demo.key}")
if result:
    retrieved, answers = result["retrieved"], result["answers"]

    st.subheader("Same model, same question")
    st.markdown(f"**Q:** {demo.agent_question}")
    if answers:
        left, right = st.columns(2)
        with left:
            st.markdown("#### 🧠 With MemCache")
            st.info(answers["with"].text or "(empty answer)")
            st.caption(f"{answers['with'].seconds:.1f}s · {rt.DEMO_AGENT_MODEL}")
        with right:
            st.markdown("#### 🚫 Without memory")
            st.warning(answers["without"].text or "(empty answer)")
            st.caption(f"{answers['without'].seconds:.1f}s · {rt.DEMO_AGENT_MODEL}")
    else:
        st.warning(
            f"`{rt.DEMO_AGENT_MODEL}` is not pulled — side-by-side skipped. "
            f"Run `ollama pull {rt.DEMO_AGENT_MODEL}` and rerun."
        )

    st.subheader("What the with-memory agent was given")
    st.code(retrieved["context"], language=None)

    st.subheader("Retrieved memory, structured")
    rows = rt.build_source_rows(retrieved["sources"])
    st.dataframe(rows, width="stretch", hide_index=True)
    c = rt.count_kinds(rows)
    st.caption(
        f"Retrieved — episodes: {c['episodes']} · entities: {c['entities']} · "
        f"goals: {c['goals']} · tool calls: {c['tool_calls']}. Every context "
        "line above is attributable to one of these rows."
    )
