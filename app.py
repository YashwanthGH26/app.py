import os
import time
from typing import Annotated, Literal
from typing_extensions import TypedDict

import httpx
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# --- STREAMLIT PAGE SETUP ---
st.set_page_config(page_title="SRE Agent Portal", page_icon="🛡️", layout="centered")

# --- MOCK LOGIN DATABASE ---
# In production, swap this out for an external database or oauth provider
USER_CREDENTIALS = {
    "admin": "password123",
    "sre_team": "securepass2026"
}

def login_form():
    """Renders the login UI widget panel."""
    st.title("🔒 SRE Operations Login")
    st.write("Please authenticate to access the automated diagnostics graph pipeline.")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign In")
        
        if submit:
            if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
                st.session_state["authenticated"] = True
                st.session_state["user"] = username
                st.success(f"Welcome back, {username}!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")

# Check authentication state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_form()
    st.stop()  # Halt execution here if not logged in

# --- LANGGRAPH RE-INITIALIZATION (Only runs if authenticated) ---

class State(TypedDict):
    messages: Annotated[list, add_messages]

def query_dynatrace_dql(query: str) -> str:
    """Execute a DQL query against Dynatrace log analytics."""
    return f"Mock Results for [{query}]: No critical anomalies found in the last 15m."

def fetch_runbook(topic: str) -> str:
    """Retrieve relevant RAG context or runbook documentation."""
    return f"Mock Runbook for [{topic}]: 1. Check DB connections. 2. Verify pool size."

TOOL_MAP = {
    "query_dynatrace_dql": query_dynatrace_dql,
    "fetch_runbook": fetch_runbook
}

client = genai.Client(api_key="AQ.Ab8RN6Jb90_esRWehTpidSqiVnyehpO0B1CpenC_cp1zFNr1xQ")
model_id = "gemini-2.5-flash"

if hasattr(client, "_api_client") and hasattr(client._api_client, "_httpx_client"):
    client._api_client._httpx_client = httpx.Client(verify=False)

gemini_tools = [types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name=name,
        description=func.__doc__,
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "query" if name == "query_dynatrace_dql" else "topic": types.Schema(type="STRING")
            },
            required=["query" if name == "query_dynatrace_dql" else "topic"]
        )
    ) for name, func in TOOL_MAP.items()
])]

def call_sre_core(state: State):
    contents = []
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=msg.content)]))
        elif isinstance(msg, AIMessage) or getattr(msg, "type", "") == "ai":
            parts = []
            if msg.content:
                parts.append(types.Part.from_text(text=msg.content))
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    parts.append(types.Part(function_call=types.FunctionCall(name=tc["name"], args=tc["args"])))
            contents.append(types.Content(role="model", parts=parts))
        elif isinstance(msg, ToolMessage) or getattr(msg, "type", "") == "tool":
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=f"Tool execution result for '{msg.name}': {msg.content}")]))

    max_retries = 3
    delay = 2
    response = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id, contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction="You are an expert SRE Agent. Use your tools to investigate issues.",
                    tools=gemini_tools
                )
            )
            break
        except ServerError as e:
            if "503" in str(e) and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise e

    tool_calls = []
    if response and response.function_calls:
        for i, call in enumerate(response.function_calls):
            tool_id = getattr(call, "id", None) or f"call_{call.name}_{i}_{int(time.time())}"
            tool_calls.append({"name": call.name, "args": dict(call.args), "id": str(tool_id), "type": "tool_call"})

    ai_msg = AIMessage(content=response.text or "" if response else "", tool_calls=tool_calls)
    return {"messages": [ai_msg]}

def execute_sre_tools(state: State):
    last_message = state["messages"][-1]
    new_messages = []
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tc in last_message.tool_calls:
            t_name = tc["name"]
            t_args = tc["args"]
            t_id = tc["id"]
            if t_name in TOOL_MAP:
                target_func = TOOL_MAP[t_name]
                func_param = t_args.get("query") if t_name == "query_dynatrace_dql" else t_args.get("topic")
                result_str = target_func(func_param)
                new_messages.append(ToolMessage(content=result_str, tool_call_id=t_id, name=t_name))
    return {"messages": new_messages}

def route_next(state: State) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

@st.cache_resource
def get_agent():
    workflow = StateGraph(State)
    workflow.add_node("sre_core", call_sre_core)
    workflow.add_node("tools", execute_sre_tools)
    workflow.add_edge(START, "sre_core")
    workflow.add_conditional_edges("sre_core", route_next)
    workflow.add_edge("tools", "sre_core")
    return workflow.compile()

sre_agent = get_agent()

# --- STREAMLIT MAIN APPLICATION CONSOLE UI ---
st.title("🤖 Live SRE Copilot Diagnostics")
st.caption(f"Logged in as: **{st.session_state['user']}**")

if st.button("Log Out", help="Clear active terminal user tokens"):
    st.session_state["authenticated"] = False
    st.rerun()

# Maintain a persistent chat logs state matrix across rendering loops
if "history" not in st.session_state:
    st.session_state["history"] = []

# Display prior historic message records cleanly inside native bubbles
for msg in st.session_state["history"]:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            if msg.content:
                st.markdown(msg.content)
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    st.info(f"⚙️ **Invoking Tool:** `{tc['name']}`\n\n**Args:** `{tc['args']}`")
    elif isinstance(msg, ToolMessage):
        with st.chat_message("assistant", avatar="⚙️"):
            st.caption(f"**Tool Output Details (`{msg.name}`):**")
            st.code(msg.content, language="json")

# User Text Entry Input Field
if user_input := st.chat_input("Describe the production incident status here..."):
    # Render user prompt immediately
    with st.chat_message("user"):
        st.markdown(user_input)
        
    human_msg = HumanMessage(content=user_input)
    st.session_state["history"].append(human_msg)
    
    # Run the compiled LangGraph workflow using updates engine matrix 
    initial_input = {"messages": st.session_state["history"]}
    
    with st.chat_message("assistant"):
        # Setup real-time feedback containers inside chat block
        status_placeholder = st.empty()
        
        for event in sre_agent.stream(initial_input, stream_mode="updates"):
            for node_name, node_output in event.items():
                for message in node_output.get("messages", []):
                    
                    # Track and append changes into global session stack
                    st.session_state["history"].append(message)
                    
                    if isinstance(message, AIMessage):
                        if message.content:
                            st.markdown(message.content)
                        if message.tool_calls:
                            for tc in message.tool_calls:
                                st.info(f"⚙️ **Invoking Tool:** `{tc['name']}`\n\n**Args:** `{tc['args']}`")
                                
                    elif isinstance(message, ToolMessage):
                        st.caption(f"**Tool Output Details (`{message.name}`):**")
                        st.code(message.content, language="json")
