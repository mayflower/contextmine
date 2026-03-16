import asyncio
import os
import re
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from prefect import task
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    repo_path: str


@tool
def bash_tool(command: str, repo_path: str) -> str:
    """Execute a bash command in the repository path."""
    import subprocess

    try:
        result = subprocess.run(  # nosemgrep: python.lang.security.audit.subprocess-shell-true
            command,
            shell=True,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=120,  # nosec B602
        )
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nEXIT CODE: {result.returncode}"
    except Exception as e:
        return f"Error executing command: {e}"


@tool
def read_tool(file_path: str, repo_path: str) -> str:
    """Read the contents of a file in the repository path."""
    import os

    full_path = os.path.join(repo_path, file_path)
    try:
        with open(full_path) as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def glob_tool(pattern: str, repo_path: str) -> str:
    """Find files matching a glob pattern in the repository path."""
    import glob
    import os

    full_pattern = os.path.join(repo_path, pattern)
    try:
        files = glob.glob(full_pattern, recursive=True)
        # return paths relative to repo_path
        relative_files = [os.path.relpath(f, repo_path) for f in files]
        return "\n".join(relative_files)
    except Exception as e:
        return f"Error finding files: {e}"


def build_agent():
    tools = [bash_tool, read_tool, glob_tool]

    # Initialize LLM
    api_key = os.environ.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_TOKEN"))
    llm = ChatAnthropic(model="claude-3-5-sonnet-latest", api_key=api_key)  # ty: ignore[missing-argument,unknown-argument]
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        messages = state["messages"]
        # Inject repo_path into tool calls if needed, or rely on agent passing it
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
    )
    graph_builder.add_edge("tools", "agent")
    return graph_builder.compile()


@task
async def deploy_target_app(repo_path: str) -> str:
    """Deploy a target app using an LLM agent."""
    agent = build_agent()

    system_prompt = f"""Analysiere das Verzeichnis. Finde Docker-Configs (Dockerfile, docker-compose.yml, package.json). 
Führe die nötigen Bash-Befehle aus, um die App im Hintergrund (detached) zu bauen und zu starten (z.B. `docker compose up -d --build`). 
Finde via `docker ps` den gemappten Host-Port heraus. 
Das repository befindet sich in: {repo_path}
WICHTIG: Wenn du Tools benutzt, übergib immer den Parameter `repo_path` mit dem Wert "{repo_path}".
Gib als finales Resultat AUSSCHLIESSLICH die lokale HTTP-URL zurück (z.B. http://localhost:8080). Keinen anderen Text."""

    state = {
        "messages": [
            ("system", system_prompt),
            ("user", "Starte das Deployment und gib die URL zurück."),
        ],
        "repo_path": repo_path,
    }

    # Run the agent
    final_state = await asyncio.to_thread(agent.invoke, state)
    final_message = final_state["messages"][-1].content

    # Extract URL using regex
    url_match = re.search(r"https?://[^\s]+", final_message)
    if url_match:
        return url_match.group(0)
    return final_message.strip()
