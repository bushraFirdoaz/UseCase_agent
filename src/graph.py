from langgraph.graph import StateGraph, END

from src.state import AgentState

from src.nodes import (
    planner_node,
    tool_executor_node,
    response_node
)


# ============================================================
# ROUTING AFTER PLANNER
# ============================================================

def route_after_planner(state: AgentState):
    """
    Decide whether the agent should execute a tool
    or generate the final answer.
    """

    action = state["plan"][0]

    if action == "ANSWER":

        return "response"

    return "tool"


# ============================================================
# ROUTING AFTER TOOL
# ============================================================

def route_after_tool(state: AgentState):
    """
    After a tool executes, return to the planner.

    The planner receives the previous tool result and
    decides what should happen next.
    """

    return "planner"


# ============================================================
# CREATE GRAPH
# ============================================================

graph = StateGraph(AgentState)


# ============================================================
# ADD NODES
# ============================================================

graph.add_node(
    "planner",
    planner_node
)

graph.add_node(
    "tool_executor",
    tool_executor_node
)

graph.add_node(
    "response",
    response_node
)


# ============================================================
# START → PLANNER
# ============================================================

graph.set_entry_point("planner")


# ============================================================
# PLANNER → TOOL OR RESPONSE
# ============================================================

graph.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "tool": "tool_executor",
        "response": "response"
    }
)


# ============================================================
# TOOL → PLANNER
# ============================================================

graph.add_edge(
    "tool_executor",
    "planner"
)


# ============================================================
# RESPONSE → END
# ============================================================

graph.add_edge(
    "response",
    END
)


# ============================================================
# COMPILE GRAPH
# ============================================================

agent_graph = graph.compile()