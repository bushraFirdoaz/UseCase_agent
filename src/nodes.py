import json
import re
import ollama
import os
from tools.calculator import calculator
from tools.weather import get_weather
from tools.datetime_tool import get_datetime
from src.knowledge.retriever import generate_answer
from src.state import AgentState

from src.bedrock_model import ask_model


# ============================================================
# CONSTANTS
# ============================================================

MAX_STEPS = 5


# ============================================================
# PLANNER NODE
# ============================================================


def planner_node(state: AgentState):
    """
    Use the configured AI provider (Ollama or Bedrock)
    to decide the next action and generate its arguments.
    """

    # MAX STEP SAFETY CHECK
    if state["current_step"] >= state["max_steps"]:
        return {
            "plan": ["ANSWER"],
            "tool_arguments": {},
            "error": (
                "The agent reached the maximum number "
                "of steps allowed."
            )
        }

    user_input = state["user_input"]
    tool_results = state["tool_results"]
    reasoning_history = state["reasoning_history"]

    # BUILD CONTEXT
    previous_context = ""

    if tool_results:
        previous_context += "\nPrevious tool results:\n"

        for item in tool_results:
            previous_context += (
                f"Step {item['step']} - "
                f"{item['action']}: "
                f"{item['result']}\n"
            )

    if reasoning_history:
        previous_context += "\nPrevious decisions:\n"

        for item in reasoning_history:
            previous_context += (
                f"Step {item['step']} - "
                f"Decision: {item['decision']}\n"
            )

    # PLANNER INSTRUCTIONS
    system_prompt = """
You are the reasoning planner of an AI agent.

Your job is to decide what action should be performed
next and provide the arguments required by that action.

Available actions:

CALCULATE
- Use ONLY for mathematical calculations.
- Required argument: expression

WEATHER
- Use for current weather information.
- Use for temperature, rain, humidity, wind,
  forecast, weather conditions, or umbrella questions.
- Required argument: city

DATETIME
- Use ONLY for current date or time questions.
- No arguments are required.

KNOWLEDGE
- Use for questions about concepts or information
  that may be available in the provided documents.
- Use for questions asking to explain, summarize,
  describe, or compare concepts from the documents.
- No arguments are required.

ANSWER
- Use for general knowledge, programming,
  conversational questions, or when the available
  information is enough to provide the final answer.
- No arguments are required.

IMPORTANT:
- Look at the original user request.
- Look at previous tool results and decisions.
- Do not repeat a tool unnecessarily.
- Return ONLY valid JSON.
- Do not use markdown.
- Do not explain your decision.

Required JSON format:
{
    "action": "ACTION_NAME",
    "arguments": {}
}

Examples:

User: What is 25 * 6?
{"action": "CALCULATE", "arguments": {"expression": "25 * 6"}}

User: What is the temperature in Hyderabad?
{"action": "WEATHER", "arguments": {"city": "Hyderabad"}}

User: What is the current time?
{"action": "DATETIME", "arguments": {}}

User: Summarize my document about machine learning.
{"action": "KNOWLEDGE", "arguments": {}}

User: What is Ollama?
{"action": "ANSWER", "arguments": {}}

If a previous weather result says temperature = 30
and the user asks to calculate 10% of that temperature:
{"action": "CALCULATE", "arguments": {"expression": "30 * 10 / 100"}}
"""

    # ASK THE CONFIGURED MODEL
    try:
        planner_prompt = (
            f"{system_prompt}\n\n"
            f"Original user request:\n{user_input}\n"
            f"{previous_context}"
        )

        planner_output = ask_model(planner_prompt).strip()
        print("DEBUG: Planner provider =", os.getenv("AI_PROVIDER", "ollama"))
        print("DEBUG: Planner output =", planner_output)

    except Exception as e:
        return {
            "plan": ["ANSWER"],
            "tool_arguments": {},
            "error": f"Planner model error: {str(e)}"
        }

    # PARSE MODEL OUTPUT
    decision = parse_planner_output(planner_output)

    action = decision["action"]
    arguments = decision["arguments"]

    valid_actions = [
        "CALCULATE",
        "WEATHER",
        "DATETIME",
        "KNOWLEDGE",
        "ANSWER"
    ]

    if action not in valid_actions:
        return {
            "plan": ["ANSWER"],
            "tool_arguments": {},
            "error": (
                f"AI model returned an invalid action: {action}"
            )
        }

    # VALIDATE TOOL ARGUMENTS
    validation_error = validate_arguments(action, arguments)

    if validation_error:
        return {
            "plan": ["ANSWER"],
            "tool_arguments": {},
            "error": validation_error
        }

    # PREVENT UNNECESSARY TOOL REPETITION
    previous_actions = [
        item["decision"]
        for item in reasoning_history
    ]

    if action in previous_actions:
        already_completed = False

        for item in tool_results:
            if (
                item["action"] == action
                and isinstance(item["result"], dict)
                and item["result"].get("success", False)
            ):
                already_completed = True
                break

        if already_completed:
            action = "ANSWER"
            arguments = {}

    # STORE REASONING HISTORY
    reasoning_history = reasoning_history + [
        {
            "step": len(reasoning_history) + 1,
            "decision": action,
            "arguments": arguments
        }
    ]

    return {
        "plan": [action],
        "tool_arguments": arguments,
        "current_step": 0,
        "reasoning_history": reasoning_history,
        "error": ""
    }

# ============================================================
# TOOL EXECUTOR NODE
# ============================================================

def tool_executor_node(state: AgentState):
    """
    Execute the action selected by Ollama.

    Tool errors are captured instead of crashing
    the entire agent.
    """

    plan = state["plan"]
    current_step = state["current_step"]
    tool_arguments = state["tool_arguments"]
    tool_results = state["tool_results"]

    if current_step >= len(plan):

        return state

    action = plan[current_step]

    result = None

    # --------------------------------------------------------
    # EXECUTE TOOL SAFELY
    # --------------------------------------------------------

    try:

        # ----------------------------------------------------
        # CALCULATOR
        # ----------------------------------------------------

        if action == "CALCULATE":

            expression = tool_arguments.get("expression")

            if not expression:

                result = {
                    "success": False,
                    "error": (
                        "No mathematical expression "
                        "was provided."
                    )
                }

            else:

                result = calculator(expression)

        # ----------------------------------------------------
        # WEATHER
        # ----------------------------------------------------

        elif action == "WEATHER":

            city = tool_arguments.get("city")

            if not city:

                result = {
                    "success": False,
                    "error": "No city was provided."
                }

            else:

                result = get_weather(city)

        # ----------------------------------------------------
        # DATE / TIME
        # ----------------------------------------------------

        elif action == "DATETIME":

            result = get_datetime()

        # ----------------------------------------------------
        # GENERAL ANSWER
        # ----------------------------------------------------
        elif action == "KNOWLEDGE":
            answer = generate_answer(state["user_input"])

            result = {
                "success": True,
                "message": answer
            }
            print("DEBUG: KNOWLEDGE ACTION CALLED")
        
        elif action == "ANSWER":

            prompt = (
                "You are a helpful AI assistant. "
                "Answer general knowledge, programming, "
                "conceptual, and conversational questions "
                "clearly and simply.\n\n"
                f"User question: {state['user_input']}"
            )
            print("DEBUG: ANSWER provider =", __import__("os").getenv("AI_PROVIDER"))
            answer = ask_model(prompt)
            
            result = {
                "success": True,
                "message": answer
            }

    except Exception as e:

        result = {
            "success": False,
            "error": str(e)
        }

    # --------------------------------------------------------
    # PROTECT AGAINST INVALID TOOL RESULT
    # --------------------------------------------------------

    if result is None:

        result = {
            "success": False,
            "error": (
                f"No result was returned for action: {action}"
            )
        }

    # --------------------------------------------------------
    # STORE RESULT
    # --------------------------------------------------------

    tool_results = tool_results + [
        {
            "step": current_step + 1,
            "action": action,
            "arguments": tool_arguments,
            "result": result
        }
    ]

    return {
        "tool_results": tool_results,
        "current_step": current_step + 1,
        "error": ""
    }


# ============================================================
# RESPONSE NODE
# ============================================================


def response_node(state: AgentState):
    """
    Generate the final response using the configured AI provider.

    If an error occurred, explain it clearly instead of exposing
    a Python traceback.
    """

    user_input = state["user_input"]
    tool_results = state["tool_results"]
    error = state.get("error", "")

    context = ""

    if tool_results:
        context += "\nInformation obtained during the task:\n"

        for item in tool_results:
            context += (
                f"Step {item['step']} - "
                f"{item['action']} - "
                f"Arguments: {item.get('arguments', {})} - "
                f"Result: {item['result']}\n"
            )

    # Return a clear error if the agent could not complete the task.
    if error:
        return {
            "final_response": (
                "I couldn't complete the request. "
                f"Reason: {error}"
            )
        }

    # Generate the final answer using Ollama or Bedrock.
    try:
        prompt = f"""
You are the final response generator for an AI agent.

Answer the user's original question using the information
collected during the task.

Rules:
- Give a clear and natural answer.
- Do not mention internal tools, planning, or reasoning.
- Do not mention DEBUG information.
- Use successful tool results accurately.
- If a tool failed, explain the failure clearly.
- If no tool was required, answer using your own knowledge.

Original question:
{user_input}

{context}
"""

        final_response = ask_model(prompt)

    except Exception as e:
        final_response = (
            "I couldn't generate the final answer. "
            f"Reason: {str(e)}"
        )

    return {
        "final_response": final_response
    }



# ============================================================
# VALIDATE TOOL ARGUMENTS
# ============================================================

def validate_arguments(action, arguments):
    """
    Validate the arguments generated by Ollama.
    """

    if not isinstance(arguments, dict):

        return "Tool arguments must be a dictionary."

    # --------------------------------------------------------
    # CALCULATE
    # --------------------------------------------------------

    if action == "CALCULATE":

        expression = arguments.get("expression")

        if not expression:

            return (
                "Calculator requires an 'expression' argument."
            )

        if not isinstance(expression, str):

            return (
                "Calculator expression must be a string."
            )

    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    elif action == "WEATHER":

        city = arguments.get("city")

        if not city:

            return (
                "Weather tool requires a 'city' argument."
            )

        if not isinstance(city, str):

            return "Weather city must be a string."

    return None


# ============================================================
# PARSE PLANNER OUTPUT
# ============================================================

def parse_planner_output(planner_output: str):
    """
    Convert Ollama's JSON response into a Python dictionary.

    Handles cases where Ollama accidentally adds markdown
    code fences around the JSON.
    """

    cleaned = planner_output.strip()

    # Remove markdown code fences
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    )

    try:

        decision = json.loads(cleaned)

        action = str(
            decision.get("action", "ANSWER")
        ).upper()

        arguments = decision.get(
            "arguments",
            {}
        )

        if not isinstance(arguments, dict):

            arguments = {}

        return {
            "action": action,
            "arguments": arguments
        }

    except (json.JSONDecodeError, TypeError):

        # Safe fallback
        upper_output = cleaned.upper()

        if "WEATHER" in upper_output:

            return {
                "action": "WEATHER",
                "arguments": {}
            }

        if "CALCULATE" in upper_output:

            return {
                "action": "CALCULATE",
                "arguments": {}
            }

        if "DATETIME" in upper_output:

            return {
                "action": "DATETIME",
                "arguments": {}
            }

        return {
            "action": "ANSWER",
            "arguments": {}
        }