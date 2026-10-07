from src.graph import agent_graph


def run_agent(user_input: str):
    """
    Run the LangGraph agent with the user's request.
    """

    initial_state = {
        "user_input": user_input,
        "plan": [],
        "tool_arguments": {},
        "current_step": 0,
        "max_steps": 5,
        "tool_results": [],
        "reasoning_history": [],
        "error": "",
        "final_response": ""
    }
    result = agent_graph.invoke(initial_state)

    return result["final_response"]


def main():

    print("=" * 60)
    print("        MULTI-STEP AI AGENT")
    print("=" * 60)

    print("\nType 'exit' to stop the agent.\n")

    while True:

        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Agent stopped.")
            break

        if not user_input.strip():
            continue

        try:

            response = run_agent(user_input)

            print("\nAgent:")
            print(response)
            print()

        except Exception as e:

            print("\nAgent Error:")
            print(e)
            print()


if __name__ == "__main__":
    main()