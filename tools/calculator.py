def calculator(expression: str):
    """
    Performs basic mathematical calculations.
    """

    try:
        # Allow only safe mathematical characters
        allowed_chars = "0123456789+-*/().% "

        if not all(char in allowed_chars for char in expression):
            return {
                "success": False,
                "error": "Invalid mathematical expression."
            }

        result = eval(expression, {"__builtins__": {}}, {})

        return {
            "success": True,
            "expression": expression,
            "result": result
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }