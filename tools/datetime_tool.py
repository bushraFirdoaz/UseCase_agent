from datetime import datetime


def get_datetime():
    """
    Returns the current local date and time.
    """

    now = datetime.now()

    return {
        "success": True,
        "date": now.strftime("%A, %d %B %Y"),
        "time": now.strftime("%I:%M:%S %p"),
        "datetime": now.strftime("%A, %d %B %Y, %I:%M:%S %p")
    }