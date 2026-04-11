import datetime

def create_log_entry(agent_name, action, detail):
    """Formats a professional log entry for the UI."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    return {
        "time": timestamp,
        "agent": agent_name,
        "action": action,
        "detail": detail
    }