import time

def task_duration(start_time: float, end_time: float) -> None:
    """
    Calculates the duration of a task given the start and end times.
    
    Parameters:
    start_time (float): The start time of the task in seconds.
    end_time (float): The end time of the task in seconds.
    
    Returns:
    None: The function does not return anything. It prints the formatted duration of the task.
    """
    elapsed_time = end_time - start_time

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = round(elapsed_time % 60, 2)

    if hours > 0:
        seconds = int(seconds)
        formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    elif minutes > 0:
        seconds = int(seconds)
        formatted_time = f"{minutes:02d}:{seconds:02d}"
    else:
        formatted_time = f"{seconds} seconds"

    print(100 * '-')
    print("Task took:", formatted_time)
    print(100 * '-')
