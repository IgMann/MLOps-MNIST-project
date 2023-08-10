import time

def task_duration(start_time, end_time):
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
