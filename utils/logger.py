
import time
import functools

GRAY_BOLD = "\033[1;30m"
WHITE_BOLD = "\033[1;37m"
GREEN_BOLD = "\033[1;32m"
RED_BOLD = "\033[1;31m"
YELLOW_BOLD = "\033[1;33m"
BLUE_BOLD = "\033[1;34m"
MAGENTA_BOLD = "\033[1;35m"
CYAN_BOLD = "\033[1;36m"
RESET = "\033[0m"

def log_execution_time(func):
    """
    함수가 실행될 때 시작과 종료 메시지를 출력하고 실행 시간을 측정하는 데코레이터

    Args:
        func (function): 데코레이터를 적용할 함수

    Returns:
        wrapper (function): 데코레이터가 적용된 함수
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"{GRAY_BOLD}[⇣]{RESET}{WHITE_BOLD} Func.{str(func.__name__).upper()} Start...{RESET}")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{GREEN_BOLD}[✔︎] Func.{str(func.__name__).upper()} Finish!{RESET} Run-Time: {end_time - start_time:.4f}s")
        return result
    return wrapper

def log(cateogry: str, msg: str):
    """
    메시지를 출력하는 함수

    Args:
        category (str): 메시지 카테고리(info, success, warning, error)
        msg (str): 출력할 메시지

    Returns:
        None
    """
    color = None
    icon = None
    if cateogry == "info":
        color = WHITE_BOLD
        icon = "☞"
    elif cateogry == "success":
        color = GREEN_BOLD
        icon = "✔︎"
    elif cateogry == "warning":
        color = YELLOW_BOLD
        icon = "⚡︎"
    elif cateogry == "error":
        color = RED_BOLD
        icon = "✘"

    print(f"{color}[{icon}] {msg}{RESET}")
    return None

def log_progress(cur_idx, last_idx, msg):
    """
    진행 상태를 출력하는 함수

    Args:
        cur_idx (int): 현재 인덱스
        last_idx (int): 마지막 인덱스
        msg (str): 출력할 메시지

    Returns:
        None
    """
    progress = round(((cur_idx) / last_idx * 100), 1)
    if cur_idx == last_idx:
        print(f" - [100.0%][{cur_idx}/{last_idx}] {msg}")
    else:
        print(f" - [{progress:5.1f}%][{cur_idx:03d}/{last_idx:03d}] {msg}", end="\r", flush=True)
    return None

def log_summary(summary: dict):
    """
    요약 정보를 출력하는 함수

    Args:
        summary (dict): 요약 정보

    Returns:
        None
    """
    print(f"{WHITE_BOLD}[⌗] Summary{RESET}")
    for key, value in summary.items():
        print(f" - {key}: {value}")
    return None