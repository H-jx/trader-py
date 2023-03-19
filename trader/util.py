from typing import List
from datetime import datetime, timedelta

def get_interval(hours = 72)  -> List[str] :
    """
    Get the start and end datetime for a given number of hours back from now.
    
    Args:
    - hours: The number of hours back to retrieve data for.
    
    Returns:
    - tuple[str, str]: start_time and end end_time.
    """
    # 获取当前时间
    now = datetime.now()

    # 计算指定小时数前的时间
    past_time = now - timedelta(hours=hours)

    # 格式化时间
    start_time = past_time.strftime('%Y-%m-%d %H:%M:%S')
    end_time = now.strftime('%Y-%m-%d %H:%M:%S')

    return start_time, end_time