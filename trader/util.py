from typing import Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def get_interval(hours = 72)  -> Tuple[str, str] :
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

def compress_data(data: pd.DataFrame, columns_to_compress=None):
    """
    将数据压缩到 0 到 1 之间，并且可以选择部分列进行压缩
    :param data: 数据
    :param columns_to_compress: 需要压缩的列，默认为 None 表示压缩所有列
    :return: 压缩后的数据
    """
    # 如果没有指定需要压缩的列，则默认压缩所有列
    if columns_to_compress is None:
        columns_to_compress = data.columns
    
    # 将需要压缩的列转换为 NumPy 数组
    columns_to_compress = np.array(columns_to_compress)
    
    # 对需要压缩的列的数据进行归一化
    for column in columns_to_compress:
        col_min = data[column].min()
        col_max = data[column].max()
        data[column] = (data[column] - col_min) / (col_max - col_min)
    
    # 返回压缩后的数据
    return data