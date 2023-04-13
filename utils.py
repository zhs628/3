import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def living_value(vec: np.ndarray[float, 2])-> float:
    x = vec[0]
    y = vec[1]
    z = (x-1) ** 2 + (y-2.5) ** 2
    return -z

def split_by_ratio(x, y):
    """
    根据 x 中的元素比例分割 y 列表，并返回分割后的子列表组成的列表。
    """
    total = sum(x)
    if total != 1:
        raise ValueError("x 中所有元素之和必须为 1。")

    lengths = [int(len(y) * r) for r in x[:-1]]
    lengths.append(len(y) - sum(lengths))

    result = []
    start = 0
    for length in lengths:
        result.append(y[start:start+length])
        start += length

    return result


def animate_decorator(func):  # 动画演示
    def wrapper(*args, **kwargs):
        # 调用算法函数
        result = func(*args, **kwargs)
        fig, ax = plt.subplots()
        # 显示动画
        for i, (points, heads) in enumerate(result):
            if len(result) <= 100 or (i in [m for m in list(range(len(result))) if m%(int(len(result)/100)) == 0]): 
                # 更新图形界面
                ax.clear()
                x_list = [point[0] for point in points] + [head[0] for head in heads]
                y_list = [point[1] for point in points] + [head[1] for head in heads]
                x_range = max(x_list) - min(x_list)
                y_range = max(y_list) - min(y_list)
                ax.set_xlim(min(x_list) - x_range, max(x_list) + x_range)
                ax.set_ylim(min(y_list) - y_range, max(y_list) + y_range)
                
                show_num = 100 if len(points)>100 else len(points)
                ax.plot([point[0] for point in points[:show_num]], [point[1] for point in points[:show_num]], 'go')
                ax.plot([point[0] for point in heads], [point[1] for point in heads], 'bo')
                ax.grid()
                ax.set_title(f"Step {i+1}")
                plt.draw()
                plt.pause(0.1)
        
        plt.close()
        # 返回算法结果
        return result

    return wrapper