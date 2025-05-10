import numpy as np
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan


def scan_to_points(scan_msg: LaserScan,
                   range_max: float = 10.0,
                   range_min: float = 0.05,
                   step: int = 1,
                   angle_range: float = 0.35):  # ±0.35弧度约±20度
    """将 LaserScan 消息中的极坐标数据转换为车辆坐标系下的点列表

    参数
    -----
    scan_msg: sensor_msgs.msg.LaserScan
        激光雷达扫描消息
    range_max: float
        只保留量程小于该值的点，单位 m
    range_min: float
        只保留量程大于该值的点，过滤掉离散噪声, 单位 m
    step: int
        对扫描线进行抽稀，每 *step* 个角度取一个点
    angle_range: float
        只保留前方在 ±angle_range 弧度范围内的点（默认±20度）

    返回
    -----
    list[geometry_msgs.msg.Point]
        转换后的点（z 坐标固定为 0）
    """
    angle = scan_msg.angle_min
    points = []
    # 遍历 ranges，按需要抽稀
    for i, r in enumerate(scan_msg.ranges):
        if i % step != 0:
            angle += scan_msg.angle_increment
            continue

        # 只保留前方指定角度范围内的点
        if abs(angle) > angle_range:
            angle += scan_msg.angle_increment
            continue
            
        # 过滤无效测距
        if not np.isfinite(r):
            angle += scan_msg.angle_increment
            continue
        if r < range_min or r > range_max:
            angle += scan_msg.angle_increment
            continue

        x = r * np.cos(angle)
        y = r * np.sin(angle)
        
        # 创建Point对象
        pt = Point()
        pt.x = float(x)
        pt.y = float(y)
        pt.z = 0.0
        points.append(pt)

        angle += scan_msg.angle_increment

    return points
