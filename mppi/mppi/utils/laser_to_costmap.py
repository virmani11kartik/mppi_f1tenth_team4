import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
import matplotlib.cm as cm
from visualization_msgs.msg import Marker
import cv2


def scan_to_grid(scan_msg, resolution=0.1):
    """将激光扫描数据转换为以车辆为中心的二维占用栅格图
    
    参数
    -----
    scan_msg: sensor_msgs.msg.LaserScan
        激光雷达消息
    resolution: float
        栅格地图分辨率 (m/cell)
        
    返回
    -----
    grid: np.ndarray
        二维栅格地图（0=空闲，100=占用，10=未知）
    origin: tuple
        栅格图左下角在车体坐标系中的位置 (x, y)
    cell_size: float
        每个栅格的实际尺寸 (m)
    """
    # 定义矩形区域范围（米）
    x_front = 8.0   # 前方8m
    x_back = 2.0    # 后方2m
    y_side = 5.0    # 左右各5m
    
    # 计算栅格尺寸
    x_size = int((x_front + x_back) / resolution)
    y_size = int((2 * y_side) / resolution)
    
    # 创建栅格地图：10表示未知，0表示空闲，100表示占用
    grid = np.full((y_size, x_size), 10, dtype=np.int8)
    
    # 计算车辆中心在栅格中的坐标
    center_x = int(x_back / resolution)  # 车辆后方有2米
    center_y = y_size // 2               # 车辆在y轴中心
    
    # 标记车辆位置为已知空闲
    car_radius = max(1, int(0.2 / resolution))  # 假设车半径为0.2m，至少1格
    cv2.circle(grid, (center_x, center_y), car_radius, 0, -1)
    
    # 第1阶段: 处理所有射线，更新空白区域
    angle = scan_msg.angle_min
    for i, r in enumerate(scan_msg.ranges):
        # 无效测量，跳过
        if not np.isfinite(r):
            angle += scan_msg.angle_increment
            continue
            
        # 计算点的坐标（以车为中心）
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        
        # 计算栅格坐标
        grid_x = int(center_x + x / resolution)
        grid_y = int(center_y + y / resolution)
        
        # 计算矩形边界的栅格坐标
        boundary_x = grid_x
        boundary_y = grid_y
        
        # 检查是否在矩形区域内
        in_rectangle = True
        
        # 超出前方边界
        if x > x_front:
            boundary_x = int(center_x + x_front / resolution)
            in_rectangle = False
        # 超出后方边界
        elif x < -x_back:
            boundary_x = int(center_x - x_back / resolution)
            in_rectangle = False
        # 超出左边界
        if y > y_side:
            boundary_y = int(center_y + y_side / resolution)
            in_rectangle = False
        # 超出右边界
        elif y < -y_side:
            boundary_y = int(center_y - y_side / resolution)
            in_rectangle = False
        
        # 对于所有射线，从中心到有效点（或边界点）的路径标记为空闲
        if in_rectangle:
            # 先标记射线路径为空闲
            ray_trace(grid, center_x, center_y, grid_x, grid_y, 0)
        else:
            # 射线超出边界，只追踪到边界
            ray_trace(grid, center_x, center_y, boundary_x, boundary_y, 0)
            
        angle += scan_msg.angle_increment
    
    # 第2阶段: 标记障碍物
    angle = scan_msg.angle_min
    for i, r in enumerate(scan_msg.ranges):
        # 无效测量，跳过
        if not np.isfinite(r):
            angle += scan_msg.angle_increment
            continue
            
        # 计算点的坐标（以车为中心）
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        
        # 只处理在矩形区域内的障碍点
        if -x_back <= x <= x_front and -y_side <= y <= y_side:
            # 换算到栅格坐标
            grid_x = int(center_x + x / resolution)
            grid_y = int(center_y + y / resolution)
            
            # 检查边界
            if 0 <= grid_x < x_size and 0 <= grid_y < y_size:
                grid[grid_y, grid_x] = 100  # 标记为占用
            
        angle += scan_msg.angle_increment
    
    # 第3阶段: 对障碍点进行膨胀处理
    grid = dilate_obstacles(grid, center_x, center_y, resolution)
    
    # 栅格图左下角的原点（在车体坐标系中）
    origin_x = -x_back
    origin_y = -y_side
    
    return grid, (origin_x, origin_y), resolution


def dilate_obstacles(grid, center_x, center_y, resolution):
    """对障碍点进行方向性膨胀
    
    参数
    -----
    grid: np.ndarray
        原始栅格地图
    center_x, center_y: int
        车辆在栅格中的坐标
    resolution: float
        栅格分辨率
        
    返回
    -----
    np.ndarray
        膨胀后的栅格地图
    """
    # 复制原始栅格图
    dilated = grid.copy()
    
    # 找出所有障碍点的坐标
    obstacle_coords = np.where(grid == 100)
    
    # 膨胀范围（格数）
    front_dilate = max(1, int(0.5 / resolution))  # 前方0.5m，约5格(0.1m/格)
    back_dilate = max(1, int(0.13 / resolution))   # 后方0.2m，约2格
    side_dilate = max(1, int(0.13 / resolution))   # 左右0.2m，约2格
    
    # 对每个障碍点进行膨胀
    for y, x in zip(obstacle_coords[0], obstacle_coords[1]):
        # 确定点相对于车的方向
        is_front = x >= center_x
        is_left = y <= center_y
        
        # 前后方向膨胀
        x_min = max(0, x - (back_dilate if is_front else front_dilate))
        x_max = min(grid.shape[1]-1, x + (front_dilate if is_front else back_dilate))
        
        # 左右方向膨胀
        y_min = max(0, y - side_dilate)
        y_max = min(grid.shape[0]-1, y + side_dilate)
        
        # 应用膨胀
        for dx in range(x_min, x_max+1):
            for dy in range(y_min, y_max+1):
                # 只膨胀未知或空闲区域，不覆盖其他障碍点
                if dilated[dy, dx] != 100:
                    dilated[dy, dx] = 100
    
    return dilated


def ray_trace(grid, x0, y0, x1, y1, value):
    """从(x0,y0)到(x1,y1)的射线上的所有栅格都标记为指定值
    
    使用Bresenham算法
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while x0 != x1 or y0 != y1:
        if 0 <= x0 < grid.shape[1] and 0 <= y0 < grid.shape[0]:
            # 不要覆盖终点
            if x0 != x1 or y0 != y1:
                grid[y0, x0] = value
                
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def costmap_to_marker(costmap, origin, resolution, frame_id="base_link"):
    """将栅格图转换为RViz可视化的Marker
    
    参数
    -----
    costmap: np.ndarray
        二维栅格地图
    origin: tuple
        栅格图左下角坐标 (x, y)
    resolution: float
        分辨率 (m/cell)
    frame_id: str
        坐标系
        
    返回
    -----
    visualization_msgs.msg.Marker
        彩色点云形式的栅格图可视化
    """
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.type = Marker.POINTS
    marker.action = Marker.ADD
    marker.scale.x = resolution
    marker.scale.y = resolution
    
    # 遍历栅格
    for y in range(costmap.shape[0]):
        for x in range(costmap.shape[1]):
            value = costmap[y, x]
            
            # 只处理障碍物区域（值为100），跳过空闲区域（值为0）和未知区域（值为10）
            if value != 100:
                continue
                
            # 计算实际坐标
            real_x = origin[0] + x * resolution
            real_y = origin[1] + y * resolution
            
            # 添加点
            p = Point()
            p.x = real_x
            p.y = real_y
            p.z = 0.01  # 稍微高于地面
            marker.points.append(p)
            
            # 设置障碍物为红色
            marker.colors.append(Marker().color)
            marker.colors[-1].r = 1.0
            marker.colors[-1].g = 0.0
            marker.colors[-1].b = 0.0
            marker.colors[-1].a = 1.0
            
    return marker 