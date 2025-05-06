#!/usr/bin/env python3
import time, os, sys
import numpy as np
import jax
import jax.numpy as jnp
import rclpy
from rclpy.node import Node
import tf_transformations
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from mppi.utils.ros_np_multiarray import to_multiarray_f32, to_numpy_f32
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from mppi.utils.laser_to_points import scan_to_points
from mppi.utils.laser_to_costmap import scan_to_grid, costmap_to_marker

from mppi.infer_env import InferEnv
from mppi.svgmppi_tracking import SVGMPPI
import mppi.utils.utils as utils
from mppi.utils.jax_utils import numpify
import mppi.utils.jax_utils as jax_utils
from mppi.utils.Track import Track
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
jax.config.update("jax_compilation_cache_dir", str(Path.home() / "jax_cache"))

class MPPI_Node(Node):
    def __init__(self):
        super().__init__('lmppi_node')
        self.config = utils.ConfigYAML()
        config_path = Path(get_package_share_directory('mppi')) / 'config' / 'config.yaml'
        self.config.load_file(str(config_path))
        self.config.norm_params = np.array(self.config.norm_params).T
        if self.config.random_seed is None:
            self.config.random_seed = np.random.randint(0, 1e6)
        jrng = jax_utils.oneLineJaxRNG(self.config.random_seed)    
        
        waypoints_dir = Path(get_package_share_directory('mppi')) / 'waypoints'
        map_info_path = waypoints_dir / 'map_info.txt'
        map_info = np.genfromtxt(str(map_info_path), delimiter='|', dtype='str')
        map_dir_str = str(waypoints_dir)
        if not map_dir_str.endswith('/'):
            map_dir_str += '/'
        self.config.map_dir = map_dir_str
        track, self.config = Track.load_map(self.config.map_dir, map_info, self.config.map_ind, self.config)
        # track.waypoints[:, 3] += 0.5 * np.pi
        self.infer_env = InferEnv(track, self.config, DT=self.config.sim_time_step)
        self.mppi = SVGMPPI(self.config, self.infer_env, jrng)

        # Do a dummy call on the MPPI to initialize the variables
        state_c_0 = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.control = np.asarray([0.0, 0.0])
        reference_traj, waypoint_ind = self.infer_env.get_refernece_traj(state_c_0.copy(), self.config.ref_vel, self.config.n_steps)
        self.mppi.update(jnp.asarray(state_c_0), jnp.asarray(reference_traj))
        self.get_logger().info('MPPI initialized')
        
        # 打印MPPI对象的所有属性
        self.get_logger().info("MPPI对象的属性:")
        for attr_name in dir(self.mppi):
            # 跳过私有属性和方法
            if attr_name.startswith('__'):
                continue
            try:
                attr = getattr(self.mppi, attr_name)
                attr_type = type(attr).__name__
                self.get_logger().info(f"  - {attr_name}: {attr_type}")
            except Exception as e:
                self.get_logger().info(f"  - {attr_name}: <无法访问> ({str(e)})")
        
        qos = rclpy.qos.QoSProfile(history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
                                   depth=1,
                                   reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
                                   durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE)
        # create subscribers
        if self.config.is_sim:
            self.pose_sub = self.create_subscription(Odometry, "/ego_racecar/odom", self.pose_callback, qos)
        else:
            self.pose_sub = self.create_subscription(Odometry, "/pf/pose/odom", self.pose_callback, qos)
        # publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", qos)
        self.reference_pub = self.create_publisher(Float32MultiArray, "/reference_arr", qos)
        self.opt_traj_pub = self.create_publisher(Float32MultiArray, "/opt_traj_arr", qos)
        # publisher for sample trajectories
        self.sampled_pub = self.create_publisher(Float32MultiArray, "/sampled_arr", qos)
        # 新增轨迹权重发布器
        self.traj_weights_pub = self.create_publisher(Float32MultiArray, "/traj_weights", qos)
        # publisher for laser points visualization
        self.laser_marker_pub = self.create_publisher(Marker, "/laser_points", qos)
        # publisher for costmap visualization
        self.costmap_marker_pub = self.create_publisher(Marker, "/costmap", qos)
        # subscriber for laser scan
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, qos)

        # 存储最新的栅格地图
        self.costmap = None
        self.costmap_origin = None
        self.costmap_resolution = 0.05

    def pose_callback(self, pose_msg):
        """
        Callback function for subscribing to particle filter's inferred pose.
        This funcion saves the current pose of the car and obtain the goal
        waypoint from the pure pursuit module.

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        """
        pose = pose_msg.pose.pose
        twist = pose_msg.twist.twist

        # Beta calculated by the arctan of the lateral velocity and the longitudinal velocity
        beta = np.arctan2(twist.linear.y, twist.linear.x)

        # For demonstration, let's assume we have these quaternion values
        quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

        # Convert quaternion to Euler angles
        euler = tf_transformations.euler_from_quaternion(quaternion)

        # Extract the Z-angle (yaw)
        theta = euler[2]  # Yaw is the third element

        state_c_0 = np.asarray([
            pose.position.x,
            pose.position.y,
            self.control[0],
            max(twist.linear.x, self.config.init_vel),
            theta,
            twist.angular.z,
            beta,
        ])
        
        # 使用None作为target_speed，让系统使用waypoints上的速度
        reference_traj, waypoint_ind = self.infer_env.get_refernece_traj(state_c_0.copy(), None, self.config.n_steps)

        ## MPPI call
        self.mppi.update(jnp.asarray(state_c_0), jnp.asarray(reference_traj))
        
        # 重新直接计算采样轨迹的权重
        weights = None
        try:
            # 获取采样轨迹和参考轨迹
            sample_trajectories = numpify(self.mppi.states)  # [n_samples, n_steps, state_dim]
            ref_traj = numpify(reference_traj)
            
            # 计算每个轨迹的代价
            all_costs = []
            for i in range(sample_trajectories.shape[0]):
                # 计算轨迹追踪代价
                tracking_cost = -np.linalg.norm(ref_traj[1:, :2] - sample_trajectories[i, :, :2], ord=1, axis=1)
                
                # 计算障碍物代价(如果有costmap)
                obstacle_costs = np.zeros_like(tracking_cost)
                if self.infer_env.costmap is not None:
                    # 转换到车体坐标系
                    traj = sample_trajectories[i]
                    vehicle_x, vehicle_y = state_c_0[0], state_c_0[1]
                    theta = state_c_0[4]
                    
                    # 转换矩阵
                    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
                    rotation_matrix = np.array([
                        [cos_theta, sin_theta],
                        [-sin_theta, cos_theta]
                    ])
                    
                    # 对轨迹点进行坐标转换
                    traj_car_frame = np.zeros_like(traj[:, :2])
                    for j in range(traj.shape[0]):
                        translated = traj[j, :2] - np.array([vehicle_x, vehicle_y])
                        traj_car_frame[j] = np.matmul(rotation_matrix, translated)
                    
                    # 计算栅格坐标
                    grid_xs = np.floor((traj_car_frame[:, 0] - self.infer_env.costmap_origin[0]) / 
                                      self.infer_env.costmap_resolution).astype(np.int32)
                    grid_ys = np.floor((traj_car_frame[:, 1] - self.infer_env.costmap_origin[1]) / 
                                      self.infer_env.costmap_resolution).astype(np.int32)
                    
                    # 获取栅格值
                    for j, (x, y) in enumerate(zip(grid_xs, grid_ys)):
                        if 0 <= x < self.infer_env.costmap.shape[1] and 0 <= y < self.infer_env.costmap.shape[0]:
                            obstacle_costs[j] = -self.infer_env.costmap[y, x] * self.infer_env.obstacle_cost_weight
                        else:
                            obstacle_costs[j] = -10 * self.infer_env.obstacle_cost_weight  # 超出范围的惩罚
                
                # 计算总代价
                total_costs = tracking_cost + obstacle_costs
                all_costs.append(np.mean(total_costs))
            
            # 转换为numpy数组
            all_costs = np.array(all_costs)
            
            # 从代价值计算权重（类似于MPPI中的weights函数）
            # 首先对代价进行标准化
            min_cost = np.min(all_costs)
            max_cost = np.max(all_costs)
            range_cost = max_cost - min_cost + 0.001  # 加小数防止除零
            
            # 使用soft-max将代价转换为权重
            temperature = 0.01  # 与MPPI中的参数保持一致
            normalized_costs = (all_costs - max_cost) / range_cost  # 值范围为[-1, 0]
            weights = np.exp(normalized_costs / temperature)  # 取指数
            weights /= np.sum(weights)  # 归一化
            
            # 打印所有权重信息
            self.get_logger().info(f"计算得到的MPPI权重: {weights}")
            # 打印权重的统计信息
            self.get_logger().info(f"计算得到的MPPI权重 - 最大值: {np.max(weights):.6f}, 最小值: {np.min(weights):.6f}, 平均值: {np.mean(weights):.6f}")
            
            # 打印最大和最小代价
            self.get_logger().info(f"采样轨迹代价 - 最小: {np.min(all_costs):.2f}, 最大: {np.max(all_costs):.2f}, 平均: {np.mean(all_costs):.2f}")
            
            # 发布轨迹权重信息
            if self.traj_weights_pub.get_subscription_count() > 0:
                weights_msg = to_multiarray_f32(weights.astype(np.float32))
                self.traj_weights_pub.publish(weights_msg)
        except Exception as e:
            self.get_logger().error(f"重新计算权重时出错: {e}")
            
        mppi_control = numpify(self.mppi.a_opt[0]) * self.config.norm_params[0, :2]/2
        self.control[0] = float(mppi_control[0]) * self.config.sim_time_step + self.control[0]
        self.control[1] = float(mppi_control[1]) * self.config.sim_time_step + twist.linear.x
        
        # 提取并打印代价信息
        try:
            # 从最优轨迹获取轨迹点
            traj_opt = numpify(self.mppi.traj_opt)
            # 计算并打印代价
            tracking_cost = np.mean(-np.linalg.norm(reference_traj[1:, :2] - traj_opt[:, :2], ord=1, axis=1))
            
            # 如果有costmap，计算障碍代价
            if self.infer_env.costmap is not None:
                # 计算栅格坐标
                # 先将全局坐标系的轨迹点转换到车体坐标系
                # 获取当前车辆位置和朝向
                vehicle_x, vehicle_y = state_c_0[0], state_c_0[1]
                theta = state_c_0[4]  # 车辆朝向角度
                
                # 转换矩阵
                cos_theta, sin_theta = np.cos(theta), np.sin(theta)
                rotation_matrix = np.array([
                    [cos_theta, sin_theta],
                    [-sin_theta, cos_theta]
                ])
                
                # 存储转换后的坐标
                traj_car_frame = np.zeros_like(traj_opt[:, :2])
                
                # 对每个轨迹点进行坐标转换
                for i in range(traj_opt.shape[0]):
                    # 1. 平移变换
                    translated = traj_opt[i, :2] - np.array([vehicle_x, vehicle_y])
                    # 2. 旋转变换
                    traj_car_frame[i] = np.matmul(rotation_matrix, translated)
                
                # 现在使用车体坐标系下的坐标计算栅格索引
                grid_xs = np.floor((traj_car_frame[:, 0] - self.infer_env.costmap_origin[0]) / 
                                  self.infer_env.costmap_resolution).astype(np.int32)
                grid_ys = np.floor((traj_car_frame[:, 1] - self.infer_env.costmap_origin[1]) / 
                                   self.infer_env.costmap_resolution).astype(np.int32)
                
                # 计算并统计不同代价值的点数量
                free_count = 0  # 代价值为0的点（空闲区域）
                unknown_count = 0  # 代价值为10的点（未知区域）
                obstacle_count = 0  # 代价值为100的点（障碍物区域）
                outside_count = 0  # 超出栅格范围的点
                
                for x, y in zip(grid_xs, grid_ys):
                    if 0 <= x < self.infer_env.costmap.shape[1] and 0 <= y < self.infer_env.costmap.shape[0]:
                        cost = self.infer_env.costmap[y, x]
                        if cost == 0:
                            free_count += 1
                        elif cost == 10:
                            unknown_count += 1
                        elif cost == 100:
                            obstacle_count += 1
                    else:
                        outside_count += 1
                
                # 计算总体代价
                costs = []
                for x, y in zip(grid_xs, grid_ys):
                    if 0 <= x < self.infer_env.costmap.shape[1] and 0 <= y < self.infer_env.costmap.shape[0]:
                        costs.append(self.infer_env.costmap[y, x])
                    else:
                        costs.append(10)  # 超出范围的惩罚
                
                obstacle_cost = -np.mean(costs) * self.infer_env.obstacle_cost_weight
                
                # 打印代价信息和点数统计
                self.get_logger().info(f"轨迹统计: 空闲点={free_count}, 未知点={unknown_count}, 障碍点={obstacle_count}, 超出范围={outside_count}")
                self.get_logger().info(f"轨迹追踪代价: {tracking_cost:.2f}, 避障代价: {obstacle_cost:.2f}")
        except Exception as e:
            self.get_logger().error(f"计算代价时出错: {e}")  # 输出具体错误，方便调试
        
        if self.reference_pub.get_subscription_count() > 0:
            ref_traj_cpu = numpify(reference_traj)
            arr_msg = to_multiarray_f32(ref_traj_cpu.astype(np.float32))
            self.reference_pub.publish(arr_msg)

        if self.opt_traj_pub.get_subscription_count() > 0:
            opt_traj_cpu = numpify(self.mppi.traj_opt)
            arr_msg = to_multiarray_f32(opt_traj_cpu.astype(np.float32))
            self.opt_traj_pub.publish(arr_msg)

        # publish sampled trajectories
        if self.sampled_pub.get_subscription_count() > 0:
            sampled_cpu = numpify(self.mppi.states)  # [n_samples, n_steps, state_dim]
            arr_msg = to_multiarray_f32(sampled_cpu.astype(np.float32))
            self.sampled_pub.publish(arr_msg)

        if twist.linear.x < self.config.init_vel:
            self.control = [0.0, self.config.init_vel * 2]

        if np.isnan(self.control).any() or np.isinf(self.control).any():
            self.control = np.array([0.0, 0.0])
            self.mppi.a_opt = np.zeros_like(self.mppi.a_opt)

        # Publish the control command
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = self.control[0]
        drive_msg.drive.speed = self.control[1]
        # self.get_logger().info(f"Steering Angle: {drive_msg.drive.steering_angle}, Speed: {drive_msg.drive.speed}")
        self.drive_pub.publish(drive_msg)
        
    def scan_callback(self, scan_msg):
        # 只在性能监控模式下显示详细日志
        performance_logging = False
        if performance_logging:
            self.get_logger().info(f'接收到激光扫描数据，角度范围: [{scan_msg.angle_min:.2f}, {scan_msg.angle_max:.2f}], 数据点数: {len(scan_msg.ranges)}')
        
        laser_frame = scan_msg.header.frame_id
        
        # 1. 生成前方±20度范围内的点云（约0.35弧度）- 仅用于栅格图生成，不再单独可视化
        angle_range = 0.35  # 约20度
        
        # 2. 生成前方栅格地图
        # 使用较粗的分辨率以提高效率
        resolution = 0.1  # 10cm分辨率
        costmap, origin, res = scan_to_grid(scan_msg, resolution=resolution)
        
        # 保存最新的栅格图
        self.costmap = costmap
        self.costmap_origin = origin
        self.costmap_resolution = res
        
        # 将costmap传递给MPPI控制器的环境
        self.infer_env.update_costmap(costmap, origin, res)
        
        # 仅在有订阅时可视化栅格地图
        if self.costmap_marker_pub.get_subscription_count() > 0:
            costmap_marker = costmap_to_marker(costmap, origin, res, frame_id=laser_frame)
            costmap_marker.header.stamp = self.get_clock().now().to_msg()
            costmap_marker.ns = "costmap"
            costmap_marker.id = 0
            
            if performance_logging:
                self.get_logger().info(f'发布栅格地图，大小: {costmap.shape}, 栅格数: {len(costmap_marker.points)}')
            
            self.costmap_marker_pub.publish(costmap_marker)

def main(args=None):
    rclpy.init(args=args)
    print("MPPI node initialized")
    mppi_node = MPPI_Node()
    rclpy.spin(mppi_node)

    mppi_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

