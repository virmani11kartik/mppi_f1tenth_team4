#!/usr/bin/env python3
import math
from dataclasses import dataclass, field
import pandas as pd

import cvxpy
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from sensor_msgs.msg import LaserScan
from mpc.utils import nearest_point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from scipy.interpolate import CubicSpline

# TODO CHECK: include needed ROS msg type headers and libraries


@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering speed, acceleration]
    TK: int = 2  # finite time horizon length kinematic

    # ---------------------------------------------------
    # TODO: you may need to tune the following matrices
    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 10.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 10.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([40.0, 40.0, 2.5, 0.0])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, delta, v, yaw, yaw-rate, beta]
    Qfk: list = field(
        default_factory=lambda: np.diag([40.0, 40.0, 2.5, 0.0])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, delta, v, yaw, yaw-rate, beta]
    # ---------------------------------------------------

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 0.03  # dist step [m] kinematic
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 6.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0

class MPC(Node):
    """ 
    Implement Kinematic MPC on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('mpc_node')
        # TODO: create ROS subscribers and publishers
        #       use the MPC as a tracker (similar to pure pursuit)
        self.subscriber = self.create_subscription(
            Odometry,
            # '/pf/pose/odom',  # for real car
            '/ego_racecar/odom',  # for simulator
            self.pose_callback,
            10)
        self.publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.marker_pub = self.create_publisher(MarkerArray, "mpc_markers", 10)
        self.marker_pub1 = self.create_publisher(MarkerArray, "mpc_markers1", 10)
        # TODO: get waypoints here
        # Load waypoints from CSV file
        # waypoints_df = pd.read_csv("map3_levine_waypoints(1).csv")
        # raw_waypoints = waypoints_df[['x', 'y']].values
        raw_waypoints = np.load("waypoints_modified.npy")
        self.waypoints = self.process_waypoints(raw_waypoints)
        self.tracking_points = np.empty((0, 2))
        if self.waypoints.size > 0:
            self.fit_and_resample()

        self.config = mpc_config()
        self.odelta_v = None
        self.odelta = None
        self.oa = None
        self.init_flag = 0

        # initialize MPC problem
        self.mpc_prob_init()
    
    def quaternion_to_yaw(self, quaternion: Quaternion):
        """ Convert quaternion to yaw angle (assuming motion on a 2D plane). """
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y**2 + quaternion.z**2)
        return math.atan2(siny_cosp, cosy_cosp)

    def process_waypoints(self, raw_points):
        """Full waypoint processing pipeline"""
        if raw_points.size == 0:
            return np.empty((0, 2))
        # mask = rdp(raw_points, 
        #          epsilon=self.get_parameter('rdp_epsilon').value,
        #          return_mask=True)
        # simplified = raw_points[np.where(mask)[0]]
        simplified = raw_points
        if not np.allclose(simplified[0], simplified[-1]):
            simplified = np.vstack([simplified, simplified[0]])
            
        return simplified

    def fit_and_resample(self):
        x = self.waypoints[:, 0]
        y = self.waypoints[:, 1]
        # Arc-length parameterization
        dx = np.diff(x)
        dy = np.diff(y)
        dist = np.cumsum(np.hypot(dx, dy))
        dist = np.insert(dist, 0, 0)
        if dist[-1] == 0:  # Handle single-point case
            self.tracking_points = self.waypoints
            return   
        # Spline fitting
        spline_x = CubicSpline(dist/dist[-1], x)
        spline_y = CubicSpline(dist/dist[-1], y)
        # Generate dense spline points
        spline_dist = np.linspace(0, dist[-1], 3000)
        spline_points = np.vstack([spline_x(spline_dist/dist[-1]), 
                                 spline_y(spline_dist/dist[-1])]).T
        # Create tracking points
        self.create_tracking_points(spline_points, spline_dist)

    def create_tracking_points(self, spline_points, spline_dist):
        interval = 0.1
        num_points = int(spline_dist[-1] / interval)
        query_dist = np.linspace(0, spline_dist[-1], num_points)        
        # Linear interpolation of spline points
        self.tracking_points = np.array([
            np.interp(query_dist, spline_dist, spline_points[:, 0]),
            np.interp(query_dist, spline_dist, spline_points[:, 1])
        ]).T
    
    def pose_callback(self, odom_msg):
        pass
        # TODO: extract pose from ROS msg
        vehicle_state = State()
        ###################################################################
        vehicle_state.x = odom_msg.pose.pose.position.x
        vehicle_state.y = odom_msg.pose.pose.position.y
        vehicle_state.yaw = self.quaternion_to_yaw(odom_msg.pose.pose.orientation)
        vehicle_state.v = np.hypot(odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y)
        #self.get_logger().info(f"Pose callback triggered. x={vehicle_state.x:.2f}, v={vehicle_state.v:.2f}")

        ref_x = self.tracking_points[:, 0]
        ref_y = self.tracking_points[:, 1]
        dx = np.gradient(ref_x)
        dy = np.gradient(ref_y)
        ref_yaw = np.arctan2(dy, dx)
        # ref_yaw = np.clip(ref_yaw, -np.pi/2, np.pi/2)
        # ref_yaw = np.unwrap(ref_yaw)  # Smooth discontinuity
        # ref_yaw = np.convolve(ref_yaw, np.ones(5)/5, mode='same')
        dtheta = np.gradient(ref_yaw)
        ds = np.sqrt(dx**2 + dy**2) + 1e-6
        curvature = np.abs(dtheta / ds)
        ref_v = np.clip(3.0 / (1 + curvature + 1e-6), self.config.MIN_SPEED, self.config.MAX_SPEED)
        ref_v = ref_v * 1.8
        # ref_v = np.full_like(ref_x, 1.5)
        ###################################################################
        # TODO: Calculate the next reference trajectory for the next T steps
        #       with current vehicle pose.
        #       ref_x, ref_y, ref_yaw, ref_v are columns of self.waypoints
        ref_path = self.calc_ref_trajectory(vehicle_state, ref_x, ref_y, ref_yaw, ref_v)
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]
        # TODO: solve the MPC control problem
        (
            self.oa,
            self.odelta_v,
            ox,
            oy,
            oyaw,
            ov,
            state_predict,
        ) = self.linear_mpc_control(ref_path, x0, self.oa, self.odelta_v)
        self.publish_mpc_prediction(state_predict)
        self.publish_yaw(ref_path)
        self.publish_path()
        # TODO: publish drive message.
        steer_output = self.odelta_v[0]
        speed_output = vehicle_state.v + self.oa[0] * self.config.DTK
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steer_output
        self.get_logger().info(f"speed={speed_output:.2f}, delta_v[0]={self.odelta_v[0]:.2f}, integrated delta={steer_output:.2f}")
        drive_msg.drive.speed = speed_output
        self.publisher.publish(drive_msg)

    def mpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html
        More QP example in CVXPY here: https://www.cvxpy.org/examples/basic/quadratic_program.html
        """
        # Initialize and create vectors for the optimization problem
        # Vehicle State Vector
        self.xk = cvxpy.Variable(
            (self.config.NXK, self.config.TK + 1)
        )
        # Control Input vector
        self.uk = cvxpy.Variable(
            (self.config.NU, self.config.TK)
        )
        objective = 0.0  # Objective value of the optimization problem
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.Rk] * self.config.TK))

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * (self.config.TK)
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block))

        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # --------------------------------------------------------
        # TODO: fill in the objectives here, you should be using cvxpy.quad_form() somehwhere
        for t in range(self.config.TK):
            objective += cvxpy.quad_form(self.uk[:, t], self.config.Rk)
        # TODO: Objective part 1: Influence of the control inputs: Inputs u multiplied by the penalty R
    
        for t in range(self.config.TK):
            objective += cvxpy.quad_form(self.xk[:, t] - self.ref_traj_k[:, t], self.config.Qk)
        objective += cvxpy.quad_form(self.xk[:, self.config.TK] - self.ref_traj_k[:, self.config.TK], self.config.Qfk)    
        objective += cvxpy.sum_squares(self.xk[3, :] - self.ref_traj_k[3, :]) * 12.0
        # TODO: Objective part 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
        
        for t in range(self.config.TK - 1):
            objective += cvxpy.quad_form(self.uk[:, t + 1] - self.uk[:, t], self.config.Rdk)
        # TODO: Objective part 3: Difference from one control input to the next control input weighted by Rd
        
        # U_vec = cvxpy.vec(self.uk)
        # X_vec = cvxpy.vec(self.xk - self.ref_traj_k)
        # U_diff = cvxpy.vec(self.uk[:, 1:] - self.uk[:, :-1])

        # objective = cvxpy.quad_form(U_vec, R_block) \
        #         + cvxpy.quad_form(X_vec, Q_block) \
        #         + cvxpy.quad_form(U_diff, Rd_block)
        # --------------------------------------------------------

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        A_block = []
        B_block = []
        C_block = []
        # init path to zeros
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], self.odelta_v[0] if self.odelta_v else 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block)).tocoo()
        B_block = block_diag(tuple(B_block)).tocoo()
        C_block = np.array(C_block)

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        # Same as A
        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        # -------------------------------------------------------------
        # TODO: Constraint part 1:
        #       Add dynamics constraints to the optimization problem
        #       This constraint should be based on a few variables:
        #       self.xk, self.Ak_, self.Bk_, self.uk, and self.Ck_
        for t in range(self.config.TK):
            constraints += [self.xk[:, t + 1] == self.Ak_[self.config.NXK * (t+1) - self.config.NXK : self.config.NXK * (t+1),
                                     self.config.NXK * t : self.config.NXK * (t+1)] @ self.xk[:, t]
                          + self.Bk_[self.config.NXK * t : self.config.NXK * (t+1),
                                     self.config.NU * t : self.config.NU * (t+1)] @ self.uk[:, t]
                          + self.Ck_[self.config.NXK * t : self.config.NXK * (t+1)]
            ]

        #       Add constraints on steering, change in steering angle
        #       cannot exceed steering angle speed limit. Should be based on:
        #       self.uk, self.config.MAX_DSTEER, self.config.DTK
        for t in range(1, self.config.TK):
            delta_rate = self.uk[1, t] - self.uk[1, t - 1]
            constraints += [
                delta_rate <= self.config.MAX_DSTEER * self.config.DTK,
                delta_rate >= -self.config.MAX_DSTEER * self.config.DTK
            ]

        # TODO: Constraint part 3:
        #       Add constraints on upper and lower bounds of states and inputs
        #       and initial state constraint, should be based on:
        #       self.xk, self.x0k, self.config.MAX_SPEED, self.config.MIN_SPEED,
        #       self.uk, self.config.MAX_ACCEL, self.config.MAX_STEER
        for t in range(self.config.TK):
            constraints += [
                self.uk[0, t] <= self.config.MAX_ACCEL,
                self.uk[0, t] >= -self.config.MAX_ACCEL,
                self.uk[1, t] <= self.config.MAX_STEER,
                self.uk[1, t] >= -self.config.MAX_STEER,
                self.xk[2, t] <= self.config.MAX_SPEED,
                self.xk[2, t] >= self.config.MIN_SPEED
            ]
        constraints += [self.xk[:, 0] == self.x0k]
        # -------------------------------------------------------------

        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        """
        calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        _, _, _, ind = nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        # travel = abs(state.v) * self.config.DTK
        travel = 0.08
        dind = travel / self.config.dlk
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        #################
        # ind_list = np.mod(ind_list, ncourse)
        #################
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]
        cyaw[cyaw - state.yaw > 4.5] = (
            cyaw[cyaw - state.yaw > 4.5] - (2 * np.pi)
        )
        cyaw[cyaw - state.yaw < -4.5] = (
            cyaw[cyaw - state.yaw < -4.5] + (2 * np.pi)
        )
        ref_traj[3, :] = cyaw[ind_list]

        return ref_traj

    def predict_motion(self, x0, oa, od, xref):
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.config.TK + 1)):
            state = self.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict

    def update_state(self, state, a, delta):

        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= -self.config.MAX_STEER:
            delta = -self.config.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK

        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

        return state

    def get_model_matrix(self, v, phi, delta):
        """
        Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax +Bu + C
        State vector: x=[x, y, v, yaw]
        :param v: speed
        :param phi: heading angle of the vehicle
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """

        # State (or system) matrix A, 4x4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        # Input Matrix B; 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)

        return A, B, C

    def mpc_prob_solve(self, ref_traj, path_predict, x0):
        self.x0k.value = x0

        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block)).tocoo()
        B_block = block_diag(tuple(B_block)).tocoo()
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        if (
            self.MPC_prob.status == cvxpy.OPTIMAL
            or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
        ):
            ox = np.array(self.xk.value[0, :]).flatten()
            oy = np.array(self.xk.value[1, :]).flatten()
            ov = np.array(self.xk.value[2, :]).flatten()
            oyaw = np.array(self.xk.value[3, :]).flatten()
            oa = np.array(self.uk.value[0, :]).flatten()
            odelta = np.array(self.uk.value[1, :]).flatten()

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov

    def linear_mpc_control(self, ref_path, x0, oa, od):
        """
        MPC contorl with updating operational point iteraitvely
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param oa: acceleration of T steps of last time
        :param od: delta of T steps of last time
        """

        if oa is None or od is None:
            oa = [0.0] * self.config.TK
            od = [0.0] * self.config.TK

        # Call the Motion Prediction function: Predict the vehicle motion for x-steps
        path_predict = self.predict_motion(x0, oa, od, ref_path)
        poa, pod = oa[:], od[:]

        # Run the MPC optimization: Create and solve the optimization problem
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(
            ref_path, path_predict, x0
        )

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict
    
    def yaw_to_quaternion(self, yaw):
        return Quaternion(
            x=0.0,
            y=0.0,
            z=np.sin(yaw / 2.0),
            w=np.cos(yaw / 2.0)
        )

    def publish_yaw(self,ref_path):
            """Visualize all path components"""
            marker_array = MarkerArray()

            arrow_length = 0.8  # meters

            x = ref_path[0, :]
            y = ref_path[1, :]
            yaw = ref_path[3, :]

            for i in range(len(x)):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "yaw_oriented_arrows"
                marker.id = i
                marker.type = Marker.ARROW
                marker.action = Marker.ADD

                marker.pose.position = Point(x=float(x[i]), y=float(y[i]), z=0.0)
                marker.pose.orientation = self.yaw_to_quaternion(yaw[i])

                # Arrow dimensions
                marker.scale.x = arrow_length     # shaft length
                marker.scale.y = 0.1             # shaft diameter
                marker.scale.z = 0.1             # head diameter

                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
                marker.lifetime.sec = 0  # keep until manually deleted

                marker_array.markers.append(marker)
        
            
            self.marker_pub.publish(marker_array)   

    def publish_path(self):
        # Original waypoints (red)
        marker_array1 = MarkerArray()
        self.add_marker(marker_array1, self.waypoints, 0, ColorRGBA(r=1.0), 0.3)
        # Tracking points (blue)
        self.add_marker(marker_array1, self.tracking_points, 1, ColorRGBA(b=1.0), 0.08)

        self.marker_pub1.publish(marker_array1)

    def add_marker(self, array, points, id, color, scale):
        """Helper to create point markers"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.POINTS
        marker.id = id
        marker.scale.x = marker.scale.y = scale
        marker.color = color
        marker.color.a = 0.3
        
        for pt in points:
            p = Point(x=float(pt[0]), y=float(pt[1]), z=0.0)
            marker.points.append(p)
        array.markers.append(marker)
    
    def publish_mpc_prediction(self, pred_path):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.LINE_STRIP
        marker.scale.x = 0.1
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        marker.id = 2
        for i in range(pred_path.shape[1]):
            p = Point()
            p.x, p.y = pred_path[0, i], pred_path[1, i]
            marker.points.append(p)
        self.marker_pub.publish(MarkerArray(markers=[marker]))

def main(args=None):
    rclpy.init(args=args)
    print("MPC Initialized")
    mpc_node = MPC()
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
