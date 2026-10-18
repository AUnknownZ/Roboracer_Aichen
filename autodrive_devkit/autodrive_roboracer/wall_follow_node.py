"""LiDAR racing controller with smooth steering and measured-speed feedback.

Compare visible turning arcs, follow the clearest smooth path, and accelerate
as curvature allows. The controller uses no map, lap count, or stored poses.
"""
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32


def clip(value, lo, hi):
    return max(lo, min(hi, value))


class ArcPlanner:
    """Compare paths while approximating the car with three overlapping discs.

    Curvature k is in 1/metres: a rear axle follows
    x = sin(k*s)/k, y = (1-cos(k*s))/k after travelling s metres.
    Intersect each footprint disc's circular path with observed wall points.
    """
    def __init__(self, radius=0.17, horizon=6.0, continuity=0.50,
                 margin_weight=0.8, curvature_cost=0.06):
        self.radius = radius
        self.horizon = horizon
        self.continuity = continuity
        self.margin_weight = margin_weight
        self.curvature_cost = curvature_cost
        self.previous = 0.0
        self.candidates = np.arange(-40, 41, dtype=float) * 0.04
        # Three overlapping discs cover the car's length without treating its
        # half-diagonal as its width. The .17 m radius includes side clearance.
        self.body_offsets = np.array([0.02, 0.17, 0.32])

    def choose(self, x, y):
        """Return curvature, clear path length, and minimum wall distance."""
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) < 20:
            return 0.0, 0.0, 0.0
        # NumPy compares every candidate, footprint disc, and obstacle together.
        # A longer horizon anticipates bends before the car reaches turn-in.
        k = self.candidates[:, None, None]
        offsets = self.body_offsets[None, :, None]
        turn_radius = 1.0/np.maximum(np.abs(k), 1e-6)
        body_radius = np.hypot(turn_radius, offsets)
        cy = turn_radius-np.sign(k)*y[None, None, :]
        radial = np.hypot(x[None, None, :], cy)
        phase = np.mod(np.arctan2(x[None, None, :], cy)
                       - np.arctan2(offsets, turn_radius), 2*np.pi)
        progress = turn_radius*phase
        clearance = np.abs(radial-body_radius)
        horizon = np.minimum(self.horizon, np.pi*turn_radius)
        margin = np.min(np.where(progress <= horizon, clearance, 2.0), axis=(1, 2))
        cosine = ((body_radius**2+radial**2-self.radius**2)
                  / (2*body_radius*np.maximum(radial, 1e-9)))
        contact = turn_radius*(phase-np.arccos(np.clip(cosine, -1.0, 1.0)))
        hit = (clearance < self.radius) & (contact >= 0.0)
        distance = np.minimum(horizon[:, 0, 0],
                              np.min(np.where(hit, contact, self.horizon), axis=(1, 2)))

        # Exact zero-curvature limit for the same footprint discs.
        forward = x[None, :]-self.body_offsets[:, None]
        inside = (forward >= 0.0) & (forward <= self.horizon)
        hit = inside & (np.abs(y[None, :]) < self.radius)
        contact = forward-np.sqrt(np.maximum(0.0, self.radius**2-y[None, :]**2))
        mid = len(self.candidates)//2
        distance[mid] = np.min(np.where(hit & (contact >= 0.0), contact, self.horizon))
        margin[mid] = np.min(np.where(inside, np.abs(y[None, :]), 2.0))
        # Reward visible progress and wall clearance; penalize unnecessary
        # curvature and changes from the previous path to prevent chatter.
        scores = (distance+self.margin_weight*np.minimum(margin, 0.7)
                  - self.continuity*np.abs(self.candidates-self.previous)
                  - self.curvature_cost*np.abs(self.candidates))
        best = int(np.argmax(scores))
        self.previous = float(self.candidates[best])
        return self.previous, float(distance[best]), float(margin[best])


class WallFollowNode(Node):
    def __init__(self):
        super().__init__('wall_follow_node')
        # Main knobs: preview distance (m), footprint radius (m), maximum
        # speed (m/s), lateral acceleration (m/s²), and steering smoothing (s).
        defaults = dict(plan_horizon=6.0, footprint_radius=0.17, continuity=0.50,
                        margin_weight=0.8, curvature_cost=0.06,
                        steering_tau=0.025, steering_rate=6.0,
                        max_speed=7.0, lateral_accel=6.0,
                        braking_accel=7.0, acceleration=1.5, speed_gain=0.035,
                        wheelbase=0.324, max_steer=0.5236)
        for name, value in defaults.items():
            self.declare_parameter(name, value)
            setattr(self, name, float(self.get_parameter(name).value))
        self.planner = ArcPlanner(self.footprint_radius, self.plan_horizon,
                                  self.continuity, self.margin_weight, self.curvature_cost)
        prefix = '/autodrive/roboracer_1/'
        self.throttle_pub = self.create_publisher(Float32, prefix+'throttle_command', 1)
        self.steering_pub = self.create_publisher(Float32, prefix+'steering_command', 1)
        self.create_subscription(LaserScan, prefix+'lidar', self.on_scan, 1)
        self.create_subscription(Odometry, prefix+'odom', self.on_odom, 1)
        self.scan = None
        self.last_scan_time = None
        self.last_control = time.monotonic()
        self.last_stamp = None
        self.steering = 0.0
        self.last_throttle = 0.0
        self.speed = 0.0
        self.last_speed_time = None
        self.clearance = 0.0
        self.margin = 0.0
        self.valid = False
        self.curvature = 0.0
        self.target_speed = 0.0
        self.timer = self.create_timer(0.04, self.control_step)
        self.get_logger().info('LiDAR swept-arc planner ready')

    def on_odom(self, msg):
        v = msg.twist.twist.linear
        self.speed = math.hypot(v.x, v.y)
        self.last_speed_time = time.monotonic()

    def on_scan(self, scan):
        self.scan = scan
        self.last_scan_time = time.monotonic()
        stamp = scan.header.stamp.sec+scan.header.stamp.nanosec*1e-9
        if stamp == self.last_stamp:
            return
        dt = clip(stamp-self.last_stamp, 0.005, 0.2) if self.last_stamp is not None else 0.04
        if self.last_stamp is not None and (stamp-self.last_stamp > 0.5 or stamp < self.last_stamp):
            self.planner.previous = 0.0
        self.last_stamp = stamp
        stride = max(1, round(math.radians(1.0)/scan.angle_increment))
        indices = np.arange(0, len(scan.ranges), stride)
        ranges = np.asarray(scan.ranges, dtype=float)[indices]
        angles = scan.angle_min+indices*scan.angle_increment
        valid = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)
        ranges, angles = ranges[valid], angles[valid]
        # Official LiDAR location: 0.2733 m ahead of the rear axle.
        x = ranges*np.cos(angles)+0.2733
        y = ranges*np.sin(angles)
        curvature, self.clearance, self.margin = self.planner.choose(x, y)
        self.valid = len(ranges) >= 20 and self.clearance > 0.0
        # Bicycle geometry converts curvature into normalized steering. Smooth
        # its changes while retaining the simulator's physical steering rate.
        target = clip(math.atan(self.wheelbase*curvature)/self.max_steer, -1.0, 1.0)
        filtered = self.steering+dt/(self.steering_tau+dt)*(target-self.steering)
        change = self.steering_rate*dt
        self.steering += clip(filtered-self.steering, -change, change)
        actual_curve = abs(math.tan(self.max_steer*self.steering)/self.wheelbase)
        self.curvature = max(abs(curvature), actual_curve)

    def front_distance(self):
        """Clearance along the chosen path, including its turn."""
        return self.clearance

    def control_step(self):
        now = time.monotonic()
        dt = clip(now-self.last_control, 0.005, 0.15)
        self.last_control = now
        throttle = 0.0
        fresh_speed = (self.last_speed_time is not None
                       and now-self.last_speed_time < 0.4
                       and math.isfinite(self.speed))
        if (self.last_scan_time is not None and now-self.last_scan_time < 0.3
                and self.valid and fresh_speed):
            # Lateral acceleration is v²*curvature. A steady curve can retain
            # positive throttle; unwinding steering raises its speed target.
            curve_speed = math.sqrt(self.lateral_accel/max(self.curvature, 0.03))
            # v² = 2*a*d gives a braking envelope before the path obstruction.
            distance_speed = math.sqrt(2*self.braking_accel*max(0.0, self.clearance-0.45))
            self.target_speed = min(self.max_speed, curve_speed, distance_speed)
            # Empirical throttle feedforward plus proportional speed feedback.
            # This corrects speed loss in curves without an accumulating I term.
            feedback = self.speed_gain*(self.target_speed-self.speed)
            target = clip(self.target_speed/24.0+feedback, -0.4, 1.0)
            if self.speed > self.target_speed+0.8:
                target = -min(0.35, 0.10*(self.speed-self.target_speed))
            throttle = min(target, self.last_throttle+self.acceleration*dt)
        self.last_throttle = throttle
        self.publish_commands(throttle, self.steering)

    def publish_commands(self, throttle, steering):
        if not math.isfinite(throttle) or not math.isfinite(steering):
            throttle, steering = 0.0, 0.0
        self.throttle_pub.publish(Float32(data=float(clip(throttle, -1, 1))))
        self.steering_pub.publish(Float32(data=float(clip(steering, -1, 1))))


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = WallFollowNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            for _ in range(3):
                node.publish_commands(0.0, 0.0)
                time.sleep(0.05)
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
