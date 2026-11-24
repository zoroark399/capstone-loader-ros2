import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import numpy as np
import cv2
from cv_bridge import CvBridge
import message_filters
from ultralytics import YOLO

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 - registers PointStamped with tf2


class OreLocalizerNode(Node):
    def __init__(self):
        super().__init__('ore_localizer_node')

        # Parameters (override these on the Jetson if topic names or model change)
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('target_class', 'bottle')  # Change to 'ore_pile' after retraining
        self.declare_parameter('conf_threshold', 0.4)
        self.declare_parameter('sample_y_ratio', 0.5)     # 0.5 = box center, 0.8 = bottom/toe of pile
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('target_frame', 'base_link')

        model_path = self.get_parameter('model_path').value
        self.target_class = self.get_parameter('target_class').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.sample_y_ratio = self.get_parameter('sample_y_ratio').value
        self.target_frame = self.get_parameter('target_frame').value

        self.get_logger().info(f'Loading YOLO model: {model_path} (target: {self.target_class})')
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        # Ensure OpenCV 4 constants exist in cv_bridge for OpenCV 5 compatibility
        for cv_val, name in [(0, '8UC1'), (8, '8UC2'), (16, '8UC3'), (24, '8UC4'),
                             (1, '8SC1'), (9, '8SC2'), (17, '8SC3'), (25, '8SC4'),
                             (2, '16UC1'), (10, '16UC2'), (18, '16UC3'), (26, '16UC4'),
                             (3, '16SC1'), (11, '16SC2'), (19, '16SC3'), (27, '16SC4'),
                             (4, '32SC1'), (12, '32SC2'), (20, '32SC3'), (28, '32SC4'),
                             (5, '32FC1'), (13, '32FC2'), (21, '32FC3'), (29, '32FC4'),
                             (6, '64FC1'), (14, '64FC2'), (22, '64FC3'), (30, '64FC4')]:
            self.bridge.cvtype_to_name.setdefault(cv_val, name)

        # Camera intrinsics (populated by camera_info callback)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publisher for the 3D target point
        self.target_pub = self.create_publisher(PointStamped, '/ore_pile/target_point', 10)
        self.debug_pub = self.create_publisher(Image, '/ore_pile/debug_image', 10)
        # Subscribe to camera_info
        self.info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self.camera_info_callback,
            10
        )

        # Synchronized RGB + Aligned Depth subscribers
        rgb_sub = message_filters.Subscriber(self, Image, self.get_parameter('rgb_topic').value)
        depth_sub = message_filters.Subscriber(self, Image, self.get_parameter('depth_topic').value)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=1, slop=0.1
        )
        self.sync.registerCallback(self.synced_callback)
        self.get_logger().info('OreLocalizerNode ready and waiting for frames...')

    def camera_info_callback(self, msg: CameraInfo):
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info(
                f'Camera intrinsics locked: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}'
            )

    def synced_callback(self, rgb_msg: Image, depth_msg: Image):
        if self.fx is None:
            return

        color_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        results = self.model(color_img, conf=self.conf_threshold, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return

        # Pick highest-confidence detection matching target_class
        best_box = None
        best_conf = -1.0
        for box in boxes:
            cls_name = self.model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if cls_name == self.target_class and conf > best_conf:
                best_conf = conf
                best_box = box

        if best_box is None:
            return

        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
        u = int((x1 + x2) / 2.0)
        v = int(y1 + (y2 - y1) * self.sample_y_ratio)

        # Sample a 7x7 patch around (u, v) and take the median valid depth
        h, w = depth_img.shape[:2]
        patch = depth_img[max(0, v - 3):min(h, v + 4), max(0, u - 3):min(w, u + 4)]
        valid_depths = patch[(patch > 0) & np.isfinite(patch)]
        if valid_depths.size == 0:
            return

        depth_val = float(np.median(valid_depths))
        # RealSense Z16 depth is in millimeters; convert to meters
        z_m = depth_val / 1000.0 if depth_img.dtype == np.uint16 else depth_val

        # Deproject 2D pixel -> 3D point in camera_color_optical_frame
        x_m = (u - self.cx) * z_m / self.fx
        y_m = (v - self.cy) * z_m / self.fy

        point_cam = PointStamped()
        point_cam.header = rgb_msg.header
        point_cam.point.x = x_m
        point_cam.point.y = y_m
        point_cam.point.z = z_m

        # Transform to base_link (falls back to camera frame if TF is not published yet)
        try:
            point_out = self.tf_buffer.transform(
                point_cam, self.target_frame, timeout=Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(
                f'TF transform to {self.target_frame} failed ({e}); publishing in {rgb_msg.header.frame_id}',
                throttle_duration_sec=5.0
            )
            point_out = point_cam

        self.target_pub.publish(point_out)

        # Draw bounding box, depth-sampling dot, and 3D coordinates
        cv2.rectangle(color_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(color_img, (u, v), 5, (0, 0, 255), -1)
        label = f'{self.target_class} ({best_conf:.2f}) X:{point_out.point.x:.2f}m Y:{point_out.point.y:.2f}m'
        cv2.putText(color_img, label, (int(x1), max(20, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        debug_msg = self.bridge.cv2_to_imgmsg(color_img, encoding='bgr8')
        debug_msg.header = rgb_msg.header
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = OreLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
