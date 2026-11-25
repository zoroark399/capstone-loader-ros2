import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments so you can easily swap models/classes without editing code
    model_path_arg = DeclareLaunchArgument(
        'model_path', default_value='yolov8n.pt',
        description='Path to YOLO weights (.pt or .engine)'
    )
    target_class_arg = DeclareLaunchArgument(
        'target_class', default_value='bottle',
        description='Target class name (change to ore_pile after retraining)'
    )
    sample_y_ratio_arg = DeclareLaunchArgument(
        'sample_y_ratio', default_value='0.5',
        description='0.5 = box center, 0.8 = bottom/toe of pile'
    )

    # 1. Intel RealSense D435i Driver (with depth alignment enabled)
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ),
        launch_arguments={'align_depth.enable': 'true'}.items()
    )

    # 2. Temporary Static TF (base_link -> camera_link)
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_static_tf',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'base_link', '--child-frame-id', 'camera_link']
    )

    # 3. Your 3D Target Localization Node
    localizer_node = Node(
        package='ore_perception',
        executable='ore_localizer',
        name='ore_localizer_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'target_class': LaunchConfiguration('target_class'),
            'sample_y_ratio': LaunchConfiguration('sample_y_ratio'),
        }]
    )

    return LaunchDescription([
        model_path_arg,
        target_class_arg,
        sample_y_ratio_arg,
        realsense_launch,
        static_tf_node,
        localizer_node,
    ])
