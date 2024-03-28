import os
from ament_index_python.packages import get_package_share_directory
from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import ExecuteProcess,DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    print("-----------Start Decentralized Controller-----------------")
    package_name='transport_mpc'
    controller_path = os.path.join(get_package_prefix(package_name),'lib',package_name,'start_controller.py')

    controller1 = ExecuteProcess(
        cmd=['python3',
             controller_path],
        output='screen'
    )

    return LaunchDescription([
        controller1
    ])