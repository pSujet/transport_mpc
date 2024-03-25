import os
from ament_index_python.packages import get_package_share_directory
from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import ExecuteProcess,DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    print("-----------Start Decentralized Controller-----------------")
    package_name='transport_mpc'
    controller_path = os.path.join(get_package_prefix(package_name),'lib',package_name,'start_controller_decen.py')

    agent1_arg = DeclareLaunchArgument(
            'agent1',
            default_value='1',
            description='Description for agent1 argument'
        )
    agent2_arg = DeclareLaunchArgument(
            'agent2',
            default_value='2',
            description='Description for agent2 argument'
        )

    controller1 = ExecuteProcess(
        cmd=['python3',
             controller_path,
             '--agent', LaunchConfiguration('agent1')],
        output='screen'
    )
    controller2 = ExecuteProcess(
        cmd=['python3', controller_path,
             '--agent', LaunchConfiguration('agent2')],
        output='screen'
    )

    return LaunchDescription([
        agent1_arg,
        agent2_arg,
        controller1,
        controller2
    ])