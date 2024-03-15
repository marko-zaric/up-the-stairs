cmake_minimum_required(VERSION 3.0.2)
project(up_the_stairs)

find_package(catkin
  REQUIRED COMPONENTS
    gazebo_ros
    gazebo_ros_control
    rospy
    roscpp
    tf2
    tf2_ros
    message_generation
)

find_package(catkin REQUIRED genmsg)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  Jump.srv
  Rollout.srv
  RolloutGaussian.srv
  BackToStart.srv
  SetStart.srv
  ActionPrototypeGen.srv
)


# Depend on system install of Gazebo
find_package(GAZEBO REQUIRED)
find_package(nav_msgs)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")



include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

catkin_python_setup()

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
)

catkin_package(
 CATKIN_DEPENDS
  gazebo_ros_control
  rospy
)

catkin_install_python(PROGRAMS
scripts/jump_node.py
scripts/robot_tf2_broadcast.py
scripts/collect_data.py
scripts/backtostart.py
scripts/get_robot_info.py
scripts/set_start_position.py
scripts/record_trajectories.py
scripts/test_wrench_angle.py
scripts/action_prototype_gen.py
scripts/pub_rviz_stairs.py
rl/dqn.py
rl/dqn_random.py
rl/sac.py
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
