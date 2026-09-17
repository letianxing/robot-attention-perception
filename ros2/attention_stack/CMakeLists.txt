cmake_minimum_required(VERSION 3.8)
project(attention_stack)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(hri_msgs REQUIRED)
find_package(perception_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(attention_stack_node
  src/attention_core.cpp
  src/attention_stack_node.cpp
)
target_compile_features(attention_stack_node PUBLIC cxx_std_17)
target_include_directories(attention_stack_node PRIVATE include)
ament_target_dependencies(attention_stack_node
  geometry_msgs hri_msgs perception_interfaces rclcpp std_msgs
)

install(TARGETS attention_stack_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/pyhri_people_adapter.py
  DESTINATION lib/${PROJECT_NAME})
ament_package()
