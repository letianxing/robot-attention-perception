cmake_minimum_required(VERSION 3.16)
project(robot_attention_perception LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(robot_attention_core
  src/async_fusion.cpp
  src/fusion.cpp
)

target_include_directories(robot_attention_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_executable(robot_attention_cli src/attention_cli.cpp)
target_link_libraries(robot_attention_cli PRIVATE robot_attention_core)

# Swappable attention algorithms. Each one is a standalone shared object behind
# the C ABI in include/robot_attention_perception/attention_plugin_abi.h, so the
# console can switch algorithms without relinking the perception host.
set(ATTENTION_PLUGIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/plugins)

function(add_attention_plugin name source)
  add_library(${name} MODULE ${source})
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins)
  set_target_properties(${name} PROPERTIES
    PREFIX "lib"
    SUFFIX ".so"
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${ATTENTION_PLUGIN_DIR})
  install(TARGETS ${name} LIBRARY DESTINATION lib/attention_plugins)
endfunction()

add_attention_plugin(attention_av_memory_language_v1 src/plugins/av_memory_language_v1.cpp)
add_attention_plugin(attention_av_memory_language_v2 src/plugins/av_memory_language_v2.cpp)

enable_testing()
add_executable(robot_attention_core_test tests/cpp_attention_test.cpp)
target_link_libraries(robot_attention_core_test PRIVATE robot_attention_core)
add_test(NAME robot_attention_core_test COMMAND robot_attention_core_test)

# The same behavioural checks run against every algorithm: an alternative
# competition layer may change timing, but not what the documented cases mean.
foreach(variant v1 v2)
  add_executable(attention_plugin_${variant}_test
    tests/cpp_attention_plugin_test.cpp src/plugins/av_memory_language_${variant}.cpp)
  target_include_directories(attention_plugin_${variant}_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins)
  add_test(NAME attention_plugin_${variant}_test COMMAND attention_plugin_${variant}_test)
endforeach()

add_executable(attention_surround_test tests/cpp_attention_surround_test.cpp src/plugins/av_memory_language_v2.cpp)
target_include_directories(attention_surround_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins)
add_test(NAME attention_surround_test COMMAND attention_surround_test)

install(TARGETS robot_attention_core robot_attention_cli
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
