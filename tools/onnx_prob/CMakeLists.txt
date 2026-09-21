cmake_minimum_required(VERSION 3.10)
project(g1_tools)
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_PREFIX_PATH /opt/nb/cpp)
find_package(ZLIB REQUIRED)

set(ORT /opt/nb/unitree_rl_mjlab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0)
list(APPEND CMAKE_PREFIX_PATH /home/jupyter/local)
find_package(unitree_sdk2 REQUIRED)

add_executable(onnx_probe onnx_probe.cpp)
target_include_directories(onnx_probe PRIVATE ${ORT}/include)
target_link_libraries(onnx_probe ${ORT}/lib/libonnxruntime.so.1)
set_target_properties(onnx_probe PROPERTIES BUILD_RPATH ${ORT}/lib)

add_executable(obs_probe obs_probe.cpp)
target_link_libraries(obs_probe unitree_sdk2)
target_include_directories(obs_probe PRIVATE /opt/nb/cpp/include/eigen3)

add_executable(policy_dry policy_dry.cpp)
target_include_directories(policy_dry PRIVATE ${ORT}/include /opt/nb/cpp/include/eigen3)
target_link_libraries(policy_dry unitree_sdk2 ${ORT}/lib/libonnxruntime.so.1)
set_target_properties(policy_dry PROPERTIES BUILD_RPATH ${ORT}/lib)

add_executable(policy_run policy_run.cpp)
target_include_directories(policy_run PRIVATE ${ORT}/include /opt/nb/cpp/include/eigen3)
target_link_libraries(policy_run unitree_sdk2 ${ORT}/lib/libonnxruntime.so.1)
set_target_properties(policy_run PROPERTIES BUILD_RPATH ${ORT}/lib)

add_executable(motion_probe motion_probe.cpp /opt/nb/unitree_g1_aivillage/thirdparty/cnpy/cnpy.cpp)
target_include_directories(motion_probe PRIVATE /opt/nb/unitree_g1_aivillage/thirdparty/cnpy)
target_link_libraries(motion_probe ZLIB::ZLIB)