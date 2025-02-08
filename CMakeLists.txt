cmake_minimum_required(VERSION 3.26)
project(acg_in_gd_lab)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(external/glfw)
add_subdirectory(external/glm)

find_package(Vulkan REQUIRED)

add_executable(acg_in_gd_lab src/main.cpp)
target_link_libraries(acg_in_gd_lab PRIVATE glfw glm::glm ${Vulkan_LIBRARIES})
target_include_directories(acg_in_gd_lab PRIVATE ${Vulkan_INCLUDE_DIR})