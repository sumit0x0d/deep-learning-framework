cmake_minimum_required(VERSION 3.22.1)

project(deep-learning-framework)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_SHARED_LIBRARY_PREFIX "")

# add_compile_options(-Wall -Wextra -g -Werror)

add_subdirectory(../data-structures data-structures EXCLUDE_FROM_ALL)

add_subdirectory(source)

target_include_directories(convolutional-neural-network PUBLIC data-structures/include include)
target_link_libraries(convolutional-neural-network PUBLIC matrix m)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
     add_subdirectory(test)
     target_link_libraries(convolutional-neural-network-test PUBLIC convolutional-neural-network)
endif()
