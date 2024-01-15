cmake_minimum_required(VERSION 3.4.1)
project(tritonserver_wrapper)

set(RAPIDJSON_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/rapidjson/include)
set(SPDLOG_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/spdlog/include)
if(TRITONSERVER_INSTALL_DIR)
    set(TRITONSERVER_INCLUDE_PATH ${TRITONSERVER_INSTALL_DIR}/include)
    set(TRITONSERVER_LIB_PATH ${TRITONSERVER_INSTALL_DIR}/lib)
else()
    set(TRITONSERVER_INCLUDE_PATH /opt/tritonserver)
    set(TRITONSERVER_LIB_PATH /opt/tritonserver/lib)
endif()

set(PROJECT_INCLUDE_PATH ${RAPIDJSON_INCLUDE_PATH} 
    ${SPDLOG_INCLUDE_PATH}
    ${TRITONSERVER_INCLUDE_PATH})

set(PROJECT_LIB_PATH ${TRITONSERVER_LIB_PATH})

file(GLOB_RECURSE TRITONSERVER_WRAPPER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_library(tritonserver_wrapper SHARED ${TRITONSERVER_WRAPPER_SRC})
target_include_directories(tritonserver_wrapper PRIVATE 
    ${PROJECT_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_directories(tritonserver_wrapper PRIVATE ${PROJECT_LIB_PATH})
target_link_libraries(tritonserver_wrapper PRIVATE tritonserver)
