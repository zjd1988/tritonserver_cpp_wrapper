cmake_minimum_required(VERSION 3.4.1)
project(tritonserver_wrapper)
set(CMAKE_CXX_STANDARD 17)

option(BUILD_TOOLS "enable build tools" ON)
option(BUILD_WITH_BACKWARD "enable with backward" ON)

# rapidjson
set(RAPIDJSON_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/rapidjson/include)
# spdlog
set(SPDLOG_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/spdlog/include)
# cxxopts
set(CXXOPTS_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/cxxopts/include)
# filesystem
set(FILESYSTEM_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/filesystem/include)
# stb
set(STB_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/stb/include)
# libnpy
set(LIBNPY_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/libnpy/include)

# backward
if (BUILD_WITH_BACKWARD)
    add_definitions(-DUSE_BACKWARD)
    add_definitions(-DBACKWARD_HAS_DW=1)
    set(BACKWARD_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/backward/include)
    set(BACKWARD_LIBS dw)
endif()


# tritonserver
if(TRITONSERVER_INSTALL_DIR)
    set(TRITONSERVER_INCLUDE_PATH ${TRITONSERVER_INSTALL_DIR}/include)
    set(TRITONSERVER_LIB_PATH ${TRITONSERVER_INSTALL_DIR}/lib)
else()
    set(TRITONSERVER_INCLUDE_PATH /opt/tritonserver/include)
    set(TRITONSERVER_LIB_PATH /opt/tritonserver/lib)
endif()

set(PROJECT_INCLUDE_PATH ${RAPIDJSON_INCLUDE_PATH} 
    ${SPDLOG_INCLUDE_PATH}
    ${CXXOPTS_INCLUDE_PATH}
    ${FILESYSTEM_INCLUDE_PATH}
    ${STB_INCLUDE_PATH}
    ${LIBNPY_INCLUDE_PATH}
    ${BACKWARD_INCLUDE_PATH}
    ${TRITONSERVER_INCLUDE_PATH})

set(PROJECT_LIB_PATH ${TRITONSERVER_LIB_PATH})

file(GLOB_RECURSE TRITONSERVER_WRAPPER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_library(tritonserver_wrapper SHARED ${TRITONSERVER_WRAPPER_SRC})
target_include_directories(tritonserver_wrapper PRIVATE 
    ${PROJECT_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_directories(tritonserver_wrapper PRIVATE ${PROJECT_LIB_PATH})
target_link_libraries(tritonserver_wrapper PRIVATE tritonserver)

if(BUILD_TOOLS)
    # common tools src
    set(COMMON_TOOLS_SRC 
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/pystring.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/tool_utils.cpp)
    
    # model infer tool
    set(MODEL_INFER_TOOL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/tools/model_infer.cpp)
    add_executable(model_infer ${MODEL_INFER_TOOL_SRC} ${COMMON_TOOLS_SRC})
    target_include_directories(model_infer PRIVATE 
        ${PROJECT_INCLUDE_PATH}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/tools)
    target_link_libraries(model_infer PRIVATE tritonserver_wrapper ${BACKWARD_LIBS})
endif()