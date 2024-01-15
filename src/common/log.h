/********************************************
 * @Author: zjd
 * @Date: 2024-01-11 
 * @LastEditTime: 2024-01-11 
 * @LastEditors: zjd
 ********************************************/
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include "spdlog/spdlog.h"

#define TRITONSERVER_LOG_LEVEL_TRACE    SPDLOG_LEVEL_TRACE
#define TRITONSERVER_LOG_LEVEL_DEBUG    SPDLOG_LEVEL_DEBUG
#define TRITONSERVER_LOG_LEVEL_INFO     SPDLOG_LEVEL_INFO
#define TRITONSERVER_LOG_LEVEL_WARN     SPDLOG_LEVEL_WARN
#define TRITONSERVER_LOG_LEVEL_ERROR    SPDLOG_LEVEL_ERROR
#define TRITONSERVER_LOG_LEVEL_FATAL    SPDLOG_LEVEL_CRITICAL
#define TRITONSERVER_LOG_LEVEL_OFF      SPDLOG_LEVEL_OFF

#define TRITONSERVER_LOG_TRACE(...)       SPDLOG_TRACE(__VA_ARGS__)
#define TRITONSERVER_LOG_DEBUG(...)       SPDLOG_DEBUG(__VA_ARGS__)
#define TRITONSERVER_LOG_INFO(...)        SPDLOG_INFO(__VA_ARGS__)
#define TRITONSERVER_LOG_WARN(...)        SPDLOG_WARN(__VA_ARGS__)
#define TRITONSERVER_LOG_ERROR(...)       SPDLOG_ERROR(__VA_ARGS__)
#define TRITONSERVER_LOG_CRITICAL(...)    SPDLOG_CRITICAL(__VA_ARGS__)

#define TRITONSERVER_LOG_IMPL(level, ...)                          \
do {                                                               \
    switch(level)                                                  \
    {                                                              \
        case TRITONSERVER_LOG_LEVEL_TRACE:                         \
            SPDLOG_TRACE(__VA_ARGS__);                             \
            break;                                                 \
        case TRITONSERVER_LOG_LEVEL_DEBUG:                         \
            SPDLOG_DEBUG(__VA_ARGS__);                             \
            break;                                                 \
        case TRITONSERVER_LOG_LEVEL_INFO:                          \
            SPDLOG_INFO(__VA_ARGS__);                              \
            break;                                                 \
        case TRITONSERVER_LOG_LEVEL_WARN:                          \
            SPDLOG_WARN(__VA_ARGS__);                              \
            break;                                                 \
        case TRITONSERVER_LOG_LEVEL_ERROR:                         \
            SPDLOG_ERROR(__VA_ARGS__);                             \
            break;                                                 \
        case TRITONSERVER_LOG_LEVEL_FATAL:                         \
            SPDLOG_CRITICAL(__VA_ARGS__);                          \
            break;                                                 \
        case TRITONSERVER_LOG_LEVEL_OFF:                           \
            break;                                                 \
        default:                                                   \
            SPDLOG_CRITICAL(__VA_ARGS__);                          \
    }                                                              \
} while(0)

#define TRITONSERVER_LOG(level, ...)  TRITONSERVER_LOG_IMPL(level, ##__VA_ARGS__)