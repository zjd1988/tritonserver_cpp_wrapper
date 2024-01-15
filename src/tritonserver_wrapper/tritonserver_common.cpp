/********************************************
 * @Author: zjd
 * @Date: 2024-01-11 
 * @LastEditTime: 2024-01-11 
 * @LastEditors: zjd
 ********************************************/
#include "tritonserver_wrapper/tritonserver_common.h"
#include "tritonserver_infer/tritonserver_infer.h"

TRITONSERVER_API ServerConfig defaultServerConfig(const char* model_repository_path, LogVerboseLevel log_verbose_level, 
    const char* log_file, const char* backend_dir, const char* repo_agent_dir)
{
    ServerConfig config = {
        .model_repository_dir = model_repository_path,
        .log_verbose_level = log_verbose_level,
        .log_format = LOG_FORMAT_DEFAULT,
        .log_file_path = log_file,
        .model_control = SERVER_MODEL_CONTROL_NONE,
        .strict_model = true,
        .backend_dir = backend_dir,
        .repo_agent_dir = repo_agent_dir,
        .check_timeout = 500,
    };
    return config;
}

TRITONSERVER_API int initTritonServerWithCustom(const ServerConfig* config)
{
    return TRITON_SERVER_INIT(config);
}

TRITONSERVER_API int initTritonServerWithDefault(const char* model_repository_path, LogVerboseLevel log_verbose_level)
{
    ServerConfig config = defaultServerConfig(model_repository_path, log_verbose_level);
    return initTritonServerWithCustom(&config);
}

TRITONSERVER_API void uninitTritonServer()
{
    TRITON_SERVER_UNINIT();
}