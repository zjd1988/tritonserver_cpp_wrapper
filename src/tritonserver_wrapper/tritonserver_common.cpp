/********************************************
 * @Author: zjd
 * @Date: 2024-01-11 
 * @LastEditTime: 2024-01-11 
 * @LastEditors: zjd
 ********************************************/
#include "common/log.h"
#include "tritonserver_wrapper/tritonserver_common.h"
#include "tritonserver_engine/tritonserver_engine.h"

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
    if (nullptr == model_repository_path)
    {
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model repo path is nullptr when triton server init");
        return -1;
    }
    ServerConfig config = defaultServerConfig(model_repository_path, log_verbose_level);
    return initTritonServerWithCustom(&config);
}

TRITONSERVER_API int tritonServerLoadModel(const char* model_name)
{
    if (nullptr == model_name)
    {
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model name is nullptr when triton server try to load model");
        return -1;
    }
    std::string model_name_str = model_name;
    return TRITON_SERVER_LOAD_MODEL(model_name_str);
}

TRITONSERVER_API int tritonServerUnloadModel(const char* model_name)
{
    if (nullptr == model_name)
    {
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model name is nullptr when triton server try to unload model");
        return -1;
    }
    std::string model_name_str = model_name;
    return TRITON_SERVER_UNLOAD_MODEL(model_name_str);
}

TRITONSERVER_API void uninitTritonServer()
{
    TRITON_SERVER_UNINIT();
    return;
}