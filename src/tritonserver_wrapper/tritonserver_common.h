/********************************************
 * @Author: zjd
 * @Date: 2024-01-11 
 * @LastEditTime: 2024-01-11 
 * @LastEditors: zjd
 ********************************************/
#pragma once
#include <stdint.h>

#ifndef TRITONSERVER_EXPORT
#ifdef _MSC_VER
#define TRITONSERVER_EXPORT __declspec(dllexport)
#else
#define TRITONSERVER_EXPORT __attribute__((visibility("default")))
#endif
#endif

#ifndef TRITONSERVER_API
#ifdef TRITONSERVER_API_EXPORTS
#define TRITONSERVER_API TRITONSERVER_EXPORT
#else
#define TRITONSERVER_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TRITON_TENSOR_MAX_DIM_NUM 8
#define TRITON_MAX_INPUT_OUTPUT_NUM 8
#define TRITON_MAX_NAME_LENGTH 127
#define TRITON_MAX_NAME_LENGTH_PLUS_ONE (TRITON_MAX_NAME_LENGTH + 1)

typedef void* ModelContext;

typedef enum ModelPlatformType
{
    TENSORFLOW_GRAPHDEF_PLATFORM_TYPE,
    TENSORFLOW_SAVEDMODEL_PLATFORM_TYPE,
    TENSORRT_PLAN_PLATFORM_TYPE,
    ONNRUNTIME_ONNX_PLATFORM_TYPE,
    PYTORCH_LIBTORCH_PLATFORM_TYPE,
    PYTHON_PLATFORM_TYPE,
    ENSEMBLE_PLATFORM_TYPE,
    RKNN_PLATFORM_TYPE,
    INVALID_PLATFORM_TYPE,
} ModelPlatformType;

// same with TRITONSERVER_DataType
//
// tensor data types
//
typedef enum TensorDataType
{
    TENSOR_TYPE_INVALID, 
    TENSOR_TYPE_BOOL, 
    TENSOR_TYPE_UINT8, 
    TENSOR_TYPE_UINT16, 
    TENSOR_TYPE_UINT32, 
    TENSOR_TYPE_UINT64, 
    TENSOR_TYPE_INT8, 
    TENSOR_TYPE_INT16, 
    TENSOR_TYPE_INT32, 
    TENSOR_TYPE_INT64, 
    TENSOR_TYPE_FP16, 
    TENSOR_TYPE_FP32, 
    TENSOR_TYPE_FP64, 
    TENSOR_TYPE_BYTES, 
    TENSOR_TYPE_BF16
} TensorDataType;

//
// model tensor attr
//
typedef struct ModelTensorAttr
{
    uint32_t                                index;
    char                                    name[TRITON_MAX_NAME_LENGTH_PLUS_ONE];
    TensorDataType                          type;
    uint32_t                                num_dim;
    int64_t                                 dims[TRITON_TENSOR_MAX_DIM_NUM];
} ModelTensorAttr;

//
// the information for MODEL_QUERY_INPUT_ATTR or MODEL_QUERY_OUTPUT_ATTR
//
typedef struct ModelTensor 
{
    uint32_t index;                                     /* the tensor index. */
    void* buf;                                          /* the buf for index. */
    uint32_t size;                                      /* the size of input buf. */
    TensorDataType type;                                /* the data type buf. */
} ModelTensor;

//
// the information for MODEL_QUERY_IN_OUT_NUM
//
typedef struct ModelInputOutputNum
{
    uint32_t n_input;                                   /* the number of input. */
    uint32_t n_output;                                  /* the number of output. */
} ModelInputOutputNum;

//
// the query command for modelQuery
//
typedef enum ModeQueryCmd
{
    MODEL_QUERY_IN_OUT_NUM = 0,                          /* query the number of input & output tensor. */
    MODEL_QUERY_INPUT_ATTR,                              /* query the attribute of input tensor. */
    MODEL_QUERY_OUTPUT_ATTR,                             /* query the attribute of output tensor. */
    MODEL_QUERY_PLATFORM_TYPE,                           /* query the deploy platform type of model. */

    MODEL_QUERY_CMD_MAX
} ModeQueryCmd;

//
// tirton server model control enum
//
typedef enum ServerModelControl
{
    SERVER_MODEL_CONTROL_NONE,
    SERVER_MODEL_CONTROL_POLL,
    SERVER_MODEL_CONTROL_EXPLICIT
} ServerModelControl;

//
// tirton server log verbose level enum
//
// "log_verbose_level" : a $number parameter that controls whether the 
// Triton server outputs verbose messages of varying degrees. This value 
// can be any integer >= 0. If "log_verbose_level" is 0, verbose logging 
// will be disabled, and no verbose messages will be output by the Triton 
// server. If "log_verbose_level" is 1, level 1 verbose messages will be 
// output by the Triton server. If "log_verbose_level" is 2, the Triton 
// server will output all verbose messages of level <= 2, etc. Attempting 
// to set "log_verbose_level" to a number < 0 will result in an error.
typedef enum LogVerboseLevel
{
    LOG_VERBOSE_LEVEL_0 = 0,
    LOG_VERBOSE_LEVEL_1 = 1,
    LOG_VERBOSE_LEVEL_2 = 2,
} LogVerboseLevel;

//
// tirton server log format
// LOG_FORMAT_DEFAULT: the log severity (L) and timestamp will be
// logged as "LMMDD hh:mm:ss.ssssss".
//
// LOG_FORMAT_ISO8601: the log format will be "YYYY-MM-DDThh:mm:ssZ L".
//
typedef enum LogFormat
{
    LOG_FORMAT_DEFAULT = 0,
    LOG_FORMAT_ISO8601 = 1,
} LogFormat;

//
// tirton server config info
//
typedef struct ServerConfig
{
    const char*                             model_repository_dir;     // model repository dir
    LogVerboseLevel                         log_verbose_level;        // log verbose level
    LogFormat                               log_format;               // log format
    const char*                             log_file_path;            // absolute log file path
    ServerModelControl                      model_control;            // model control, NONE/POLL/EXPLICIT
    bool                                    strict_model;             // strict config model
    const char*                             backend_dir;              // triton server backends dir
    const char*                             repo_agent_dir;           // triton server repo agent dir
    int                                     check_timeout;            // triton server check ready timeout
} ServerConfig;

inline static const char* getTypeString(TensorDataType type)
{
    switch(type)
    {
        case TENSOR_TYPE_INVALID: return "INVALID";
        case TENSOR_TYPE_BOOL: return "BOOL";
        case TENSOR_TYPE_UINT8: return "UINT8";
        case TENSOR_TYPE_UINT16: return "UINT16";
        case TENSOR_TYPE_UINT32: return "UINT32";
        case TENSOR_TYPE_UINT64: return "UINT64";
        case TENSOR_TYPE_INT8: return "INT8";
        case TENSOR_TYPE_INT16: return "INT16";
        case TENSOR_TYPE_INT32: return "INT32";
        case TENSOR_TYPE_INT64: return "INT64";
        case TENSOR_TYPE_FP16: return "FP16";
        case TENSOR_TYPE_FP32: return "FP32";
        case TENSOR_TYPE_FP64: return "FP64";
        case TENSOR_TYPE_BYTES: return "BYTES";
        case TENSOR_TYPE_BF16: return "BF16";
        default: return "UNKNOW";
    }
}

inline static const char* getModelPlatformTypeString(ModelPlatformType type)
{
    switch(type)
    {
        case TENSORFLOW_GRAPHDEF_PLATFORM_TYPE: return "tensorflow_graphdef";
        case TENSORFLOW_SAVEDMODEL_PLATFORM_TYPE: return "tensorflow_savedmodel";
        case TENSORRT_PLAN_PLATFORM_TYPE: return "tensorrt_plan";
        case ONNRUNTIME_ONNX_PLATFORM_TYPE: return "onnxruntime_onnx";
        case PYTORCH_LIBTORCH_PLATFORM_TYPE: return "pytorch_libtorch";
        case PYTHON_PLATFORM_TYPE: return "python";
        case ENSEMBLE_PLATFORM_TYPE: return "ensemble";
        case RKNN_PLATFORM_TYPE: return "rknn";
        case INVALID_PLATFORM_TYPE: return "invalid";
        default: return "unknown";
    }
}

TRITONSERVER_API ServerConfig defaultServerConfig(const char* model_repository_path, 
    LogVerboseLevel log_verbose_level = LOG_VERBOSE_LEVEL_0, const char* log_file = "", 
    const char* backend_dir = "", const char* repo_agent_dir = "/opt/tritonserver/repoagents/");

TRITONSERVER_API int initTritonServerWithCustom(const ServerConfig* config);

TRITONSERVER_API int initTritonServerWithDefault(const char* model_repository_path, 
    LogVerboseLevel log_verbose_level = LOG_VERBOSE_LEVEL_0);

TRITONSERVER_API int tritonServerLoadModel(const char* model_name);

TRITONSERVER_API int tritonServerUnloadModel(const char* model_name);

TRITONSERVER_API void uninitTritonServer();

#ifdef __cplusplus
}
#endif