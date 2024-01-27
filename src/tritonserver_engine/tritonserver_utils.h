/********************************************
 * @Author: zjd
 * @Date: 2024-01-11 
 * @LastEditTime: 2024-01-11 
 * @LastEditors: zjd
 ********************************************/
#pragma once
#include "triton/core/tritonserver.h"

namespace TRITON_SERVER
{

    #define FAIL(MSG)                                                            \
    do {                                                                         \
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "error: {}", (MSG));      \
        exit(1);                                                                 \
    } while (false)

    #define FAIL_IF_ERR(X, MSG)                                                  \
    do {                                                                         \
        TRITONSERVER_Error* err__ = (X);                                         \
        if (err__ != nullptr) {                                                  \
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "error: {}: {}-{}",   \
                (MSG), TRITONSERVER_ErrorCodeString(err__),                      \
                TRITONSERVER_ErrorMessage(err__));                               \
            TRITONSERVER_ErrorDelete(err__);                                     \
            exit(1);                                                             \
        }                                                                        \
    } while (false)

    #define LOG_IF_ERR(X, MSG)                                                   \
    do {                                                                         \
        TRITONSERVER_Error* err__ = (X);                                         \
        if (err__ != nullptr) {                                                  \
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "error: {}: {}-{}",   \
                (MSG), TRITONSERVER_ErrorCodeString(err__),                      \
                TRITONSERVER_ErrorMessage(err__));                               \
            TRITONSERVER_ErrorDelete(err__);                                     \
        }                                                                        \
    } while (false)

    #define LOG_AND_RET_IF_ERR(X, MSG)                                           \
    if ((X) != nullptr) {                                                        \
        TRITONSERVER_Error* err__ = (X);                                         \
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "error: {}: {}-{}",       \
            (MSG), TRITONSERVER_ErrorCodeString(err__),                          \
            TRITONSERVER_ErrorMessage(err__));                                   \
        TRITONSERVER_ErrorDelete(err__);                                         \
        return -1;                                                               \
    }

#ifdef TRITON_ENABLE_GPU
    #define FAIL_IF_CUDA_ERR(X, MSG)                                             \
    do {                                                                         \
        cudaError_t err__ = (X);                                                 \
        if (err__ != cudaSuccess) {                                              \
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "error: {}:{}",       \
                (MSG), cudaGetErrorString(err__));                               \
            exit(1);                                                             \
        }                                                                        \
    } while (false)
#endif  // TRITON_ENABLE_GPU

} // namespace TRITON_SERVER