/********************************************
 * @Author: zjd
 * @Date: 2024-01-11 
 * @LastEditTime: 2024-01-11 
 * @LastEditors: zjd
 ********************************************/
#pragma once
#include "tritonserver_wrapper/tritonserver_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  modelInit

    initial the model context.

    input:
        ModelContext* context           the pointer of context handle.
        const char* model_name          model name.
        int64_t  model version          model version.
    return:
        int                             error code.
*/
TRITONSERVER_API int modelInit(ModelContext* context, const char* model_name, int64_t model_version = -1);


/*  modelDestroy

    destroy model context.

    input:
        ModelContext context        the handle of context.
    return:
        int                         0:success -1:fail.
*/
TRITONSERVER_API void modelDestroy(ModelContext context);


/*  modelQuery

    query the information about model. see ModeQueryCmd.

    input:
        ModelContext context        the handle of context.
        ModelQueryCmd cmd           the command of query.
        void* info                  the buffer point of information.
        uint32_t size               the size of information.
    return:
        int                         success:0, fail:-1
*/
TRITONSERVER_API int modelQuery(ModelContext context, ModeQueryCmd cmd, void* info, uint32_t size);


/*  modelInputsSet

    set inputs information by input index of triton model.
    inputs information see ModelTensor.

    input:
        ModelContext context        the handle of context.
        uint32_t n_inputs           the number of inputs.
        ModelTensor inputs[]        the arrays of inputs information, see ModelTensor.
    return:
        int                         success:0, fail:-1
*/
TRITONSERVER_API int modelInputsSet(ModelContext context, uint32_t n_inputs, ModelTensor inputs[]);


/*  modelRun

    run the model to execute inference.

    input:
        ModelContext context        the handle of context.
        async                       call model async api
    return:
        int                         success:0, fail:-1
*/
TRITONSERVER_API int modelRun(ModelContext context, bool async = false);


/*  modelOutputsGet

    wait the inference to finish and get the outputs.
    the results will set to outputs[].

    input:
        ModelContext context        the handle of context.
        uint32_t n_outputs          the number of outputs.
        ModelTensor outputs[]       the arrays of output, see ModelTensor.
    return:
        int                         success:0, fail:-1
*/
TRITONSERVER_API int modelOutputsGet(ModelContext context, uint32_t n_outputs, ModelTensor outputs[]);


/*  modelOutputsRelease

    release the outputs that get by modelOutputsGet.
    after called, the outputs[x].buf will be free from modelOutputsGet

    input:
        ModelContext context        the handle of context.
        uint32_t n_ouputs           the number of outputs.
        ModelTensor outputs[]       the arrays of output.
    return:
        int                         success:0, fail:-1
*/
TRITONSERVER_API int modelOutputsRelease(ModelContext context, uint32_t n_ouputs, ModelTensor outputs[]);


#ifdef __cplusplus
}
#endif