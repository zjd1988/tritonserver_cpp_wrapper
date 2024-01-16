/********************************************
 * @Author: zjd
 * @Date: 2024-01-11 
 * @LastEditTime: 2024-01-11 
 * @LastEditors: zjd
 ********************************************/
#include <memory>
#include "common/log.h"
#include "tritonserver_wrapper/tritonserver_c_wrapper.h"
#include "tritonserver_wrapper/tritonserver_cpp_wrapper.h"
using namespace TRITON_SERVER;

TRITONSERVER_API int initModel(ModelContext* context, const char* model_name, int64_t model_version)
{
    *context = nullptr;
    std::unique_ptr<TritonModel> model_inst(new TritonModel(model_name, model_version));
    if (nullptr == model_inst.get() || model_inst->status())
        return -1;
    *context = (ModelContext)model_inst.release();
    return 0;
}

TRITONSERVER_API void modelDestroy(ModelContext context)
{
    if (nullptr == context)
        return;
    TritonModel* model_inst = (TritonModel*)context;
    delete model_inst;
    return 0;
}

TRITONSERVER_API int modelQuery(ModelContext context, ModeQueryCmd cmd, void* info, uint32_t size)
{
    if (nullptr == context)
        return -1;
    TritonModel* model_inst = (TritonModel*)context;
    return model_inst->query(cmd, info, size);
}

TRITONSERVER_API int modelInputsSet(ModelContext context, uint32_t n_inputs, ModelTensor inputs[])
{
    if (nullptr == context)
        return -1;
    TritonModel* model_inst = (TritonModel*)context;
    return model_inst->inputsSet(n_inputs, inputs);
}

TRITONSERVER_API int modelRun(ModelContext context)
{
    if (nullptr == context)
        return -1;
    TritonModel* model_inst = (TritonModel*)context;
    return  model_inst->run();
}

TRITONSERVER_API int modelOutputsGet(ModelContext context, uint32_t n_outputs, ModelTensor outputs[])
{
    if (nullptr == context)
        return -1;
    TritonModel* model_inst = (TritonModel*)context;
    return model_inst->outputsGet(n_outputs, outputs);
}

TRITONSERVER_API int modelOutputsRelease(ModelContext context, uint32_t n_outputs, ModelTensor outputs[])
{
    if (nullptr == context)
        return -1;
    TritonModel* model_inst = (TritonModel*)context;
    return model_inst->outputsRelease(n_outputs, outputs);
}