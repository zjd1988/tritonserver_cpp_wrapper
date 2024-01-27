/********************************************
 * @Author: zjd
 * @Date: 2024-01-11 
 * @LastEditTime: 2024-01-11 
 * @LastEditors: zjd
 ********************************************/
#include <iostream>
#include "triton/core/tritonserver.h"
#include "common/log.h"
#include "tritonserver_engine/tritonserver_utils.h"
#include "tritonserver_engine/tritonserver_engine.h"
#include "tritonserver_wrapper/tritonserver_cpp_wrapper.h"

namespace TRITON_SERVER
{

    std::map<std::string, ModelPlatformType> gStrToModelPlatformType {
        {"tensorflow_graphdef", TENSORFLOW_GRAPHDEF_PLATFORM_TYPE},
        {"tensorflow_savedmodel", TENSORFLOW_SAVEDMODEL_PLATFORM_TYPE},
        {"tensorrt_plan", TENSORRT_PLAN_PLATFORM_TYPE},
        {"onnxruntime_onnx", ONNRUNTIME_ONNX_PLATFORM_TYPE},
        {"pytorch_libtorch", PYTORCH_LIBTORCH_PLATFORM_TYPE},
        {"python", PYTHON_PLATFORM_TYPE},
        {"ensemble", ENSEMBLE_PLATFORM_TYPE},
        {"rknn", RKNN_PLATFORM_TYPE},
    };

    TritonModel::TritonModel(const char* model_name, int64_t model_version, bool support_async)
    {
        if (nullptr == model_name)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model name is nullptr");
            return;
        }
        if (model_version != -1 && model_version <= 0)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model version:{} is invalid", model_version);
            return;
        }
        m_model_name = model_name;
        m_model_version = model_version;
        std::string model_paltform;
        if (0 != TritonServerEngine::Instance().getModelMetaInfo(model_name, model_version, model_paltform, 
            m_model_input_attrs, m_model_output_attrs))
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "get model {}:{} info fail", model_name, model_version);
            return;
        }

        if (gStrToModelPlatformType.end() == gStrToModelPlatformType.find(model_paltform))
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "get unsuppored platform for model {}:{}", 
                model_paltform, model_name, model_version);
            return;
        }
        m_model_platform = gStrToModelPlatformType[model_paltform];

        if (0 == m_model_input_attrs.size() || 0 == m_model_output_attrs.size())
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model {}:{} input number:{} output number:{}", 
                model_name, model_version, sizeof(m_model_input_attrs.size()),
                sizeof(m_model_output_attrs.size()));
            return;
        }

        // init triton server allocator
        if (0 != TritonServerEngine::Instance().createResponseAllocator(&m_response_allcator))
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "creating response allocator for {}:{} fail", 
                model_name, model_version);
            return;
        }

        if (support_async)
        {
            // init triton server request
            if (0 != TritonServerEngine::Instance().createInferenceRequest(model_name, model_version, 
                &m_inference_request))
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "creating inference reqeust for {}:{} fail", 
                    model_name, model_version);
                return;
            }
            m_async = support_async;
        }
        m_model_status = true;
    }

    TritonModel::~TritonModel()
    {
        m_model_input_attrs.clear();
        m_model_output_attrs.clear();
        m_input_tensors.clear();
        m_output_tensors.clear();
        m_inference_response_barrier.reset();
        m_inference_request_barrier.reset();
        if (nullptr != m_inference_request)
        {
            TritonServerEngine::Instance().deleteInferenceRequest(m_model_name, m_model_version, m_inference_request);
        }
        if (nullptr != m_response_allcator)
        {
            TritonServerEngine::Instance().destroyResponseAllocator(m_response_allcator);
        }
    }

    int TritonModel::query(ModeQueryCmd cmd, void* info, uint32_t size)
    {
        if (false == m_model_status)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model status is false");
            return -1;
        }
        if (nullptr == info)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "query result info is nullptr");
            return -1;
        }
        uint32_t input_num = m_model_input_attrs.size();
        uint32_t output_num = m_model_output_attrs.size();
        switch (cmd)
        {
            case MODEL_QUERY_IN_OUT_NUM:
            {
                uint32_t expect_size = sizeof(ModelInputOutputNum);
                if (expect_size != size)
                {
                    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "query MODEL_QUERY_IN_OUT_NUM size expect {}"
                        " but get {}", expect_size, size);
                    return -1;
                }
                auto tensor_num = (ModelInputOutputNum*)info;
                tensor_num->n_input = input_num;
                tensor_num->n_output = output_num;
                break;
            }
            case MODEL_QUERY_INPUT_ATTR:
            {
                uint32_t expect_size = sizeof(ModelTensorAttr) * input_num;
                if (expect_size != size)
                {
                    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "query MODEL_QUERY_INPUT_ATTR size expect {}"
                        " but get {}", expect_size, size);
                    return -1;
                }
                memcpy(info, &m_model_input_attrs[0], size);
                break;
            }
            case MODEL_QUERY_OUTPUT_ATTR:
            {
                uint32_t expect_size = sizeof(ModelTensorAttr) * output_num;
                if (expect_size != size)
                {
                    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "query MODEL_QUERY_OUTPUT_ATTR size expect {}"
                        " but get {}", expect_size, size);
                    return -1;
                }
                memcpy(info, &m_model_output_attrs[0], size);
                break;
            }
            case MODEL_QUERY_PLATFORM_TYPE:
            {
                uint32_t expect_size = sizeof(ModelPlatformType);
                if (expect_size != size)
                {
                    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "query MODEL_QUERY_PLATFORM_TYPE size expect {}"
                        " but get {}", expect_size, size);
                    return -1;
                }
                memcpy(info, &m_model_platform, size);
                break;
            }
            default:
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "query cmd {} not supported", int(cmd));
                return -1;
            }
        }
        return 0;
    }

    int TritonModel::inputsSet(uint32_t n_inputs, ModelTensor* inputs)
    {
        m_input_tensors.clear();
        if (false == m_model_status)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model status is false");
            return -1;
        }
        if (0 >= n_inputs)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model input tensor number:{} is invalid", n_inputs);
            return -1;
        }
        if (nullptr == inputs)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model input tensor is nullptr");
            return -1;
        }
        if (n_inputs != m_model_input_attrs.size())
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "input tensor number:{} not equal to tensor attr number:{}", 
                n_inputs, m_model_input_attrs.size());
            return -1;
        }

        for (uint32_t i = 0; i < n_inputs; i++)
        {
            ModelTensor* model_tensor = inputs + i;
            uint32_t index =  model_tensor->index;
            const auto& tensor_attr = m_model_input_attrs[index];
            std::string tensor_name = std::string(tensor_attr.name);
            // check input datatype
            TRITONSERVER_DataType input_dtype = (TRITONSERVER_DataType)model_tensor->type;
            TRITONSERVER_DataType expect_dtype = (TRITONSERVER_DataType)tensor_attr.type;
            if (input_dtype != expect_dtype)
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "tensor:{} expect datatype:{} but get {}", 
                    tensor_name, getTypeString(tensor_attr.type), getTypeString(model_tensor->type));
                return -1;
            }
            // check input tensor shape with model input tensor shape
            std::vector<int64_t> expect_shape(&tensor_attr.dims[0], &tensor_attr.dims[0] + tensor_attr.num_dim);
            std::vector<int64_t> input_shape(&model_tensor->dims[0], &model_tensor->dims[0] + model_tensor->num_dim);
            if (expect_shape.size() != input_shape.size())
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "checke tensor:{} shape size fail, expect shape:{} but get {}", 
                    tensor_name, fmt::join(expect_shape, " "), fmt::join(input_shape, " "));
                return -1;                
            }
            for (auto dim_index = 0; dim_index < expect_shape.size(); dim_index++)
            {
                if (0 < expect_shape[dim_index] && 
                    input_shape[dim_index] != expect_shape[dim_index])
                {
                    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "check tensor:{} shape dims fail, expect shape:{} but get {}", 
                        tensor_name, fmt::join(expect_shape, " "), fmt::join(input_shape, " "));
                    return -1;
                }
            }
            // construct tensor with datatype/shape/data
            void* data = model_tensor->buf;
            std::shared_ptr<TritonTensor> triton_tensor(new TritonTensor(input_dtype, input_shape, data));
            if (nullptr == triton_tensor.get())
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "construct tensor:{} fail", tensor_name);
                return -1;
            }
            m_input_tensors[tensor_name] = triton_tensor;
        }
        if (m_async)
        {

            m_inference_request_barrier = std::make_unique<std::promise<void>>();
            m_inference_response_barrier = std::make_unique<std::promise<void*>>();
            if (nullptr == m_inference_request_barrier.get() || nullptr == m_inference_response_barrier.get())
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "creating inference reqeust/response barrier for {}:{} fail", 
                    m_model_name, m_model_version);
                return -1;
            }
            void* request_barrier = (void*)m_inference_request_barrier.get();
            void* response_barrier = (void*)m_inference_response_barrier.get();
            void* response_allocator = m_response_allcator;
            if (0 != TritonServerEngine::Instance().prepareModelInferRequestResponse(m_inference_request, m_model_name, 
                m_model_version, response_allocator, request_barrier, response_barrier, m_model_input_attrs, 
                m_model_output_attrs, m_input_tensors))
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "praprae model input/output tensors for request fail");
                return -1;
            }
        }
        return 0;
    }

    int TritonModel::run()
    {
        if (false == m_model_status)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model status is false");
            return -1;
        }
        if (m_async)
        {
            return TRITON_SERVER_INFER_ASYNC(m_model_name, m_model_version, m_inference_request);
        }
        else
        {
            return TRITON_SERVER_INFER(m_model_name, m_model_version, m_model_input_attrs, m_model_output_attrs, 
                m_input_tensors, m_output_tensors, m_response_allcator);
        }
    }

    int TritonModel::outputsGet(uint32_t n_outputs, ModelTensor* outputs)
    {
        if (false == m_model_status)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model status is false");
            return -1;
        }
        if (0 >= n_outputs)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model output tensor number:{} is invalid", n_outputs);
            return -1;
        }
        if (nullptr == outputs)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model output tensor is nullptr");
            return -1;
        }
        if (n_outputs != m_model_output_attrs.size())
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "output tensor number:{} not equal to tensor attr number:{}", 
                n_outputs, m_model_output_attrs.size());
            return -1;
        }
        if (m_async)
        {
            std::future<void*> completed = m_inference_response_barrier->get_future();
            void* completed_response = completed.get();
            if (0 != TritonServerEngine::Instance().parseModelInferResponse(completed_response, m_model_name, 
                m_model_version, m_model_output_attrs, m_output_tensors, true))
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "parse model output tensors from response fail");
                return -1;
            }

            std::future<void> request_release_future = m_inference_request_barrier->get_future();
            request_release_future.get();
        }
        for (auto i = 0; i < m_model_output_attrs.size(); i++)
        {
            const auto& tensor_attr = m_model_output_attrs[i];
            int index = tensor_attr.index;
            std::string tensor_name = std::string(tensor_attr.name);
            if (m_output_tensors.end() == m_output_tensors.find(tensor_name))
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "cannot find model output tensor {}", tensor_name);
                return -1;
            }
            ModelTensor* model_tensor = outputs + index;
            model_tensor->index = index;
            model_tensor->buf = m_output_tensors[tensor_name]->base<void>();
            model_tensor->size = m_output_tensors[tensor_name]->byteSize();
            model_tensor->type = (TensorDataType)m_output_tensors[tensor_name]->dataType();
            auto tensor_shape = m_output_tensors[tensor_name]->shape();
            model_tensor->num_dim = (uint32_t)tensor_shape.size();
            memcpy(&model_tensor->dims[0], &tensor_shape[0], tensor_shape.size() * sizeof(int64_t));
        }
        return 0;
    }

    int TritonModel::outputsRelease(uint32_t n_outputs, ModelTensor* outputs)
    {
        if (false == m_model_status)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model status is false");
            return -1;
        }
        m_output_tensors.clear();
        return 0;
    }

} // TRITON_SERVER