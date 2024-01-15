/********************************************
 * @Author: zjd
 * @Date: 2024-01-11 
 * @LastEditTime: 2024-01-11 
 * @LastEditors: zjd
 ********************************************/
#include "common/log.h"
#include "tritonserver_infer/tritonserver_infer.h"
#include "tritonserver_wrapper/tritonserver_cpp_wrapper.h"

namespace TRITON_SERVER
{

    TritonModel::TritonModel(const char* model_name, int64_t model_version)
    {
        if (nullptr == model_name)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model name is nullptr");
            return -1;
        }
        if (model_version != -1 && model_version <= 0)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model version:{} is invalid", model_version);
            return -1;
        }
        m_model_name = model_name;
        m_model_version = model_version;
        if (0 != TritonServerInfer::Instance().getModelInfo(model_name, model_verison, 
            m_model_input_attrs, m_model_output_attrs))
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "get model {}:{} info fail", model_name, model_version);
            return;
        }
        if (0 == m_model_input_attrs.size() || 0 == m_model_output_attrs.size())
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model {}:{} input number:{} output number:{}", 
                model_name, model_version, sizeof(m_model_input_attrs.size()),
                sizeof(m_model_output_attrs.size()));
            return;
        }
        m_response_allcator = TritonServerInfer::Instance().createResponseAllocator();
        if (nullptr == m_response_allcator)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "creating response allocator for {}:{} fail", 
                model_name, model_version);
            return;
        }
        m_status = true;
    }

    TritonModel::~TritonModel()
    {
        if (nullptr != m_response_allcator)
        {
            TritonServerInfer::Instance().destroyResponseAllocator(m_response_allcator);
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
            default:
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "query cmd {} not supported", int(cmd));
                return -1;
            }
        }
        return 0;
    }

    // int TritonModel::inputsSet(uint32_t n_inputs, ModelTensor* inputs)
    // {
    //     if (false == m_model_status)
    //     {
    //         TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model status is false");
    //         return -1;
    //     }
    //     m_input_tensors.clear();
    //     if (0 >= n_inputs)
    //     {
    //         TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model input tensor number:{} is invalid", n_inputs);
    //         return -1;
    //     }
    //     if (nullptr == inputs)
    //     {
    //         TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model input tensor is nullptr");
    //         return -1;
    //     }
    //     memcpy(&m_input_tensors[0], inputs, sizeof(ModelTensor) * n_inputs);
    //     return 0;
    // }

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
        m_input_tensors.resize(n_inputs);
        for (uint32_t i = 0; i < n_inputs; i++)
        {
            ModelTensor* model_tensor = inputs[i];
            uint32_t index =  model_tensor->index;
            // check input datatype
            TRITONSERVER_DataType dtype = (TRITONSERVER_DataType)model_tensor->type;
            TRITONSERVER_DataType expect_dtype = (TRITONSERVER_DataType)m_model_input_attrs[index].type;
            if (dtype != expect_dtype)
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "tensor:{} expect datatype:{} but get {}", 
                    index, getTypeString(m_model_input_attrs[index].type), getTypeString(model_tensor->type));
                return -1;
            }
            // get expect shape
            uint32_t num_dim = m_model_input_attrs[index].num_dim;
            std::vector<int64_t> expect_shape(&m_model_input_attrs[index].dims[0], 
                &m_model_input_attrs[index].dims[0] + num_dim);
            void* data = model_tensor->buf;
            std::shared_ptr<TritonTensor> triton_tensor(expect_dtype, expect_shape, data);
            if (nullptr == triton_tensor.get())
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "construct tensor:{} fail", index);
                return -1;
            }
            // check input data size 
            uint32_t expect_size = triton_tensor->byteSize();
            if (model_tensor->size != expect_size)
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "tensor:{} expect size:{} but get {}", 
                    index, expect_size, model_tensor->size);
                return -1;
            }
            m_input_tensors[index] = triton_tensor;
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
        std::string model_version = std::to_string(m_model_version);
        return TritonServerInfer::Instance().infer(m_model_name, model_version, 
            m_model_input_attrs, m_model_output_attrs, m_input_tensors, m_output_tensors);
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
        for (auto i = 0; i < m_output_tensors.size(); i++)
        {
            int index = i;
            ModelTensor* model_tensor = outputs[i];
            model_tensor->index = index;
            model_tensor->buf = m_output_tensors[i]->base<void>();
            model_tensor->size = m_output_tensors[i]->byteSize();
            model_tensor->type = (TensorDataType)m_output_tensors[i]->dataType();
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