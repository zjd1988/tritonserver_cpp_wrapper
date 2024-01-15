/********************************************
 * @Author: zjd
 * @Date: 2024-01-11 
 * @LastEditTime: 2024-01-11 
 * @LastEditors: zjd
 ********************************************/
#pragma once
#include <vector>
#include <memory>
#include <string>
#include "tritonserver_wrapper/tritonserver_common.h"

namespace TRITON_SERVER
{

    class TritonTensor;

    class TritonModel
    {
    public:
        TritonModel(const char* model_name, int64_t model_version = -1);
        ~TritonModel();

    public:
        bool status() { return m_model_status; }
        int query(ModeQueryCmd cmd, void* info, uint32_t size);
        int inputsSet(uint32_t n_inputs, ModelTensor* inputs);
        int run();
        int outputsGet(uint32_t n_outputs, ModelTensor* outputs);
        int outputsRelease(uint32_t n_outputs, ModelTensor* outputs);

    private:
        std::string                                       m_model_name;
        int64_t                                           m_model_version;
        bool                                              m_model_status = false;
        std::vector<ModelTensorAttr>                      m_model_input_attrs;
        std::vector<ModelTensorAttr>                      m_model_output_attrs;
        // std::vector<ModelTensor>                          m_input_tensors;
        // std::vector<ModelTensor>                          m_output_tensors;
        std::vector<std::shared_ptr<TritonTensor>>        m_input_tensors;
        std::vector<std::shared_ptr<TritonTensor>>        m_output_tensors;
        void*                                             m_response_allcator = nullptr;
    };

} // namespace TRITON_SERVER