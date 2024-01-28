/********************************************
 * @Author: zjd
 * @Date: 2024-01-11 
 * @LastEditTime: 2024-01-11 
 * @LastEditors: zjd
 ********************************************/
#pragma once
#include <map>
#include <future>
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
        int run(bool async = false);
        int outputsGet(uint32_t n_outputs, ModelTensor* outputs);
        int outputsRelease(uint32_t n_outputs, ModelTensor* outputs);

    private:
        std::string                                               m_model_name;
        int64_t                                                   m_model_version;
        ModelPlatformType                                         m_model_platform;
        bool                                                      m_model_status = false;
        std::vector<ModelTensorAttr>                              m_model_input_attrs;
        std::vector<ModelTensorAttr>                              m_model_output_attrs;
        std::map<std::string, std::shared_ptr<TritonTensor>>      m_input_tensors;
        std::map<std::string, std::shared_ptr<TritonTensor>>      m_output_tensors;

        // barrier for async model run
        std::unique_ptr<std::promise<void*>>                      m_inference_response_barrier;
        std::unique_ptr<std::promise<void>>                       m_inference_request_barrier;
        void*                                                     m_inference_request = nullptr;
        void*                                                     m_response_allcator = nullptr;
    };

} // namespace TRITON_SERVER