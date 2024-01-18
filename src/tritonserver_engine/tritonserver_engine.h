/********************************************
 * @Author: zjd
 * @Date: 2024-01-11 
 * @LastEditTime: 2024-01-11 
 * @LastEditors: zjd
 ********************************************/
#pragma once
#include <map>
#include "triton/core/tritonserver.h"
#include "common/log.h"
#include "common/non_copyable.h"
#include "tritonserver_wrapper/tritonserver_common.h"

namespace TRITON_SERVER
{

    class TritonTensor : public NonCopyable
    {
    public:
        TritonTensor(const TRITONSERVER_DataType dtype, const std::vector<int64_t>& shape, void* data = nullptr);
        ~TritonTensor();

        TRITONSERVER_DataType dataType() const { return m_dtype; }
        std::vector<int64_t> shape() const { return m_shape; }

        template<class T>
        T* base() const { return (T*)m_base; }

        size_t byteSize() const { return m_byte_size; }

    private:
        TRITONSERVER_DataType                m_dtype;
        std::vector<int64_t>                 m_shape;
        size_t                               m_byte_size;
        char*                                m_base = nullptr;
        bool                                 m_own = false;
    };

    #define TRITON_SERVER_INIT(config) TRITON_SERVER::TritonServerEngine::Instance().init(config)

    #define TRITON_SERVER_INFER(model_name, model_verison, inputs_attr, outputs_attr, inputs, outputs, allocator) \
        TRITON_SERVER::TritonServerEngine::Instance().infer(model_name, model_verison, inputs_attr, outputs_attr, inputs, outputs, allocator)

    #define TRITON_SERVER_UNINIT() TRITON_SERVER::TritonServerEngine::Instance().uninit()

    class TritonServerEngine : public NonCopyable
    {
    public:
        static TritonServerEngine& Instance();
        int init(const ServerConfig* config);
        void uninit();
        void* createResponseAllocator();
        void destroyResponseAllocator(void* allocator);
        int infer(const std::string model_name, 
            const int64_t model_version, 
            const std::vector<ModelTensorAttr>& input_attrs, 
            const std::vector<ModelTensorAttr>& output_attrs, 
            const std::map<std::string, std::shared_ptr<TritonTensor>>& input_tensors, 
            std::map<std::string, std::shared_ptr<TritonTensor>>& output_tensors, 
            void* response_allocator = nullptr);
        int getModelInfo(const std::string model_name, const int64_t model_version, 
            std::vector<ModelTensorAttr>& input_attrs, std::vector<ModelTensorAttr>& output_attrs);

    private:
        TritonServerEngine() = default;
        ~TritonServerEngine() = default;

        void parseModelInferResponse(TRITONSERVER_InferenceResponse* response, 
            const std::string model_name, const int64_t model_version, 
            const std::map<std::string, ModelTensorAttr>& output_attrs, 
            std::map<std::string, std::shared_ptr<TritonTensor>>& output_tensors);

    private:
        std::string                                                 m_model_repository_path;
        int32_t                                                     m_verbose_level;
        std::shared_ptr<TRITONSERVER_Server>                        m_server;
    };

}  // namespace TRITON_SERVER
