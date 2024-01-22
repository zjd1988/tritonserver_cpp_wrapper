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

    #define TRITON_SERVER_LOAD_MODEL(model_name) TRITON_SERVER::TritonServerEngine::Instance().loadModel(model_name)

    #define TRITON_SERVER_UNLOAD_MODEL(model_name) TRITON_SERVER::TritonServerEngine::Instance().unloadModel(model_name)

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
        int getModelMetaInfo(const std::string model_name, const int64_t model_version, std::string& model_platform, 
            std::vector<ModelTensorAttr>& input_attrs, std::vector<ModelTensorAttr>& output_attrs);

        // load/unload model by name
        int loadModel(const std::string model_name);
        int unloadModel(const std::string model_name);

    private:
        TritonServerEngine() = default;
        ~TritonServerEngine() = default;

        void parseModelInferResponse(TRITONSERVER_InferenceResponse* response, 
            const std::string model_name, const int64_t model_version, 
            const std::map<std::string, ModelTensorAttr>& output_attrs, 
            std::map<std::string, std::shared_ptr<TritonTensor>>& output_tensors);

    private:
        // triton server option
        std::string                                                 m_model_repository_path; // model repository dir
        int32_t                                                     m_verbose_level;         // log verbose level
        TRITONSERVER_LogFormat                                      m_log_format;            // log format
        std::string                                                 m_log_file_path;         // absolute log file path
        TRITONSERVER_ModelControlMode                               m_model_control;         // model control, NONE/POLL/EXPLICIT
        bool                                                        m_strict_model;          // strict config model
        std::string                                                 m_backend_dir;           // triton server backends dir
        std::string                                                 m_repo_agent_dir;        // triton server repo agent dir
        int                                                         m_check_timeout;         // triton server check ready timeout
        // triton server
        std::shared_ptr<TRITONSERVER_Server>                        m_server;
    };

}  // namespace TRITON_SERVER
