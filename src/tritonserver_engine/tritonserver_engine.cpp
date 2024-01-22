/********************************************
 * @Author: zjd
 * @Date: 2024-01-11 
 * @LastEditTime: 2024-01-11 
 * @LastEditors: zjd
 ********************************************/
#include <stdio.h>
#include <set>
#include <memory>
#include <future>
#include <sstream>
#include <numeric>
#include <functional>
#include <algorithm>
#ifdef TRITON_ENABLE_GPU
    #include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "tritonserver_engine/tritonserver_engine.h"

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

    static size_t getTritonDataTypeByteSize(TRITONSERVER_DataType dtype)
    {
        size_t byte_size = 0;
        switch (dtype)
        {
            case TRITONSERVER_TYPE_UINT8:
            case TRITONSERVER_TYPE_INT8:
                byte_size = sizeof(int8_t);
                break;
            case TRITONSERVER_TYPE_UINT16:
            case TRITONSERVER_TYPE_INT16:
                byte_size = sizeof(int16_t);
                break;
            case TRITONSERVER_TYPE_UINT32:
            case TRITONSERVER_TYPE_INT32:
            case TRITONSERVER_TYPE_FP32:
                byte_size = sizeof(int32_t);
                break;
            case TRITONSERVER_TYPE_UINT64:
            case TRITONSERVER_TYPE_INT64:
            case TRITONSERVER_TYPE_FP64:
                byte_size = sizeof(int64_t);
                break;
            default:
                FAIL("get invalid datatype " + std::to_string(int(dtype)) + " when get datatype bytesize");
        }
        return byte_size;
    }

    static TRITONSERVER_DataType convertStrToTritonDataType(std::string datatype_str)
    {
        if (0 == strcmp(datatype_str.c_str(), "UINT8"))
            return TRITONSERVER_TYPE_UINT8;
        else if (0 == strcmp(datatype_str.c_str(), "UINT16"))
            return TRITONSERVER_TYPE_UINT16;
        else if (0 == strcmp(datatype_str.c_str(), "UINT32"))
            return TRITONSERVER_TYPE_UINT32;
        else if (0 == strcmp(datatype_str.c_str(), "UINT64"))
            return TRITONSERVER_TYPE_UINT64;
        else if (0 == strcmp(datatype_str.c_str(), "INT8"))
            return TRITONSERVER_TYPE_INT8;
        else if (0 == strcmp(datatype_str.c_str(), "INT16"))
            return TRITONSERVER_TYPE_INT16;
        else if (0 == strcmp(datatype_str.c_str(), "INT32"))
            return TRITONSERVER_TYPE_INT32;
        else if (0 == strcmp(datatype_str.c_str(), "INT64"))
            return TRITONSERVER_TYPE_INT64;
        else if (0 == strcmp(datatype_str.c_str(), "FP32"))
            return TRITONSERVER_TYPE_FP32;
        else if (0 == strcmp(datatype_str.c_str(), "FP64"))
            return TRITONSERVER_TYPE_FP64;
        else
            return TRITONSERVER_TYPE_INVALID;
    }

    TritonTensor::TritonTensor(const TRITONSERVER_DataType dtype, const std::vector<int64_t>& shape, void* data)
    {
        if (TRITONSERVER_TYPE_INVALID == m_dtype)
        {
            FAIL("invalid triton tensor datatype " + std::to_string(dtype));
        }
        m_dtype = dtype;
        m_shape = shape;
        int64_t ele_count = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
        m_byte_size = ele_count * getTritonDataTypeByteSize(m_dtype);
        if (nullptr == data)
        {
            m_base = new char[m_byte_size];
            if (nullptr == m_base)
            {
                FAIL("malloc triton tensor memory buffer fail");
                return;
            }
            m_own = true;
        }
        else
            m_base = (char*)data;
    }

    TritonTensor::~TritonTensor()
    {
        if (m_own)
            delete []m_base;
    }

    static int parseModelMetadata(const rapidjson::Document& model_metadata, 
        const std::string model_name, const std::string model_version, 
        std::vector<ModelTensorAttr>& input_attrs, std::vector<ModelTensorAttr>& output_attrs)
    {
        std::string model_key = model_name + ":" + model_version;
        uint32_t input_index = 0;
        uint32_t output_index = 0;
        for (const auto& input : model_metadata["inputs"].GetArray())
        {
            std::string name = input["name"].GetString();
            std::string datatype_str = input["datatype"].GetString();
            TRITONSERVER_DataType datatype = convertStrToTritonDataType(datatype_str);
            if (TRITONSERVER_TYPE_INVALID == datatype)
            {
                FAIL("model " + model_key + " input:" + name + " contain unsupported datatype " + datatype_str);
            }
            std::vector<int64_t> shape_vec;
            for (const auto &shape_item : input["shape"].GetArray())
            {
                int64_t dim_value = shape_item.GetInt64();
                shape_vec.push_back(dim_value);
            }
            ModelTensorAttr input_attr = {0};
            input_attr.index = input_index++;
            snprintf(&input_attr.name[0], TRITON_MAX_NAME_LENGTH, "%s", name.c_str());
            if (shape_vec.size() > TRITON_TENSOR_MAX_DIM_NUM)
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "{}:{} input tensor:{} shape dims number:{} "
                    "exceed max dim:{}", model_name, model_version, name, 
                    shape_vec.size(), TRITON_TENSOR_MAX_DIM_NUM);
                return -1;   
            }
            input_attr.num_dim = shape_vec.size();
            memcpy(&input_attr.dims[0], &shape_vec[0], sizeof(int64_t) * shape_vec.size());
            input_attr.type = (TensorDataType)datatype;
            input_attrs.push_back(input_attr);
        }

        for (const auto& output : model_metadata["outputs"].GetArray())
        {
            std::string name = output["name"].GetString();
            std::string datatype_str = output["datatype"].GetString();
            TRITONSERVER_DataType datatype = convertStrToTritonDataType(datatype_str);
            if (TRITONSERVER_TYPE_INVALID == datatype)
            {
                FAIL("model " + model_key + " output:" + name + " contain unsupported datatype " + datatype_str);
            }
            std::vector<int64_t> shape_vec;
            for (const auto &shape_item : output["shape"].GetArray())
            {
                int64_t dim_value = shape_item.GetInt64();
                shape_vec.push_back(dim_value);
            }
            ModelTensorAttr output_attr = {0};
            output_attr.index = output_index++;
            snprintf(&output_attr.name[0], TRITON_MAX_NAME_LENGTH, "%s", name.c_str());
            if (shape_vec.size() > TRITON_TENSOR_MAX_DIM_NUM)
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "{}:{} output tensor:{} shape dims number:{} "
                    "exceed max dim:{}", model_name, model_version, name, 
                    shape_vec.size(), TRITON_TENSOR_MAX_DIM_NUM);
                return -1;   
            }
            output_attr.num_dim = shape_vec.size();
            memcpy(&output_attr.dims[0], &shape_vec[0], sizeof(int64_t) * shape_vec.size());
            output_attr.type = (TensorDataType)datatype;
            output_attrs.push_back(output_attr);
        }

        if (0 == input_attrs.size() || 0 == output_attrs.size())
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "get model input attrs number:{} output attrs number:{}",
                input_attrs.size(), output_attrs.size());
            return -1;
        }
        return 0;
    }

    static TRITONSERVER_Error* ResponseAlloc(TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
        size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
        int64_t preferred_memory_type_id, void* userp, void** buffer,
        void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
        int64_t* actual_memory_type_id)
    {
        // Initially attempt to make the actual memory type and id that we
        // allocate be the same as preferred memory type
        *actual_memory_type = preferred_memory_type;
        *actual_memory_type_id = preferred_memory_type_id;

        // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
        // need to do any other book-keeping.
        if (byte_size == 0)
        {
            *buffer = nullptr;
            *buffer_userp = nullptr;
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_DEBUG, "allocated {} bytes for result tensor {}", 
                byte_size, tensor_name);
        }
        else
        {
            void* allocated_ptr = nullptr;

            switch (*actual_memory_type)
            {
            #ifdef TRITON_ENABLE_GPU
                case TRITONSERVER_MEMORY_CPU_PINNED: {
                    auto err = cudaSetDevice(*actual_memory_type_id);
                    if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
                        (err != cudaErrorInsufficientDriver)) {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INTERNAL,
                        std::string(
                            "unable to recover current CUDA device: " +
                            std::string(cudaGetErrorString(err)))
                            .c_str());
                    }

                    err = cudaHostAlloc(&allocated_ptr, byte_size, cudaHostAllocPortable);
                    if (err != cudaSuccess) {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INTERNAL,
                        std::string(
                            "cudaHostAlloc failed: " +
                            std::string(cudaGetErrorString(err)))
                            .c_str());
                    }
                    break;
                }

                case TRITONSERVER_MEMORY_GPU: {
                    auto err = cudaSetDevice(*actual_memory_type_id);
                    if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
                        (err != cudaErrorInsufficientDriver)) {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INTERNAL,
                        std::string(
                            "unable to recover current CUDA device: " +
                            std::string(cudaGetErrorString(err)))
                            .c_str());
                    }

                    err = cudaMalloc(&allocated_ptr, byte_size);
                    if (err != cudaSuccess) {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INTERNAL,
                        std::string(
                            "cudaMalloc failed: " + std::string(cudaGetErrorString(err)))
                            .c_str());
                    }
                    break;
                }
            #endif  // TRITON_ENABLE_GPU

                // Use CPU memory if the requested memory type is unknown
                // (default case).
                case TRITONSERVER_MEMORY_CPU:
                default:
                {
                    *actual_memory_type = TRITONSERVER_MEMORY_CPU;
                    allocated_ptr = malloc(byte_size);
                    break;
                }
            }

            // Pass the tensor name with buffer_userp so we can show it when
            // releasing the buffer.
            if (allocated_ptr != nullptr)
            {
                *buffer = allocated_ptr;
                *buffer_userp = new std::string(tensor_name);
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_DEBUG, "allocated {} bytes in {} for result tensor {}", 
                    byte_size, TRITONSERVER_MemoryTypeString(*actual_memory_type), tensor_name);
            }
        }

        return nullptr;  // Success
    }

    static TRITONSERVER_Error* ResponseRelease(TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
        size_t byte_size, TRITONSERVER_MemoryType memory_type, int64_t memory_type_id)
    {
        std::string* name = nullptr;
        if (buffer_userp != nullptr)
        {
            name = reinterpret_cast<std::string*>(buffer_userp);
        }
        else
        {
            name = new std::string("<unknown>");
        }

        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_DEBUG, "Releasing response buffer of size {} in {} for result tensor {}", 
            byte_size, TRITONSERVER_MemoryTypeString(memory_type), *name);

        switch (memory_type)
        {
            case TRITONSERVER_MEMORY_CPU:
                free(buffer);
                break;
            #ifdef TRITON_ENABLE_GPU
                case TRITONSERVER_MEMORY_CPU_PINNED: 
                {
                    auto err = cudaSetDevice(memory_type_id);
                    if (err == cudaSuccess)
                    {
                        err = cudaFreeHost(buffer);
                    }
                    if (err != cudaSuccess)
                    {
                        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "error: failed to cudaFree {x}:{}", 
                            buffer, cudaGetErrorString(err));
                    }
                    break;
                }
                case TRITONSERVER_MEMORY_GPU:
                {
                    auto err = cudaSetDevice(memory_type_id);
                    if (err == cudaSuccess)
                    {
                        err = cudaFree(buffer);
                    }
                    if (err != cudaSuccess)
                    {
                        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "error: failed to cudaFree {x}:{}", 
                            buffer, cudaGetErrorString(err));
                    }
                    break;
                }
            #endif  // TRITON_ENABLE_GPU
            default:
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "error: unexpected buffer allocated in CUDA managed memory");
                break;
        }
        delete name;
        return nullptr;  // Success
    }

    static void InferRequestRelease(TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
    {
        // TRITONSERVER_InferenceRequestDelete(request);
        std::promise<void>* barrier = reinterpret_cast<std::promise<void>*>(userp);
        barrier->set_value();
    }

    static void InferResponseComplete(TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
    {
        if (response != nullptr)
        {
            // Send 'response' to the future.
            std::promise<TRITONSERVER_InferenceResponse*>* p =
                reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
            p->set_value(response);
            delete p;
        }
    }

    TritonServerEngine& TritonServerEngine::Instance()
    {
        static TritonServerEngine triton_server;
        return triton_server;
    }

    void* TritonServerEngine::createResponseAllocator()
    {
        // When triton needs a buffer to hold an output tensor, it will ask
        // us to provide the buffer. In this way we can have any buffer
        // management and sharing strategy that we want. To communicate to
        // triton the functions that we want it to call to perform the
        // allocations, we create a "response allocator" object. We pass
        // this response allocate object to triton when requesting
        // inference. We can reuse this response allocate object for any
        // number of inference requests.
        TRITONSERVER_ResponseAllocator* allocator = nullptr;
        LOG_IF_ERR(TRITONSERVER_ResponseAllocatorNew(&allocator, ResponseAlloc, ResponseRelease, nullptr /* start_fn */),
            "creating response allocator");
        return allocator;
    }

    void TritonServerEngine::destroyResponseAllocator(void* allocator)
    {
        TRITONSERVER_ResponseAllocator* response_allocator = (TRITONSERVER_ResponseAllocator*)allocator;
        if (nullptr != response_allocator)
        {
            LOG_IF_ERR(TRITONSERVER_ResponseAllocatorDelete(response_allocator), 
                "deleting response allocator");
        }
        return;
    }

    void TritonServerEngine::uninit()
    {
        m_server.reset();
    }

    int TritonServerEngine::init(const ServerConfig* config)
    {
        uint32_t api_version_major;
        uint32_t api_version_minor;
        FAIL_IF_ERR(TRITONSERVER_ApiVersion(&api_version_major, &api_version_minor), 
            "getting Triton API version");
        if ((TRITONSERVER_API_VERSION_MAJOR != api_version_major) || 
            (TRITONSERVER_API_VERSION_MINOR > api_version_minor))
        {
            FAIL("triton server API version mismatch");
        }

        std::string model_repository_path = config->model_repository_dir;
        int verbose_level = (int)config->log_verbose_level;
        TRITONSERVER_LogFormat log_format = (TRITONSERVER_LogFormat)config->log_format;
        std::string log_file;
        if (config->log_file_path)
            log_file = config->log_file_path;

        std::string backend_dir = config->backend_dir;
        std::string repo_agent_dir = config->repo_agent_dir;
        int timeout = config->check_timeout;
        TRITONSERVER_ModelControlMode model_control = (TRITONSERVER_ModelControlMode)config->model_control;
        bool strict_model = config->strict_model;

        m_model_repository_path = model_repository_path;   // model repository dir
        m_verbose_level = verbose_level;                   // log verbose level
        m_log_format = log_format;                         // log format
        m_log_file_path = log_file;                        // absolute log file path
        m_model_control = model_control;                   // model control, NONE/POLL/EXPLICIT
        m_strict_model = strict_model;                     // strict config model
        m_backend_dir = backend_dir;                       // triton server backends dir
        m_repo_agent_dir = repo_agent_dir;                 // triton server repo agent dir
        m_check_timeout = timeout;                         // triton server check ready timeout

        TRITONSERVER_ServerOptions* server_options = nullptr;
        FAIL_IF_ERR(TRITONSERVER_ServerOptionsNew(&server_options), 
            "creating server options");
        FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetModelRepositoryPath(server_options, model_repository_path.c_str()), 
            "setting model repository path");
        FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetLogVerbose(server_options, verbose_level), 
            "setting verbose logging level");
        FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetLogFormat(server_options, log_format), 
            "settiong log format");
        FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetLogFile(server_options, log_file.c_str()), 
            "setting log file");
        FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetBackendDirectory(server_options, backend_dir.c_str()), 
            "setting backend directory");
        FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetRepoAgentDirectory(server_options, repo_agent_dir.c_str()), 
            "setting repository agent directory");
        FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetModelControlMode(server_options, model_control), 
            "setting model control model");
        FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetStrictModelConfig(server_options, true),
            "setting strict model configuration");

    #ifdef TRITON_ENABLE_GPU
        double min_compute_capability = TRITON_MIN_COMPUTE_CAPABILITY;
    #else
        double min_compute_capability = 0;
    #endif  // TRITON_ENABLE_GPU
        FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(server_options, min_compute_capability),
            "setting minimum supported CUDA compute capability");

        // Create the server object using the option settings. The server
        // object encapsulates all the functionality of the Triton server
        // and allows access to the Triton server API. Typically only a
        // single server object is needed by an application, but it is
        // allowed to create multiple server objects within a single
        // application. After the server object is created the server
        // options can be deleted.
        TRITONSERVER_Server* server_ptr = nullptr;
        FAIL_IF_ERR(TRITONSERVER_ServerNew(&server_ptr, server_options),
            "creating server object");
        FAIL_IF_ERR(TRITONSERVER_ServerOptionsDelete(server_options),
            "deleting server options");

        std::shared_ptr<TRITONSERVER_Server> server(server_ptr, TRITONSERVER_ServerDelete);
        m_server = std::move(server);
        // Wait until the server is both live and ready. The server will not
        // appear "ready" until all models are loaded and ready to receive
        // inference requests.
        size_t health_iters = 0;
        while (true)
        {
            bool live = false, ready = false;
            FAIL_IF_ERR(TRITONSERVER_ServerIsLive(m_server.get(), &live),
                "unable to get server liveness");
            FAIL_IF_ERR(TRITONSERVER_ServerIsReady(m_server.get(), &ready),
                "unable to get server readiness");
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "Server Health: live {}, ready {}", live, ready);
            if (live && ready)
            {
                break;
            }
            if (++health_iters >= 10)
            {
                FAIL("failed to find healthy inference server");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(timeout));
        }

        // log server metadata
        {
            TRITONSERVER_Message* server_metadata_message;
            FAIL_IF_ERR(TRITONSERVER_ServerMetadata(m_server.get(), &server_metadata_message),
                "unable to get server metadata message");

            const char* buffer;
            size_t byte_size;
            FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(server_metadata_message, &buffer, &byte_size),
                "unable to serialize server metadata message");

            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "Triton Server Metadata:\n{}", buffer);

            FAIL_IF_ERR(TRITONSERVER_MessageDelete(server_metadata_message),
                "deleting server metadata message");
        }

        // log server models metadata
        {
            // get model statistic message
            TRITONSERVER_Message* models_statistic_message;
            FAIL_IF_ERR(TRITONSERVER_ServerModelStatistics(m_server.get(), "", -1, &models_statistic_message),
                "unable to get models statistic message");

            const char* buffer;
            size_t byte_size;
            FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(models_statistic_message, &buffer, &byte_size),
                "unable to serialize models statistic message");

            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "Triton Server Models Statistics:\n{}", buffer);

            // parse model statistic message
            rapidjson::Document models_statistic_metadata;
            models_statistic_metadata.Parse(buffer, byte_size);
            if (models_statistic_metadata.HasParseError())
            {
                FAIL("error: failed to parse models statistic from JSON: " +
                    std::string(GetParseError_En(models_statistic_metadata.GetParseError())) +
                    " at " + std::to_string(models_statistic_metadata.GetErrorOffset()));
            }

            // delete model statistic message
            FAIL_IF_ERR(TRITONSERVER_MessageDelete(models_statistic_message),
                "deleting models statistic message");

            // init models info
            const rapidjson::Value &model_stats = models_statistic_metadata["model_stats"];
            for (auto &model_item : model_stats.GetArray())
            {
                std::string model_name = model_item["name"].GetString();
                std::string model_version = model_item["version"].GetString();

                // get model metadata
                std::stringstream ss;
                int64_t model_version_int;
                ss << model_version;
                ss >> model_version_int;
                TRITONSERVER_Message* model_metadata_message;
                FAIL_IF_ERR(TRITONSERVER_ServerModelMetadata(m_server.get(), model_name.c_str(), model_version_int, &model_metadata_message), 
                    "unable to get model metadata message");
                const char* model_metadata_buffer;
                size_t model_metadata_byte_size;
                FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(model_metadata_message, &model_metadata_buffer, &model_metadata_byte_size),
                    "unable to serialize model metadata");

                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "model {}:{} metadata:\n{}", 
                    model_name, model_version, model_metadata_buffer);

                // parse model metadata json
                rapidjson::Document model_metadata;
                model_metadata.Parse(model_metadata_buffer, model_metadata_byte_size);
                if (model_metadata.HasParseError())
                {
                    FAIL("error: failed to parse model " + model_name + std::string(" metadata from JSON: ") + 
                        std::string(GetParseError_En(model_metadata.GetParseError())) +
                        " at " + std::to_string(model_metadata.GetErrorOffset()));
                }

                // delete model metadata message
                FAIL_IF_ERR(TRITONSERVER_MessageDelete(model_metadata_message),
                    "deleting model metadata message");
            }
        }
        return 0;
    }

    int TritonServerEngine::getModelMetaInfo(const std::string model_name, const int64_t model_version, 
        std::string& model_platform, std::vector<ModelTensorAttr>& input_attrs, 
        std::vector<ModelTensorAttr>& output_attrs)
    {
        input_attrs.clear();
        output_attrs.clear();
        std::string model_key = model_name + ":" + std::to_string(model_version);

        // get specific model info
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "get model {} info", model_key);
        if (nullptr == m_server.get())
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "triton server not init or init failed, please init first");
            return -1;
        }

        // get model metadata
        TRITONSERVER_Message* model_metadata_message = nullptr;
        FAIL_IF_ERR(TRITONSERVER_ServerModelMetadata(m_server.get(), model_name.c_str(), model_version, 
            &model_metadata_message), "unable to get model metadata message");

        // serialize model metadata to json str
        const char* model_metadata_buffer;
        size_t model_metadata_byte_size;
        FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(model_metadata_message, &model_metadata_buffer, 
            &model_metadata_byte_size), "unable to serialize model metadata");

        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "model {} metadata:\n{}", model_key, model_metadata_buffer);

        // parse model metadata json
        rapidjson::Document model_metadata;
        model_metadata.Parse(model_metadata_buffer, model_metadata_byte_size);
        if (model_metadata.HasParseError())
        {
            FAIL("error: failed to parse model " + model_key + std::string(" metadata from JSON: ") + 
                std::string(GetParseError_En(model_metadata.GetParseError())) +
                " at " + std::to_string(model_metadata.GetErrorOffset()));
        }

        // delete model metadata message
        FAIL_IF_ERR(TRITONSERVER_MessageDelete(model_metadata_message), "deleting model metadata message");

        // check model name
        if (strcmp(model_metadata["name"].GetString(), model_name.c_str()))
        {
            FAIL("unable to find metadata for model " + model_key);
        }

        // get model platform
        model_platform = model_metadata["platform"].GetString();

        // check model version
        bool found_version = false;
        if (model_metadata.HasMember("versions"))
        {
            for (const auto& version : model_metadata["versions"].GetArray())
            {
                if (strcmp(version.GetString(), std::to_string(model_version).c_str()) == 0)
                {
                    found_version = true;
                    break;
                }
            }
        }
        if (-1 == model_version || found_version)
        {
            if (0 != parseModelMetadata(model_metadata, model_name, std::to_string(model_version), 
                input_attrs, output_attrs))
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "parsing model {} metadata fail", model_key);
                input_attrs.clear();
                output_attrs.clear();
                return -1;
            }
        }
        else
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "unable to find version {} status for model {}", 
                model_version, model_name);
            return -1;
        }

        return 0;
    }

    void TritonServerEngine::parseModelInferResponse(TRITONSERVER_InferenceResponse* response, 
        const std::string model_name, const int64_t model_version, 
        const std::map<std::string, ModelTensorAttr>& expect_outputs, 
        std::map<std::string, std::shared_ptr<TritonTensor>>& output_tensors)
    {
        std::string model_key = model_name + ":" + std::to_string(model_version);
        // get model output count
        uint32_t output_count;
        FAIL_IF_ERR(TRITONSERVER_InferenceResponseOutputCount(response, &output_count),
            "getting number of response outputs for model " + model_key);
        if (output_count != expect_outputs.size())
        {
            FAIL("expecting " + std::to_string(expect_outputs.size()) + " response outputs, got " + 
                std::to_string(output_count) + " for model " + model_key);
        }

        for (uint32_t idx = 0; idx < output_count; ++idx)
        {
            const char* cname;
            TRITONSERVER_DataType datatype;
            const int64_t* shape;
            uint64_t dim_count;
            const void* base;
            size_t byte_size;
            TRITONSERVER_MemoryType memory_type;
            int64_t memory_type_id;
            void* userp;

            FAIL_IF_ERR(TRITONSERVER_InferenceResponseOutput(
                response, idx, &cname, &datatype, &shape, &dim_count, &base,
                &byte_size, &memory_type, &memory_type_id, &userp), "getting output info");

            if (cname == nullptr)
            {
                FAIL("unable to get output name for model " + model_key);
            }
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "parse {} tensor {}", model_key, cname);

            std::string name(cname);
            if (expect_outputs.end() == expect_outputs.find(name))
            {
                FAIL("output " + name + " not in output tensor attr for model " + model_key);
            }

            TRITONSERVER_DataType expected_datatype = (TRITONSERVER_DataType)expect_outputs.at(name).type;
            if (datatype != expected_datatype)
            {
                FAIL("output " + name + " have unexpected datatype " + 
                    std::string(TRITONSERVER_DataTypeString(datatype)) + " for model " + model_key);
            }

            // parepare output tensor
            std::vector<int64_t> tensor_shape(shape, shape + dim_count);
            std::shared_ptr<TritonTensor> output_tensor(new TritonTensor(datatype, tensor_shape));
            if (nullptr == output_tensor.get() || nullptr == output_tensor->base<void>())
            {
                FAIL("malloc buff to output " + name + " fail for model " + model_key);
            }
            // We make a copy of the data here... which we could avoid for
            // performance reasons but ok for this simple example.
            switch (memory_type)
            {
                case TRITONSERVER_MEMORY_CPU:
                {
                    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_DEBUG, "{} is stored in system memory for model {}", 
                        name, model_key);
                    memcpy(output_tensor->base<void>(), base, byte_size);
                    break;
                }

                case TRITONSERVER_MEMORY_CPU_PINNED:
                {
                    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_DEBUG, "{} is stored in pinned memory for model {}", 
                        name, model_key);
                    memcpy(output_tensor->base<void>(), base, byte_size);
                    break;
                }

            #ifdef TRITON_ENABLE_GPU
                case TRITONSERVER_MEMORY_GPU:
                {
                    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_DEBUG, "{} is stored in GPU memory for model {}", 
                        name, model_key);
                    FAIL_IF_CUDA_ERR(cudaMemcpy(output_tensor->base<void>(), base, byte_size, cudaMemcpyDeviceToHost),
                        "getting " + name + " data from GPU memory for model " + model_key);
                    break;
                }
            #endif

                default:
                    FAIL("unexpected memory type for model " + model_key);
            }
            output_tensors[name] = output_tensor;
        }
        return;
    }

    int TritonServerEngine::infer(const std::string model_name, const int64_t model_version, 
        const std::vector<ModelTensorAttr>& input_attrs, 
        const std::vector<ModelTensorAttr>& output_attrs, 
        const std::map<std::string, std::shared_ptr<TritonTensor>>& input_tensors, 
        std::map<std::string, std::shared_ptr<TritonTensor>>& output_tensors, 
        void* response_allocator)
    {
        output_tensors.clear();
        std::string model_key = model_name + ":" + std::to_string(model_version);
        // check input tensors size equal to model inputs
        if (input_tensors.size() != input_attrs.size())
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "input tensors size:{} not equal to"
                " input tensors attr size:{} for model {}", input_tensors.size(), 
                model_key, input_attrs.size(), model_key);
            return -1;
        }
        // When triton needs a buffer to hold an output tensor, it will ask
        // us to provide the buffer. In this way we can have any buffer
        // management and sharing strategy that we want. To communicate to
        // triton the functions that we want it to call to perform the
        // allocations, we create a "response allocator" object. We pass
        // this response allocate object to triton when requesting
        // inference. We can reuse this response allocate object for any
        // number of inference requests.
        TRITONSERVER_ResponseAllocator* allocator = nullptr;
        if (nullptr != response_allocator)
            allocator = (TRITONSERVER_ResponseAllocator*)response_allocator;
        else
            FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorNew(&allocator, ResponseAlloc, ResponseRelease, nullptr /* start_fn */),
                "creating response allocator for model " + model_key);

        // Create an inference request object. The inference request object
        // is where we set the name of the model we want to use for
        // inference and the input tensors.
        TRITONSERVER_InferenceRequest* irequest = nullptr;
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestNew(&irequest, m_server.get(), model_name.c_str(), model_version),
            "creating inference request for model " + model_key);

        std::unique_ptr<std::promise<void>> barrier = std::make_unique<std::promise<void>>();
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetReleaseCallback(irequest, InferRequestRelease,
            reinterpret_cast<void*>(barrier.get())), "setting request release callback for model " + model_key);
        std::future<void> request_release_future = barrier->get_future();

        // Add the model inputs to the request...
        for (size_t i = 0; i < input_attrs.size(); i++)
        {
            std::string tensor_name = std::string(input_attrs[i].name);
            std::vector<int64_t> input_shape = input_tensors.at(tensor_name)->shape();
            TRITONSERVER_DataType datatype = input_tensors.at(tensor_name)->dataType();
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(irequest, tensor_name.c_str(), 
                datatype, &input_shape[0], input_shape.size()), "assigning input: " + tensor_name + 
                " meta-data to request for model " + model_key);
            size_t input_size = input_tensors.at(tensor_name)->byteSize();
            const void* input_base = input_tensors.at(tensor_name)->base<void>();
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(irequest, tensor_name.c_str(), input_base, input_size, 
                TRITONSERVER_MEMORY_CPU, 0 /* memory_type_id */),
                "assigning input: " + tensor_name + " data to request for model " + model_key);
        }

        // Add the model outputs to the request...
        std::map<std::string, ModelTensorAttr> expected_outputs;
        for (size_t i = 0; i < output_attrs.size(); i++)
        {
            std::string tensor_name = std::string(output_attrs[i].name);
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, tensor_name.c_str()), 
                "assigning output: " + tensor_name + " to request for model " + model_key);
            expected_outputs[tensor_name] = output_attrs[i];
        }

        // Perform inference by calling TRITONSERVER_ServerInferAsync. This
        // call is asynchronous and therefore returns immediately. The
        // completion of the inference and delivery of the response is done
        // by triton by calling the "response complete" callback functions
        // (InferResponseComplete in this case).
        {
            auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
            std::future<TRITONSERVER_InferenceResponse*> completed = p->get_future();

            FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback( 
                irequest, allocator, nullptr /* response_allocator_userp */, InferResponseComplete, 
                reinterpret_cast<void*>(p)), "setting response callback for model " + model_key);

            FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_server.get(), irequest, nullptr /* trace */),
                "running inference for model " + model_key);

            // The InferResponseComplete function sets the std::promise so
            // that this thread will block until the response is returned.
            TRITONSERVER_InferenceResponse* completed_response = completed.get();
            FAIL_IF_ERR(TRITONSERVER_InferenceResponseError(completed_response), 
                "response status for model " + model_key);

            // parse model infer output from response
            parseModelInferResponse(completed_response, model_name, model_version, expected_outputs, output_tensors);

            // delete model infer response
            FAIL_IF_ERR(TRITONSERVER_InferenceResponseDelete(completed_response), 
                "deleting inference response for model " + model_key);
        }

        request_release_future.get();
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestDelete(irequest), 
            "deleting inference request for model " + model_key);

        if (nullptr == response_allocator)
        {
            FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorDelete(allocator), 
                "deleting response allocator for model " + model_key);
        }

        return 0;
    }

    int TritonServerEngine::loadModel(const std::string model_name)
    {
        if (nullptr == m_server.get())
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "triton server not init or init failed, please init first");
            return -1;
        }
        LOG_IF_ERR(TRITONSERVER_ServerLoadModel(m_server.get(), model_name.c_str()), 
            "load model " + model_name + " fail");
        return 0;
    }

    int TritonServerEngine::unloadModel(const std::string model_name)
    {
        if (nullptr == m_server.get())
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "triton server not init or init failed, please init first");
            return -1;
        }
        LOG_IF_ERR(TRITONSERVER_ServerUnloadModel(m_server.get(), model_name.c_str()), 
            "unload model " + model_name + " fail");
        return 0;
    }

} // namespace TRITON_SERVER