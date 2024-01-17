/***********************************
******  tool_utils.cpp
******
******  Created by zhaojd on 2024/01/16.
***********************************/
#include <random>
#include <numeric>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "npy.hpp"
#include "tool_utils.h"
#include "common/log.h"

namespace TRITON_SERVER
{
    #define ALIAN_SIZE 64
    #define MEM_ALIGN(x, align) (((x) + ((align)-1)) & ~((align)-1))

    int64_t tensorElementCount(ModelTensorAttr tensor_attr)
    {
        if (0 == tensor_attr.num_dim)
            return 0;
        else
            return std::accumulate(&tensor_attr.dims[0], &tensor_attr.dims[0] + tensor_attr.num_dim, 
                1, std::multiplies<int64_t>());
    }

    int tensorElementSize(ModelTensorAttr tensor_attr)
    {
        int element_size = 1;
        switch (tensor_attr.type)
        {
            case TENSOR_TYPE_INT32:
            case TENSOR_TYPE_UINT32:
            case TENSOR_TYPE_FP32:
                element_size = sizeof(float);
                break;
            case TENSOR_TYPE_BOOL:
            case TENSOR_TYPE_BYTES:
            case TENSOR_TYPE_INT8:
            case TENSOR_TYPE_UINT8:
                element_size = sizeof(char);
                break;
            case TENSOR_TYPE_INT16:
            case TENSOR_TYPE_UINT16:
            case TENSOR_TYPE_FP16:
                element_size = sizeof(short);
                break;
            case TENSOR_TYPE_INT64:
            case TENSOR_TYPE_UINT64:
            case TENSOR_TYPE_FP64:
                element_size = sizeof(int64_t);
            default:
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "cannot get {}'s element size", 
                    getTypeString(tensor_attr.type));
                element_size = 0;
                break;
        }
        return element_size;
    }

    template<typename T>
    void randomInitTensorData(void* data, int count, std::default_random_engine& eng)
    {
        T* tensor_data = (T*)data;
        std::uniform_real_distribution<float> distr_f(0.0f, 1.0f);
        std::uniform_int_distribution<int> distr_i(0, 255);
        if (typeid(int8_t) == typeid(T) || typeid(int16_t) == typeid(T) || 
            typeid(int32_t) == typeid(T) || typeid(int64_t) == typeid(T))
        {
            for (int i = 0; i < count; i++)
            {
                tensor_data[i] = (T)(128 - distr_i(eng));
            }
        }
        else if (typeid(uint8_t) == typeid(T) || typeid(uint16_t) == typeid(T) || 
            typeid(uint32_t) == typeid(T) || typeid(uint64_t) == typeid(T))
        {
            for (int i = 0; i < count; i++)
            {
                tensor_data[i] = (T)distr_i(eng);
            }
        }
        else
        {
            for (int i = 0; i < count; i++)
            {
                tensor_data[i] = (T)distr_f(eng);
            }
        }
        return;
    }

    int loadRandomDataToModelTensor(ModelTensorAttr tensor_attr, ModelTensor* tensor)
    {
        std::random_device rd;
        std::default_random_engine eng(rd());
        int element_count = tensorElementCount(tensor_attr);
        int element_size = tensorElementSize(tensor_attr);
        uint8_t* tensor_data = (uint8_t*)aligned_alloc(ALIAN_SIZE, element_count * element_size);
        if (nullptr == tensor_data)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "malloc memory buffer for tensor {} fail", 
                &tensor_attr.name[0]);
            return -1;
        }
        switch (tensor_attr.type)
        {
            case TENSOR_TYPE_FP32:
                randomInitTensorData<float>(tensor_data, element_count, eng);
                break;
            case TENSOR_TYPE_INT8:
                randomInitTensorData<int8_t>(tensor_data, element_count, eng);
                break;
            case TENSOR_TYPE_UINT8:
                randomInitTensorData<uint8_t>(tensor_data, element_count, eng);
                break;
            case TENSOR_TYPE_INT16:
                randomInitTensorData<int16_t>(tensor_data, element_count, eng);
                break;
            case TENSOR_TYPE_UINT16:
                randomInitTensorData<uint16_t>(tensor_data, element_count, eng);
                break;
            case TENSOR_TYPE_INT32:
                randomInitTensorData<int32_t>(tensor_data, element_count, eng);
                break;
            case TENSOR_TYPE_UINT32:
                randomInitTensorData<uint32_t>(tensor_data, element_count, eng);
                break;
            case TENSOR_TYPE_INT64:
                randomInitTensorData<int64_t>(tensor_data, element_count, eng);
                break;
            case TENSOR_TYPE_UINT64:
                randomInitTensorData<uint64_t>(tensor_data, element_count, eng);
                break;
            default:
            {
                free(tensor_data);
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "not support random init {} type for tensor {}", 
                    getTypeString(tensor_attr.type), &tensor_attr.name[0]);
                return -1;
            }
        }
        tensor->buf = tensor_data;
        tensor->size = element_count * element_size;
        tensor->type = tensor_attr.type;
        return 0;
    }

    int loadStbDataToModelTensor(const std::string file_name, ModelTensor* tensor)
    {
        if (nullptr == tensor)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "load {} fail, input tensor is nullptr", 
                file_name);
            return -1;
        }
        int height = 0;
        int width = 0;
        int channel = 0;
        unsigned char *image_data = stbi_load(file_name.c_str(), &width, &height, &channel, 3);
        if (nullptr == image_data)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "stb load data from {} failed!", file_name);
            return -1;
        }
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "load image from {}, h*w*c={}*{}*{}", 
            file_name, height, width, channel);
        // stb load image as rgb, need to convert rgb to bgr
        uint8_t* bgr_data = (uint8_t*)aligned_alloc(ALIAN_SIZE, height * width * channel);
        if (nullptr == bgr_data)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "load {} fail, malloc memory buffer is nullptr", 
                file_name);
            stbi_image_free(image_data);
            return -1;
        }
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                for (int k = 0; k < channel; k++)
                {
                    int src_index = i * width * channel + j * channel + k;
                    int dst_index = i * width * channel + j * channel + channel - k - 1;
                    bgr_data[dst_index] = image_data[src_index];
                }
            }
        }
        stbi_image_free(image_data);

        tensor->buf = bgr_data;
        tensor->size = height * width * channel;
        tensor->type = TENSOR_TYPE_UINT8;
        return 0;
    }

    int loadNpyDataToModelTensor(const std::string file_name, ModelTensor* tensor)
    {
        if (nullptr == tensor)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "load {} fail, input tensor is nullptr", 
                file_name);
            return -1;
        }
        std::ifstream stream(file_name, std::ifstream::binary);
        std::string header_str = npy::read_header(stream);
        npy::header_t npy_header = npy::parse_header(header_str);
        if (npy_header.fortran_order)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "load {} fail, currently not support fortran order", 
                file_name);
            return -1;
        }
        std::type_index type_index = std::type_index(typeid(unsigned char));
        std::string type_str = npy_header.dtype.str();
        int item_size = sizeof(char);
        for (auto it = npy::dtype_map.begin(); it != npy::dtype_map.end(); it++)
        {
            if (0 == it->second.str().compare(type_str))
            {
                type_index = it->first;
                item_size = it->second.itemsize;
            }
        }
        TensorDataType tensor_dtype = TENSOR_TYPE_UINT8;
        if (std::type_index(typeid(float)) == type_index)
            tensor_dtype = TENSOR_TYPE_FP32;
        if (std::type_index(typeid(double)) == type_index)
            tensor_dtype = TENSOR_TYPE_FP64;
        else if (std::type_index(typeid(int64_t)) == type_index)
            tensor_dtype = TENSOR_TYPE_INT64;
        else if (std::type_index(typeid(int32_t)) == type_index)
            tensor_dtype = TENSOR_TYPE_INT32;
        else if (std::type_index(typeid(int16_t)) == type_index)
            tensor_dtype = TENSOR_TYPE_INT16;
        else if (std::type_index(typeid(char)) == type_index)
            tensor_dtype = TENSOR_TYPE_INT8;
        else if (std::type_index(typeid(uint64_t)) == type_index)
            tensor_dtype = TENSOR_TYPE_UINT64;
        else if (std::type_index(typeid(uint32_t)) == type_index)
            tensor_dtype = TENSOR_TYPE_UINT32;
        else if (std::type_index(typeid(uint16_t)) == type_index)
            tensor_dtype = TENSOR_TYPE_UINT16;
        else if (std::type_index(typeid(unsigned char)) == type_index)
            tensor_dtype = TENSOR_TYPE_UINT8;
        else
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "load {} fail, unsupported npy data type {}", 
                file_name, type_str);
            return -1;
        }
        auto element_count = static_cast<size_t>(npy::comp_size(npy_header.shape));
        int tensor_len = item_size * element_count;
        uint8_t* tensor_data = (uint8_t*)aligned_alloc(ALIAN_SIZE, tensor_len);
        if (nullptr == tensor_data)
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "load {} fail, malloc memory buffer is nullptr", 
                file_name);
            return -1;
        }
        // read the data
        stream.read((char*)tensor_data, tensor_len);
        stream.close();

        tensor->buf = tensor_data;
        tensor->size = tensor_len;
        tensor->type = tensor_dtype;        
        return 0;
    }

    int saveModelTensorToNpyFile(const std::string file_name, ModelTensorAttr tensor_attr, ModelTensor* tensor)
    {
        std::ofstream stream(file_name, std::ofstream::binary);
        if (!stream.is_open())
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "failed to open output file {}", file_name);
            return -1;
        }
        std::type_index tensor_index = std::type_index(typeid(float));
        if (TENSOR_TYPE_FP32 == tensor_attr.type)
            tensor_index = std::type_index(typeid(float));
        else if (TENSOR_TYPE_INT8 == tensor_attr.type)
            tensor_index = std::type_index(typeid(int8_t));
        else if (TENSOR_TYPE_INT16 == tensor_attr.type)
            tensor_index = std::type_index(typeid(int16_t));
        else if (TENSOR_TYPE_INT32 == tensor_attr.type)
            tensor_index = std::type_index(typeid(int32_t));
        else if (TENSOR_TYPE_INT64 == tensor_attr.type)
            tensor_index = std::type_index(typeid(int64_t));
        else if (TENSOR_TYPE_UINT8 == tensor_attr.type)
            tensor_index = std::type_index(typeid(uint8_t));
        else if (TENSOR_TYPE_UINT16 == tensor_attr.type)
            tensor_index = std::type_index(typeid(uint16_t));
        else if (TENSOR_TYPE_UINT32 == tensor_attr.type)
            tensor_index = std::type_index(typeid(uint32_t));
        else if (TENSOR_TYPE_UINT64 == tensor_attr.type)
            tensor_index = std::type_index(typeid(uint64_t));
        else
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "tensor:{} data type: {} not supported", 
                &tensor_attr.name[0], getTypeString(tensor_attr.type));
            return -1;
        }
        std::vector<npy::ndarray_len_t> shape_v;
        for (auto i = 0; i < tensor_attr.num_dim; i++)
        {
            shape_v.push_back(tensor_attr.dims[i]);
        }
        bool fortran_order = false;
        npy::dtype_t dtype = npy::dtype_map.at(tensor_index);
        npy::header_t header{ dtype, fortran_order, shape_v };
        npy::write_header(stream, header);
        size_t before = stream.tellp(); //current pos
        stream.write((const char*)tensor->buf, tensor->size);
        if (!stream.bad())
        {
            size_t curr_pos = stream.tellp();
            size_t write_len = curr_pos - before;
            if (tensor->size != write_len)
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "expect write {} bytes, actually write {} bytes to file {}",
                    tensor->size, write_len, file_name);
                return -1;
            }
        }
        else
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "write tensor data to file {} fail", file_name);
            return -1;
        }
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "write tensor data to file {} success", file_name);
        return 0;
    }

    void releaseModelTensor(ModelTensor* tensor)
    {
        if (nullptr == tensor)
            return;
        free(tensor->buf);
        tensor->buf = nullptr;
    }

} // namespace TRITON_SERVER