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
    #define MEM_ALIGN(x, align) (((x) + ((align)-1)) & ~((align)-1))

    void ModelTensorData::randomInitData()
    {
        std::random_device rd;
        std::default_random_engine eng(rd());
        std::uniform_real_distribution<> distr_f(0.0f, 1.0f);
        std::uniform_int_distribution<> distr_i(0, 127);
        int element_count = tensorElementCount();
        if (TIMVX_TENSOR_FLOAT32 == m_type)
        {
            float* tensor_data = (float*)m_data;
            for (int i = 0; i < element_count; i++)
            {
                tensor_data[i] = distr_f(eng);
            }
        }
        else if (TIMVX_TENSOR_INT8 == m_type)
        {
            int8_t* tensor_data = (int8_t*)m_data;
            for (int i = 0; i < element_count; i++)
            {
                tensor_data[i] = distr_i(eng);
            }
        }
        else if (TIMVX_TENSOR_UINT8 == m_type)
        {
            uint8_t* tensor_data = (uint8_t*)m_data;
            for (int i = 0; i < element_count; i++)
            {
                tensor_data[i] = distr_i(eng);
            }
        }
        else if (TIMVX_TENSOR_INT16 == m_type)
        {
            int16_t* tensor_data = (int16_t*)m_data;
            for (int i = 0; i < element_count; i++)
            {
                tensor_data[i] = distr_i(eng);
            }
        }
        else
            TIMVX_LOG(TIMVX_LEVEL_WARN, "not support random init {} type", getTypeString(m_type));
    }

    int ModelTensorData::tensorElementCount()
    {
        if (0 == m_shape.size())
            return 0;
        else
            return std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<int>());
    }

    int ModelTensorData::tensorElementSize()
    {
        int element_size = 1;
        switch (m_type)
        {
            case TIMVX_TENSOR_FLOAT32:
                element_size = sizeof(float);
                break;
            case TIMVX_TENSOR_INT8:
                element_size = sizeof(char);
                break;
            case TIMVX_TENSOR_UINT8:
                element_size = sizeof(unsigned char);
                break;
            case TIMVX_TENSOR_INT16:
                element_size = sizeof(short);
                break;
            default:
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "cannot get {}'s element size", getTypeString(m_type));
                element_size = 0;
                break;
        }
        return element_size;
    }

    int loadStbDataToModelTensor(const std::string file_name, ModelTensor* tensor)
    {
        int height = 0;
        int width = 0;
        int channel = 0;

        unsigned char *image_data = stbi_load(file_name, &width, &height, &channel, 3);
        if (nullptr == image_data)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "stb load data from {} failed!", file_name);
            return -1;
        }
        TIMVX_LOG(TIMVX_LEVEL_INFO, "load image from {}, h*w*c={}*{}*{}", file_name, height, width, channel);
        // stb load image as rgb, need to convert rgb to bgr
        int aligned_size = MEM_ALIGN(height * width * channel, 64);
        uint8_t* bgr_data = (uint8_t*)aligned_alloc(64, aligned_size);
        if (nullptr == bgr_data)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "malloc memory for bgr data fail when load from {}", file_name);
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
        
        std::vector<int> image_shape = {1, height, width, channel};
        m_shape = image_shape;
        m_data = (uint8_t*)bgr_data;
        m_type = TIMVX_TENSOR_UINT8;
        m_format = TIMVX_TENSOR_NHWC;
        m_data_len = height * width * channel;
        return 0;
    }

    int loadNpyDataToModelTensor(const std::string file_name, ModelTensor* tensor)
    {
        std::ifstream stream(file_name, std::ifstream::binary);
        std::string header_str = npy::read_header(stream);
        npy::header_t npy_header = npy::parse_header(header_str);
        if (npy_header.fortran_order)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "currently not support fortran order npy file");
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
        TimvxTensorType tensor_dtype = TIMVX_TENSOR_UINT8;
        if (std::type_index(typeid(float)) == type_index)
            tensor_dtype = TIMVX_TENSOR_FLOAT32;
        else if (std::type_index(typeid(short)) == type_index)
            tensor_dtype = TIMVX_TENSOR_INT16;
        else if (std::type_index(typeid(char)) == type_index)
            tensor_dtype = TIMVX_TENSOR_INT8;
        else if (std::type_index(typeid(unsigned char)) == type_index)
            tensor_dtype = TIMVX_TENSOR_UINT8;
        else
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "unsupported npy data type {}", type_str);
            return -1;
        }
        auto element_count = static_cast<size_t>(npy::comp_size(npy_header.shape));
        int tensor_len = item_size * element_count;
        int aligned_size = MEM_ALIGN(tensor_len, 64);
        uint8_t* tensor_data = (uint8_t*)aligned_alloc(64, aligned_size);
        if (nullptr == tensor_data)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "malloc memory for tensor data fail when load from {}", file_name);
            return -1;
        }
        // read the data
        stream.read((char*)tensor_data, tensor_len);
        stream.close();
        // compute the data size based on the shape
        std::vector<int> tensor_shape(npy_header.shape.begin(), npy_header.shape.end());
        m_shape = tensor_shape;
        m_type = tensor_dtype;
        m_format = TIMVX_TENSOR_NCHW;
        m_data = (uint8_t*)tensor_data;
        m_data_len = tensor_len;
        return 0;
    }

    int saveModelTensorToNpyFile(const std::string file_name, ModelTensor* tensor)
    {
        std::ofstream stream(file_name, std::ofstream::binary);
        if (!stream)
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "failed to open output file {}", file_name);
            return -1;
        }
        std::type_index tensor_index = std::type_index(typeid(float));
        if (TIMVX_TENSOR_FLOAT32 == m_type)
            tensor_index = std::type_index(typeid(float));
        else if (TIMVX_TENSOR_INT8 == m_type)
            tensor_index = std::type_index(typeid(int8_t));
        else if (TIMVX_TENSOR_UINT8 == m_type)
            tensor_index = std::type_index(typeid(uint8_t));
        else if (TIMVX_TENSOR_INT16 == m_type)
            tensor_index = std::type_index(typeid(int16_t));
        else
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "unsupported tensor data type: {}", getTypeString(m_type));
            return -1;
        }
        std::vector<npy::ndarray_len_t> shape_v(m_shape.begin(), m_shape.end());
        bool fortran_order = false;
        npy::dtype_t dtype = npy::dtype_map.at(tensor_index);
        npy::header_t header{ dtype, fortran_order, shape_v };
        npy::write_header(stream, header);
        size_t before = stream.tellp(); //current pos
        stream.write((const char*)m_data, m_data_len);
        if (!stream.bad())
        {
            size_t curr_pos = stream.tellp();
            size_t write_len = curr_pos - before;
            if (m_data_len != write_len)
            {
                TIMVX_LOG(TIMVX_LEVEL_ERROR, "expect write {} bytes, actually write {} bytes to file {}",
                    m_data_len, write_len, file_name);
                return -1;
            }
        }
        else
        {
            TIMVX_LOG(TIMVX_LEVEL_ERROR, "write tensor data to file {} fail", file_name);
            return -1;
        }
        TIMVX_LOG(TIMVX_LEVEL_DEBUG, "write tensor data to file {} success", file_name);
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