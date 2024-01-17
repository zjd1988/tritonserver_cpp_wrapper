/***********************************
******  tool_utils.h
******
******  Created by zhaojd on 2024/01/16.
***********************************/
#pragma once
#include "cxxopts.hpp"
#include "ghc/filesystem.h"
#include "common/non_copyable.h"
#include "tritonserver_wrapper/tritonserver_c_wrapper.h"

namespace TRITON_SERVER
{

    typedef struct CmdLineArgOption
    {
        // model infer
        std::string                    model_name;
        int64_t                        model_version;
        std::vector<std::string>       input_files;
        bool                           output_flag = false;
        // model benchmark
        int                            benchmark_times = 0;
        // common
        std::string                    model_repo_path;
        std::string                    backends_path;
        std::string                    repo_agent_path;
        int                            log_verbose_level;
        bool                           help_flag = false;
    } CmdLineArgOption;

    int loadRandomDataToModelTensor(ModelTensorAttr tensor_attr, ModelTensor* tensor);
    int loadStbDataToModelTensor(const std::string file_name, ModelTensor* tensor);
    int loadNpyDataToModelTensor(const std::string file_name, ModelTensor* tensor);
    int saveModelTensorToNpyFile(const std::string file_name, ModelTensorAttr tensor_attr, ModelTensor* tensor);
    void releaseModelTensor(ModelTensor* tensor);

} // namespace TRITON_SERVER