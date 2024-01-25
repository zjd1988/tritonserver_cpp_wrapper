/***********************************
******  model_infer.cpp
******
******  Created by zhaojd on 2022/04/26.
***********************************/
#include <chrono>
#include "tool_utils.h"
#include "pystring.h"
#include "common/log.h"
#include "tritonserver_wrapper/tritonserver_c_wrapper.h"

using namespace TRITON_SERVER;

int parseModelInferOption(int argc, char* argv[], CmdLineArgOption& arg_opt)
{
    // 1 init arg options
    cxxopts::Options arg_options("model_infer", "model inference test");
    arg_options.add_options()
        // model name
        ("name", "Model name", cxxopts::value<std::string>())
        // model version
        ("version", "Model version", cxxopts::value<int64_t>())
        // model input data files, if multi inputs separate with comma
        ("input", "Input data files", cxxopts::value<std::vector<std::string>>())
        // model output data file
        ("output", "Output tesnor data to file")
        // model repo path
        ("model_repo_path", "Model repository path for triton server", 
            cxxopts::value<std::string>()->default_value("/models"))
        // backend path
        ("backends_path", "Backends path for triton server", 
            cxxopts::value<std::string>()->default_value("/opt/tritonserver/backends"))
        // repo agent dir
        ("repo_agent_path", "repo agent path for triton server", 
            cxxopts::value<std::string>()->default_value("/opt/tritonserver/repoagents"))
        // benchmark test
        ("benchmark", "benchmark test")
        // benchmark test times
        ("benchmark_num", "benchmark test number", cxxopts::value<int>()->default_value("1"))
        // log level, default is info level
        ("log_level", "Log verbose level for triton server", cxxopts::value<int>()->default_value("0"))
        // help
        ("help", "Print usage");
    arg_options.allow_unrecognised_options();

    // 2 parse arg
    auto parse_result = arg_options.parse(argc, argv);

    // 3 check help arg
    arg_opt.help_flag = false;
    if (parse_result.count("help"))
    {
        arg_opt.help_flag = true;
        std::cout << arg_options.help() << std::endl;
        return -1;
    }

    // 4 check unmatch arg
    const std::vector<std::string>& unmatch = parse_result.unmatched();
    if (parse_result.unmatched().size() > 0)
    {
        std::cout << "contain unsupported options:" << std::endl;
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "contain unsupported options:");
        for (int i = 0; i < unmatch.size(); i++)
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "{}", unmatch[i]);
        return -1;
    }

    // 5 chcek model name/version arg
    if (0 == parse_result.count("name"))
    {
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model name should be set");
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "{}", arg_options.help());
        return -1;
    }
    arg_opt.model_name = parse_result["name"].as<std::string>();

    if (0 == parse_result.count("version"))
    {
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model version should be set");
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "{}", arg_options.help());
        return -1;
    }
    arg_opt.model_version = parse_result["version"].as<int64_t>();

    // 6 check input files and output flag arg
    if (0 != parse_result.count("input"))
        arg_opt.input_files = parse_result["input"].as<std::vector<std::string>>();

    if (0 != parse_result.count("output"))
        arg_opt.output_flag = true;

    // 7 check triton server config
    arg_opt.model_repo_path = parse_result["model_repo_path"].as<std::string>();
    arg_opt.backends_path = parse_result["backends_path"].as<std::string>();
    arg_opt.repo_agent_path = parse_result["repo_agent_path"].as<std::string>();

    // 8 check benchmark config
    if (parse_result.count("benchmark"))
    {
        arg_opt.benchmark = true;
        arg_opt.benchmark_number = parse_result["benchmark_num"].as<int>();
    }

    // LOG_VERBOSE_LEVEL_0 = 0,
    // LOG_VERBOSE_LEVEL_1,
    // LOG_VERBOSE_LEVEL_2,
    arg_opt.log_verbose_level = parse_result["log_level"].as<int>();

    return 0;
}

int main(int argc, char* argv[])
{
    CmdLineArgOption cmd_option;
    ModelContext model_context = nullptr;
    ModelInputOutputNum input_output_num;
    ModelTensor tensor;
    std::vector<ModelTensorAttr> input_attrs;
    std::vector<ModelTensorAttr> output_attrs;
    std::vector<ModelTensor> input_tensors;
    std::vector<ModelTensor> output_tensors;

    if (0 != parseModelInferOption(argc, argv, cmd_option))
        return -1;

    // log cmd options
    std::string input_files= pystring::join(",", cmd_option.input_files);
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "     parsed cmd options:");
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "             model name: {}", cmd_option.model_name);
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "          model version: {}", cmd_option.model_version);
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "            input files: {}", input_files);
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "            output flag: {}", cmd_option.output_flag);
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "        model repo path: {}", cmd_option.model_repo_path);
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "          backends path: {}", cmd_option.backends_path);
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "        repo agent path: {}", cmd_option.repo_agent_path);
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "         benchmark test: {}", cmd_option.benchmark);
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "       benchmark number: {}", cmd_option.benchmark_number);
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "      log verbose level: {}", cmd_option.log_verbose_level);

    // get triton server config
    ServerConfig config = defaultServerConfig(cmd_option.model_repo_path.c_str(), 
        (LogVerboseLevel)cmd_option.log_verbose_level, "", cmd_option.backends_path.c_str(), 
        cmd_option.repo_agent_path.c_str());

    // start triton server with config
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "1 start triton server with config");
    if (0 != initTritonServerWithCustom(&config))
    {
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "init triton server fail");
        goto FINAL;
    }

    // init model context
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "2 init model context");
    if (0 != modelInit(&model_context, cmd_option.model_name.c_str(), cmd_option.model_version))
    {
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "init model context fail");
        goto FINAL;
    }

    // get model info
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "3 get model info");
    if (0 != getModelInfo(model_context, input_output_num, input_attrs, output_attrs))
    {
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "get model info fail");
        goto FINAL;
    }

    // load input files
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "4 load model input tensors");
    if (0 != loadInputTensors(cmd_option.input_files, input_output_num, input_attrs, input_tensors))
    {
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "load model input tensors fail");
        goto FINAL;
    }

    // check model input number equalt to input files number
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "5 check model input number equalt to input files number");
    if (input_output_num.n_input != input_tensors.size())
    {
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model expect inputs number:{}, but get input tensors number:{}",
            input_output_num.n_input, input_tensors.size());
        goto FINAL;
    }

    // model inference
    if (cmd_option.benchmark)
    {
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "6 model benchmark inference");
        if (0 != modelInference(model_context, input_output_num, input_tensors, output_tensors))
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model warmup inference fail");
            goto FINAL;
        }
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        for (auto index = 0; index < cmd_option.benchmark_number; index++)
        {
            if (0 != modelInference(model_context, input_output_num, input_tensors, output_tensors))
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model inference fail");
                goto FINAL;
            }
        }
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::high_resolution_clock::duration elapsed = end - start;
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "model average inference time is {}ms", 
            std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() / cmd_option.benchmark_number);
    }
    else
    {
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "6 model inference");
        if (0 != modelInference(model_context, input_output_num, input_tensors, output_tensors))
        {
            TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model inference fail");
            goto FINAL;
        }
    }

    // save model output
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "7 save model output");
    if (cmd_option.output_flag)
    {
        for (int index = 0; index < output_tensors.size(); index++)
        {
            std::string tensor_name = std::string(output_attrs[index].name);
            std::string output_file = tensor_name + ".npy";
            if (saveModelTensorToNpyFile(output_file, &output_tensors[index]))
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "save tensor {} to file {} fail", 
                    tensor_name, output_file);
                break;
            }
        }
    }

    // release model outputs
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "8 release model outputs");
    if (0 != modelOutputsRelease(model_context, input_output_num.n_output, &output_tensors[0]))
    {
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model outputs release fail");
        goto FINAL;
    }

FINAL:
    // release input tensor buf
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "9 release input tensor buf");
    for (int index = 0; index < input_tensors.size(); index++)
    {
        releaseModelTensor(&input_tensors[index]);
    }

    // destroy model context
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "10 destroy model context");
    modelDestroy(model_context);
    model_context = nullptr;

    // uninit triton server
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "11 uninit triton server");
    uninitTritonServer();
    return 0;
}