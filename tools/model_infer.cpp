/***********************************
******  model_infer.cpp
******
******  Created by zhaojd on 2022/04/26.
***********************************/
#include "tool_utils.h"
#include "pystring.h"
#include "common/log.h"
#include "tritonserver_wrapper/tritonserver_c_wrapper.h"

using namespace TRITON_SERVER;
namespace fs = ghc::filesystem;

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

    // LOG_VERBOSE_LEVEL_0 = 0,
    // LOG_VERBOSE_LEVEL_1,
    // LOG_VERBOSE_LEVEL_2,
    arg_opt.log_verbose_level = parse_result["log_level"].as<int>();

    return 0;
}

int main(int argc, char* argv[])
{
    int index = 0;
    std::string file_type;
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

    // query model input output num
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "3 query model input output num");
    if (0 != modelQuery(model_context, MODEL_QUERY_IN_OUT_NUM, &input_output_num, sizeof(ModelInputOutputNum)))
    {
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "query model input output number fail");
        goto FINAL;
    }

    // query model input tensor attr
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "4 query model input tensor attr");
    input_attrs.resize(input_output_num.n_input);
    if (0 != modelQuery(model_context, MODEL_QUERY_INPUT_ATTR, &input_attrs[0], 
        input_output_num.n_input * sizeof(input_attrs[0])))
    {
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "query model input tensor attr fail");
        goto FINAL;
    }
    for (index = 0; index < input_output_num.n_input; index++)
    {
        std::vector<int64_t> tensor_shape(&input_attrs[index].dims[0], 
            &input_attrs[index].dims[0] + input_attrs[index].num_dim);
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "---------------");
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "         index: {}", input_attrs[index].index);
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "          name: {}", &input_attrs[index].name[0]);
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "      datatype: {}", getTypeString(input_attrs[index].type));
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "         shape: {}", fmt::join(tensor_shape, " "));
    }

    // query model output tensor attr
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "5 query model output tensor attr");
    output_attrs.resize(input_output_num.n_output);
    if (0 != modelQuery(model_context, MODEL_QUERY_OUTPUT_ATTR, &output_attrs[0], 
        input_output_num.n_output * sizeof(output_attrs[0])))
    {
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "query model output tensor attr fail");
        goto FINAL;
    }
    for (index = 0; index < input_output_num.n_output; index++)
    {        
        std::vector<int64_t> tensor_shape(&output_attrs[index].dims[0], 
            &output_attrs[index].dims[0] + output_attrs[index].num_dim);
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "---------------");
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "         index: {}", output_attrs[index].index);
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "          name: {}", &output_attrs[index].name[0]);
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "      datatype: {}", getTypeString(output_attrs[index].type));
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "         shape: {}", fmt::join(tensor_shape, " "));
    }

    // load input files
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "6 load input files");
    if (0 == cmd_option.input_files.size())
    {
        for (index = 0; index < input_output_num.n_input; index++)
        {
            if(0 != loadRandomDataToModelTensor(input_attrs[index], &tensor))
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "load tensor from number:{} tensor attr fail", index);
                break;
            }
            tensor.index = index;
            input_tensors.push_back(tensor);
        }
    }
    else
    {
        for (index = 0; index < cmd_option.input_files.size(); index++)
        {
            fs::path file_path = cmd_option.input_files[index];
            file_type = file_path.extension().string();
            if (!fs::exists(file_path))
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "file {} not exists", file_path.string());
                break;
            }
            if (file_type == ".jpg" || file_type == ".bmp")
            {
                if (0 != loadStbDataToModelTensor(file_path.string(), &tensor))
                {
                    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "load tensor from {} fail", file_path.string());
                    break;
                }
            }
            else if (file_type == ".npy")
            {
                if (0 != loadNpyDataToModelTensor(file_path.string(), &tensor))
                {
                    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "load tensor from {} fail", file_path.string());
                    break;
                }
            }
            else
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "unsupported file type {}", file_path.string());
                break;
            }
            tensor.index = index;
            input_tensors.push_back(tensor);
        }
    }

    // check model input number equalt to input files number
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "7 check model input number equalt to input files number");
    if (input_output_num.n_input != input_tensors.size())
    {
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model expect inputs number:{}, but get input tensors number:{}",
            input_output_num.n_input, input_tensors.size());
        goto FINAL;
    }

    // set model inputs
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "8 set model inputs");
    if (0 != modelInputsSet(model_context, input_output_num.n_input, &input_tensors[0]))
    {
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model inputs set fail");
        goto FINAL;
    }

    // model run
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "9 model run");
    if (0 != modelRun(model_context))
    {
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model run fail");
        goto FINAL;
    }

    // get model output
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "10 get model output");
    output_tensors.resize(input_output_num.n_output);
    if (0 != modelOutputsGet(model_context, input_output_num.n_output, &output_tensors[0]))
    {
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model outputs get fail");
        goto FINAL;
    }

    // save model output
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "11 save model output");
    if (cmd_option.output_flag)
    {
        for (index = 0; index < output_tensors.size(); index++)
        {
            std::string tensor_name = std::string(output_attrs[index].name);
            std::string output_file = tensor_name + ".npy";
            if (saveModelTensorToNpyFile(output_file, output_attrs[index], &output_tensors[index]))
            {
                TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "save tensor {} to file {} fail", 
                    tensor_name, output_file);
                break;
            }
        }
    }

    // release model outputs
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "12 release model outputs");
    if (0 != modelOutputsRelease(model_context, input_output_num.n_output, &output_tensors[0]))
    {
        TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_ERROR, "model outputs release fail");
        goto FINAL;
    }

FINAL:
    // release input tensor buf
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "13 release input tensor buf");
    for (index = 0; index < input_tensors.size(); index++)
    {
        releaseModelTensor(&input_tensors[index]);
    }

    // destroy model context
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "14 destroy model context");
    modelDestroy(model_context);
    model_context = nullptr;

    // uninit triton server
    TRITONSERVER_LOG(TRITONSERVER_LOG_LEVEL_INFO, "15 uninit triton server");
    uninitTritonServer();
    return 0;
}