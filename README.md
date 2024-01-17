# tritonserver_cpp_wrapper

## 3rd_party
```
https://github.com/imageworks/pystring.git v1.1.4
https://github.com/gulrak/filesystem.git v1.5.14
https://github.com/nothings/stb/blob/master/stb_image.h v2.29
https://github.com/Tencent/rapidjson.git v1.1.0
https://github.com/gabime/spdlog.git v1.12.0
https://github.com/jarro2783/cxxopts.git v3.3.0
https://github.com/llohse/libnpy.git v1.0.1
```

## build
```
docker pull nvcr.io/nvidia/tritonserver:23.12-py3
git clone https://github.com/zjd1988/tritonserver_cpp_wrapper.git
docker run -it --gpus="device=0" -v $PWD:/workspace/tritonserver_wrapper \
    nvcr.io/nvidia/tritonserver:23.12-py3 /bin/bash
cd /workspace/tritonserver_cpp_wrapper
mkdir build && cd build
cmake .. && make -j4
```


