# convert image to npy

## convert mobilenetv2 model test img
```
python convert.py --input ./images/bell.jpg --resize 224 224 --mean 123.675 116.28 103.53 --std 58.395 57.12 57.375 --output ./data/bell.npy
```

## convert yolov5n model test img
```
python convert.py --input ./images/bus.jpg --resize 640 640 --mean 0 0 0 --std 255 255 255 --output ./data/bus.npy --bgr2rgb
```

## mobilenetv2 postprcess 
```
<!-- run model get model output -->
cd build
./model_infer --name mobilenetv2 --version 1 --model_repo_path /workspace/tritonserver_wrapper/test/models/ --output --input ../test/data/bell.npy

<!-- post process -->
cd test
python ./postprocess/mobilenetv2.py ../build/output.npy ./models/mobilenetv2/synset.txt

<!-- result -->
-----TOP 5-----
[494] score=0.98 class="n03017168 chime, bell, gong"
[696] score=0.01 class="n03876231 paintbrush"
[505] score=0.01 class="n03063689 coffeepot"
[792] score=0.00 class="n04208210 shovel"
[899] score=0.00 class="n04560804 water jug"
```

## yolov5n postprcess 
```
<!-- run model get model output -->
./model_infer --name yolov5n --version 1 --model_repo_path /workspace/tritonserver_wrapper/test/models/ --output --input ../test/data/bus.npy

<!-- post process -->
cd test
python ./postprocess/yolov5n.py ./images/bus.jpg ../build/output0.npy ../build/343.npy ../build/345.npy ./yolov5n_result.jpg

<!-- result -->
person @ (210 242 286 511) 0.827
person @ (111 233 203 528) 0.814
person @ (482 222 562 518) 0.614
bus @ (111 140 565 457) 0.513
Detection result save to ./yolov5n_result.jpg

![yolov5n测试结果](https://github.com/zjd1988/tritonserver_cpp_wrapper/tree/main/test/images/yolov5n_result.jpg)
```
