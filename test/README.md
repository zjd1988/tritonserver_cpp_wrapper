# convert image to npy

## convert mobilenetv2 model test img
```
python convert.py --input ./images/bell.jpg --resize 224 224 --mean 123.675 116.28 103.53 --std 58.395 57.12 57.375 --output ./data/bell.npy
```

## convert yolov5n model test img
```
python convert.py --input ./images/bus.jpg --resize 640 640 --mean 0 0 0 --std 1 1 1 --output ./data/bus.npy
```
