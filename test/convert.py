import cv2
import argparse
import numpy as np

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert image file to numpy file', add_help=True)
    parser.add_argument('--input', type=str, default="./images/test.jpg", help='input image file path')
    parser.add_argument('--resize', type=int, nargs='+', help='resize size')
    parser.add_argument('--bgr2rgb', action='store_true', help='bgr to rgb')
    parser.add_argument('--nhwc2nchw', action='store_true', help='nhwc to nchw')
    parser.add_argument('--mean', type=float, nargs='+', default=[0.0, 0.0, 0.0], help='mean values')
    parser.add_argument('--std', type=float, nargs='+', default=[1.0, 1.0, 1.0], help='std values')
    parser.add_argument('--output', type=str, default="./data/test.npy", help='output numpy file path')
    parser.add_argument('--dtype', type=str, default='fp32', help='dtype of npy, u8/i8/fp32')
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if len(args.resize):
        resized_image, _, _ = letterbox(image, args.resize)
    else:
        resized_image = image
    if args.bgr2rgb:
        resized_image = resized_image[:, :, ::-1]

    img_np = np.ascontiguousarray(resized_image)
    std = np.array(args.std).reshape((1,1,3))
    mean = np.array(args.mean).reshape((1,1,3))
    norm_np = (img_np - mean) / std
    if args.nhwc2nchw:
        norm_np = norm_np.transpose(2, 0, 1)
    nptypes_map = {"u8": np.uint8, "i8": np.int8, "fp32": np.float32}
    if args.dtype in nptypes_map.keys():
        output_np = np.expand_dims(norm_np, 0).astype(nptypes_map[args.dtype])
    else:
        print("input unsupported dtype {}, current only support {}".format(args.dtype, nptypes_map))
    np.save(args.output, output_np)
