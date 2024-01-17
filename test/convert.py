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
    parser.add_argument('--mean', type=float, nargs='+', default=[0.0, 0.0, 0.0], help='mean values')
    parser.add_argument('--std', type=float, nargs='+', default=[1.0, 1.0, 1.0], help='std values')
    parser.add_argument('--output', type=str, default="./data/test.npy", help='output numpy file path')
    # parser.add_argument('--dtype', type=str, default='i8', help='dtype of npy, i8/fp32')
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if len(args.resize):
        resized_image, _, _ = letterbox(image, args.resize)
    else:
        resized_image = image
    if args.bgr2rgb:
        chw_img = resized_image[:, :, ::-1].transpose(2, 0, 1)
    else:
        chw_img = resized_image.transpose(2, 0, 1)
    chw_np = np.ascontiguousarray(chw_img)
    std = np.array(args.std).reshape((3,1,1))
    mean = np.array(args.mean).reshape((3,1,1))
    norm_np = (chw_np - mean) / std
    output_np = np.expand_dims(norm_np, 0).astype(np.float)
    np.save(args.output, output_np)