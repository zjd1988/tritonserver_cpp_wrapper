import sys
import numpy as np
import cv2
from scipy.special import softmax

if __name__ == "__main__":
    model_output_path = sys.argv[1]
    labes_path = sys.argv[2]
    mobilenetv2_output = np.load(model_output_path)
    with open(labes_path, 'r') as f:
        labels = [l.rstrip() for l in f]

    scores = softmax(mobilenetv2_output)
    # print the top-5 inferences class
    scores = np.squeeze(scores)
    a = np.argsort(scores)[::-1]
    print('-----TOP 5-----')
    for i in a[0:5]:
        print('[%d] score=%.2f class="%s"' % (i, scores[i], labels[i]))