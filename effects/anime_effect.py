from scipy import stats  # statistical tools from SciPy
import numpy as np       # matrix manipulation
import cv2               # image processing
import argparse          # command-line arguments
from collections import defaultdict
from tqdm import tqdm    # progress bar
import time


# K-means algorithm to cluster the histogram of image
# Value of K is auto-selected
def animefy(input_image, old=0):
    output = np.array(input_image)
    x, y, channel = output.shape

    # Apply bilateral filter on each channel
    for i in range(channel):
        output[:, :, i] = cv2.bilateralFilter(output[:, :, i], 5, 50, 50)

    # Edge detection
    edge = cv2.Canny(output, 100, 200)

    # Convert image to HSV
    output = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)

    # Create HSV histograms
    hists = [
        np.histogram(output[:, :, 0], bins=181, range=(0, 180))[0],
        np.histogram(output[:, :, 1], bins=256, range=(0, 255))[0],
        np.histogram(output[:, :, 2], bins=256, range=(0, 255))[0]
    ]

    Collect = []
    for h in tqdm(hists, desc="Progress 1 of 2"):
        Collect.append(KHist(h))

    output = output.reshape((-1, channel))
    for i in tqdm(range(channel), desc="Progress 2 of 2"):
        channel1 = output[:, i]
        index = np.argmin(np.abs(channel1[:, np.newaxis] - Collect[i]), axis=1)
        output[:, i] = Collect[i][index]
    output = output.reshape((x, y, channel))
    output = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)

    # Find and draw contours on the filtered image
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output, contours, -1, 0, thickness=1)

    # Convert to other color spaces
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    output2 = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2XYZ)
    output3 = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2HLS)

    if old == 0:
        return output, output2, output3
    else:
        output1 = output.copy()
        output = np.array(output, dtype=np.float64)
        output = cv2.transform(output, np.matrix([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ]))
        output[np.where(output > 255)] = 255
        output = np.array(output, dtype=np.uint8)
        return output1, output2, output3, output


def update_C(C, histogram):
    while True:
        groups = defaultdict(list)
        for i in range(len(histogram)):
            if histogram[i] == 0:
                continue
            d = np.abs(C - i)
            index = np.argmin(d)
            groups[index].append(i)

        new_C = np.array(C)
        for i, indice in groups.items():
            if np.sum(histogram[indice]) == 0:
                continue
            new_C[i] = int(np.sum(indice * histogram[indice]) / np.sum(histogram[indice]))

        if np.sum(new_C - C) == 0:
            break
        C = new_C
    return C, groups


def KHist(hist):
    alpha = 0.001  # p-value threshold
    N = 80         # minimum group size
    C = np.array([128])

    while True:
        C, groups = update_C(C, hist)
        new_C = set()

        for i, indice in groups.items():
            if len(indice) < N:
                new_C.add(C[i])
                continue

            if len(hist[indice]) >= 8:
                _, pval = stats.normaltest(hist[indice])
            else:
                pval = 1.0

            if pval < alpha:
                left = 0 if i == 0 else C[i - 1]
                right = len(hist) - 1 if i == len(C) - 1 else C[i + 1]
                delta = right - left
                if delta >= 3:
                    new_C.add((C[i] + left) / 2)
                    new_C.add((C[i] + right) / 2)
                else:
                    new_C.add(C[i])
            else:
                new_C.add(C[i])

        if len(new_C) == len(C):
            break
        else:
            C = np.array(sorted(new_C))
    return C


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to input image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    if image is None:
        raise ValueError("Error: Cannot read image. Check the file path.")

    start_time = time.time()
    print("Wait, work is in progress...")
    output, _, _, _ = animefy(image, 1)
    end_time = time.time()
    print('Processing time: {:.2f}s'.format(end_time - start_time))

    cv2.imwrite("assets/anime_effect.jpg", output)
    print("Your results are ready!")
