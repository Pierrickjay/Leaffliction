import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs
import cv2
import os
import argparse
import pandas as pd
from sys import argv

def histogram_with_colors(img, color_spaces):
    plt.figure(figsize=(8, 6))
    for color_space in color_spaces:
        if color_space == "blue":
            channel = img[1:, :, 0]
        elif color_space == "blue-yellow":
            blue_yellow = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, 2]
            channel = cv2.subtract(img[:, :, 2], blue_yellow)
        elif color_space == "green":
            channel = img[1:, :, 1]
        elif color_space == "green-magenta":
            green_magenta = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, 1]
            channel = cv2.subtract(img[:, :, 1], green_magenta)
        elif color_space == "hue":
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            channel = hsv[1:, :, 0]
        elif color_space == "lightness":
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            channel = lab[1:, :, 0]
        elif color_space == "red":
            channel = img[1:, :, 2]
        elif color_space == "saturation":
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            channel = hsv[1:, :, 1]
        elif color_space == "value":
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            channel = hsv[1:, :, 2]
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist = hist / np.sum(hist) * 100
        plt.plot(hist, label= color_space)
    plt.title('Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency in %')
    plt.xlim([0, 256])
    plt.grid(True)
    plt.legend()
    plt.show()

def gaussian_blur(gray_img):
    thresh = pcv.threshold.binary(gray_img=gray_img, threshold=115, object_type="light")
    pcv.plot_image(thresh)

def mask_img(img, thresh):
    result = np.ones_like(img) * 255
    result[thresh == 255] = img[thresh == 255]
    mask_img = result
    pcv.plot_image(result)

def roi_img(img, thresh) :
    gray_img = pcv.rgb2gray_lab(rgb_img=img, channel='a')
    thresh2 = pcv.threshold.triangle(gray_img=gray_img, object_type="dark", xstep= 100)
    edge_ok = pcv.fill(bin_img=thresh2, size=5000)
    mask_with_no_buffer = pcv.fill(bin_img=pcv.invert(gray_img=edge_ok), size=1000)
    green = np.ones_like(img) * 255
    green[mask_with_no_buffer == 255] = img[mask_with_no_buffer == 255]
    green[thresh == 255] = img[thresh == 255]
    green[thresh == 0] = [93, 255, 51]
    green[mask_with_no_buffer == 255] = img[mask_with_no_buffer == 255]
    x, y, h, w = 0, 0, 256, 256
    roi = green[y:y+h, x:x+w]
    cv2.rectangle(green, (x, y), (x+w, y+h), (255, 0, 0),10)
    pcv.plot_image(green)

def pseudo_landmarks(img, thresh):
    pcv.params.debug = "plot"
    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(img=img, mask=thresh)

def analyze_object(img, thresh):
    a_fill_image = pcv.fill(bin_img=thresh, size=3)
    roi = pcv.roi.rectangle(img=img, x=0, y=0, h=256, w=256)
    mask = pcv.roi.filter(mask=a_fill_image, roi=roi, roi_type="partial")
    shape_img = pcv.analyze.size(img=img,labeled_mask=mask, n_labels=1)
    pcv.plot_image(shape_img)

def transfo_img(path):
    img = np.array(Image.open(path))
    gray_img = pcv.rgb2gray_cmyk(rgb_img=img, channel='y')
    thresh = pcv.threshold.binary(gray_img=gray_img, threshold=115, object_type="light")
    # need to apply a filter to tresh to harmonize the pixel

    pcv.plot_image(thresh)
    mask_img(img, thresh)
    roi_img(img, thresh)
    analyze_object(img, thresh)
    pseudo_landmarks(img, thresh)
    histogram_with_colors(img, color_spaces=["blue", "blue-yellow", "green", "green-magenta", "hue", "lightness", "red", "saturation", "value"])


def main(file=None, src=None, dst=None):
    try:
        if file:
            if os.path.isfile(file):
                print("Source File:", src)
        elif src:
            if os.path.isdir(src):
                print("Source Directory:", src)
                if dst:
                    print("Destination Directory:", dst)
                else:
                    print("Precise a src with -dst ")
                    exit(0)
            else:
                print("Precise a src with -src ")
                exit(0)
        else:
            raise FileNotFoundError("Invalid source path:", src)
        pcv.params.debug = "None"
        transfo_img(file)
    except Exception as err:
        print("Error: ", err)
        return 1


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(description="Transformation program")
    parser.add_argument("file", nargs="?", type=str, help="File path")
    parser.add_argument("-src", type=str, help="Source file path or directory")
    parser.add_argument("-dst", type=str, help="Optional destination directory")
    args = parser.parse_args()
    main(args.file, args.src, args.dst)
