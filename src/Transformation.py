import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from Distribution import retrieve_file_subdir
from Augmentation import save_in
from Pseudo_landmark_change import x_axis_pseudolandmarks
from plantcv import plantcv as pcv
import cv2
import sys
import os
import argparse
import pandas as pd
from sys import argv


def histogram_with_colors(img, color_spaces, to_save=False):
    histograms = []
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
        histograms.append((color_space, hist))
    return histograms

def plot_hist(histo, output_path=None, to_save=False):
    plt.figure(figsize=(8, 6))
    for hist in (histo):
        color_space = hist[0]  # Assign a default label if color space info is not available
        plt.plot(hist[1], label=color_space)
    plt.title('Histograms')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.grid(True)
    plt.legend()
    if to_save == True:
        plt.savefig(output_path)  # Save the plot as an image file
        plt.close()  # Close the plot to free up memory
    else:
        plt.show()


def gaussian_blur(gray_img):
    thresh = pcv.threshold.binary(gray_img=gray_img, threshold=115, object_type="light")
    pcv.plot_image(thresh)

def mask_img(img, thresh):
    result = np.ones_like(img) * 255
    result[thresh == 255] = img[thresh == 255]
    return result

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
    cv2.rectangle(green, (x, y), (x+w, y+h), (255, 0, 0),10)
    return green

def pseudo_landmarks(img, thresh, plot=True):
    top, bottom, center_v, img2 = x_axis_pseudolandmarks(img=img, mask=thresh)
    return img2

def analyze_object(img, thresh):
    a_fill_image = pcv.fill(bin_img=thresh, size=3)
    roi = pcv.roi.rectangle(img=img, x=0, y=0, h=256, w=256)
    mask = pcv.roi.filter(mask=a_fill_image, roi=roi, roi_type="partial")
    shape_img = pcv.analyze.size(img=img,labeled_mask=mask, n_labels=1)
    return shape_img

def plot_img(imgs):
    for img in imgs:
        pcv.plot_image(img)

def transfo_all(src, dest):
    df = retrieve_file_subdir(src)
    print(df)
    print(df.index)
    for column in df.columns:
        if not os.path.isdir(dest + "/" + os.path.dirname(df.loc[0, column])):
            os.makedirs(dest + "/" + os.path.dirname(df.loc[0, column]))
        for value in df[column]:
            # print(value)
            img = np.array(Image.open(value))
            gray_img = pcv.rgb2gray_cmyk(rgb_img=img, channel='y')
            thresh = pcv.threshold.binary(gray_img=gray_img, threshold=115, object_type="light")
            # need to apply a filter to tresh to harmonize the pixel
            mask = mask_img(img, thresh)
            roi = roi_img(img, thresh)
            analy = analyze_object(img, thresh)
            pseud = pseudo_landmarks(img, thresh, False)
            histo = histogram_with_colors(img, color_spaces=["blue", "blue-yellow", "green", "green-magenta", "hue", "lightness", "red", "saturation", "value"], to_save=True)
            plot_hist(histo, dest + "/" + value + "_histo.png", to_save=True)
            cv2.imwrite(dest + "/" + value + "_tresh.png" , thresh)
            cv2.imwrite(dest + "/" + value + "_mask.png" , mask)
            cv2.imwrite(dest + "/" + value + "_analyze.png", analy)
            cv2.imwrite(dest + "/" + value + "_roi.png", roi)
            cv2.imwrite(dest + "/" + value + "_pseu.png", pseud)
            cv2.imwrite(dest + "/" + value + "_tresh.png" ,thresh)


def transfo_img(path=None, src=None, dest=None):
    if src:
        transfo_all(src, dest)
    else:
        img = np.array(Image.open(path))
        gray_img = pcv.rgb2gray_cmyk(rgb_img=img, channel='y')
        thresh = pcv.threshold.binary(gray_img=gray_img, threshold=115, object_type="light")
        # need to apply a filter to tresh to harmonize the pixel
        mask_imag = mask_img(img, thresh)
        roi_imag = roi_img(img, thresh)
        analyze = analyze_object(img, thresh)
        histo = histogram_with_colors(img, color_spaces=["blue", "blue-yellow", "green", "green-magenta", "hue", "lightness", "red", "saturation", "value"])
        pseud = pseudo_landmarks(img, thresh)
        plot_img([img, thresh, mask_imag, roi_imag, analyze, pseud])
        plot_hist(histo)


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
        transfo_img(file, src, dst)
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
