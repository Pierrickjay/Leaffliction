import argparse
import zipfile
import os
from tensorflow.keras.models import load_model
from Train import removeBack
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def processArgs(**kwargs):
    img_path = None
    learning_zip = None
    for key, value in kwargs.items():
        if value is not None:
            match key:
                case 'img':
                    img_path = value
                case 'path_learning_zip':
                    learning_zip = value
    return img_path, learning_zip


def extract_file_from_zip(zip_file_name, internal_file_path, destination_directory):
    with zipfile.ZipFile(zip_file_name, 'r') as zipf:
        zipf.extract(internal_file_path, destination_directory)


def printResult(img, imgModified, predictedClass):
    fig = plt.figure(figsize=(7.5, 6))
    fig.patch.set_facecolor('black')
    gs = GridSpec(nrows=2, ncols=2, width_ratios=[1, 1], height_ratios=[2, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img, aspect='auto')
    ax0.axis('off')
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(imgModified, aspect='auto')
    ax1.axis('off')
    txt_part1 = "===       DL classification       ==="
    txt_part2 = "Class predicted : " + predictedClass
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_facecolor('black')
    ax2.text(0.5, 0.7, txt_part1, color='white', fontsize=25, ha='center', va='center')
    ax2.text(0.5, 0.4, txt_part2, color='white', fontsize=14, ha='center', va='center')
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98,wspace=0.05, hspace=0.2)
    manager = plt.get_current_fig_manager()
    manager.set_window_title("Leaffliction prediction result")
    plt.show()


def main(**kwargs):
    try:
        img_path, learning_zip = processArgs(**kwargs)
        if learning_zip is None:
            learning_zip = 'Learning.zip'
        assert img_path is not None, "Please enter a directory path as parametter"
        assert os.path.isfile(img_path), "Please enter a file path for the image"
        assert os.path.isfile(learning_zip), "Something wrong with the learning zip file"
        print("\n")
        #Recuperation du model
        extract_file_from_zip(learning_zip, "model_param.keras", ".")
        extract_file_from_zip(learning_zip, "class_names.csv", ".")

        # Loading the model
        model = load_model('model_param.keras')
        class_names = np.genfromtxt('class_names.csv', delimiter=',', dtype=str)
        os.remove('model_param.keras')
        os.remove('class_names.csv')

        # Loading the image
        img = np.array(Image.open(img_path, "r"))
        imgModified = removeBack(img, 5000, 1, 10)
        modelWidth = model.layers[0].get_config()['batch_input_shape'][1]
        modelHeigh = model.layers[0].get_config()['batch_input_shape'][2]
        y_pred = model.predict(imgModified.reshape(1, modelWidth, modelHeigh, 3))
        predictedClass = class_names[np.argmax(y_pred)]

        printResult(img, imgModified, predictedClass)

    except Exception as err:
        print("Error: ", err)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training program for Leaffliction")
    parser.add_argument("--img", "-i", type=str,
                        help="Path to the image for witch we want a prediction")
    parser.add_argument("--path_learning_zip", "-p", type=str,
                        help="Path to the zip file containing the learning")

    args = parser.parse_args()
    kwargs = {key: getattr(args, key) for key in vars(args)}
    main(**kwargs)
