import argparse
import zipfile
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from PIL import Image
import numpy as np


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


def main(**kwargs):
    try:
        img_path, learning_zip = processArgs(**kwargs)
        if learning_zip is None:
            learning_zip = 'Learning.zip'
        assert img_path is not None, "Please enter a directory path as parametter"
        assert os.path.isfile(img_path), "Please enter a file path for the image"
        assert os.path.isfile(learning_zip), "Something wrong with the learning zip file"
        
        #Recuperation du model
        extract_file_from_zip(learning_zip, "model_param.keras", ".")

        # Loading the model
        model = load_model('model_param.keras')

        # Loading the image
        img = np.array(Image.open("img_path", "r"))

        

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
