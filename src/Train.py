import numpy as np
from PIL import Image, ImageEnhance
from plantcv import plantcv as pcv
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from Balance import balance
import os
import argparse
import zipfile
import shutil


def loadDataset(path, img_size):
    return tf.keras.preprocessing.image_dataset_from_directory(
        path,
        shuffle=True,
        image_size=(img_size, img_size)
    )


def processImgDataSet(path):
    img_path_list = [
         [[foldername, fn, '/'.join(
              [e for e in foldername.split("/") if e not in ["..", "."]])]
          for fn in filenames]
         for foldername, subdirectory, filenames in os.walk(path)
         if len(filenames)]
    img_path_list = np.array([element for sous_liste in
                              img_path_list for element in sous_liste])
    list_path_long = list(set([img[2] for img in img_path_list]))
    img_path_list = [[img[0], img[1], img[2].replace(
        os.path.commonpath(list_path_long) + '/', '')]
         for img in img_path_list]
    img_array = np.array(
         [np.array(Image.open(str(img_path[0] + "/" + img_path[1]), "r"))
          for img_path in img_path_list])
    img_back_removed = [removeBack(img, 5000, 1, 10) for img in img_array]
    img_back_removed_IMG = [Image.fromarray(img_array)
                            for img_array in img_back_removed]
    [os.makedirs("increased/" + path, exist_ok=True)
     for path in list(set([img[2] for img in img_path_list]))]
    [img.save(os.path.join(
        "increased", path[2], path[1].split(".")[0] + ".png"), format="PNG")
     for path, img in zip(img_path_list, img_back_removed_IMG)]
    return


def removeBack(img, size_fill, enhance_val, buffer_size):
    img_img = Image.fromarray(img, mode="RGB")
    contr_img = ImageEnhance.Contrast(img_img).enhance(enhance_val)
    gray_img = pcv.rgb2gray_lab(rgb_img=np.array(contr_img), channel='a')
    thresh = pcv.threshold.triangle(
        gray_img=gray_img, object_type="dark", xstep=100)
    edge_ok = pcv.fill(bin_img=thresh, size=5000)
    mask = pcv.fill(bin_img=pcv.invert(gray_img=edge_ok), size=size_fill)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask_buf = mask.copy()
    if (len(contours)):
        cv2.drawContours(mask_buf,
                         contours[np.argmax([len(c) for c in contours])],
                         -1, (0, 0, 0), buffer_size)
    if ([mask_buf[0, 0], mask_buf[0, -1],
         mask_buf[0, -1], mask_buf[-1, 0]] == [0, 0, 0, 0]):
        mask_buf[0:11, 0:11] = 255
        mask_buf[-11:, -11:] = 255
        mask_buf[0:11, -11:] = 255
        mask_buf[-11:, 0:11] = 255
    mask_buf[0:1, :] = 255
    mask_buf[-1:, :] = 255
    mask_buf[:, 0:1] = 255
    mask_buf[:, -1:] = 255
    mask_buf = pcv.fill(bin_img=mask_buf, size=size_fill)
    img_modified = np.ones_like(img) * 255
    img_modified[mask_buf == 0] = img[mask_buf == 0]
    return img_modified


def getDatasetPartitionTf(ds, train_split=0.85,
                          shuffle=True, shuffle_size=10000):
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    len_train_dataset = int(len(ds) * train_split)
    train_dataset = ds.take(len_train_dataset)
    cv_dataset = ds.skip(len_train_dataset)
    return train_dataset, cv_dataset


def createFinalZip(zipFileName):
    learningFilePath = "model_param.keras"
    classNamesCsv = "class_names.csv"
    imgDir = "increased"
    with zipfile.ZipFile(zipFileName, 'w') as zipf:
        for rootDir, subDir, files in os.walk(imgDir):
            for file in files:
                fullPath = os.path.join(rootDir, file)
                zipf.write(fullPath, os.path.relpath(fullPath, imgDir))
        zipf.write(learningFilePath)
        zipf.write(classNamesCsv)


def processArgs(**kwargs):
    batch_size = 32
    epochs = 15
    path = None
    save_name = "Learning"
    img_size = 256
    for key, value in kwargs.items():
        if value is not None:
            match key:
                case 'batch_size':
                    batch_size = value
                case 'epochs':
                    epochs = value
                case 'path':
                    path = value
                case 'save_name':
                    save_name = value
    return batch_size, epochs, path, save_name, img_size


def main(**kwargs):
    try:
        batch, epochs, path, saveN, imgSize = processArgs(**kwargs)
        assert path is not None, "Please enter a directory path as parametter"
        assert os.path.isdir(path), "Please enter a directory as a parametter"
        input_shape = (imgSize, imgSize, 3)

        # Balance the dataset
        print("Balancing the dataset.....................")
        balance(path)
        print("..........................................done !\n")

        # Modify the dataset before the learning
        print("\nRemoving img background...................")
        processImgDataSet(path)
        print("..........................................done !\n")

        # Datasets
        print("Loading dataset...........................")
        dataset = loadDataset("increased", imgSize)
        train_ds, validation_ds = getDatasetPartitionTf(dataset)
        train_ds = train_ds.cache().shuffle(1000).prefetch(
            buffer_size=tf.data.AUTOTUNE)
        validation_ds = validation_ds.cache().shuffle(1000).prefetch(
            buffer_size=tf.data.AUTOTUNE)
        class_names = dataset.class_names
        print("..........................................done !\n")

        # CNN Model definition
        print("Defining CNN model........................")
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(len(class_names), activation='softmax')
        ])
        print("..........................................done !\n")

        # CNN Learning
        print("Learning phase............................")
        model.build(input_shape=input_shape)
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False),
            optimizer='adam',
            metrics=['accuracy'])
        model.fit(
            train_ds,
            epochs=epochs,
            batch_size=batch,
            verbose=1,
            validation_data=validation_ds
        )
        print("..........................................done !\n")

        # Saving the model
        print("Saving the model..........................")
        model.save('model_param.keras')
        np.savetxt("class_name.csv", class_names, delimiter=',', fmt='%s')
        print("..........................................done !\n")

        # Besoin de creer le zip avec les learning et les images
        print("Creating Learning.zip.....................")
        createFinalZip(saveN + '.zip')
        print("..........................................done !\n")

        # Besoin de creer le zip avec les learning et les images
        print("Removing tmp files........................")
        shutil.rmtree("increased")
        os.remove('model_param.keras')
        os.remove('class_names.csv')
        print("..........................................done !\n")

    except Exception as err:
        print("Error: ", err)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training program for Leaffliction")
    parser.add_argument("--batch_size", "-b", type=int,
                        help="Size of the images batch")
    parser.add_argument("--epochs", "-e", type=int,
                        help="Number of epochs for the training")
    parser.add_argument("--path", "-p", type=str,
                        help="Path to the dataset directory")
    parser.add_argument("--save_name", "-sn", type=str,
                        help="Name of the learning saving file")

    args = parser.parse_args()
    kwargs = {key: getattr(args, key) for key in vars(args)}
    main(**kwargs)
