import numpy as np
from PIL import Image, ImageEnhance
from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs
import cv2
import tensorflow as tf
import os
import argparse


def loadDataset(path, img_size, batch_size):
    return tf.keras.preprocessing.image_dataset_from_directory(
	    path, 
	    shuffle = True, 
	    image_size = (img_size, img_size), 
	    batch_size = batch_size
    )


def remove_back(img, size_fill, enhance_val, buffer_size):
	img_img = Image.fromarray(img, mode="RGB")
	contr_img = ImageEnhance.Contrast(img_img).enhance(enhance_val)
	gray_img = pcv.rgb2gray_lab(rgb_img=np.array(contr_img), channel='a')
	thresh = pcv.threshold.triangle(gray_img=gray_img, object_type="dark", xstep= 100)
	edge_ok = pcv.fill(bin_img=thresh, size=5000)
	mask = pcv.fill(bin_img=pcv.invert(gray_img=edge_ok), size=size_fill)
	contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	mask_with_buffer = mask.copy()
	if (len(contours)):
		cv2.drawContours(mask_with_buffer, contours[np.argmax([len(c) for c in contours])], -1, (0, 0, 0), buffer_size)
	if (mask_with_buffer[0,0] == 0 and mask_with_buffer[0,-1] == 0 and mask_with_buffer[0,-1] == 0 and mask_with_buffer[-1,0] == 0):
		mask_with_buffer[0:11,0:11] = 255
		mask_with_buffer[-11:,-11:] = 255
		mask_with_buffer[0:11,-11:] = 255
		mask_with_buffer[-11:,0:11] = 255
	mask_with_buffer[0:1,:] = 255
	mask_with_buffer[-1:,:] = 255
	mask_with_buffer[:,0:1] = 255
	mask_with_buffer[:,-1:] = 255
	mask_with_buffer = pcv.fill(bin_img=mask_with_buffer, size=size_fill)
	img_modified = np.ones_like(img) * 255
	img_modified[mask_with_buffer == 0] = img[mask_with_buffer == 0]
	return img_modified


def get_dataset_partition_tf(ds, train_split = 0.85, shuffle = True, shuffle_size = 10000):
	if shuffle:
		ds = ds.shuffle(shuffle_size, seed = 12)
	len_train_dataset = int(len(ds) * train_split)
	train_dataset = ds.take(len_train_dataset)
	cv_dataset = ds.skip(len_train_dataset)
	return train_dataset, cv_dataset


def main(**kwargs):
    batch_size = 32
    epochs = 15
    path = None
    save_dir = ""
    save_name = "learnings"
    img_size = 256
    for key, value in kwargs.items():
        match key:
            case 'batch_size':
                batch_size = value
            case 'epochs':
                epochs = value
            case 'path':
                path = value
            case 'save_dir':
                save_dir = value
            case 'save_name':
                save_name = value
    try:
        assert path is not None, "Please enter a directory path as parametter"
        assert os.path.isdir(path), "Please enter a directory as a parametter"
        input_shape = (batch_size, img_size, img_size, 3)

        # Modify the dataset before the learning
        # to do Besoin de prendre tous les fichiers du dossier et de les modifier
        # Datasets
        dataset = loadDataset(path, img_size, batch_size)
        train_ds, validation_ds = get_dataset_partition_tf(dataset)
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
        validation_ds = validation_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
        class_names = dataset.class_names

        # CNN Model definition
        model = Sequential([
            Conv2D(32, (3,3), activation = 'relu', input_shape = input_shape),
            MaxPooling2D((2,2)),
            Conv2D(32, (3,3),  activation = 'relu'),
            MaxPooling2D((2,2)),
            Flatten(),
            Dense(64, activation = 'relu'), 
            Dense(len(class_names), activation = 'softmax')
        ])

        # CNN Learning
        model.build(input_shape = input_shape)
        model.compile(
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
            optimizer = 'adam',
            metrics = ['accuracy'])
        history = model.fit(
            train_ds,
            epochs = epochs,
            batch_size = batch_size,
            verbose = 1, 
            validation_data = validation_ds
        )
        model.save(save_dir + save_name + '.keras')

    except Exception as err:
        print("Error: ", err)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training program for Leaffliction")
    parser.add_argument("--batch_size", "-b", type=int, help="Size of the images batch")
    parser.add_argument("--epochs", "-e", type=int, help="Number of epochs for the training")
    parser.add_argument("--path", "-p", type=str, help="Path to the dataset directory")
    parser.add_argument("--save_dir", "-sd", type=str, help="Path to the learning saving directory")
    parser.add_argument("--save_name", "-sn", type=str, help="Name of the learning saving file")

    args = parser.parse_args()
    kwargs = {key: getattr(args, key) for key in vars(args)}
    main(**kwargs)