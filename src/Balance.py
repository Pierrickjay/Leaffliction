import os
from sys import argv
import random
from PIL import Image
from Augmentation import create_dir_if_needed, augment_images
from Augmentation import rotating_img, fliping_img, bluring_img
from Augmentation import illuminating_img, scaling_img, increase_contrast
from Distribution import retrieve_file_subdir


def inc_to_numb(to_add, img):
    img_created = Image.open(img)
    augmented_func = [
        rotating_img,
        fliping_img,
        bluring_img,
        illuminating_img,
        scaling_img,
        increase_contrast
    ]
    for i, augment_func in enumerate(augmented_func):
        if i <= to_add:
            img_saved = augment_func(img_created, img)
            img_saved.close()


def inc(max, nameDir, actualsize):
    to_add = 0
    print("max to reach = ",max)
    print("nb of image now = ", actualsize)
    fn = os.listdir(nameDir)
    a = 0
    create_dir_if_needed(nameDir)
    while actualsize + to_add <= max and a < len(fn):
        img = f"{nameDir}/{fn[a]}"
        if actualsize + to_add + 6 <= max:
            augment_images(img, False)
            to_add += 6
        else:
            inc_to_numb(max - (actualsize + to_add), img)
            to_add += max - (actualsize + to_add)
            break
        a += 1
    if actualsize + to_add < max:
        inc(max, nameDir, actualsize + to_add)

def increment_to_balance(path, file_count):
    max = file_count.max()
    print(file_count)
    a = 0
    print(path)
    for index in file_count.index:
        print(path + index)
        inc(max, path + index, file_count.iloc[a])
        a += 1


def main():
    try:
        assert len(argv) == 2, "Please enter a directory path as parametter"
        if not os.path.isdir(argv[1]):
            print("Please enter a directory as a parametter")
            return 1
        df = retrieve_file_subdir(argv[1])
        filecount = df.count()
        increment_to_balance(argv[1], filecount)
    except Exception as err:
        print("Error: ", err)
        return 1


if (__name__ == "__main__"):
    main()
