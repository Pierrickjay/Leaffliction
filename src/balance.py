import os
from sys import argv
import random
from Augmentation import augment_images, create_dir_if_needed
from Distribution import retrieve_file_subdir


def inc(max, nameDir, actualsize):
    to_add = 0
    while actualsize + to_add < max :
        print(nameDir)
        img = f"leaves/images/{nameDir}/image (" + str(random.randrange(1, actualsize, 1)) + ").JPG"
        print(img)
        create_dir_if_needed(nameDir)
        if actualsize + to_add + 6 < max:
            augment_images(img, False)

def increment_to_balance(file_count):
    max = file_count.max()
    print(file_count)
    a = 0
    for index in file_count.index:
        inc(max, index, file_count.iloc[a])
        a += 1


def main():
    try:
        if not os.path.isdir(argv[1]):
            print("Please enter a directory as a parametter")
            return 1
        df = retrieve_file_subdir(argv[1])
        filecount = df.count()
        print(filecount.max())
        print(filecount.loc["Grape_Esca"])
        increment_to_balance(filecount)
    except Exception as err:
        print("Error: ", err)
        return 1


if (__name__ == "__main__"):
    main()


