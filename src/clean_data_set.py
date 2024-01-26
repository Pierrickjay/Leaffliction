from sys import argv
import os

def delete_augmented_data(dir):
    for foldername, subdirectory, filenames in os.walk(dir):
            for filename in filenames:
                if filename.find("_") != -1 :
                    os.remove(foldername + "/" + filename)

def main():
    try:
        if not os.path.isdir(argv[1]):
            print("Please enter a directory as a parametter")
            return 1
        delete_augmented_data(argv[1])
    except Exception as err:
        print("Error: ", err)
        return 1


if (__name__ == "__main__"):
    main()
