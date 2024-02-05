from sys import argv
import os

def delete_augmented_data(dir, removedir):
    for foldername, subdirectory, filenames in os.walk(dir):
            for filename in filenames:
                if filename.find("_") != -1 :
                    os.remove(foldername + "/" + filename)
    if removedir :
        os.rmdir(dir)
    if os.path.isdir("augmented_directory"):
        len_dir = os.listdir("augmented_directory")
        if len(len_dir) == 0 :
            print("ouin")
            os.rmdir("augmented_directory")
        else :
            for subdir in len_dir:
                subdir_path = os.path.join("augmented_directory", subdir)
                delete_augmented_data(subdir_path, True)

def main():
    try:
        assert len(argv) == 2, "Please enter a directory path as parametter"
        assert os.path.isdir(argv[1]), "Please enter a directory as a parametter"
        delete_augmented_data(argv[1], False)
    except Exception as err:
        print("Error: ", err)
        return 1


if (__name__ == "__main__"):
    main()
