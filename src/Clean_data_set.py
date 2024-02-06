from sys import argv
import shutil
import os

def delete_augmented_data(dir, removedir):
    for foldername, subdirectorys, filenames in os.walk(dir):
            if subdirectorys:
                for subdirectory in subdirectorys:
                    for filename in filenames:
                        if filename.find("_") != -1 :
                            print(foldername + "/" + subdirectory +"/"+ filename)
                            os.remove(foldername + "/" + subdirectory +"/"+ filename)
            else:
                for filename in filenames:
                        if filename.find("_") != -1 :
                            os.remove(foldername + "/"+ filename)
    if removedir :
        os.rmdir(dir)
    if os.path.isdir("augmented_directory"):
        shutil.rmtree("augmented_directory")

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
