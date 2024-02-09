def processImgDataSet(path):
    if os.path.isdir("increased"):
        print("The increased directory already exists. No modification made.")
        return
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
