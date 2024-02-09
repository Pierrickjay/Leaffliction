import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from sys import argv


def save_in(file_path, img, _type):
    filepath_split = os.path.splitext(file_path)
    new_path = filepath_split[0] + _type + filepath_split[1]
    img.save(new_path)


def rotating_img(img, file_path):
    rotated_img = img.rotate(30, fillcolor="#FFFFFF")
    save_in(file_path, rotated_img, "_rotated")
    return rotated_img


def fliping_img(img, file_path):
    flipped_img = ImageOps.flip(img)
    save_in(file_path, flipped_img, "_fliped")
    return flipped_img


def bluring_img(img, file_path):
    blur_img = img.filter(ImageFilter.BLUR)
    save_in(file_path, blur_img, "_blured")
    return blur_img


def scaling_img(img, file_path):
    w, h = img.size
    zoom2 = 5
    img_crop = img.crop(((w // 2) - w / zoom2, (h // 2) - h / zoom2,
                         (w // 2) + w / zoom2, (h // 2) + h / zoom2))
    img_zoomed = img_crop.resize((w, h), Image.LANCZOS)
    save_in(file_path, img_zoomed, "_scaled")
    return img_zoomed


def increase_contrast(img, file_path):
    contr_img = ImageEnhance.Contrast(img).enhance(1.5)
    save_in(file_path, contr_img, "_contrasted")
    return contr_img


def illuminating_img(img, file_path):

    bright_img = ImageEnhance.Brightness(img).enhance(1.5)
    save_in(file_path, bright_img, "_illuminated")
    return bright_img


def close_all(img, rot_img, flip_img,
              blur_img, illum_img, contr_img, zoom_img):
    img.close()
    rot_img.close()
    flip_img.close()
    blur_img.close()
    illum_img.close()
    contr_img.close()
    zoom_img.close()


def plot_img(img, filename, rot_img, flip_img,
             blur_img, illum_img, contr_img, scal_img):
    fig = plt.figure(figsize=(15, 2))
    fig.add_subplot(1, 7, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Original")
    fig.add_subplot(1, 7, 3)
    plt.imshow(rot_img)
    plt.axis('off')
    plt.title("Rotation")
    fig.add_subplot(1, 7, 4)
    plt.imshow(flip_img)
    plt.axis('off')
    plt.title("Flip")
    fig.add_subplot(1, 7, 2)
    plt.imshow(blur_img)
    plt.axis('off')
    plt.title("Blur")
    fig.add_subplot(1, 7, 5)
    plt.imshow(illum_img)
    plt.axis('off')
    plt.title("Illumination")
    fig.add_subplot(1, 7, 6)
    plt.imshow(contr_img)
    plt.axis('off')
    plt.title("Contrast")
    fig.add_subplot(1, 7, 7)
    plt.imshow(scal_img)
    plt.axis('off')
    plt.title("Scaling")
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.set_window_title("Augmentation for " + filename)
    plt.show()
    close_all(img, rot_img, flip_img, blur_img, illum_img, contr_img, scal_img)


def augment_images(filename, show):
    img = Image.open(filename)
    rot_img = rotating_img(img, filename)
    flip_img = fliping_img(img, filename)
    blur_img = bluring_img(img, filename)
    illum_img = illuminating_img(img, filename)
    scal_img = scaling_img(img, filename)
    contr_img = increase_contrast(img, filename)
    if show:
        plot_img(img, filename, rot_img, flip_img,
                 blur_img, illum_img, contr_img, scal_img)
    close_all(img, rot_img, flip_img, blur_img, illum_img, contr_img, scal_img)


def main():
    try:
        assert len(argv) == 2, "Please enter a file path as parametter"
        assert os.path.isfile(argv[1]), "Please enter a file as parametter"
        print(os.path.dirname(argv[1]))
        augment_images(argv[1], True)
    except Exception as err:
        print("Error: ", err)
        return 1


if (__name__ == "__main__"):
    main()
