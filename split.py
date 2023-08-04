import os # os 관련 기능 사용
import random
import shutil
from glob import glob


def split_yolo_dataset(path, images_path, labels_path, train_pct=0.8, val_pct=0.1):
    os.makedirs(os.path.join(path,"train/images"), exist_ok=True)
    os.makedirs(os.path.join(path,"train/labels"), exist_ok=True)
    
    os.makedirs(os.path.join(path,"test/images"), exist_ok=True)
    os.makedirs(os.path.join(path,"test/labels"), exist_ok=True)
    
    os.makedirs(os.path.join(path,"val/images"), exist_ok=True)
    os.makedirs(os.path.join(path,"val/labels"), exist_ok=True)

    images = glob(images_path + "/*.png")
    random.shuffle(images)
    num_images = len(images)
    num_train = int(num_images * train_pct)
    num_val = int(num_images * val_pct)
    num_test = num_images - (num_train + num_val)

    train_imgs = images[:num_train]
    val_imgs = images[num_train:num_train + num_val]
    test_imgs = images[num_train + num_val:]

    for train_img in train_imgs:
        basename = os.path.basename(train_img).split(".")[0]
        shutil.copy(train_img, os.path.join(path,"train/images"))
        shutil.copy(os.path.join(labels_path, basename + ".txt"), os.path.join(path,"train/labels"))

    for val_img in val_imgs:
        basename = os.path.basename(val_img).split(".")[0]
        shutil.copy(val_img, os.path.join(path,"val/images"))
        shutil.copy(os.path.join(labels_path, basename + ".txt"), os.path.join(path,"val/labels"))

    for test_img in test_imgs:
        basename = os.path.basename(test_img).split(".")[0]
        shutil.copy(test_img, os.path.join(path,"test/images"))
        shutil.copy(os.path.join(labels_path, basename + ".txt"), os.path.join(path,"test/labels"))


if __name__ == "__main__":
    path = "./ultralytics/cfg/car_plate_dataset"
    images_path = os.path.join(path,"images")
    labels_path = os.path.join(path,"labels")

    train_pct = 0.85
    val_pct = 0.1
    split_yolo_dataset(path, images_path, labels_path, train_pct, val_pct)
