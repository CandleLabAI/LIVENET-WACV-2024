# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

"""
dataloader for lolv1 dataset
"""
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os


class Lolv1Dataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.low_light_images = os.listdir(opt["low_images"])
        self.low_light_images = [image for image in self.low_light_images if image.endswith(".png")]
        self.normal_light_images = os.listdir(opt["normal_images"])
        self.normal_light_images = [image for image in self.normal_light_images if image.endswith(".png")]
        self.transform = transforms.Compose(
            [
                transforms.Resize((opt["img_size_h"], opt["img_size_w"])),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.low_light_images)

    def __getitem__(self, index):
        low_img = self.low_light_images[index]
        low_img_path = os.path.join(self.opt["low_images"], low_img)
        low_image_rgb = Image.open(low_img_path)
        
        normal_img = self.normal_light_images[index]
        normal_img_path = os.path.join(self.opt["normal_images"], normal_img)
        normal_image_rgb = Image.open(normal_img_path)
        normal_image_gray = normal_image_rgb.convert('L')

        low_image_rgb = self.transform(low_image_rgb)
        normal_image_rgb = self.transform(normal_image_rgb)
        normal_image_gray = self.transform(normal_image_gray)

        return low_image_rgb, normal_image_rgb, normal_image_gray

def get_data(opt):
    train_opt = opt["datasets"]["train"]
    val_opt = opt["datasets"]["val"]

    train_dataset = Lolv1Dataset(opt=train_opt)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_opt["batch_size"],
        shuffle=train_opt["use_shuffle"],
        num_workers=train_opt["num_workers"],
    )
    
    val_dataset = Lolv1Dataset(opt=val_opt)
    val_loader = DataLoader(val_dataset, batch_size=val_opt["batch_size"], shuffle=val_opt["use_shuffle"])

    return train_loader, val_loader


