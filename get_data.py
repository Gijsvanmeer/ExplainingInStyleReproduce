import numpy as np
import torch
import torchvision.models as models
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# for data with labels (afhq, celeba)
def create_dataset_classes(img_folder, size, classes):

    data = []

    for i, dir1 in enumerate(classes):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            img_path = os.path.join(img_folder, dir1, file)
            img = Image.open(img_path)
            img = torch.from_numpy(np.array(img.resize((size, size)))).to(torch.float)
            img = (img - 127.5) / 127.5
            data.append(img)

    return torch.stack(data)

# for data without labels (ffhq)
def create_dataset(img_folder, size):

    data = []

    for file in os.listdir(os.path.join(img_folder)):
        img_path = os.path.join(img_folder, file)
        img = Image.open(img_path)
        img = torch.from_numpy(np.array(img.resize((size, size)))).to(torch.float)
        img = (img - 127.5) / 127.5
        data.append(img)

    return torch.stack(data)

if __name__ == "__main__":
    # extract the image array and class name

    # ffhq
    train_path_thumb = 'thumbnails128x128/train/'
    train_data_thumb = create_dataset(train_path_thumb, 64)
    train_loader_thumb = DataLoader(train_data_thumb, batch_size=4, shuffle=True)

    # afhq
    train_path = 'afhq/train/'
    val_path = 'afhq/val/'
    train_data = create_dataset_classes(train_path, 64, ['cat', 'dog'])
    val_data = create_dataset_classes(val_path, 64, ['cat', 'dog'])
    val_loader = DataLoader(val_data, batch_size=4, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

    # classifier = models.mobilenet_v2(pretrained=False, num_classes=2)
    # classifier.load_state_dict(torch.load("classifier_model_permute_lr.pt"))
    # classifier.eval()
