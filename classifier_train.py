import numpy as np
import torch
import torchvision.models as models
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def create_dataset(img_folder, size, classes):

    data = []

    for i, dir1 in enumerate(classes):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            img_path = os.path.join(img_folder, dir1, file)
            img = Image.open(img_path)
            img = torch.from_numpy(np.array(img.resize((size, size)))).to(torch.float)
            img = (img - 127.5) / 127.5
            data.append(img)

    return torch.stack(data)

if __name__ == "__main__":
    # extract the image array and class name
    train_path = 'afhq/train/'
    val_path = 'afhq/val/'

    train_data = create_dataset(train_path, 64, ['cat', 'dog'])
    val_data = create_dataset(val_path, 64, ['cat', 'dog'])
    val_loader = DataLoader(val_data, batch_size=4, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

    classifier = models.mobilenet_v2(pretrained=False, num_classes=2)
    classifier.load_state_dict(torch.load("classifier_model.pt"))
    classifier.eval()
