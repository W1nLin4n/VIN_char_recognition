import os.path
import random
import pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def class_to_index(cls):
    value = ord(cls)
    index = -1
    if 48 <= value <= 57:
        index = value - 48
    else:
        index = value - 65 + 10
        if value > ord('I'):
            index -= 1
        if value > ord('O'):
            index -= 1
        if value > ord('Q'):
            index -= 1
    return index


def index_to_class(index):
    asc = -1
    if 0 <= index <= 9:
        asc = index + 48
    else:
        asc = index - 10 + 65
        if asc >= ord('I'):
            asc += 1
        if asc >= ord('O'):
            asc += 1
        if asc >= ord('Q'):
            asc += 1
    return chr(asc)


class VINNumbersDataset(Dataset):
    def __init__(self, root, image_names, labels, transform=None):
        self.root = root
        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.root.joinpath(self.image_names[idx])
        image = Image.open(image_path)

        label = class_to_index(self.labels[idx])

        if self.transform is not None:
            image = self.transform(image)
        return image, label


class VINNumbersClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6400, 256)
        self.fc2 = nn.Linear(256, 33)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


def train(model, device, model_file,
          criterion, optimizer, scheduler, train_loader, valid_loader, epochs):
    min_valid_loss = np.inf
    for epoch in range(1, epochs + 1):
        print("-------------Epoch {}-------------".format(epoch))

        # Training loop
        train_loss = 0.0
        train_run_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            print(labels)
            optimizer.zero_grad()

            labels_predicted = model(images)
            loss = criterion(labels_predicted, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_run_loss += loss.item()
            if i % 10 == 9:
                print("Epoch {} batch {}: loss - {}".format(epoch, i+1, train_run_loss))
                train_run_loss = 0.0
        print("Epoch {} train loss - {}".format(epoch, train_loss))

        # Validation loop
        valid_loss = 0.0
        with torch.no_grad():
            model.eval()
            for _, data in enumerate(valid_loader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                labels_predicted = model(images)
                loss = criterion(labels_predicted, labels)

                valid_loss += loss.item()

            scheduler.step(valid_loss)

            print("Epoch {} validation loss - {}".format(epoch, valid_loss))
            if min_valid_loss > valid_loss:
                print("Validation loss decreased from {} to {}, saving the model".format(min_valid_loss, valid_loss))
                min_valid_loss = valid_loss
                torch.save(model.state_dict(), model_file)


def test(model, device, test_loader):
    # Testing loop
    correct_guesses = 0
    total_guesses = 0
    with torch.no_grad():
        model.eval()
        for _, data in enumerate(test_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            _, labels_predicted = torch.max(model(images).data, 1)
            total_guesses += labels.size(0)
            correct_guesses += (labels_predicted == labels).sum().item()
    print("Accuracy of a model is {}".format(100 * correct_guesses / total_guesses))


def main():
    # Constants
    app_root = pathlib.Path(__file__).parent.absolute()
    data_root = app_root.parent.joinpath("datasets", "emnist_modified")
    data_csv_file = "emnist_modified.csv"
    model_file = "model.pth"

    random_state = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_size = 0.1
    valid_size = 0.2
    batch_size = 2048
    epochs = 50
    lr = 0.01
    lr_drop_factor = 0.3
    lr_patience = 3
    momentum = 0.9

    # Setting random seeds
    torch.manual_seed(random_state)
    random.seed(random_state)

    # Declaring transforms
    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomAffine(degrees=0,
                                translate=(0.2, 0.2),
                                scale=(0.8, 1.2),
                                interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])
    valid_test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    # Loading raw training data
    data = pd.read_csv(data_root.joinpath(data_csv_file))
    X = data['file'].to_numpy()
    y = data['label'].to_numpy()

    # Splitting data into (train+valid)/test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        shuffle=True,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                          shuffle=True,
                                                          test_size=valid_size,
                                                          random_state=random_state,
                                                          stratify=y_train)

    # Creating datasets with needed transforms
    train_dataset = VINNumbersDataset(data_root, X_train, y_train, train_transform)
    valid_dataset = VINNumbersDataset(data_root, X_valid, y_valid, valid_test_transform)
    test_dataset = VINNumbersDataset(data_root, X_test, y_test, valid_test_transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True)

    # Creating class weights for all classes
    weights = torch.FloatTensor([len(y) / y.tolist().count(index_to_class(x)) for x in range(33)]).to(device)

    # Model initialization
    model = VINNumbersClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weights)
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=lr_drop_factor,
                                                     patience=lr_patience,
                                                     verbose=True)

    # Training
    train(model, device, app_root.joinpath(model_file),
          criterion, optimizer, scheduler, train_loader,
          valid_loader, epochs)

    # Testing
    model = VINNumbersClassifier()
    model.load_state_dict(torch.load(app_root.joinpath(model_file)))
    model.to(device)
    test(model, device, test_loader)


if __name__ == "__main__":
    main()
