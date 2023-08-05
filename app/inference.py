import argparse
import pathlib
import shutil
from PIL import Image
import torch
from torchvision import transforms
import train


def copy_images(source, destination):
    files = source.glob("*")
    for file in files:
        if file.is_file():
            try:
                im = Image.open(file)
                im.save(destination.joinpath(file.name))
            except Exception:
                pass


def main(input_folder):
    # Defining all paths to necessary files and folders
    app_root = pathlib.Path(__file__).parent.absolute()
    data_folder = app_root.joinpath(input_folder.stem)
    model_path = app_root.joinpath("model.pth")

    if data_folder.exists():
        shutil.rmtree(data_folder)
    data_folder.mkdir()

    # Defining necessary transformations for input data
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.CenterCrop((28, 28)),
        transforms.ToTensor()
    ])

    # Copying all valid images to local folder
    copy_images(input_folder, data_folder)

    # Initializing input
    images = data_folder.glob("*")
    images = [data_folder.joinpath(image) for image in images]
    images_data = torch.cat([transform(Image.open(image)).unsqueeze(0) for image in images])

    # Initializing model
    model = train.VINNumbersClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # Getting labels
    _, labels_predicted = torch.max(model(images_data).data, 1)
    labels_predicted = [train.index_to_class(label_predicted) for label_predicted in labels_predicted]

    # Printing result
    for i in range(len(labels_predicted)):
        print("{:03d}, {}".format(ord(labels_predicted[i]), images[i].absolute().as_posix()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Set images source folder")
    args = parser.parse_args()
    main(pathlib.Path(args.input).absolute())
