# VIN characters recognition app
## Task analysis and solution ideas
### Task summary
We have small squared grayscale images of some VIN code characters, and we 
need to be able to classify these characters. Each image contains small 
white border representing the box and inside it - the character itself.

### Main problems and their solutions
#### We don't know what dimensions of images to use
The problem states that images are small squares, but it does not tell us
the length of their sides.  
Solution: all samples in our dataset have the same size(28, 28), therefore
it is logical to train model on this size, and later (down/up)scale input
images to this size.

#### Input images contain white border, but images in our dataset don't
There are 2 possible solutions:
1. Transform dataset images to have white border
2. Remove white border during inference

Both first and second options are problematic, because we don't know the size
of the border. But the second option is more reasonable because it doesn't
create dead neurons in the network.  
Solution: downscale input image to (32, 32) and then cut 2px from each side
to get (28, 28) image.

#### Data in dataset is not balanced between classes
Our dataset is very unbalanced, so in order to train on it properly we need
to do some tweaks.  
Solution: introduce weighted loss. Weight for each class is calculated like
this: sum/x, where x is total number of samples of current class in training
dataset and sum is total number of samples in training dataset.

### Proposed model architecture
It is obvious that this is an image classification task, therefore the model
we propose is a convolutional neural network with 2 convolutional and
2 fully connected layers:
1. Input layer: input layer accepts (1, 28, 28) (C, H, W) tensor representing 
image
2. Convolutional layer 1(relu activation): convolutional layer with 5x5 kernel,
stride 1 and no padding.  
(1, 28, 28) -> (32, 24, 24)
3. Convolutional layer 2(relu activation): convolutional layer with 5x5 kernel,
stride 1 and no padding.  
(32, 28, 28) -> (64, 20, 20)
4. Max pooling layer: max pooling layer with 2x2 kernel.  
(64, 20, 20) -> (64, 10, 10)
5. Flattening layer.  
   (64, 10, 10) -> (6400)
6. Dropout layer 1: dropout layer with dropout probability equal to 50%.
7. Linear layer 1(relu activation): Fully connected layer with 6400 input and
256 output features.
   (6400) -> (256)
8. Dropout layer 2: dropout layer with dropout probability equal to 50%.
9. Linear layer 2: Fully connected layer with 256 input and 33
output features.  
   (256) -> (33)
10. Output layer: softmax function is used to calculate probability 
distribution for each of 33 classes.

### Training process
#### Data split and augmentation
All data was split into 3 parts: train/valid/test = 72/18/10. Grayscale
transformation was applied on each dataset, because all of our images are
black & white. Additionally, to create more diverse training data and adapt
model to real life situations, we added random affine transformation to 
training dataset. This transformation generates images, where characters are 
randomly scaled and moved.

#### Training loop
Our training loop uses cross entropy loss as criterion, SGD as optimizer and
learning rate scheduler with learning rate reduction on plateau.  
Hyperparameters:
* number of epochs - 50
* batch size - 2048(images are very small, so this is ok)
* learning rate - 0.01
* learning rate drop factor - 0.3
* learning rate patience - 3 epochs
* momentum - 0.9

#### Final results
After training, model showed 96.04% accuracy on testing dataset. It could
be improved by training for more epochs, but these improvements wouldn't
be very significant. Therefore, we consider this result to be final.

## Datasets
The only dataset used to train and test the model is the EMNIST dataset[^1].
Original dataset consists of 28x28 grayscale images of digits, lower and upper
case letters with images and their labels being encoded in binary format.  

To use this dataset for our purposes it had to be modified a bit:
1. All images containing lowercase letters or I, O, Q were removed, because 
these characters are not used in VIN codes. 
2. Train and test datasets were merged together in order to use custom
train/valid/test split during training and evaluation processes. 
3. Images were saved as png files and all data about image classes was moved to
csv file

## User Guide
To run the program follow these steps:
1. Build the docker image
2. Use following command:  
`docker run -it --rm -v "data_folder:/data" image python3 inference.py --input /data`,
where `image` is the name of the docker image and `data_folder` is 
the folder containing input data
  
[^1]: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: 
an extension of MNIST to handwritten letters. 
Retrieved from http://arxiv.org/abs/1702.05373