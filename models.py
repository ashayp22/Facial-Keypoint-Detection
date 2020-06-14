## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        #input size: 224 * 224
        
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 26
#         self.conv1 = nn.Conv2d(1, 32, 4) #output size (32, 220, 220)
#         self.pool1 = nn.MaxPool2d(2, 2) #output size (32, 110, 110)
#         self.conv1_drop = nn.Dropout(p=0.4)
        
#         self.conv2 = nn.Conv2d(32, 64, 3) #output size (64, 110, 220)
#         self.pool2 = nn.MaxPool2d(2, 2) #output size (64, 55, 55)
#         self.conv2_drop = nn.Dropout(p=0.4)

#         self.conv3 = nn.Conv2d(64, 128, 2) #output size (128, 55, 55)
#         self.pool3 = nn.MaxPool2d(2, 2) #output size (128, 27, 27)
#         self.conv3_drop = nn.Dropout(p=0.4)
        
#         self.conv4 = nn.Conv2d(128, 256, 1) #output size (256, 27, 27)
#         self.pool4 = nn.MaxPool2d(2, 2) #output size (256, 13, 13)
#         self.conv4_drop = nn.Dropout(p=0.4)
        
#         #flatten here
        
#         self.fc1 = nn.Linear(256*13*13, 1000)
#         self.fc1_drop = nn.Dropout(p=0.4)
        
#         self.fc2 = nn.Linear(1000, 500)
#         self.fc2_drop = nn.Dropout(p=0.4)
        
#         self.fc3 = nn.Linear(500, 250)
#         self.fc3_drop = nn.Dropout(p=0.4)

#         self.fc4 = nn.Linear(250, 136) #output 

         #defining maxpool block
        self.maxpool = nn.MaxPool2d(2, 2)
               
        #defining dropout block
        self.dropout = nn.Dropout(p=0.2)
        
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        #defining second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        #defining third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        #defining linear output layer
        self.fc1 = nn.Linear(128*26*26, 136)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        #convolution
        
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = self.pool3(F.relu(self.conv3(x)))
#         x = self.pool4(F.relu(self.conv4(x)))
        
#         #flatten
#         # prep for linear layer
#         # this line of code is the equivalent of Flatten in Keras
#         x = x.view(x.size(0), -1)
        
#         #dense
        
#         x = F.relu(self.fc1(x))
#         x = self.fc1_drop(x)
#         x = F.relu(self.fc2(x))
#         x = self.fc2_drop(x)
#         x = F.relu(self.fc3(x))
#         x = self.fc3_drop(x)
#         x = self.fc4(x)
        #passing tensor x through first conv layer
        x = self.maxpool(F.relu(self.conv1(x)))
     
        #passing tensor x through second conv layer
        x = self.maxpool(F.relu(self.conv2(x)))
        
        #passing tensor x through third conv layer
        x = self.maxpool(F.relu(self.conv3(x)))
        
        #flattening x tensor
        x = x.view(x.size(0), -1)
        
        #applying dropout
        x = self.dropout(x)
     
        #passing x through linear layer
        x = self.fc1(x)
                
        # a modified x, having gone through all the layers of your model, should be returned
        return x
