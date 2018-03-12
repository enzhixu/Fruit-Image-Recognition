
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torch.utils.data as Data

from datetime import datetime
from torch.autograd import Variable
from torchvision import transforms

# Hyper Parameters
EPOCH = 5  # train the training data 5 times
BATCH_SIZE = 4
LR = 0.001  # learning rate

if __name__ == "__main__":
    start_time = datetime.now()
    train_image_data = dset.ImageFolder("../fruits-360/Training", transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_image_data,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=2)
    val_image_data = dset.ImageFolder("../fruits-360/Validation", transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(val_image_data,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=2)

    end_time = datetime.now()
    print("Time spent:", end_time - start_time)

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
                nn.Conv2d(
                    in_channels=3,      # input height
                    out_channels=16,    # n_filters
                    kernel_size=5,      # filter size
                    stride=1,           # filter movement/step
                    padding=2,
                ),
                nn.ReLU(),                  # activation
                nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (x, x, x)
            )
            self.conv2 = nn.Sequential(     # input shape (16, 14, 14)
                nn.Conv2d(16, 32, 5, 1, 2), # output shape (32, 14, 14)
                nn.ReLU(),                  # activation
                nn.MaxPool2d(2),            # output shape (32, 7, 7)
            )
            self.out = nn.Linear(32 * 25 * 25, 60)  # fully connected layer, output 60 classes

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
            output = self.out(x)
            return output, x  # return x for visualization


    cnn = CNN()
    print(cnn)  # net architecture

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            b_x = Variable(x)  # batch x
            b_y = Variable(y)  # batch y

            output = cnn(b_x)[0]  # cnn output

            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 500 == 499:
                print('step:', step, datetime.now())
                correct = 0
                total = 0
                for index, data in enumerate(test_loader):
                    images, labels = data
                    outputs = cnn(Variable(images))
                    _, predicted = torch.max(outputs[0], 1)
                    predicted = predicted.data
                    total += labels.size(0)

                    correct += (predicted == labels).sum()
                print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
