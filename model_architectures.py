import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas as pd
import os
import pickle
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

current_directory = os.getcwd()
mnist_directory = os.path.join(current_directory,"./MNIST_CSV")
cifar_directory = os.path.join(current_directory,"./CIFAR-10")
svhn_directory = os.path.join(current_directory,"./SVHN")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes * self.expansion, stride)  # Adjusted out_channels here
        self.bn1 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes * self.expansion, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# Define the ResNet model
class ResNetCIFAR(nn.Module):
    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNetCIFAR, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas
    

# Define the ResNet model
class ResNetMNIST(nn.Module):
    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

# Define the ResNet model
class ResnetSVHN(nn.Module):
    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResnetSVHN, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
    
        return logits, probas
    
class Load():
    def load_mnist_test_images(self):
        mnist_test = pd.read_csv(os.path.join(mnist_directory, "mnist_test.csv"), header=None)
        test_labels = mnist_test.iloc[:, 0].values
        test_images = mnist_test.iloc[:, 1:].values / 255.0  # Normalize pixel values to the range [0, 1]

        return test_images.reshape((10000,1,28,28)), test_labels

    def load_cifar10_test_images(self):
        def load_cifar10_batch(file_path):
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
            return batch

        # Helper function which concatenate the data from the 5 files into one large file
        def load_cifar10_data(data_dir):
            train_batches = []
            for i in range(1, 6):
                file_path = f"{data_dir}/data_batch_{i}"
                train_batches.append(load_cifar10_batch(file_path))

            test_batch = load_cifar10_batch(f"{data_dir}/test_batch")

            # Concatenate training batches to get the full training dataset
            train_data = np.concatenate([batch[b'data'] for batch in train_batches])
            # Data Normalization
            train_data = train_data
            # print(train_data)
            train_labels = np.concatenate([batch[b'labels'] for batch in train_batches])

            # Extract test data and labels
            test_data = test_batch[b'data']
            # Data Normalization
            test_data = test_data 
            # print(test_data)
            test_labels = test_batch[b'labels']

            return train_data, train_labels, test_data, test_labels

        def convert_to_pytorch_images(data_array):
            """
            Convert a single record into a pytorch image. Pytorch takes in a very specific input where each record is 3x32x32.

            Parameters:
            - One-hot vector the first 1024 values represent the red channel of pixels, the second 1024 values represent the green channel of pixels, and the last 1024 values represent the blue channel of pixels

            Returns: 
            - a numpy array representing the image data in pytorch format
            """

            num_images = data_array.shape[0]
            image_size = 32

            # Split the array into three parts
            split_size = data_array.shape[1] // 3
            red_channel = data_array[:, :split_size].reshape((num_images, 1, image_size, image_size))
            green_channel = data_array[:, split_size:2*split_size].reshape((num_images, 1, image_size, image_size))
            blue_channel = data_array[:, 2*split_size:].reshape((num_images, 1, image_size, image_size))

            # Stack the channels along the second axis to get the final shape (num_images, 3, 32, 32)
            return np.concatenate([red_channel, green_channel, blue_channel], axis=1)
            
        # Load the data from CSVs using pandas
        train_data, train_labels, test_data, test_labels = load_cifar10_data(cifar_directory)
        # Append the labels
        columns = [f"pixel_{i+1}" for i in range(train_data.shape[1])]
        cifar_test = pd.DataFrame(test_data, columns=columns)
        cifar_test['label'] = test_labels

        # Extract labels and pixel values
        test_labels = cifar_test.iloc[:, -1].values
        test_images = cifar_test.iloc[:, :-1].values 
        test_images = convert_to_pytorch_images(test_images)

        return test_images, test_labels

    def load_svhn_test_images(self):
        def flatten(images):
            """
            Flattens images back to a one hot vector format

            Parameters:
            - images: Numpy array representing the images with shape (num_images, height, width, channels)

            Returns:
            - Flat array where the first 1024 values represent the red channel, the next 1024 values represent the green channel,
            and the last 1024 values represent the blue channel of pixels for each image.
            Example:    
                flat_array = flatten(train_images)
            """
            num_images, _, _, _ = images.shape

            # Reshape the images array to (num_images, 1024, 3)
            reshaped_images = images.reshape((num_images, -1, 3))

            # Split the reshaped array into red, green, and blue channels
            red_channel = reshaped_images[:, :, 0]
            green_channel = reshaped_images[:, :, 1]
            blue_channel = reshaped_images[:, :, 2]

            # Stack the three channels horizontally
            stacked_channels = np.hstack([red_channel, green_channel, blue_channel])

            return stacked_channels

        def load_svhn_data_mat(file_path):
            mat_data = scipy.io.loadmat(file_path)

            # Extract data and labels
            images = mat_data['X']
            labels = mat_data['y']

            # Reshape the images to (num_samples, height, width, channels)
            images = np.transpose(images, (3, 0, 1, 2))

            # This replaces the label 10 with 0. For some reason the CUDA toolkit does not work if the labels are indexed 1-10 instead of 0-9.
            labels[labels == 10] = 0

            return images, labels

        def convert_to_pytorch_images(data_array):
            """
            Convert a single record into a pytorch image. Pytorch takes in a very specific input where each record is 3x32x32.

            Parameters:
            - One-hot vector the first 1024 values represent the red channel of pixels, the second 1024 values represent the green channel of pixels, and the last 1024 values represent the blue channel of pixels

            Returns: 
            - a numpy array representing the image data in pytorch format
            """
            num_images = data_array.shape[0]
            image_size = 32

            # Split the array into three parts
            split_size = data_array.shape[1] // 3
            red_channel = data_array[:, :split_size].reshape((num_images, 1, image_size, image_size))
            green_channel = data_array[:, split_size:2*split_size].reshape((num_images, 1, image_size, image_size))
            blue_channel = data_array[:, 2*split_size:].reshape((num_images, 1, image_size, image_size))

            # Stack the channels along the second axis to get the final shape (num_images, 3, 32, 32)
            return np.concatenate([red_channel, green_channel, blue_channel], axis=1)

        filepath = os.path.join(svhn_directory,"test_32x32.mat")
        test_data, test_labels = load_svhn_data_mat(filepath)

        flattened_test_data = flatten(test_data)
        data_dict = {'pixel_{}'.format(i+1): flattened_test_data[:, i] for i in range(flattened_test_data.shape[1])}
        data_dict['label'] = [i[0] for i in test_labels]
        flattened_test_data = pd.DataFrame(data_dict)

        test_labels = flattened_test_data.iloc[:, -1].values
        test_images = flattened_test_data.iloc[:, :-1].values 

        test_images = convert_to_pytorch_images(test_images)

        return test_images, test_labels
    
    def convert_mnist_numpy_to_tensor(self, images, labels):
        test_images_tensor = torch.tensor(images, dtype=torch.float32)
        test_labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
        test_images_tensor = test_images_tensor.view(-1, 1, 28, 28)

        # Move the test data to the GPU
        test_images_tensor, test_labels_tensor = test_images_tensor.to(device), test_labels_tensor.to(device)

        batch_size = 256
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
        return test_loader
    
    def convert_cifar10_numpy_to_tensor(self, images, labels):
        test_images_tensor = torch.tensor(images, dtype=torch.float32)
        test_labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
        test_images_tensor = test_images_tensor.view(-1, 3, 32, 32)

        # Move the test data to the GPU
        test_images_tensor, test_labels_tensor = test_images_tensor.to(device), test_labels_tensor.to(device)

        batch_size = 256
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
        return test_loader
    
    def convert_svhn_numpy_to_tensor(self, images, labels):
        return self.convert_cifar10_numpy_to_tensor(images,labels)
    
    
class Visualizer():
    def show(self, image_data):
        image_data = np.transpose(image_data, (1, 2, 0))
        plt.imshow(image_data)
        plt.axis('off') 
        plt.show()

class Tester():
    def test(self, model, test_loader):
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                # print(images.shape)
                outputs, _ = model(images)
                _, predicted = torch.max(outputs.data, 1)
                # print(f"Label: {labels}, Predicted{predicted}")
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        # print(f'Test Accuracy: {accuracy * 100:.2f}%')
        return accuracy