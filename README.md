# CS490: Data Science Capstone Project

### Datasets:
The datasets used are MNIST, CIFAR10, and SVHN. They can be found in the following links. 

MNIST: https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/ 

CIFAR10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

SVHN: http://ufldl.stanford.edu/housenumbers/ 

#### Directory Structure:
```
src:
│   cifar-10_summary_statistics.ipynb
│   cifar10_resnet18.ipynb
│   cifar10_summary_statistics.txt
│   mnist_resnet18.ipynb
│   mnist_summary_statistics.ipynb
│   mnist_summary_statistics.txt
│   resnet18_cifar_model.pth
│   resnet18_mnist_model.pth
│   resnet18_svhn_model.pth
│   svhn_resnet18.ipynb
│   svhn_summary_statistics.ipynb
│   svhn_summary_statistics.txt
│
├───CIFAR-10
│       batches.meta
│       data_batch_1
│       data_batch_2
│       data_batch_3
│       data_batch_4
│       data_batch_5
│       readme.html
│       test_batch
│
├───images
│       resnet18ex1.png
│       resnet18ex2.png
│
├───MNIST_CSV
│       generate_mnist_csv.py
│       mnist_test.csv
│       mnist_train.csv
│       readme.md
│
├───Summary_Statistics_pdfs
│       cifar-10_summary_statistics.pdf
│       mnist_summary_statistics.pdf
│       svhn_summary_statistics.pdf
│
└───SVHN
    │   extra_32x32.mat
    │   test_32x32.mat
    │   train_32x32.mat
```

### Summary Statistics:
Brief summary statistics have been created for each dataset in their respective ipynb files. These notebooks include metadata, data transformation (one-hot vectors to human interpretable images), "mean" images, and traditional summary statistics on the euclidean distance from the mean image and records. The latter was outputted to text files to declutter the notebooks.

### Resnet18 Models:
Our control group for this project are basic resnet models trained on our datasets. Details about the model can be found within their respective notebooks. The resnet19 models were recreated using pytorch. GPU acceleration is recommended for training. Models were saved to .pth files.
