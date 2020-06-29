from torchvision import datasets

trn_data = datasets.CIFAR10(root="data", train=True, download=True)