import torch
from torchvision import datasets, transforms


def get_dataset(dir, name):
    if name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(
            dir, train=True, download=True, transform=transform)
        eval_dataset = datasets.MNIST(dir, train=False, transform=transform)
    elif name == 'CIFAR-10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(
            dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.CIFAR10(
            dir, train=False, transform=transform_test)
    elif name == 'Fashion-MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.FashionMNIST(
            dir, train=True, download=True, transform=transform)
        eval_dataset = datasets.FashionMNIST(
            dir, train=False, transform=transform)
    elif name == 'CIFAR-100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = datasets.CIFAR100(
            dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.CIFAR100(
            dir, train=False, transform=transform_test)
    return train_dataset, eval_dataset
