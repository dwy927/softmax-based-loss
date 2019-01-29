import matplotlib
matplotlib.use('Agg')
import torchvision
import matplotlib.pyplot as plt
import sys
from src.common import load_dataset

def view_data(dataset):
    data_loader_train, data_loader_test, data_train, data_test = load_dataset(dataset, 300)

    images, labels = next(iter(data_loader_train))
    plt.figure(1, figsize=(8, 20))
    for i in range(10):
        plt.subplot(10, 1, i+1)
        image = images[labels == i][:20, :, :]
        img = torchvision.utils.make_grid(image, nrow=10)
        img = img.numpy().transpose(1, 2, 0)
        std = [0.5, 0.5, 0.5]
        mean = [0.5, 0.5, 0.5]
        img = img * std + mean
        plt.axis('off')
        plt.imshow(img)
        plt.title(i)
    print('saving file in data/{}-train.png'.format(dataset))
    plt.savefig('data/{}-train.png'.format(dataset))
    plt.clf()

    images, labels = next(iter(data_loader_test))
    plt.figure(1, figsize=(8, 20))
    for i in range(10):
        plt.subplot(10, 1, i + 1)
        image = images[labels == i][:20, :, :]
        img = torchvision.utils.make_grid(image, nrow=10)
        img = img.numpy().transpose(1, 2, 0)
        std = [0.5, 0.5, 0.5]
        mean = [0.5, 0.5, 0.5]
        img = img * std + mean
        plt.axis('off')
        plt.imshow(img)
        plt.title(i)
    print('saving file in data/{}-train.png'.format(dataset))
    plt.savefig('data/{}-test.png'.format(dataset))
    plt.clf()
    plt.close('all')


if __name__ == '__main__':
    dataset = sys.argv[1]
    assert dataset in ['mnist', 'fashion_mnist']
    view_data(dataset)