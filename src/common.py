import torch
from torchvision import datasets, transforms
import numpy as np
import logging
from IPython import embed

import sys
import os
sys.path.append(os.path.dirname(__file__))
from misc import *


class Session:
    def __init__(self, net=None, optimizer=None):

        self.log_root = os.path.realpath(os.path.join(os.path.abspath(__file__), '../../train_log', os.path.basename(os.getcwd())))
        self.log_model_dir = os.path.join(self.log_root, 'models')
        self.log_curv_dir = os.path.join(self.log_root, 'curve')

        make_symlink_if_not_exists(self.log_root, 'train_log', True)

        ensure_dir(self.log_model_dir)
        ensure_dir(self.log_curv_dir)

        self.logger = logging.getLogger()
        self.set_logger()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.clock = TrainClock()

        self.net = net
        self.optimizer = optimizer

        self.best_state = {
            # 'loss': np.inf,
            'accuracy': -np.inf,
            'epoch': -1
        }

        self.curv_stat = {
            'train_epoch': {'accuracy': [], 'loss': [], 'acc_cls':[]},
            'train_step': {'accuracy': [], 'loss': [], },
            'val_epoch': {'accuracy': [], 'loss': [], 'acc_cls':[]},
        }

    def update_curv_state(self, stage, acc, loss, acc_cls=None):
        assert stage in ['train_epoch', 'train_step', 'val_epoch']
        self.curv_stat[stage]['accuracy'].append(acc)
        self.curv_stat[stage]['loss'].append(loss)
        if 'epoch' in stage:
            self.curv_stat[stage]['acc_cls'].append(acc_cls)

    def update_best_state(self, acc):
        if acc > self.best_state['accuracy']:
            self.best_state['accuracy'] = acc
            self.best_state['epoch'] = self.clock.epoch
            self.save_checkpoint('best-accuracy')

    def save_checkpoint(self, name):
        ckp_path = os.path.join(self.log_model_dir, name)

        torch.save(
            {
                'epoch': self.clock.epoch,
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            },
            ckp_path
        )

    def set_logger(self):
        self.logger.setLevel('DEBUG')
        formatter = logging.Formatter(fmt="%(message)s")
        chlr = logging.StreamHandler()  # 输出到控制台的handler
        chlr.setFormatter(formatter)
        chlr.setLevel('INFO')
        fhlr = logging.FileHandler(os.path.join(self.log_root, 'worklog.txt')) # 输出到文件的handler
        fhlr.setFormatter(formatter)
        fhlr.setLevel('INFO')

        self.logger.addHandler(chlr)
        self.logger.addHandler(fhlr)


    def end(self):
        self.logger.info("Finish training!")
        self.logger.info("The best accuracy model is: {}".format(self.best_state))

        weight = self.net.state_dict()['pred.weight'].to('cpu').data
        weight_norm = torch.norm(weight, dim=1)
        self.logger.info("weight_norm: {}".format(weight_norm.data))

        torch.save(self.curv_stat, os.path.join(self.log_curv_dir, 'curve_info'))



class TrainClock:
    def __init__(self):
        self.epoch = 0
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0



class Backbone(torch.nn.Module):

    def __init__(self):
        super(Backbone, self).__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=7, stride=1, padding=3),
            # 28 * 28
            torch.nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2),
            # 14 * 14
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2),
            # 7 * 7
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )


    def forward(self, x):
        x = self.backbone(x)
        return x

def load_dataset(data_set='fashion_mnist', batch_size=64):
    assert data_set in ['fashion_mnist', 'mnist']
    print('loading dataset {}'.format(data_set))

    data_set_func = datasets.FashionMNIST if data_set == 'fashion_mnist' else datasets.MNIST

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )

    with change_dir(os.path.dirname(os.path.realpath(__file__))):
        data_train = data_set_func(root="../data/{}".format(data_set),
                                    transform=transform,
                                    train=True,
                                    download=True)

        print("len of train data:", len(data_train))

        data_test = data_set_func(root="../data/{}".format(data_set),
                                   transform=transform,
                                   train=False)

        print("len of test data:", len(data_test))

    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=2)

    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)

    return data_loader_train, data_loader_test, data_train, data_test


def get_label_list(dataset):
    f_mnist_label = ['T-shirt', 'kuzi', 'taoshan', 'qunzi', 'waitao', 'liangxie', 'hanshan', 'yundongxie', 'bao', 'luoxue']
    mnist_label = [str(x) for x in range(10)]
    return f_mnist_label if dataset == 'fashion_mnist' else mnist_label


if __name__ == '__main__':
    load_dataset()