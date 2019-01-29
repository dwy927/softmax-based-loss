import torch
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
from common import load_dataset, get_label_list
from config import *
from misc import *

color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def draw_feature(path='train_log/models/best-accuracy', scale=1, data_set='fashion_mnist'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    checkpoint = torch.load(path, map_location=device)
    data_loader_train, data_loader_test, data_train, data_test = load_dataset(data_set=data_set)

    net = Model(n_feature)
    net.load_state_dict(checkpoint['model_state_dict'])
    exact_list = ['feature']
    feature_extractor = FeatureExtractor(net, exact_list)
    feature_extractor.to(device)

    # get weight
    weight = checkpoint['model_state_dict']['pred.weight'].to('cpu').data
    weight_norm = weight / torch.norm(weight, dim=1, keepdim=True)
    print("weight_norm: ", torch.norm(weight, dim=1))

    # get feature
    features = []
    labels = []
    for data in data_loader_train:
        X_train, y_train = data
        X_train = Variable(X_train).to(device)
        outputs = feature_extractor(X_train)['feature'].data
        features.append(outputs)
        labels.append(y_train)
    features = torch.cat(features, dim=0).to('cpu').data
    features_norm =  features / torch.norm(features, dim=1, keepdim=True)
    features = features.numpy()
    features_norm = features_norm.numpy()
    labels = torch.cat(labels, dim=0).to('cpu').data.numpy()

    # draw features
    label_list = get_label_list(data_set)

    plt.figure(1, figsize=(20, 20))
    plt.subplot(221)
    for i in range(10):
        plt.plot([0, scale*weight[i, 0]], [0, scale*weight[i, 1]], color=color_list[i])
        feature = features[labels == i]
        plt.scatter(feature[:, 0], feature[:, 1], c=color_list[i], marker='.', label= label_list[i], s=1)
        plt.legend()

    plt.subplot(223)
    for i in range(10):
        plt.plot([0, weight_norm[i, 0]], [0, weight_norm[i, 1]], color=color_list[i])
        feature = features_norm[labels == i]
        plt.scatter(feature[:, 0], feature[:, 1], c=color_list[i], marker='.', label= label_list[i], s=1)
        plt.legend()

    # get feature
    features = []
    labels = []
    for data in data_loader_test:
        X_test, y_test = data
        X_test = Variable(X_test).to(device)
        outputs = feature_extractor(X_test)['feature'].data
        features.append(outputs)
        labels.append(y_test)
    features = torch.cat(features, dim=0).to('cpu').data
    features_norm = features / torch.norm(features, dim=1, keepdim=True)
    features = features.numpy()
    features_norm = features_norm.numpy()
    labels = torch.cat(labels, dim=0).to('cpu').data.numpy()

    plt.subplot(222)
    for i in range(10):
        plt.plot([0, scale * weight[i, 0]], [0, scale * weight[i, 1]], color=color_list[i])
        feature = features[labels == i]
        plt.scatter(feature[:, 0], feature[:, 1], c=color_list[i], marker='.', label= label_list[i], s=1)
        plt.legend()

    plt.subplot(224)
    for i in range(10):
        plt.plot([0, weight_norm[i, 0]], [0, weight_norm[i, 1]], color=color_list[i])
        feature = features_norm[labels == i]
        plt.scatter(feature[:, 0], feature[:, 1], c=color_list[i], marker='.', label= label_list[i], s=1)
        plt.legend()

    title = os.path.basename(os.getcwd()) + '-' + os.path.basename(path)
    plt.suptitle(title)

    fname = 'train_log/feature-{}'.format(os.path.basename(path))
    figname = 'train_log/{}.png'.format(fname)

    os.remove(figname) if os.path.exists(figname) else None
    plt.savefig(fname)
    plt.close('all')


def draw_curve(curv_state='train_log/curve/curve_info', root='.', data_set='fashion_mnist'):
    if isinstance(curv_state, dict):
        curv_state = curv_state
    else:
        root = os.path.dirname(curv_state)
        curv_state = torch.load(curv_state)

    label_list = get_label_list(data_set)
    title = os.path.basename(os.getcwd())

    with change_dir(root):
        plt.figure(1, figsize=(10, 20))
        plt.subplot(411)
        plt.plot(curv_state['train_epoch']['loss'], linewidth=1.0, label='train')
        plt.plot(curv_state['val_epoch']['loss'], linewidth=1.0, label='val')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')

        plt.subplot(412)
        plt.plot(curv_state['train_epoch']['accuracy'], linewidth=1.0, label='train')
        plt.plot(curv_state['val_epoch']['accuracy'], linewidth=1.0, label='val')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()

        plt.subplot(413)
        train_acc_cls = np.array(curv_state['train_epoch']['acc_cls'])
        for i in range(10):
            plt.plot(train_acc_cls[:, i], linewidth=1.0, label=label_list[i])
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('train-acc')

        plt.subplot(414)
        val_acc_cls = np.array(curv_state['val_epoch']['acc_cls'])
        for i in range(10):
            plt.plot(val_acc_cls[:, i], linewidth=1.0, label=label_list[i])
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('test-acc')

        plt.suptitle(title)
        plt.savefig('curve-epoch')
        # plt.show()

        plt.figure(2, figsize=(10, 10))
        plt.subplot(211)
        plt.plot(curv_state['train_step']['loss'], linewidth=1.0, label='train')
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.legend()

        plt.subplot(212)
        plt.plot(curv_state['train_step']['accuracy'], linewidth=1.0, label='train')
        plt.xlabel('step')
        plt.ylabel('accuravy')
        plt.legend()
        plt.suptitle(title)
        plt.savefig('curve-step')
        plt.close('all')

def draw_R(R):
    if not isinstance(R, list):
        R = torch.load(R)
    plt.figure(1, figsize=(10, 20))
    plt.plot(R, linewidth=1.0)
    plt.xlabel('step')
    plt.ylabel('R')
    plt.savefig('train_log/curve/R')
    plt.close('all')



def main():
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    p = subparsers.add_parser('draw_feature')
    p.set_defaults(func = draw_feature)
    p.add_argument('--path', default='train_log/models/best-accuracy')
    p.add_argument('--scale', default=1, type=int)
    p.add_argument('--data_set', default=data_set)

    p = subparsers.add_parser('draw_curve')
    p.set_defaults(func=draw_curve)
    p.add_argument('--curv_state', default='train_log/curve/curve_info')
    p.add_argument('--root', default='.')
    p.add_argument('--data_set', default=data_set)

    args = parser.parse_args()
    # args.func(args)
    args.func(args.path, args.scale, args.data_set)

if __name__ == "__main__":
   main()


