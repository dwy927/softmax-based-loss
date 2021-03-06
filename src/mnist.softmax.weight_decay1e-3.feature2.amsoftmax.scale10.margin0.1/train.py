import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from IPython import embed
sys.path.append('..')
from common import Session, load_dataset, get_label_list
from config import *
from draw import draw_curve, draw_feature


def main():

    net = Model(n_feature)
    optimizer = torch.optim.Adam(net.parameters(), weight_decay=weight_decay)

    session = Session(net, optimizer)
    device = session.device
    clock = session.clock
    logger = session.logger

    net.to(device)
    logger.info(net)

    # prepare data
    data_loader_train, data_loader_test,  data_train, data_test = load_dataset(data_set=data_set)

    cost = torch.nn.CrossEntropyLoss()

    while True:
        clock.tock()
        if clock.epoch > n_epochs:
            break

        logger.info("Epoch {}/{}".format(clock.epoch, n_epochs))
        logger.info("-" * 10)

        train_loss_amsoftmax, train_loss_ce, train_correct = 0.0, 0.0, 0.0
        train_correct_cls, train_num_cls = [0] * 10, [0] * 10

        for idx, data in enumerate(data_loader_train):
            X_train, y_train = data
            X_train, y_train = Variable(X_train).to(device), Variable(y_train).to(device)

            net.train()
            outputs = net(X_train)

            _, pred = torch.max(outputs.data, 1)

            for i in range(10):
                index = y_train == i
                pred_i = pred[index]
                label_i = y_train[index].data
                train_num_cls[i] += len(pred_i)
                train_correct_cls[i] += torch.sum(pred_i == label_i).item()

            optimizer.zero_grad()
            outputs_2 = am_softmax(outputs, y_train, scale, margin)
            loss_amsoftmax = cost(outputs_2, y_train)
            loss_ce = cost(scale * outputs, y_train)

            loss_amsoftmax.backward()
            optimizer.step()

            step_correct = torch.sum(pred == y_train.data).item()
            train_loss_amsoftmax += loss_amsoftmax.item()
            train_loss_ce += loss_ce.item()
            train_correct += step_correct

            if idx % 10 == 0: # update for every 10 step
                session.update_curv_state('train_step', step_correct/len(y_train), loss_amsoftmax.item())

            if idx % 100 == 0: # print train info
                logger.info("step: {}, train am-softmax loss: {:.4f}, ce loss: {:.4f}, train acc: {:.4f}".format(idx, loss_amsoftmax.item(), loss_ce.item(), step_correct / len(y_train)))
            clock.tick()

        test_loss, test_correct = 0.0, 0.0
        test_correct_cls, test_num_cls = [0] * 10, [0] * 10

        for data in data_loader_test:
            X_test, y_test = data
            X_test, y_test = Variable(X_test).to(device), Variable(y_test).to(device)
            net.eval()
            outputs = net(X_test)
            _, pred = torch.max(outputs.data, 1)

            for i in range(10):
                idx = y_test == i
                pred_i = pred[idx]
                label_i = y_test[idx].data
                test_num_cls[i] += len(pred_i)
                test_correct_cls[i] += torch.sum(pred_i == label_i).item()

            test_correct += torch.sum(pred == y_test.data).item()
            test_loss += cost(scale*outputs, y_test).item()

        train_acc = train_correct / len(data_train)
        train_loss_amsoftmax = 64 * train_loss_amsoftmax / len(data_train)
        train_acc_cls = np.array(train_correct_cls) / np.array(train_num_cls)
        assert np.sum(np.array(train_num_cls)) == len(data_train)
        assert np.sum(np.array(train_correct_cls)) == train_correct

        test_acc = test_correct /len(data_test)
        test_loss = 64 * test_loss / len(data_test)
        test_acc_cls = np.array(test_correct_cls) / np.array(test_num_cls)
        assert np.sum(np.array(test_num_cls)) == len(data_test)
        assert np.sum(np.array(test_correct_cls)) == test_correct


        session.update_best_state(test_acc)
        session.update_curv_state('train_epoch', train_acc, train_loss_amsoftmax, train_acc_cls)
        session.update_curv_state('val_epoch', test_acc, test_loss, test_acc_cls)

        logger.info("Loss is:{:.4f}, Train Accuracy is:{:.2f}%, Test Accuracy is:{:.2f}%, {}".format(
            train_loss_amsoftmax, 100 * train_acc, 100 * test_acc, session.best_state))
        logger.info(', '.join([ '{:.4f}'.format(x) for x in train_acc_cls]))
        logger.info(', '.join(['{:.4f}'.format(x) for x in test_acc_cls]))


        if clock.epoch in [5, 20, 50, 100]:
            session.save_checkpoint('epoch-{}'.format(clock.epoch))
        session.save_checkpoint('latest')

    session.end()

    print('drawing curve')
    draw_curve(session.curv_stat, session.log_curv_dir, data_set)
    if n_feature == 2:
        print('drawing featue')
        draw_feature(os.path.join(session.log_model_dir, 'best-accuracy'), 1, data_set)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted!")
        sys.exit(1)
