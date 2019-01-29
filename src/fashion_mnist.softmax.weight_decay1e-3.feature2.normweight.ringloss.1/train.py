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
from draw import draw_curve, draw_feature, draw_R


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

    list_R = []
    while True:
        clock.tock()
        if clock.epoch > n_epochs:
            break

        logger.info("Epoch {}/{}".format(clock.epoch, n_epochs))
        logger.info("-" * 10)

        train_loss, train_correct = 0.0, 0.0
        train_ringloss, train_celoss = 0.0, 0.0
        train_correct_cls, train_num_cls = [0] * 10, [0] * 10


        for idx, data in enumerate(data_loader_train):
            X_train, y_train = data
            X_train, y_train = Variable(X_train).to(device), Variable(y_train).to(device)

            net.train()
            outputs, feature_norm, R = net(X_train)
            list_R.append(R.item())

            _, pred = torch.max(outputs.data, 1)

            for i in range(10):
                index = y_train == i
                pred_i = pred[index]
                label_i = y_train[index].data
                train_num_cls[i] += len(pred_i)
                train_correct_cls[i] += torch.sum(pred_i == label_i).item()

            optimizer.zero_grad()
            ringloss = torch.mean(((feature_norm - R) ** 2 + 1e-6).sqrt())

            if torch.isnan(ringloss):
                print(clock.epoch)
                print(clock.step)
                print(R)
                print(torch.isnan(feature_norm).sum())
                print(torch.isnan(feature_norm - R).sum())
                print(torch.isnan((feature_norm - R)**2).sum())
                # print(torch.isnan((feature_norm - R) ** 2).sqrt())
                print("ERROR")
                os._exit(1)

            loss_ce = cost(outputs, y_train)
            loss = loss_ce + lamb * ringloss
            loss.backward()
            optimizer.step()

            step_correct = torch.sum(pred == y_train.data).item()

            train_loss += loss.item()
            train_celoss += loss_ce.item()
            train_ringloss += ringloss.item()

            train_correct += step_correct

            if idx % 10 == 0: # update for every 10 step
                session.update_curv_state('train_step', step_correct/len(y_train), loss.item())

            if idx % 100 == 0: # print train info
                logger.info(R.item())
                logger.info("step: {}, ce loss: {:.4f}, ringloss:{:.4f}, loss:{:.4f}, train acc: {:.4f}".format(idx, loss_ce.item(), ringloss.item(), loss.item(), step_correct / len(y_train)))
            clock.tick()

        test_loss, test_correct = 0.0, 0.0
        test_correct_cls, test_num_cls = [0] * 10, [0] * 10

        for data in data_loader_test:
            X_test, y_test = data
            X_test, y_test = Variable(X_test).to(device), Variable(y_test).to(device)
            net.eval()
            outputs, _, _= net(X_test)
            _, pred = torch.max(outputs.data, 1)

            for i in range(10):
                idx = y_test == i
                pred_i = pred[idx]
                label_i = y_test[idx].data
                test_num_cls[i] += len(pred_i)
                test_correct_cls[i] += torch.sum(pred_i == label_i).item()

            test_correct += torch.sum(pred == y_test.data).item()
            test_loss += cost(outputs, y_test).item()

        train_acc = train_correct / len(data_train)
        train_loss = 64 * train_loss / len(data_train)
        train_ringloss = 64 * train_ringloss / len(data_train)
        train_celoss = 64 * train_celoss / len(data_train)
        train_acc_cls = np.array(train_correct_cls) / np.array(train_num_cls)
        assert np.sum(np.array(train_num_cls)) == len(data_train)
        assert np.sum(np.array(train_correct_cls)) == train_correct

        test_acc = test_correct /len(data_test)
        test_loss = 64 * test_loss / len(data_test)
        test_acc_cls = np.array(test_correct_cls) / np.array(test_num_cls)
        assert np.sum(np.array(test_num_cls)) == len(data_test)
        assert np.sum(np.array(test_correct_cls)) == test_correct


        session.update_best_state(test_acc)
        session.update_curv_state('train_epoch', train_acc, train_loss, train_acc_cls)
        session.update_curv_state('val_epoch', test_acc, test_loss, test_acc_cls)

        logger.info("Loss is:{:.4f}/{:.4f}, Train Accuracy is:{:.2f}%, Test Accuracy is:{:.2f}%, {}".format(
            train_celoss, train_ringloss, 100 * train_acc, 100 * test_acc, session.best_state))
        logger.info(', '.join([ '{:.4f}'.format(x) for x in train_acc_cls]))
        logger.info(', '.join(['{:.4f}'.format(x) for x in test_acc_cls]))


        if clock.epoch in [5, 20, 50, 100]:
            session.save_checkpoint('epoch-{}'.format(clock.epoch))
        session.save_checkpoint('latest')

    session.end()
    torch.save(list_R, 'train_log/curve/R_info')

    print('drawing curve')
    draw_curve(session.curv_stat, session.log_curv_dir, data_set)
    if n_feature == 2:
        print('drawing featue')
        draw_feature(os.path.join(session.log_model_dir, 'best-accuracy'), 1, data_set)

    draw_R(list_R)




if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted!")
        sys.exit(1)
