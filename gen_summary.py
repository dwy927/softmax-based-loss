from glob import glob
import os
import argparse
import re
from tabulate import tabulate

from IPython import embed

filter_keys = ['weight_decay1e-5', 'withbias', 'lr.1e-4']
filter_lists = [
    'mnist.softmax.weight_decay1e-5.feature2',
    'mnist.softmax.weight_decay1e-5.feature2.withbias',
    'mnist.softmax.weight_decay1e-5.feature2.normweight',
    'mnist.softmax.weight_decay1e-5.feature2.lr.1e-4',
    'mnist.softmax.weight_decay1e-5.feature2.lr.1e-4.withbias',
    'mnist.softmax.weight_decay1e-3.feature2.normweight.ringloss.0.1.withbias',
    'mnist.softmax.weight_decay1e-3.feature2.withbias',
]

def gen_summary(dataset='mnist'):
    assert dataset in ['mnist', 'fashion_mnist']
    os.chdir('train_log')

    experiments = glob('{}.*'.format(dataset))
    experiments = [x for x in experiments if x not in filter_lists]

    header = ['experiment', 'best-acc', 'best-epoch']
    result = []
    for element in experiments:
        file = '{}/worklog.txt'.format(element)
        ret = re.search("The best accuracy model is: {'accuracy': (.*), 'epoch': (.*)}", open(file, mode='r').read(), flags=0)
        if ret != None:
            acc = ret.group(1)
            epoch = ret.group(2)
            result.append([element,float(acc), epoch])
            continue
        ret = re.search("The best accuracy model is: {'epoch': (.*), 'accuracy': (.*)}", open(file, mode='r').read(), flags=0)
        if ret != None:
            acc = ret.group(2)
            epoch = ret.group(1)
            result.append([element, float(acc), epoch])
            continue
        print('not find result: {}'.format(element))

    table = sorted(result, key=lambda item: item[1])

    t = tabulate(table, header, tablefmt='orgtbl')
    lines = t.split('\n')
    lines[1] = lines[1].replace('+', '|')
    t_wiki = '\n'.join(lines)

    print(t_wiki)
    fname = '{}_summary.txt'.format(dataset)
    print('the summary file is saved at {}'.format(os.getcwd()))
    with open(fname, 'w') as fout:
        print(t_wiki, file=fout)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices={'mnist', 'fashion_mnist'}, default='mnist')
    args = parser.parse_args()
    gen_summary(args.dataset)


