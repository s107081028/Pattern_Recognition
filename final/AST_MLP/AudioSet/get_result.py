import argparse
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp_path", type=str, default='', help="the root path of the experiment")

if __name__ == '__main__':
    args = parser.parse_args()
    mAP_list = []
    acc_list = []
    for fold in range(1, 6):
        result = np.loadtxt(args.exp_path+'/fold' + str(fold) + '/result.csv', delimiter=',')
        if fold == 1:
            cum_result = np.zeros([result.shape[0], result.shape[1]])
        cum_result = cum_result + result
    result = cum_result / 5
    np.savetxt(args.exp_path+'/result.csv', result, delimiter=',')
    best_epoch = np.argmax(result[:, 0])
    np.savetxt(args.exp_path + '/best_result.csv', result[best_epoch, :], delimiter=',')

    acc_fold = []
    print('--------------Result Summary--------------')
    for fold in range(1, 6):
        result = np.loadtxt(args.exp_path+'/fold' + str(fold) + '/result.csv', delimiter=',')
        acc_fold.append(result[best_epoch, 0])
        print('Fold {:d} accuracy: {:.4f}'.format(fold, result[best_epoch, 0]))
    acc_fold.append(np.mean(acc_fold))
    print('The averaged accuracy of 5 folds is {:.3f}'.format(acc_fold[-1]))
    np.savetxt(args.exp_path + '/acc_fold.csv', acc_fold, delimiter=',')
