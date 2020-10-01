# -*- coding: utf-8 -*-
import poisoner
from my_args import setup_argparse
from torch.autograd import Variable
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy.random as nr
import numpy as np
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from math import ceil, floor
from time import sleep
import os
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
np.set_printoptions(threshold=100)
matplotlib.use('Agg')


def ember(args):
    repeat = 1
    current = 1
    while (current <= repeat):
        print(current, file=sys.stderr)
        nDatas = 400
        nFeatures = 2351
        goodware_csv = pd.read_csv(
            "ember+microsoft/ember_goodware.csv", header=None)
        goodware_csv = pd.concat([pd.DataFrame(np.zeros(
            goodware_csv.shape[0], dtype=np.int64)), goodware_csv], axis=1, ignore_index=True)
        malware_csv = pd.read_csv(
            "ember+microsoft/microsoft_malware_selected.csv", header=None)

        # feature selection
        '''
        0_ByteHistogram(256)[0:1], 1_ByteEntropyHistogram(256)[0:1], 2_SectionInfo(255)
        3_ImportsInfo(1280), 4_ExportsInfo(128), 5_GeneralFileInfo(10)
        6_HeaderFileInfo(62), 7_StringExtractor(104)
        '''
        fl = [list(range(1, 257)), list(range(257, 513)), list(range(513, 768)), list(range(768, 2048)), list(
            range(2048, 2176)), list(range(2176, 2186)), list(range(2186, 2248)), list(range(2248, 2352))]
        features = fl[0]  # + fl[1] + fl[2] + fl[3] + fl[4] + fl[5] + fl[7]
        goodware_csv = goodware_csv.iloc[:, np.insert(features, 0, 0)]
        malware_csv = malware_csv.iloc[:, np.insert(features, 0, 0)]
        goodware_csv.columns = range(goodware_csv.shape[1])
        malware_csv.columns = range(malware_csv.shape[1])

        # split malware and goodware
        good = goodware_csv.sample(
            n=200, random_state=args.random_state)
        mal_0 = malware_csv[malware_csv[0] == args.mal_0_id].sample(
            n=50, random_state=args.random_state)
        mal_1 = malware_csv[malware_csv[0] != args.mal_0_id].sample(
            n=150, random_state=args.random_state)
        concat = pd.concat([good, mal_0, mal_1])
        print("dataset size:{}      (goodware:{}, malware_0:{}, malware_1:{})".format(
            concat.shape[0], good.shape[0], mal_0.shape[0], mal_1.shape[0]))
        print("malware_0 ID :{}".format(args.mal_0_id))

        # 2:1:1 train, valid, test
        mal0_test = mal_0.iloc[35:50]
        mal0_train = mal_0.iloc[0:20]
        mal0_valid = mal_0.iloc[20:35]

        if (args.mal_0_id == 9) & (args.parameter > 0):
            xvalid = pd.concat([mal0_valid, mal_1.iloc[115:115 + args.parameter]]
                               ).sample(frac=1, random_state=args.random_state)
            yvalid = pd.DataFrame(np.concatenate([np.zeros(mal0_valid.shape[0]), np.ones(
                args.parameter)], axis=0)).sample(frac=1, random_state=args.random_state).values.T[0]
        else:
            xvalid = mal0_valid[list(
                range(1, malware_csv.shape[1]))].values
            if args.multi_mode == 0:
                yvalid = np.ones(xvalid.shape[0])
            else:
                yvalid = np.zeros(xvalid.shape[0])
        xtest = pd.concat([good.iloc[0:50], mal0_test, mal_1.iloc[0:35]]).sample(
            frac=1, random_state=args.random_state)
        ytest = np.array([i if i == 0 else 1 for i in xtest[0].values])

        xtrain = pd.concat([good.iloc[50:150], mal0_train, mal_1.iloc[35:115]]).sample(
            frac=1, random_state=args.random_state)
        ytrain = np.array(
            [i if i == 0 else 1 for i in xtrain[0].values])

        xtest_indices = xtest[0].values
        xtrain_indices = xtrain[0].values
        xtrain = xtrain[list(range(1, malware_csv.shape[1]))].values
        xtest = xtest[list(range(1, malware_csv.shape[1]))].values
        if (args.mal_0_id == 9) & (args.parameter > 0):
            xvalid = xvalid[list(
                range(1, malware_csv.shape[1]))].values
        nClasses = int(ytrain.max() + 1)
        nFeatures = xtrain.shape[1]
        xmal0 = mal0_train[list(range(1, malware_csv.shape[1]))].values
        test_xmal0 = mal0_test[list(
            range(1, malware_csv.shape[1]))].values
        train_good = good.iloc[50:150][list(
            range(1, malware_csv.shape[1]))].values

        if args.scaler == 0:
            scaler = MinMaxScaler()
        xtrain = scaler.fit_transform(xtrain)
        xtest = scaler.transform(xtest)
        xvalid = scaler.transform(xvalid)
        xmal0 = scaler.transform(xmal0)
        test_xmal0 = scaler.transform(test_xmal0)
        train_good = scaler.transform(train_good)
        eta = args.eta
        termination = args.termination
        eps = args.epsilon
        steps = args.steps
        T = args.T
        if(args.phi_map):
            dataset_id = 2
        else:
            dataset_id = -1
        poisoning_num = args.poisoning_num
        specific = args.specific
        multi = args.multi
        epochs = args.epochs
        learning_rate = args.learning_rate
        centroid = np.mean(train_good, axis=0)
        radius = calc_radius(centroid, train_good)
        centroid_t = np.mean(xmal0, axis=0)
        radius_t = calc_radius(centroid_t, xmal0)

        if args.use_cuda == 0:
            torch.cuda.set_device(0)
            device = torch.device("cuda:0")
        elif args.use_cuda == 1:
            torch.cuda.set_device(1)
            device = torch.device("cuda:1")
        else:
            device = torch.device("cpu")

        # np.array -> tensor
        xtrain = torch.from_numpy(xtrain).float().to(device=device)
        ytrain = torch.from_numpy(ytrain).long().to(device=device)
        xvalid = torch.from_numpy(xvalid).float().to(device=device)
        yvalid = torch.from_numpy(yvalid).long().to(device=device)
        xtest = torch.from_numpy(xtest).float().to(device=device)
        ytest = torch.from_numpy(ytest).long().to(device=device)
        xmal0 = torch.from_numpy(xmal0).float().to(device=device)
        ymal0 = torch.from_numpy(np.zeros(xmal0.shape[0])).long().to(device=device)
        test_xmal0 = torch.from_numpy(test_xmal0).float().to(device=device)
        test_ymal0 = torch.from_numpy(np.zeros(test_xmal0.shape[0])).long().to(device=device)

        if (poisoning_num > 0) & (not args.eval_poisoning):
            poison = poisoner.Back_gradient_poisoner(xtrain, ytrain, xvalid, yvalid,
                                                     eta=eta, eps=eps, steps=steps, T=T,
                                                     dataset_id=dataset_id, termination_cond=termination,
                                                     specific=specific, multi=multi, epochs=epochs,
                                                     learning_rate=learning_rate, device=device,
                                                     spmode=args.specific_mode, mulmode=args.multi_mode,
                                                     init_point_list=xtrain_indices, mal_0_id=args.mal_0_id,
                                                     decay=args.decay, constraint=args.constraint, beta=args.beta,
                                                     step_max=args.step_max, init_seed=args.init_seed,
                                                     centroid=centroid, radius=radius, centroid_t=centroid_t, radius_t=radius_t, s_def=args.sphere_defense, flip=args.label_flip, solver=args.solver)

            if (args.solver or args.label_flip):
                pxtrain, pytrain = poison.make_poisoning_data_using_solver(num=poisoning_num)
            else:
                pxtrain, pytrain = poison.make_poisoning_data(num=poisoning_num)

            selected_init_list = poison.get_selected_init_indices()
            del(poison)

            good_n = scaler.transform(
                good[list(range(1, goodware_csv.shape[1]))].values)
            mal0_n = scaler.transform(
                mal_0[list(range(1, malware_csv.shape[1]))].values)
            mal1_n = scaler.transform(
                mal_1[list(range(1, malware_csv.shape[1]))].values)
            pxtrain_n = pxtrain.cpu().numpy()
            good_g = good_n.sum(axis=0) / good_n.shape[0]
            mal0_g = mal0_n.sum(axis=0) / mal0_n.shape[0]
            mal1_g = mal1_n.sum(axis=0) / mal1_n.shape[0]
            pxtrain_g = pxtrain_n.sum(axis=0) / pxtrain_n.shape[0]
            dist_good = np.linalg.norm(good_g - pxtrain_g)
            dist_mal0 = np.linalg.norm(mal0_g - pxtrain_g)
            dist_mal1 = np.linalg.norm(mal1_g - pxtrain_g)

            init_pxtrain_n = xtrain[selected_init_list].cpu().numpy()
            init_pxtrain_g = init_pxtrain_n.sum(
                axis=0) / init_pxtrain_n.shape[0]
            dist_good_init = np.linalg.norm(good_g - init_pxtrain_g)
            dist_mal0_init = np.linalg.norm(mal0_g - init_pxtrain_g)
            dist_mal1_init = np.linalg.norm(mal1_g - init_pxtrain_g)
            pxtrain_numpy = pxtrain.cpu().numpy()
            inv_pxtrain = scaler.inverse_transform(pxtrain_numpy)

            if(args.save_directory == '.'):
                f = open('poisoningDatas_direct.csv', 'w')
            else:
                f = open(args.save_directory + '_poisoningDatas_direct.csv', 'w')
            for i in list(range(pxtrain_numpy.shape[0])):
                for j in list(range(pxtrain_numpy.shape[1])):
                    if j == pxtrain_numpy.shape[1] - 1:
                        f.write("{}\n".format(pxtrain_numpy[i][j]))
                    else:
                        f.write("{}, ".format(pxtrain_numpy[i][j]))
            f.close()
            if(args.save_directory == '.'):
                f = open('poisoningDatas.csv', 'w')
            else:
                f = open(args.save_directory + '_poisoningDatas.csv', 'w')
            for i in list(range(inv_pxtrain.shape[0])):
                for j in list(range(inv_pxtrain.shape[1])):
                    if j == inv_pxtrain.shape[1] - 1:
                        f.write("{}\n".format(inv_pxtrain[i][j]))
                    else:
                        f.write("{}, ".format(inv_pxtrain[i][j]))
            f.close()

        if args.eval_poisoning:
            pdata_csv = pd.read_csv(args.pdata_path, header=None)
            pxtrain_numpy = scaler.transform(
                pdata_csv.iloc[0:poisoning_num].values)
            pxtrain = torch.from_numpy(pxtrain_numpy).float().to(device=device)
            pytrain = torch.from_numpy(
                np.zeros(pxtrain.shape[0])).long().to(device=device)

        epochs = args.epochs
        learning_rate = args.learning_rate
        acc_data, prec0to1, prec1to0, rec0to1, rec1to0 = [
        ], [], [], [], []
        mal0to0, mal1to0, acc_limited = [], [], []
        train_mal0_loss_data, test_mal0_loss_data, valid_loss_data = [], [], []
        n = -1
        xtrain_pre = xtrain
        ytrain_pre = ytrain
        while(n < poisoning_num):
            print('\n --- {0} points poisoning ({1}% poisoning)'.format(
                n + 1, (n + 1) / (xtrain.shape[0] + 1) * 100))
            model = LogisticRegression(nFeatures, nClasses).to(device=device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            if (n != -1):
                xtrain = xtrain_pre
                ytrain = ytrain_pre
                xtrain = torch.tensor(
                    torch.cat((xtrain, pxtrain[n].reshape(1, nFeatures)), 0).detach())
                ytrain = torch.tensor(
                    torch.cat((ytrain, pytrain[n].reshape(1))).detach())
                xtrain_pre = xtrain
                ytrain_pre = ytrain

            list_to_delete = []
            for i, (x, y) in enumerate(zip(xtrain, ytrain)):
                if (y == 0):
                    distanse = np.linalg.norm(x - centroid)
                    if (distanse > radius + 0.01):
                        list_to_delete.append(i)
            s = 0
            for i in list_to_delete:
                if (i >= len(xtrain) - n):
                    s += 1
            print("Poison Removal Rate:", s / len(pxtrain))
            for i in sorted(list_to_delete, reverse=True):
                xtrain = torch.cat([xtrain[0:i], xtrain[i + 1:]])
                ytrain = torch.cat([ytrain[0:i], ytrain[i + 1:]])

            for epoch in range(epochs):
                if args.shuffle:
                    perm = np.arange(xtrain.shape[0])
                    np.random.shuffle(perm)
                    xtrain = xtrain[perm]
                    ytrain = ytrain[perm]
                optimizer.zero_grad()
                loss = criterion(model(xtrain), ytrain)
                loss.backward()
                optimizer.step()
                train_loss = loss.data
            train_mal0_loss = criterion(model(xmal0), ymal0).data

            def test(xtest, ytest):
                outputs = model(xtest)
                test_loss = criterion(outputs, ytest)
                test_mal0_loss = criterion(model(test_xmal0), test_ymal0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == ytest.data).sum()
                acc = int(correct) / ytest.size(0)
                np_predicted = predicted.cpu().numpy()
                np_ytest = ytest.cpu().numpy()
                each_pred = np.zeros([3, nClasses])
                acc_l = 0

                for i, v in enumerate(np_ytest):
                    if v == 0:
                        each_pred[0][np_predicted[i]] += 1
                    elif v == 1:
                        if xtest_indices[i] == args.mal_0_id:
                            each_pred[1][np_predicted[i]] += 1
                        elif (xtest_indices[i] != args.mal_0_id) & (xtest_indices[i] != 0):
                            each_pred[2][np_predicted[i]] += 1
                each_pred[0] = each_pred[0] / \
                    len([i for i in xtest_indices if i == 0])
                each_pred[1] = each_pred[1] / \
                    len([i for i in xtest_indices if i == args.mal_0_id])
                each_pred[2] = each_pred[2] / \
                    len([i for i in xtest_indices if i !=
                         0 and i != args.mal_0_id])
                num_mal1_good = len(
                    [i for i in xtest_indices if i != args.mal_0_id])

                for i, v in enumerate(xtest_indices):
                    if ((v == 0) & (np_predicted[i] == 0)) | ((v != 0) & (v != args.mal_0_id) & (np_predicted[i] == 1)):
                        acc_l += 1

                acc_l /= num_mal1_good
                prec = precision(np_predicted, np_ytest, nClasses)
                rec = recall(np_predicted, np_ytest, nClasses)
                return test_loss.data, test_mal0_loss.data, acc, each_pred, prec, rec, acc_l, np_predicted

            test_loss, test_mal0_loss, acc, each_pred, prec, rec, acc_l, test_predict = test(
                xtest, ytest)
            valid_loss = criterion(model(xvalid), yvalid).data
            acc_data.append(acc)
            prec0to1.append(prec[0][1])
            prec1to0.append(prec[1][0])
            rec0to1.append(rec[0][1])
            rec1to0.append(rec[1][0])
            mal0to0.append(each_pred[1][0])
            mal1to0.append(each_pred[2][0])
            acc_limited.append(acc_l)
            print("Epochs %d, Train loss: %.5f Test loss: %.5f Test accuracy: %.5f" % (epochs, train_loss, test_loss, acc))
            print("Prediction Result  Correct(good, mal_B, mal_NB) * Prediction(good, mal)")
            print(each_pred)
            del(model)
            del(optimizer)
            del(criterion)
            sys.stdout.flush()
            n += 1

        if(current == 1):
            Y1 = np.array(rec0to1)
            Y2 = np.array(mal0to0)
            Y3 = np.array(mal1to0)
        else:
            Y1 += np.array(rec0to1)
            Y2 += np.array(mal0to0)
            Y3 += np.array(mal1to0)
        current += 1
    Y1 /= repeat
    Y2 /= repeat
    Y3 /= repeat
    x = np.array([i / (i + xtrain.shape[0] - poisoning_num)
                  for i in range(poisoning_num + 1)])
    print()
    print("poisoning_rate", "good->mal", "mal_B->good", "mal_NB->good")
    for i in range(len(x)):
        print(x[i], Y1[i], Y2[i], Y3[i])
    return


def calc_radius(cent, data):
    distanse = [np.linalg.norm(i - cent) for i in data]
    sorted_distanse = sorted(distanse)
    elem = sorted_distanse[floor(len(distanse) * (1 - args.eliminate))]
    return elem


def precision(pred, test, nClasses):
    prec = np.zeros((nClasses, nClasses))
    for i, v in enumerate(pred):
        prec[v][test[i]] += 1
    each = np.zeros(nClasses)
    for v in pred:
        each[v] += 1
    return prec / each.reshape(-1, 1)


def recall(pred, test, nClasses):
    rec = np.zeros((nClasses, nClasses))
    for i, v in enumerate(test):
        rec[v][pred[i]] += 1

    each = np.zeros(nClasses)
    for v in test:
        each[v] += 1
    return rec / each.reshape(-1, 1)


class LogisticRegression(nn.Module):
    def __init__(self, nFeatures, nClasses):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(nFeatures, nClasses)

    def forward(self, x):
        out = self.linear(x)
        return out


if __name__ == '__main__':
    parser = setup_argparse()
    args = parser.parse_args()
    sys.stdout.flush()
    ember(args)
