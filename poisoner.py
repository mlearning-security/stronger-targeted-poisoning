# -*- coding: utf-8 -*-
import random
from numpy.random import Generator, MT19937
import time
import math
import cvxpy as cp
import numpy.random as nr
import io
import sys
import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
np.set_printoptions(threshold=100)


class Back_gradient_poisoner():
    def __init__(self, xtrain, ytrain, xvalid, yvalid, eta, eps, steps, T, dataset_id, termination_cond, specific, multi, epochs, learning_rate, device, spmode, mulmode, decay, centroid, centroid_t, radius, radius_t, s_def, flip, solver, init_point_list=np.array([]), mal_0_id=-1, constraint=0, beta=0.1, step_max=-1, init_seed=0):
        self.nFeatures = xtrain.shape[1]
        self.nClasses = int(torch.max(ytrain)) + 1
        self.point = torch.empty(
            1, self.nFeatures, device=device)
        self.label = torch.empty(
            1, dtype=torch.long, device=device)
        self.model = None
        self.criterion = None
        self.dataset_id = dataset_id
        self.w_0 = torch.empty(self.nClasses, self.nFeatures, device=device)
        self.b_0 = torch.empty(self.nClasses, device=device)
        self.dx = torch.empty(1, self.nFeatures, device=device)
        self.dt = torch.empty(self.nClasses, self.nFeatures + 1, device=device)
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.w_indices = torch.tensor(
            list(range(0, self.nFeatures)), device=device)
        self.b_indices = torch.tensor([self.nFeatures], device=device)
        if decay:
            self.eta = torch.tensor(0., device=device)
            self.init_eta = torch.tensor(eta, device=device)
            self.decay = None
        else:
            self.eta = torch.tensor(eta, device=device)
        self.termination_cond = termination_cond
        self.decay_on = decay
        self.eps = eps
        self.steps = steps
        self.T = T
        self.specific = specific
        self.multi = multi
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.spmode = spmode
        self.mulmode = mulmode
        self.init_point_list = init_point_list
        self.mal_0_id = mal_0_id
        self.selected_init = []
        self.constraint = constraint
        self.beta = torch.tensor(beta, device=device)
        self.step_max = step_max
        self.init_seed = random.randint(1, 10000)
        self.init_pdata_indices = None
        self.init_plabel_list = None
        self.init_goodware_indices = None
        self.xvalid = xvalid
        self.yvalid = yvalid
        self.centroid = torch.tensor(centroid, device=device).float()
        self.radius = torch.tensor(radius, device=device).float()
        self.centroid_t = torch.tensor(centroid, device=device).float()
        self.radius_t = torch.tensor(radius_t, device=device).float()
        self.s_def = s_def
        self.solver = solver
        self.phi_iris = torch.tensor([0., 10.], device=device)
        self.phi_ransom = torch.tensor([0., 1., 0.5], device=device)
        self.eta2 = torch.tensor(eta, device=device)
        eta_limit = 0.1
        self.eta_limit = torch.tensor(eta_limit, device=device)
        self.init_eta_limit = torch.tensor(eta_limit, device=device)
        self.decay_limit = None

    def phi_map(self, option=0):
        if(self.dataset_id == -1):
            return
        elif(self.dataset_id == 1):
            if(option == 0):
                for i in range(self.nFeatures):
                    x = self.point[0][i]
                    if(x <= self.phi_ransom[0]):
                        x.copy_(self.phi_ransom[0])
                    elif(x >= self.phi_ransom[1]):
                        x.copy_(self.phi_ransom[1])
            elif(option == 1):  # final
                for i in range(self.nFeatures):
                    x = self.point[0][i]
                    if(x <= self.phi_ransom[2]):
                        x.copy_(self.phi_ransom[0])
                    else:
                        x.copy_(self.phi_ransom[1])
        elif(self.dataset_id == 2):  # ember
            if(option == 0):
                for i in range(self.nFeatures):
                    x = self.point[0][i]
                    if(x <= self.phi_ransom[0]):
                        x.copy_(self.phi_ransom[0])
                    elif(x >= self.phi_ransom[1]):
                        x.copy_(self.phi_ransom[1])
        return

    def gradient_descent(self):
        [w, b] = list(self.model.parameters())
        w.data.copy_(self.w_0)
        b.data.copy_(self.b_0)
        t = 0
        while(t < self.T):
            self.model.zero_grad()
            loss = self.criterion(self.model(
                torch.cat((self.xtrain, self.point))), torch.cat((self.ytrain, self.label)))
            loss.backward()
            for param in self.model.parameters():
                param.data.sub_(param.grad.data * self.eta)
            t += 1
        return

    def back_gradient_descent(self):
        self.dx.copy_(torch.zeros(1, self.nFeatures))
        [w, b] = list(self.model.parameters())
        self.model.zero_grad()
        self.point.requires_grad_(True)
        val_loss = self.criterion(self.model(self.xvalid), self.yvalid)
        w_grad_val_loss = grad(val_loss, w, create_graph=True)[0]
        b_grad_val_loss = grad(val_loss, b, create_graph=True)[0]
        g = w_grad_val_loss[0].reshape(self.nFeatures, 1)
        gT = torch.transpose(g, 0, 1)

        def jacobian(y, x, create_graph=False):
            jac = []
            flat_y = y.reshape(-1)
            grad_y = torch.zeros_like(flat_y)
            for i in range(len(flat_y)):
                grad_y[i] = 1.
                grad_x, = torch.autograd.grad(
                    flat_y, x, grad_y, retain_graph=True, create_graph=create_graph, allow_unused=True)
                jac.append(grad_x[0].reshape(x.shape[1]))
                grad_y[i] = 0.
            return torch.stack(jac).reshape(y.shape[0], x.shape[1])

        self.model.zero_grad()
        loss = self.criterion(self.model(
            torch.cat((self.xtrain, self.point))), torch.cat((self.ytrain, self.label)))
        w_grad_train_loss = grad(loss, w, create_graph=True)[0]
        H = jacobian(w_grad_train_loss[0], w)
        delta = 10
        H = delta * torch.eye(self.nFeatures, self.nFeatures) + H
        Hinv = H.inverse()
        self.model.zero_grad()
        ploss = self.criterion(self.model(self.point), self.label)
        p_w_grad_loss = grad(ploss, w, create_graph=True)[0]
        p_xw_grad_loss = jacobian(p_w_grad_loss[0], self.point)
        pT = torch.transpose(p_xw_grad_loss, 0, 1)
        dwdx = torch.mm(Hinv, p_xw_grad_loss)
        dt = torch.mm(gT, dwdx)
        x = dt.to('cpu').detach().numpy().copy()
        dt = torch.from_numpy(x.astype(np.float32)).clone()
        self.dx.copy_(dt)
        self.point.requires_grad_(False)
        return self.dx

    def make_poisoning_point_using_solver(self):

        def sigmoid(x):
            return 1. / (1. + math.exp(-x))

        def necessary_sphere(cent, radius, data):
            distanse = torch.sqrt(torch.sum((data - cent)**2))
            return distanse > radius

        def vec2centroid(cent, radius, data):
            to_cent = cent - data
            distanse = torch.sqrt(torch.sum((data - cent)**2))
            return to_cent * (distanse - radius) / distanse

        self.init_poisoning_point()

        if (self.solver):
            [w, b] = list(self.model.parameters())
            w = w[0].detach().numpy()
            c = b[0].detach().numpy()
            mu1 = self.centroid
            mu2 = self.centroid_t
            mu2 = self.point[0]
            r1 = self.radius
            r2 = self.radius_t
            r2 = 0.7
            n = len(w)
            x = cp.Variable(n)
            objective = cp.Minimize(w * x + c)
            print("constraints: ||x|| <= r , r = {:.5f}".format(r1))
            print("r2 = {:.5f}".format(r2))
            constraints = [cp.sum_squares(
                x - mu1) <= r1 * r1, cp.sum_squares(x - mu2) <= r2 * r2, x >= 0, x <= 1]
            problem = cp.Problem(objective, constraints)
            result = problem.solve()
            print("min_x ( w * x + c ) : {:.2f} ||x-mu1|| : {:.5f} ||x-mu2|| : {:.5f}".format(
                result, cp.sum_squares(x - mu1).value, cp.sum_squares(x - mu2).value))
            print("x : {}".format(x.value))
            p = sigmoid(result)
            self.point[0] = torch.tensor(x.value)

        if necessary_sphere(self.centroid, self.radius, self.point[0]):
            self.point[0].add_(vec2centroid(
                self.centroid, self.radius, self.point[0]))
            loss = self.criterion(self.model(self.point), self.label).item()
        self.phi_map(1)

        if necessary_sphere(self.centroid, self.radius, self.point[0]):
            v = self.centroid - self.point[0]
            for i in range(1, 10):
                tempx = self.point[0] + 0.1 * i * v
                for i in range(self.nFeatures):
                    if(tempx[i] <= self.phi_ransom[2]):
                        tempx[i].copy_(self.phi_ransom[0])
                    else:
                        tempx[i].copy_(self.phi_ransom[1])
                if (not necessary_sphere(self.centroid, self.radius, tempx)):
                    self.point[0] = tempx
                    break
        loss = self.criterion(self.model(self.point), self.label).item()
        sys.stdout.flush()
        return

    def make_poisoning_point(self):
        def sigmoid(x):
            return 1. / (1. + math.exp(-x))

        def necessary_sphere(cent, radius, data):
            distanse = np.linalg.norm(data - cent)
            return distanse, (distanse > radius + 0.1)

        def vec2centroid(cent, radius, data):
            to_cent = cent - data
            distanse = torch.sqrt(torch.sum((data - cent)**2))
            return to_cent * (distanse - radius) / distanse

        self.init_poisoning_point()
        self.model = LogisticRegression(
            self.nFeatures, self.nClasses).to(device=self.device)
        self.criterion = nn.CrossEntropyLoss()
        [w, b] = list(self.model.parameters())
        self.w_0.copy_(w.data)
        self.b_0.copy_(b.data)

        if self.constraint != 0:
            const_goodware = self.xtrain[self.init_goodware_indices.pop(0)]
        init_point = torch.tensor(self.point)

        loss = sys.float_info.max
        i = 0
        while(True):
            if self.step_max == i:
                break
            if self.decay_on:
                self.decay = torch.tensor(
                    1 / np.sqrt(i + 1), device=self.device)
                self.eta.copy_(self.init_eta * self.decay)
            pre_point = torch.tensor(self.point)
            pre_loss = loss
            self.gradient_descent()

            if self.constraint == 1:
                self.point.requires_grad_(True)
                l2norm_square = torch.norm(self.point - const_goodware)**2
                grad_l2 = grad(l2norm_square, self.point)[0]
                self.point.detach_()
                self.point.add_(self.eta * self.back_gradient_descent())
                self.point.sub_(self.eta * self.beta * grad_l2)
            elif self.constraint == 0:
                self.point.add_(
                    self.eta2 * self.back_gradient_descent())

            if self.s_def:
                dist_defense, is_defense = necessary_sphere(
                    self.centroid, self.radius, self.point[0])
                if is_defense:
                    vec_defense = vec2centroid(
                        self.centroid, self.radius, self.point[0])
                    self.point[0].add_(vec_defense)
            self.phi_map()
            print("Poisoning Point (steps:{}): {}".format(
                i + 1, self.point.cpu().numpy()))
            print("eta: {}".format(self.eta))
            sys.stdout.flush()

            i += 1

            [w, b] = list(self.model.parameters())
            w.data.copy_(torch.ones(self.nClasses, self.nFeatures))
            b.data.copy_(torch.ones(self.nClasses))
            for epoch in range(self.epochs):
                self.model.zero_grad()
                loss = self.criterion(self.model(
                    torch.cat((self.xtrain, self.point))), torch.cat((self.ytrain, self.label)))
                loss.backward()
                for param in self.model.parameters():
                    param.data.sub_(param.grad.data * self.learning_rate)
            if self.constraint == 1:
                l2norm_square = torch.norm(self.point[0] - const_goodware)**2
                loss = self.criterion(self.model(self.xvalid), self.yvalid).item(
                ) + self.beta.item() * l2norm_square.item()
            elif self.constraint == 0:
                loss = self.criterion(self.model(
                    self.xvalid), self.yvalid).item()
            if(pre_loss - loss < self.eps):
                break
        self.phi_map(1)
        del(self.model)
        del(self.criterion)
        return

    def init_poisoning_point(self):
        self.point[0].copy_(
            self.xtrain[self.init_pdata_indices.pop(0)])
        self.label.copy_(torch.tensor(
            [self.init_plabel_list.pop(0)]).long())
        return

    def init_selection(self, num):
        def necessary_sphere(cent, radius, data):
            distanse = torch.sqrt(torch.sum((data - cent)**2))
            return distanse > radius
        nr.seed(self.init_seed)
        l = np.where(self.init_point_list == self.mal_0_id)[0]

        for i in reversed(range(len(l))):
            if (necessary_sphere(self.centroid, self.radius, self.xtrain[l[i]])):
                l = np.delete(l, i)

        self.init_pdata_indices = nr.choice(l, num).tolist()
        self.init_plabel_list = [0] * num
        self.selected_init = self.init_pdata_indices[:]
        nr.seed(self.init_seed)
        l = np.where(self.init_point_list == 0)[0]
        self.init_goodware_indices = nr.choice(l, num).tolist()
        return

    def make_poisoning_data_using_solver(self, num):
        start = time.time()
        poisoning_data_x = torch.empty(num, self.nFeatures, device=self.device)
        poisoning_data_y = torch.empty(
            num, dtype=torch.long, device=self.device)
        self.init_selection(num)
        fraction = torch.tensor(num / self.xtrain.shape[0], device=self.device)
        self.model = LogisticRegression(
            self.nFeatures, self.nClasses).to(device=self.device)
        self.criterion = nn.CrossEntropyLoss()
        print(self.model.parameters())
        [w, b] = list(self.model.parameters())
        w.data.fill_(0)
        b.data.fill_(0)
        burn = 2000
        n = 1
        p = 0
        while(n <= num + burn):
            if (n > burn):
                print("\n --- creating point ({})".format(p + 1))
                self.make_poisoning_point_using_solver()
                sys.stdout.flush()
            self.decay_limit = torch.tensor(1 / np.sqrt(n + 1), device=self.device)
            self.eta_limit.copy_(self.init_eta_limit * self.decay_limit)
            self.model.zero_grad()
            loss1 = self.criterion(self.model(self.xtrain), self.ytrain)
            [w, b] = list(self.model.parameters())
            w_grad_loss1 = grad(loss1, w, create_graph=True)[0]
            b_grad_loss1 = grad(loss1, b, create_graph=True)[0]
            w.detach().sub_(self.eta_limit * (w_grad_loss1.detach()))
            b.detach().sub_(self.eta_limit * (b_grad_loss1.detach()))
            if (n > burn):
                loss2 = self.criterion(self.model(self.point), self.label)
                w_grad_loss2 = grad(loss2, w, create_graph=True)[0]
                b_grad_loss2 = grad(loss2, b, create_graph=True)[0]
                w.detach().sub_(self.eta_limit * (w_grad_loss2.detach() * fraction))
                b.detach().sub_(self.eta_limit * (b_grad_loss2.detach() * fraction))
            sys.stdout.flush()
            poisoning_data_x[p].copy_(self.point[0])
            poisoning_data_y[p].copy_(self.label[0])
            if (n > burn):
                p += 1
            n += 1
        [w, b] = list(self.model.parameters())
        del(self.model)
        del (self.criterion)
        return poisoning_data_x, poisoning_data_y

    def make_poisoning_data(self, num):
        start = time.time()
        poisoning_data_x = torch.empty(num, self.nFeatures, device=self.device)
        poisoning_data_y = torch.empty(
            num, dtype=torch.long, device=self.device)
        self.init_selection(num)
        n = 0
        while(n < num):
            print("\n --- creating point ({})".format(n + 1))
            sys.stdout.flush()
            self.make_poisoning_point()
            if(torch.sum(self.point != self.point).item() != 0):
                continue
            else:
                poisoning_data_x[n].copy_(self.point[0])
                poisoning_data_y[n].copy_(self.label[0])
            n += 1
        return poisoning_data_x, poisoning_data_y

    def get_selected_init_indices(self):
        return self.selected_init


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out
