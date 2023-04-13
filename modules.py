
from torch import nn
import torch
from tqdm import tqdm
import random
import numpy as np
import sys

class GossipLearning(nn.Module):
    def __init__(self, user_item_pairs: list, latent_dim = 500, update_epochs = 100, lr = 1e-4, probs_for_send = 0.5, node_number = 10, device = torch.device('cuda'), train_set_size = 0.8) -> None:
        super(GossipLearning, self).__init__()
        ## user_item_pairs = [username, itemname, rating]
        random.shuffle(user_item_pairs)
        self.users = list(set([i[0] for i in user_item_pairs]))
        self.items = list(set([i[1] for i in user_item_pairs]))
        self.train_pairs = user_item_pairs[:int(len(user_item_pairs) * train_set_size)]
        self.test_pairs = user_item_pairs[int(len(user_item_pairs) * train_set_size):]
        self.R_test = self.create_R(self.test_pairs)
        self.node_number = node_number
        self.latent_dim = latent_dim
        self.update_epochs = update_epochs
        self.lr = lr
        self.probs_for_send = probs_for_send
        self.device = device
        ## create nodes
        self.nodes = []
        self.rmses = []
        for id in tqdm(range(node_number), desc = 'Generating Nodes'):
            self.nodes.append(Node(id, latent_dim = latent_dim, user_dim = len(self.users), item_dim = len(self.items), R = self.create_R(self.train_pairs[int(id * (len(self.train_pairs) / node_number)): int((id + 1) * (len(self.train_pairs) / node_number))]), 
                                    R_test = self.R_test, nodes = [], update_epochs = self.update_epochs, lr = self.lr, probs_for_send = self.probs_for_send, device = device).to(device))
        for id in range(node_number):
            self.nodes[id].nodes = self.nodes
    def create_R(self, pairs: list):
        R = np.zeros((len(self.users), len(self.items))) - 1
        for pair in pairs:
            R[self.users.index(pair[0]), self.items.index(pair[1])] = pair[2]
        return R
    def train(self, epoches = 100):
        self.transmission = []
        with tqdm(total = epoches) as t:
            for epoch in range(epoches):
                temp_rmse = 0
                temp_transmission = 0
                for node in self.nodes:
                    node.to(self.device)
                    temp_transmission += (node.send_model())
                for node in self.nodes:
                    temp_rmse += min(node.model.rmses)
                t.set_postfix(loss = temp_rmse / len(self.nodes))
                t.update()
                self.rmses.append(temp_rmse / len(self.nodes))
                self.transmission.append(temp_transmission)
        self.transmission = np.cumsum(self.transmission)
class Node(nn.Module):
    def __init__(self, id, latent_dim, user_dim, item_dim, R: np.array, R_test: np.array, nodes, update_epochs = 100, lr = 1e-4, probs_for_send = 0.5, device = torch.device('cuda')) -> None:
        super(Node, self).__init__()
        self.age = 0
        self.id = id
        self.model = PMF(latent_dim, user_dim, item_dim, R, R_test)
        self.model.to(device)
        self.nodes = nodes # neighbor node
        self.update_epochs = update_epochs
        self.lr = lr
        self.probs_send = probs_for_send
        self.device = device
        self.model.train(epochs = self.update_epochs, learning_rate = self.lr, device = device)
    def send_model(self):
        transmission_quantity = 0
        for node in self.nodes:
            if np.random.random() > self.probs_send and node.id != self.id:
                node.receive_model(self.model.U, self.model.V, self.model.age)
                transmission_quantity += (sys.getsizeof(self.model.U) + sys.getsizeof(self.model.V) + sys.getsizeof(self.model.age))
        return transmission_quantity
    def update(self):
        self.model.train(epochs = self.update_epochs, learning_rate = self.lr)
    def receive_model(self, U, V, age):
        self.merge(U, V, age)
        self.update()
    def merge(self, U, V, age):
        propotion = age / (age + self.model.age + 1e-9)
        self.model.age = max(age, self.model.age)
        self.model.U = nn.parameter.Parameter(propotion * U + self.model.U * (1 - propotion))
        self.model.V = nn.parameter.Parameter(propotion * V + self.model.V * (1 - propotion))
    def add_node(self, node):
        self.nodes.append(node)
class PMF(nn.Module):
    def __init__(self, latent_dim, user_dim, item_dim, R: np.array, R_test: np.array) -> None:
        super(PMF, self).__init__()
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.latent_dim = latent_dim
        self.U = nn.parameter.Parameter(torch.randn((user_dim, latent_dim)))
        self.V = nn.parameter.Parameter(torch.randn((item_dim, latent_dim)))
        self.R = torch.tensor(R, requires_grad = False)
        self.I = (self.R >= 0).to(torch.int32)
        self.I.requires_grad = False
        self.R_test = torch.tensor(R_test, requires_grad = False)
        self.I_test = (self.R_test >= 0).to(torch.int32)
        self.I_test.requires_grad = False
        self.r_std = torch.std(self.R[self.R >= 0])
        self.r_std.requires_grad = False
        self.v_std = nn.parameter.Parameter(torch.tensor([np.random.random()]))
        self.u_std = nn.parameter.Parameter(torch.tensor([np.random.random()]))
        self.losses = []
        self.rmses = []
        self.age = 0
    def to_device(self, device = torch.device('cuda')):
        self.I = self.I.to(device)
        self.R = self.R.to(device)
        self.U = self.U.to(device)
        self.V = self.V.to(device)
        self.R_test = self.R_test.to(device)
        self.I_test = self.I_test.to(device)
        self.r_std = self.r_std.to(device)
        self.u_std = self.u_std.to(device)
        self.v_std = self.v_std.to(device)
    def train(self, epochs, learning_rate = 1e-4, device = torch.device('cuda')):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        self.to_device(device)
        for epoch in range(epochs):
            self.age += torch.sum(self.I)
            rmse = self.rmse()
            self.rmses.append(rmse.item())
            loss = self.compute_loss()
            self.losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.gradient_descent()
            # assert(self.compute_loss() < loss)
    # def gradient_descent(self):
        # self.prev_gradient = self.gradient
        # self.gradient = [self.learning_rate * (np.matmul(self.I * (self.R - np.matmul(self.U, self.V.T)), self.V * -1) / (self.r_std ** 2) + self.U / (self.u_std ** 2)) + self.momentum * self.prev_gradient[0],
        #                  self.learning_rate * (np.matmul((self.I * (self.R - np.matmul(self.U, self.V.T))).T, self.U * -1) / (self.r_std ** 2) + self.V / (self.v_std ** 2)) + self.momentum * self.prev_gradient[1]]
        # self.U = self.U - self.gradient[0]
        # self.V = self.V - self.gradient[1]
    def compute_loss(self) -> torch.Tensor:
        return torch.norm((self.I * (self.R - self.U @ self.V.T)), p = 2) / (2 * self.r_std ** 2) + torch.norm(self.U) / (2 * self.u_std[0] ** 2) + torch.norm(self.V) / (2 * self.v_std[0] ** 2)
    @torch.no_grad()
    def rmse(self):
        return torch.mean(self.I_test * (self.R_test - self.U @ self.V.T) ** 2) ** 0.5