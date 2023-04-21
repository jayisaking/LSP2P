
from torch import nn
import torch
from tqdm import tqdm
import random
import numpy as np
import sys

class GossipLearning(nn.Module):
    def __init__(self, user_item_pairs: list, latent_dim = 500, update_epochs = 100, lr = 1e-4, probs_for_send = 0.5, node_number = 10, device = torch.device('cuda'), 
                train_set_size = 0.8, momentum = 0, weight_decay = 0) -> None:
        # `super(GossipLearning, self).__init__()` is calling the constructor of the parent class
        # `nn.Module` to initialize the `GossipLearning` object. This is necessary because
        # `GossipLearning` is a subclass of `nn.Module` and needs to inherit its properties and
        # methods. The rest of the code block initializes various attributes of the `GossipLearning`
        # object, such as the user-item pairs, the number of nodes, the latent dimension, the update
        # epochs, the learning rate, and the device. It also creates a list of `Node` objects and
        # assigns them to the `nodes` attribute of the `GossipLearning` object. Finally, it sets the
        # `nodes` attribute of each `Node` object to the list of all `Node` objects in the network.
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
                                    R_test = self.R_test, nodes = [], update_epochs = self.update_epochs, lr = self.lr, probs_for_send = self.probs_for_send, device = device,
                                    momentum = momentum, weight_decay = weight_decay).to(device))
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
                # This code block is the main training loop for the Gossip Learning algorithm.
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
    def __init__(self, id, latent_dim, user_dim, item_dim, R: np.array, R_test: np.array, nodes, update_epochs = 100, lr = 1e-4, probs_for_send = 0.5, device = torch.device('cuda'), momentum = 0, weight_decay = 0) -> None:
        super(Node, self).__init__()
        self.age = 0
        self.id = id
        self.model = PMF(latent_dim, user_dim, item_dim, R, R_test, momentum = momentum, weight_decay = weight_decay)
        self.model.to(device)
        self.nodes = nodes # neighbor node
        self.update_epochs = update_epochs
        self.lr = lr
        self.probs_send = probs_for_send
        self.device = device
        self.model.train(epochs = self.update_epochs, learning_rate = self.lr, device = device)
    def send_model(self):
        # This code block is responsible for sending the model parameters (U, V, age) to neighboring
        # nodes in the Gossip Learning algorithm. It loops through all the nodes in the network, and
        # for each node, it checks if a random number generated from a uniform distribution is greater
        # than the probability of sending (probs_send) and if the node is not the current node. If the
        # condition is satisfied, it calls the receive_model method of the neighboring node, passing
        # the current node's model parameters (U, V, age) as arguments. It also calculates the
        # transmission quantity by adding the size of the U, V, and age tensors using the
        # sys.getsizeof() function. Finally, it returns the total transmission quantity.
        transmission_quantity = 0
        for node in self.nodes:
            if np.random.random() > self.probs_send and node.id != self.id:
                node.receive_model(self.model.U, self.model.V, self.model.age)
                transmission_quantity += (sys.getsizeof(self.model.U) + sys.getsizeof(self.model.V) + sys.getsizeof(self.model.age))
        return transmission_quantity
    def update(self):
        self.model.train(epochs = self.update_epochs, learning_rate = self.lr)
    def receive_model(self, U, V, age):
        # `self.merge(U, V, age)` is a method in the `Node` class that merges the model parameters of
        # the current node with the model parameters received from a neighboring node. It calculates
        # the proportion of the age of the received model parameters to the sum of the ages of the
        # received and current model parameters, and uses this proportion to update the current node's
        # model parameters.
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
    def __init__(self, latent_dim, user_dim, item_dim, R: np.array, R_test: np.array, momentum, weight_decay) -> None:
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
        self.momentum = momentum
        self.weight_decay = weight_decay
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
       # This is the training method for the PMF (Probabilistic Matrix Factorization) model. It takes
       # in the number of epochs to train for (`epochs`), the learning rate (`learning_rate`), and the
       # device to use (`device`).
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.SGD(self.parameters(), lr = learning_rate, momentum = self.momentum, weight_decay = self.weight_decay)
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
    def compute_loss(self) -> torch.Tensor:
        return torch.norm((self.I * (self.R - self.U @ self.V.T)), p = 2) / (2 * self.r_std ** 2) + torch.norm(self.U) / (2 * self.u_std[0] ** 2) + torch.norm(self.V) / (2 * self.v_std[0] ** 2)
    @torch.no_grad()
    def rmse(self):
        return torch.mean(self.I_test * (self.R_test - self.U @ self.V.T) ** 2) ** 0.5