from modules import *
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pickle
import torch
import random

def get_args_parser():

    # In this code, `argparse.ArgumentParser` is used to create an argument parser object. The
    # `add_argument` method is then used to specify the arguments that the script can accept. Each
    # argument is defined with a name, a default value, and a help string that describes what the
    # argument does. When the script is run, the argument parser reads the command-line arguments and
    # converts them into Python objects that can be used in the script.
    parser = argparse.ArgumentParser('Train Gossip Learning Based on PMF', add_help = True)
    parser.add_argument('--user_item_pairs_path', default = './book_ratings.pickle', help = 'pickle file containing list of [user id, item id, rating] (this parameter is required)')
    parser.add_argument('--used_data_size', default = 1, help = 'the proportion of pairs you want to use', type = float)
    parser.add_argument('--update_epochs', default = 100, help = 'update epochs for each call to PMF train', type = int)
    parser.add_argument('--epochs', default = 100, help = 'epochs for gossip learning', type = int)
    parser.add_argument('--latent_dim', default = 500, help = 'latent dim for PMF', type = int)
    parser.add_argument('--lr', default = 1e-4, help = 'learning rate for PMF', type = float)
    parser.add_argument('--probs_for_send', default = 0.5, help = 'the probability to select and send model in gossip learning', type = float)
    parser.add_argument('--node_number', default = 10, help = 'the number of nodes', type = int)
    parser.add_argument('--device', default = 'cuda')
    parser.add_argument('--train_set_size', default = 0.8, type = float)
    parser.add_argument('--transmission_graph_file', default = 'lsp2p_trans.png')
    parser.add_argument('--rmse_graph_file', default = 'lsp2p_rmses.png', type = str)
    parser.add_argument('--momentum', default = 0, help = "momentum for SGD", type = float)
    parser.add_argument('--weight_decay', default = 0, help = "weight decay (l2 penalty) for SGD", type = float)
    parser.add_argument('--cluster_number', default = 2, help = "number of clusters", type = int)
    parser.add_argument('--log_file', default = "lsp2p_log.csv", type = str)
    parser.add_argument('--evaluate_every', default = 5, help = "evaluate every n seconds", type = int)
    args = parser.parse_args()
    return args
def main(args):
    # This code block is loading a pickled file containing a list of user-item pairs and their
    # corresponding ratings. The `with open(args.user_item_pairs_path, 'rb') as f:` statement opens
    # the file in binary read mode and assigns it to the variable `f`. The `pickle.load(f)` method is
    # then used to load the contents of the file into the variable `pairs`.
    with open(args.user_item_pairs_path, 'rb') as f:
        pairs = pickle.load(f)
    random.shuffle(pairs)
    pairs = pairs[:int(len(pairs) * args.used_data_size)]
    device = torch.device(args.device)
    lsp2p = LSP2P(user_item_pairs = pairs, latent_dim = args.latent_dim, update_epochs = args.update_epochs, lr = args.lr, 
                  probs_for_send = args.probs_for_send, node_number = args.node_number, device = device, train_set_size = args.train_set_size, 
                  momentum = args.momentum, weight_decay = args.weight_decay, cluster_number = args.cluster_number)
    lsp2p.train(epochs = args.epochs, evaluate_every = args.evaluate_every)

    plt.plot(np.arange(0, len(lsp2p.regular_transmission) * args.evaluate_every, args.evaluate_every), np.cumsum(lsp2p.regular_transmission))
    plt.xlabel("Seconds")
    plt.ylabel("Bytes Transmitted")
    plt.savefig(args.transmission_graph_file)
    plt.clf()

    plt.plot(np.arange(0, len(lsp2p.regular_rmses) * args.evaluate_every, args.evaluate_every), lsp2p.regular_rmses)
    plt.xlabel("Seconds")
    plt.ylabel("RMSE")
    plt.savefig(args.rmse_graph_file)
    plt.clf()
    
    # Write rmses and transmission to log file
    log = pd.DataFrame({"rmses": lsp2p.regular_rmses, "transmission": lsp2p.regular_transmission})
    log.to_csv(args.log_file, index = False)
if __name__ == '__main__':
    main(get_args_parser())