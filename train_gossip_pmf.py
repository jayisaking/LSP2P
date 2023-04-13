from modules import *
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pickle
import torch
def get_args_parser():
    parser = argparse.ArgumentParser('Train Gossip Learning Based on PMF', add_help = True)
    parser.add_argument('--user_item_pairs_path', default = './book_ratings.pickle', help = 'pickle file contains list of [user id, item id, rating] (this parameter is required)')
    parser.add_argument('--update_epochs', default = 100, help = 'update epochs for each call to PMF train')
    parser.add_argument('--epochs', default = 100, help = 'epoches for gossip learning')
    parser.add_argument('--latent_dim', default = 500, help = 'latent dim for PMF')
    parser.add_argument('--lr', default = 1e-4, help = 'learning rate for PMF')
    parser.add_argument('--probs_for_send', default = 0.5, help = 'the probability to select and send model in gossip learning')
    parser.add_argument('--node_number', default = 10, help = 'the number of nodes')
    parser.add_argument('--device', default = 'cuda')
    parser.add_argument('--train_set_size', default = 0.8)
    args = parser.parse_args()
    return args
def main(args):
    with open(args.user_item_pairs_path, 'rb') as f:
        pairs = pickle.load(f)
    device = torch.device(args.device)
    gl = GossipLearning(user_item_pairs = pairs, latent_dim = args.latent_dim, update_epochs = args.update_epochs,
                        lr = args.lr, probs_for_send = args.probs_for_send, node_number = args.node_number, device = device, train_set_size = args.train_set_size)
    gl.train(epoches = args.epochs)

    plt.plot(np.arange(1, len(gl.transmission) + 1), gl.transmission)
    plt.xlabel("Epoch")
    plt.ylabel("Bytes Transmitted")
    plt.savefig('trans.png')
    plt.clf()

    plt.plot(np.arange(1, len(gl.rmses) + 1), gl.rmses)
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.savefig('rmses.png')
    plt.clf()
if __name__ == '__main__':
    main(get_args_parser())