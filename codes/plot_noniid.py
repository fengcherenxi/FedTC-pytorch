import torch
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(42)
from utils.ourdatasets import GA_train_targets,IM_train_targets,Covid_targets
np.random.seed(42)
def dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels==y).flatten() for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    return client_idcs
if __name__ == "__main__":
    N_CLIENTS = 30
    DIRICHLET_ALPHA = 100

    train_labels = Covid_targets
    print('num_train_data:',len(train_labels))
    train_labels = np.array(train_labels)
    client_idcs = dirichlet_split_noniid(train_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)

    plt.figure(figsize=(20,10))
    plt.hist([train_labels[idc]for idc in client_idcs], stacked=True, 
            bins=np.arange(min(train_labels)-0.5, max(train_labels) + 1.5, 1),
            label=["Client {}".format(i) for i in range(N_CLIENTS)], rwidth=0.5)
    # plt.xticks(np.arange(10), ['0','1','2','3','4','5','6','7','8','9'])
    plt.xticks(np.arange(4), ['0','1','2','3'])
    
    plt.legend(loc = 'upper left')
    plt.savefig('Covidnoniid_100.pdf')