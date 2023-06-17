import random
import numpy as np
import torch

def iid(dataset, num_users, server_data_ratio):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    
    if server_data_ratio > 0.0:
        dict_users['server'] = set(np.random.choice(all_idxs, int(len(dataset)*server_data_ratio), replace=False))
    
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    
    return dict_users
def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    return: dict & distribution
    '''
    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    
    class_dict = [np.argwhere(train_labels==y).flatten() for y in range(n_classes)]
    client_dict = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_dict, label_distribution):
        for i, item_index in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_dict[i] += [item_index]
    client_list = [np.concatenate(item) for item in client_dict]
    client_dict = {}
    for i in range(len(client_list)):
        client_dict[i] = client_list[i]
    return client_dict, np.transpose(label_distribution)
def dirichlet_noniid(num_users,targets,alpha=0.1):
    print('dirichlet_noniid-数据生成中......')
    train_labels = np.array(targets)
    client_dict, label_distribution = dirichlet_split_noniid(train_labels, alpha=alpha, n_clients=num_users)
    return client_dict,label_distribution

def noniid_our(dataset, num_users, shard_per_user, server_data_ratio,targets ,rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users, all_idxs = {i: np.array([], dtype='int64') for i in range(num_users)}, [i for i in range(len(dataset))]
    idxs_dict = {}
    for i in range(len(targets)):
        label = targets[i]
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)
    num_classes = len(idxs_dict)
    shard_per_class = int(shard_per_user * num_users / num_classes)
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        # 求余数
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(targets)[value])
        assert(len(x)) <= shard_per_user
        test.append(value)

    test = np.concatenate(test)
    print('test:',len(test))
    assert(len(test) == len(dataset))
    assert(len(set(list(test))) == len(dataset))

    if server_data_ratio > 0.0:
        dict_users['server'] = set(np.random.choice(all_idxs, int(len(dataset)*server_data_ratio), replace=False))
    return dict_users, rand_set_all