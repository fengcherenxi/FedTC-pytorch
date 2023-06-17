import copy
import numpy as np
from scipy import stats
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    '''
    params: 总数据集，对应的子索引
    return: client对应的训练或测试数据集
    '''
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
def test_img(net_g, datatest, args, return_probs=False, user_idx=-1):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    probs = []

    for idx,(data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        
        probs.append(log_probs)

        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    if args.verbose:
        if user_idx < 0:
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                test_loss, correct, len(data_loader.dataset), accuracy))
        else:
            print('Local model {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                user_idx, test_loss, correct, len(data_loader.dataset), accuracy))

    return accuracy, test_loss

dataloader_kwargs = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': 4,
    }

def test_img_local(net_g, dataset, args, user_idx=-1, idxs=None):
    net_g.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=8, shuffle=False,**dataloader_kwargs)
    for idx,(data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).sum()
    if len(data_loader.dataset)!=0:
        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    else:
        test_loss = -1
        accuracy = -1
    return accuracy, test_loss

def test_img_cur_model(net_g, dataset, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False,**dataloader_kwargs)
    for idx,(data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).sum()#.long().cpu().sum()
    if len(data_loader.dataset)!=0:
        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
    else:
        test_loss = -1
        accuracy = -1
    return accuracy, test_loss  
def test_img_cur_newmodel(net_g, dataset, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    # data_loader = DataLoader(dataset, batch_size=args.bs)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False,**dataloader_kwargs)
    # print(set([i[1] for i in list(data_loader.dataset)]))
    for idx,(data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs,_ = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    if len(data_loader.dataset)!=0:
        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    else:
        test_loss = -1
        accuracy = -1
    return accuracy, test_loss 
def ood_test_img_local(net_g, dataset, args, user_idx=-1, idxs=None, user_train_targets=None):
    net_g.eval()
    # testing
    per_total = 0
    per_correct = 0
    ood_total = 0
    ood_correct = 0
    # data_loader = DataLoader(dataset, batch_size=args.bs)
    data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.bs, shuffle=False)
    l = len(data_loader)
    
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
            user_train_targets = user_train_targets.to(args.device)
        log_probs = net_g(data)
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        
        # get the index of the max log-probability
        target_dup = torch.cat([target.view(-1, 1)]*len(user_train_targets), dim=1)
        user_train_targets_dup = torch.cat([user_train_targets.view(1, -1)]*len(target), dim=0)
        per_ood = torch.sum(target_dup == user_train_targets_dup, dim=1)
        
        per_idx = torch.where(per_ood == 1)
        ood_idx = torch.where(per_ood == 0)
        
        per_pred = y_pred[per_idx]
        ood_pred = y_pred[ood_idx]
        
        per_target = target[per_idx]
        ood_target = target[ood_idx]
        
        per_total += len(per_target)
        ood_total += len(ood_target)
        
        per_correct += per_pred.eq(per_target.data.view_as(per_pred)).long().cpu().sum()
        ood_correct += ood_pred.eq(ood_target.data.view_as(ood_pred)).long().cpu().sum()

    if args.verbose:
        print('Local model {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            user_idx, test_loss, correct, len(data_loader.dataset), accuracy))
    else:
        return per_correct.item()/per_total*100, ood_correct.item()/ood_total*100
    
def distance_test_img_local(net_g, dataset_train, dataset_test, args, user_idx=-1, train_idxs=None, test_idxs=None):
    net_g.eval()
    
    train_data_loader = DataLoader(DatasetSplit(dataset_train, train_idxs), batch_size=args.bs, shuffle=False)    
    for idx, (data, target) in enumerate(train_data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        feature = net_g.extract_features(data)
        
        if idx == 0:
            features, targets = feature.detach().cpu(), target.detach().cpu()
        else:
            features = torch.cat([features, feature.detach().cpu()])
            targets = torch.cat([targets, target.detach().cpu()])
            
    if args.model == 'cnn':
        template = -99 * torch.ones([10, 256])
    elif args.model == 'mobile':
        template = -99 * torch.ones([100, 1024])
    for i in range(len(template)):
        if i in targets:
            template[i] = torch.mean(features[targets==i], dim=0)
        
    test_data_loader = DataLoader(DatasetSplit(dataset_test, test_idxs), batch_size=args.bs, shuffle=False)    
    for idx, (data, target) in enumerate(test_data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        feature = net_g.extract_features(data)
        
        if idx == 0:
            features, targets = feature.detach().cpu(), target.detach().cpu()
        else:
            features = torch.cat([features, feature.detach().cpu()])
            targets = torch.cat([targets, target.detach().cpu()])
            
    predicted = torch.argmin(torch.cdist(features, template), dim=1)
    return sum(predicted==targets).item()
    
def test_img_local_all(net_local_list, args, dataset_test, dict_users_test, return_all=False):
    # 输入：客户端模型组成的list；测试数据集和测试用户字典
    # 输入测试准确度和loss值
    acc_test_local = np.zeros(args.num_users)
    loss_test_local = np.zeros(args.num_users)
    for idx in range(args.num_users):
        # 取出对应idx的本地模型，并开启测试模式
        net_local = net_local_list[idx]
        net_local.eval()
        
        a, b = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx])
        if a!=-1:
            acc_test_local[idx] = a
            loss_test_local[idx] = b

    data_ratio_local = np.zeros(args.num_users)
    for idx in range(args.num_users):
        idxs = dict_users_test[idx]
        data_ratio_local[idx] = len(DatasetSplit(dataset_test, idxs)) / len(dataset_test)
    
    if return_all:
        return acc_test_local, loss_test_local
#     return acc_test_local.mean(), loss_test_local.mean()
    return (acc_test_local*data_ratio_local).sum(), (loss_test_local*data_ratio_local).sum()

def test_img_avg_all(net_glob, net_local_list, args, dataset_test, return_net=False):
    net_glob_temp = copy.deepcopy(net_glob)
    w_keys_epoch = net_glob.state_dict().keys()
    w_glob_temp = {}
    for idx in range(args.num_users):
        net_local = net_local_list[idx]
        w_local = net_local.state_dict()
        
        if len(w_glob_temp) == 0:
            w_glob_temp = copy.deepcopy(w_local)
        else:
            for k in w_keys_epoch:
                w_glob_temp[k] += w_local[k]

    for k in w_keys_epoch:
        w_glob_temp[k] = torch.div(w_glob_temp[k], args.num_users)
    net_glob_temp.load_state_dict(w_glob_temp)
    acc_test_avg, loss_test_avg = test_img(net_glob_temp, dataset_test, args)

    if return_net:
        return acc_test_avg, loss_test_avg, net_glob_temp
    return acc_test_avg, loss_test_avg

criterion = nn.CrossEntropyLoss()

def test_img_ensemble_all(net_local_list, args, dataset_test):
    probs_all = []
    preds_all = []
    for idx in range(args.num_users):
        net_local = net_local_list[idx]
        net_local.eval()
        # _, _, probs = test_img(net_local, dataset_test, args, return_probs=True, user_idx=idx)
        acc, loss, probs = test_img(net_local, dataset_test, args, return_probs=True, user_idx=idx)
        # print('Local model: {}, loss: {}, acc: {}'.format(idx, loss, acc))
        probs_all.append(probs.detach())

        preds = probs.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
        preds_all.append(preds)

    labels = np.array(dataset_test.targets)
    preds_probs = torch.mean(torch.stack(probs_all), dim=0)

    # ensemble (avg) metrics
    preds_avg = preds_probs.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
    loss_test = criterion(preds_probs, torch.tensor(labels).to(args.device)).item()
    acc_test_avg = (preds_avg == labels).mean() * 100

    # ensemble (maj)
    preds_all = np.array(preds_all).T
    preds_maj = stats.mode(preds_all, axis=1)[0].reshape(-1)
    acc_test_maj = (preds_maj == labels).mean() * 100

    return acc_test_avg, loss_test, acc_test_maj

def test_img_cur(net_local, args, dataset_test, dict_users_test, return_all=False):
    acc_test_local = np.zeros(args.num_users)
    loss_test_local = np.zeros(args.num_users)
    for idx in range(args.num_users):
        net_local = net_local
        net_local.eval()
        a, b = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx])

        acc_test_local[idx] = a
        loss_test_local[idx] = b

    data_ratio_local = np.zeros(args.num_users)
    for idx in range(args.num_users):
        idxs = dict_users_test[idx]
        data_ratio_local[idx] = len(DatasetSplit(dataset_test, idxs)) / len(dataset_test)
    
    if return_all:
        return acc_test_local, loss_test_local
#     return acc_test_local.mean(), loss_test_local.mean()
    return (acc_test_local*data_ratio_local).sum(), (loss_test_local*data_ratio_local).sum()
