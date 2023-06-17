import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import torch.nn.functional as F



class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

dataloader_kwargs = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': 4,
    }    
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, 
                                    shuffle=True if len(idxs)!=0 else False,
                                    **dataloader_kwargs)
        self.pretrain = pretrain

    def train(self, net, body_lr, head_lr, idx=-1, local_eps=None):
        net.train()
        body_params = [p for name, p in net.named_parameters() if 'fc' not in name]
        head_params = [p for name, p in net.named_parameters() if 'fc' in name]
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)
        epoch_loss = []
        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)
                loss = self.loss_func(logits, labels)
                
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            if len(batch_loss)!=0:
                print('local epoch:',sum(batch_loss)/len(batch_loss))
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
        if len(epoch_loss) == 0:
            return net.state_dict(), 0
        else:
            return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
def softmax_kl_loss(input_logits, target_logits):
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, reduction='batchmean')
class LocalUpdateFedTKF(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, 
                                    shuffle=True if len(idxs)!=0 else False,
                                    **dataloader_kwargs)
        self.pretrain = pretrain

    def train(self, net, body_lr, head_lr, w_helper_list, cur_epoch, idx=-1, local_eps=None):
        net.train()
        body_params = [p for name, p in net.named_parameters() if self.args.UDL not in name ]
        head_params = [p for name, p in net.named_parameters() if self.args.UDL in name]
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)
        epoch_loss = []
        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)
                loss = self.loss_func(logits, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            if len(batch_loss)!=0:
                print('local epoch:',sum(batch_loss)/len(batch_loss))
                epoch_loss.append(sum(batch_loss)/len(batch_loss))    
        if len(epoch_loss) == 0:
            return net.state_dict(), 0
        else:
            return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
class LocalUpdateFedProx(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, 
                                    shuffle=True if len(idxs)!=0 else False,
                                    **dataloader_kwargs)
                                    
        self.pretrain = pretrain

    def train(self, net, body_lr, head_lr):
        net.train()
        g_net = copy.deepcopy(net)
        
        body_params = [p for name, p in net.named_parameters() if 'fc' not in name]
        head_params = [p for name, p in net.named_parameters() if 'fc' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)

                loss = self.loss_func(logits, labels)
                
                # for fedprox
                fed_prox_reg = 0.0
                for l_param, g_param in zip(net.parameters(), g_net.parameters()):
                    fed_prox_reg += (self.args.mu / 2 * torch.norm((l_param - g_param)) ** 2)
                loss += fed_prox_reg
                
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            if len(batch_loss)!=0:
                print('local epoch:',sum(batch_loss)/len(batch_loss))
                epoch_loss.append(sum(batch_loss)/len(batch_loss))    
        if len(epoch_loss) == 0:
            return net.state_dict(), 0
        else:
            return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
class LocalUpdateFedBN(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, 
                                    shuffle=True if len(idxs)!=0 else False,
                                    **dataloader_kwargs)
        self.pretrain = pretrain

    def train(self, net, body_lr, head_lr, idx=-1, local_eps=None):
        net.train()
        body_params = [p for name, p in net.named_parameters() if 'bn' not in name]
        head_params = [p for name, p in net.named_parameters() if 'bn' in name]
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)
        epoch_loss = []
        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)
                loss = self.loss_func(logits, labels)
                
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            if len(batch_loss)!=0:
                print('local epoch:',sum(batch_loss)/len(batch_loss))
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
        if len(epoch_loss) == 0:
            return net.state_dict(), 0
        else:
            return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
   
class LocalUpdateDitto(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
            
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, 
                                    shuffle=True if len(idxs)!=0 else False,
                                    **dataloader_kwargs)
                                    

    def train(self, net, w_ditto=None, lam=0, idx=-1, lr=0.1, last=False, momentum=0.9):
        net.train()
        # train and update
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
                
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

        local_eps = self.args.local_ep
        args = self.args 
        epoch_loss=[]
        num_updates = 0
        
        for iter in range(local_eps):
            done=False
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                w_0 = copy.deepcopy(net.state_dict())
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if w_ditto is not None:
                    w_net = copy.deepcopy(net.state_dict())
                    for key in w_net.keys():
                        w_net[key] = w_net[key] - args.lr*lam*(w_0[key] - w_ditto[key])
                    net.load_state_dict(w_net)
                    optimizer.zero_grad()
                
                num_updates += 1
                batch_loss.append(loss.item())
            if len(batch_loss)!=0:
                print('local epoch:',sum(batch_loss)/len(batch_loss))
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
        if len(epoch_loss) == 0:
            return net.state_dict(), 0
        else:
            return net.state_dict(), sum(epoch_loss) / len(epoch_loss)