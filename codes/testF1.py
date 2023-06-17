from sklearn.metrics import f1_score, recall_score,precision_score
import torch
import os
import copy
from utils.train_utils import get_data, get_model
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from utils.options import args_parser
import pickle
import numpy as np
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
torch.backends.cudnn.deterministic = True
np.random.seed(2023)
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
def test_img_cur_model(net_g, dataset, args):
    net_g.eval()
    y_predls,y_true = [],[]
    test_loss ,correct = 0,0
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False,**dataloader_kwargs)
    for idx,(data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        y_predls+=[list(i)[0] for i in y_pred.cpu().numpy()]
        y_true+=list(target.cpu().numpy())
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    if len(data_loader.dataset)!=0:
        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    else:
        test_loss = -1
        accuracy = -1
        
    average = 'micro'
    print('f1_score',f1_score(y_true,y_predls,average=average)*100)
    return accuracy, test_loss  
def test_img_local_all(net_local_list, args, dataset_test, dict_users_test):
    correct  = 0
    y_predls,y_true = [],[]
    
    for idx in range(args.num_users):
        net_local = net_local_list[idx]
        net_local.eval()
        data_loader = DataLoader(DatasetSplit(dataset_test, idxs=dict_users_test[idx]), batch_size=16, shuffle=False,**dataloader_kwargs)
        for idx,(data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            log_probs = net_local(data)
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).sum()
            y_true+=list(target.cpu().numpy())
            y_predls+=[list(i)[0] for i in y_pred.cpu().numpy()]
            
    average = 'micro'
    print('acc:',correct/len(dataset_test))        
    print('f1_score',f1_score(y_true,y_predls,average=average)*100)
    return correct, len(dataset_test)
     
args = args_parser()
args.dataset = 'GA'
dataloader_kwargs = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': 4,
    }
args.device = torch.device('cpu')
dataset_train, dataset_test, _,dict_users_train, dict_users_test = get_data(args)
def testF1(dict_save_path,base_dir):
    with open(dict_save_path, 'rb') as handle:
        dict_users_train, dict_users_test = pickle.load(handle)
        net_glob = get_model(args)
        net_glob.train()
        net_local_list = []
        model_save_path = os.path.join(base_dir, 'best_model.pt')
        ditto = False
        if ditto:
            for user_idx in range(args.num_users):
                net_local_list.append(copy.deepcopy(net_glob))
            for user_idx in range(args.num_users):
                model_save_path_local = os.path.join(base_dir, 'best_local_'+str(user_idx)+'.pt')
                net_local_list[user_idx].load_state_dict(torch.load(model_save_path_local,map_location='cpu'), strict=False)
            acc_test, loss_test = test_img_local_all(net_local_list, args, dataset_test, dict_users_test)
            
        else:
            net = copy.deepcopy(net_glob)
            net.load_state_dict(torch.load(model_save_path,map_location='cpu'), strict=True)
            acc_test_cur, _ = test_img_cur_model(net,dataset_test, args)
            print(acc_test_cur)
        
base_dirls = [
            '/home/FedTC/save/GA/resnet18_iidFalse_num30_C0.04_le10_m0.9_wd0.0/shard2_sdr0.0/FedPer/local_upt_full_aggr_body/',
              ]
dict_save_pathls = [i+'dict_users.pkl' for i in base_dirls]
for i in range(len(base_dirls)):
    testF1(dict_save_pathls[i],base_dirls[i])