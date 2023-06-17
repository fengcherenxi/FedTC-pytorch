import copy
import pickle
import numpy as np
import torch

from utils.options import args_parser
from utils.train_utils import get_data,get_model
from models.Update import LocalUpdateDitto
from models.test import test_img_cur_model
from models.Fed import FedAvg
import os

if __name__ == '__main__':
    # parse args
    print('Ditto-Model')
    args = args_parser()
    print('dirichlet:',args.dirichlet)
    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if args.unbalanced:
        if args.partition == 'shard':
            base_dir = './save/{}/{}_iid{}_num{}_C{}_le{}/shard{}_unbalanced_bu{}_md{}/{}/'.format(
                args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.shard_per_user, args.num_batch_users, args.moved_data_size, args.results_save)
        elif args.partition == "dirichlet":
            base_dir = './save/{}/{}_iid{}_num{}_C{}_le{}/dir{}/{}/'.format(
                args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.dd_beta, args.results_save)
    else:
        base_dir = './save/{}/{}_iid{}_num{}_C{}_le{}/shard{}/{}/'.format(
            args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.shard_per_user, args.results_save)
    algo_dir = "ditto"
    
    if not os.path.exists(os.path.join(base_dir, algo_dir)):
        os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)

    dataset_train, dataset_test, dataset_val, dict_users_train, dict_users_test = get_data(args)

    dict_save_path = os.path.join(base_dir, algo_dir, 'dict_users.pkl')
    with open(dict_save_path, 'wb') as handle:
        pickle.dump((dict_users_train, dict_users_test), handle)

    # build a global model
    net_glob = get_model(args)
    net_glob.train()

    # build local models
    net_local_list = []
    for user_idx in range(args.num_users):
        net_local_list.append(copy.deepcopy(net_glob))
    
    # training
    results_save_path = os.path.join(base_dir, algo_dir, 'results.csv')

    loss_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    results = []
    w_glob = copy.deepcopy(net_glob.state_dict())
    lam = 0.75
    
    for iter in range(args.epochs):
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = list(np.random.choice(range(args.num_users), m, replace=False))
        
        w_locals = []
        for idx in idxs_users:
            print(idx)
            local = LocalUpdateDitto(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            
            net_global = copy.deepcopy(net_glob)
            
            _ = local.train(net=net_global.to(args.device), idx=idx, lr=args.lr)
            
            w_glob_k = copy.deepcopy(net_global.state_dict()) 
            
            net_local = net_local_list[idx]
            
            w, loss = local.train(net=net_local.to(args.device), idx=idx, lr=args.lr, w_ditto=w_glob_k, lam=lam)
            
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            
        # update global weights
        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)
        
        if (iter + 1) in [args.epochs//2, (args.epochs*3)//4]:
            lr *= 0.1
            
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        if (iter + 1) % args.test_freq == 0:
            # acc_test, loss_test = test_img_local_all(net_local_list, args, dataset_test, dict_users_test, return_all=False)
            acc_test, loss_test = test_img_cur_model(net_local_list[0],dataset_test,args)
            
            print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))

            if best_acc is None or acc_test > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = acc_test
                best_epoch = iter
                
                for user_idx in range(args.num_users):
                    best_save_path = os.path.join(base_dir, algo_dir, 'best_local_{}.pt'.format(user_idx))
                    torch.save(net_local_list[user_idx].state_dict(), best_save_path)
        # rollback global model
        for user_idx in range(args.num_users):
            net_local_list[user_idx].load_state_dict(w_glob, strict=False)
    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))