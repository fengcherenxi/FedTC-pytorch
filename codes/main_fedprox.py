import copy
import pickle
import numpy as np
import pandas as pd
import torch
import os

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdateFedProx
from models.test import test_img_cur_model
from models.Fed import FedAvg

if __name__ == '__main__':
    args = args_parser()
    print('FedProx-Model')
    print('GPU:',args.gpu)
    print('local_upt_part:',args.local_upt_part)
    print('aggr_part:',args.aggr_part)
    print('dirichlet',args.dirichlet)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    assert args.local_upt_part in ['body', 'head', 'full'] and args.aggr_part in ['body', 'head', 'full']
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if args.unbalanced:
        base_dir = './save/{}/{}_iid{}_num{}_C{}_le{}_m{}_wd{}/shard{}_sdr{}_unbalanced_bu{}_md{}/{}/'.format(
            args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.momentum, args.wd, args.shard_per_user, args.server_data_ratio, args.num_batch_users, args.moved_data_size, args.results_save)
    else:
        base_dir = './save/{}/{}_iid{}_num{}_C{}_le{}_m{}_wd{}/shard{}_sdr{}/{}/'.format(
            args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.momentum, args.wd, args.shard_per_user, args.server_data_ratio, args.results_save)
    algo_dir = 'local_upt_{}_aggr_{}'.format(args.local_upt_part, args.aggr_part)
    if not os.path.exists(os.path.join(base_dir, algo_dir)):
        os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)
    dataset_train, dataset_test, dataset_val, dict_users_train, dict_users_test = get_data(args)
    dict_users_train = {int(k): np.array(v, dtype=int) for k, v in dict_users_train.items()}
    dict_users_test = {int(k): np.array(v, dtype=int) for k, v in dict_users_test.items()}
    dict_save_path = os.path.join(base_dir, algo_dir, 'dict_users.pkl')
    with open(dict_save_path, 'wb') as handle:
        pickle.dump((dict_users_train, dict_users_test), handle)
    net_glob = get_model(args)
    net_glob.train()
    net_local_list = []
    for user_idx in range(args.num_users):
        net_local_list.append(copy.deepcopy(net_glob))
    results_save_path = os.path.join(base_dir, algo_dir, 'results.csv')
    loss_train = []
    best_loss = None
    best_acc = None
    best_epoch = None
    lr = args.lr
    results = []
    for iter in range(args.epochs):
        w_glob = None
        loss_locals,w_locals = [],[]
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print("Round {}, lr: {:.6f}, {}".format(iter, lr, idxs_users))
        for idx in idxs_users:
            local = LocalUpdateFedProx(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            net_local = copy.deepcopy(net_local_list[idx])
            if args.local_upt_part == 'body':
                w_local, loss = local.train(net=net_local.to(args.device), body_lr=lr, head_lr=0.)
            if args.local_upt_part == 'head':
                w_local, loss = local.train(net=net_local.to(args.device), body_lr=0., head_lr=lr)
            if args.local_upt_part == 'full':
                w_local, loss = local.train(net=net_local.to(args.device), body_lr=lr, head_lr=lr)
            
            loss_locals.append(copy.deepcopy(loss))
            w_locals.append(copy.deepcopy(w_local))
        # Aggregation
        w_glob = FedAvg(w_locals)
        
        # Broadcast
        update_keys = list(w_glob.keys())
        if args.aggr_part == 'body':
            if args.server_data_ratio > 0.0:
                pass
            else:
                update_keys = [k for k in update_keys if 'fc' not in k]
        elif args.aggr_part == 'head':
            update_keys = [k for k in update_keys if 'fc' in k]
        elif args.aggr_part == 'full':
            pass
        w_glob = {k: v for k, v in w_glob.items() if k in update_keys}
        for user_idx in range(args.num_users):
            net_local_list[user_idx].load_state_dict(w_glob, strict=False)
        
        if (iter + 1) in [args.epochs//2, (args.epochs*3)//4]:
            lr *= 0.1
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        if (iter + 1) % args.test_freq == 0:
            acc_test, loss_test = test_img_cur_model(net_local_list[0],dataset_test,args)
                       
            print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))
            if best_acc is None or acc_test > best_acc:
                best_acc = acc_test
                best_epoch = iter
                best_save_path = os.path.join(base_dir, algo_dir, 'best_model.pt')
                torch.save(net_local_list[0].state_dict(), best_save_path)
            results.append(np.array([iter, loss_avg, loss_test, acc_test, best_acc]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc'])
            final_results.to_csv(results_save_path, index=False)
    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))