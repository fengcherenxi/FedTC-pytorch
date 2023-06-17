import argparse
import torch
def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=32, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=30, help="number of users: K")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=16, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=16, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")
    parser.add_argument('--wd', type=float, default=0.0, help="weight decay (default: 0.0)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--grad_norm', action='store_true', help='use_gradnorm_avging')
    parser.add_argument('--local_ep_pretrain', type=int, default=10, help="the number of pretrain local ep")
    parser.add_argument('--lr_decay', type=float, default=1.0, help="learning rate decay per round")
    parser.add_argument('--fl_alg', type=str, default='FedAvg', help="federated learning algorithm")
    parser.add_argument('--mu', type=float, default=1.0, help="parameter for proximal local SGD")
    parser.add_argument('--gpu', type=int, default=1, help="GPU ID, -1 for CPU")
    parser.add_argument('--dataset', type=str, default='IM', help="name of dataset")
    parser.add_argument('--dirichlet', type=bool, default=True, help="False or True")
    parser.add_argument('--seed', type=int, default=2023, help='random seed (default: 1)')
    
    # model hyperparameters
    parser.add_argument('--Top', type=int, default=1, help="parameter for Top")
    parser.add_argument('--UDL', type=str, default='fc1', help="parameter for weight")

    # model arguments
    parser.add_argument('--model', type=str, default='resnet18', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_layers_keep', type=int, default=1, help='number layers to keep')

    # other arguments
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--print_freq', type=int, default=100, help="print loss frequency during training")
    parser.add_argument('--test_freq', type=int, default=1, help='how often to test on val set')
    parser.add_argument('--load_fed', type=str, default='', help='define pretrained federated model path')
    parser.add_argument('--results_save', type=str, default='/', help='define fed results save folder')
    parser.add_argument('--start_saving', type=int, default=0, help='when to start saving models')
    
    # evaluation arguments
    parser.add_argument('--ft_ep', type=int, default=5, help="the number of epochs for fine-tuning")
    parser.add_argument('--fine_tuning', action='store_true', help='whether fine-tuning before evaluation')
    

    # additional arguments
    parser.add_argument('--local_upt_part', type=str, default='full', help='body, head, or full')
    parser.add_argument('--aggr_part', type=str, default='full', help='body, head, or full')
    parser.add_argument('--unbalanced', action='store_true', help='unbalanced data size')
    parser.add_argument('--num_batch_users', type=int, default=0, help='when unbalanced dataset setting, batch users (same data size)')
    parser.add_argument('--moved_data_size', type=int, default=0, help='when unbalanced dataset setting, moved data size')
    
    parser.add_argument('--server_data_ratio', type=float, default=0.0, help='The percentage of data that servers also have across data of all clients.')
    
    # arguments for a single model
    parser.add_argument('--opt', type=str, default='SGD', help="optimizer")
    parser.add_argument('--body_lr', type=float, default=None, help="learning rate for the body of the model")
    parser.add_argument('--head_lr', type=float, default=None, help="learning rate for the head of the model")
    parser.add_argument('--body_m', type=float, default=None, help="momentum for the body of the model")
    parser.add_argument('--head_m', type=float, default=None, help="momentum for the head of the model")
        
    args = parser.parse_args()
    
    return args
