from torchvision import datasets, transforms
from models.ResNet import ResNet18
from utils.sampling import noniid_our,dirichlet_noniid
from utils.ourdatasets import IM_image_datasets,GA_image_datasets,train_covid,test_covid,val_covid
from utils.ourdatasets import IM_train_targets,IM_test_targets,GA_train_targets,GA_test_targets
from utils.options import args_parser
args = args_parser()
dataset = args.dataset
dirichlet = args.dirichlet

if dataset=='IM':
    targets_train = IM_train_targets
    targets_test = IM_test_targets
    dataset_train_our = IM_image_datasets['train']
    dataset_test_our = IM_image_datasets['test']
    dataset_val_our = IM_image_datasets['val']

    print('train:',len(IM_image_datasets['train']))
    print('test:',len(IM_image_datasets['test']))
    print('val:',len(IM_image_datasets['val']))
elif dataset=='GA':
    targets_train = GA_train_targets
    targets_test = GA_test_targets
    dataset_train_our = GA_image_datasets['train']
    dataset_test_our = GA_image_datasets['test']
    dataset_val_our = GA_image_datasets['val']

    print('train:',len(GA_image_datasets['train']))
    print('test:',len(GA_image_datasets['test']))
    print('val:',len(GA_image_datasets['val']))
elif dataset=='covid':
    targets_train = [i[1] for i in train_covid]
    targets_test = [i[1] for i in test_covid]
    print('train:',len(train_covid))
    print('test:',len(test_covid))
    print('val:',len(val_covid))
    
else:
    exit('Error: unrecognized dataset')


trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])

def get_data(args, env='fed'):
    if env == 'fed':
        if args.dataset == 'IM':      
            dataset_train =  dataset_train_our
            dataset_test =  dataset_test_our
            dataset_val =  dataset_val_our
            if dirichlet:
                dict_users_train, _ = dirichlet_noniid(args.num_users,targets_train,0.1)
                dict_users_test, _ = dirichlet_noniid(args.num_users,targets_test,0.1)
            else:
                dict_users_train, rand_set_all = noniid_our(dataset_train, args.num_users, args.shard_per_user, args.server_data_ratio,targets=targets_train)
                dict_users_test, rand_set_all = noniid_our(dataset_test, args.num_users, args.shard_per_user, args.server_data_ratio,targets=targets_test, rand_set_all=rand_set_all)
        elif args.dataset == 'GA':      
            dataset_train =  dataset_train_our
            dataset_test =  dataset_test_our
            dataset_val =  dataset_val_our
            if dirichlet:
                dict_users_train, _ = dirichlet_noniid(args.num_users,targets_train,0.1)
                dict_users_test, _ = dirichlet_noniid(args.num_users,targets_test,0.1)
            else:
                dict_users_train, rand_set_all = noniid_our(dataset_train, args.num_users, args.shard_per_user, args.server_data_ratio,targets=targets_train)
                dict_users_test, rand_set_all = noniid_our(dataset_test, args.num_users, args.shard_per_user, args.server_data_ratio,targets=targets_test, rand_set_all=rand_set_all)
        elif args.dataset == 'covid':
            dataset_train = train_covid
            dataset_test = test_covid
            dataset_val = val_covid
            if dirichlet:
                dict_users_train, _ = dirichlet_noniid(args.num_users,targets_train,0.1)
                dict_users_test, _ = dirichlet_noniid(args.num_users,targets_test,0.1)
            else:
                dict_users_train, rand_set_all = noniid_our(dataset_train, args.num_users, args.shard_per_user, args.server_data_ratio,targets=targets_train)
                dict_users_test, rand_set_all = noniid_our(dataset_test, args.num_users, args.shard_per_user, args.server_data_ratio,targets=targets_test, rand_set_all=rand_set_all)
        else:
            exit('Error: unrecognized dataset')

    return dataset_train, dataset_test, dataset_val,dict_users_train, dict_users_test
def get_model(args):
    if args.model == 'resnet18' and args.dataset in ['cifar10', 'cifar100','IM','GA','covid']:
        net_glob = ResNet18().to(args.device)
    else:
        exit('Error: unrecognized model')
    return net_glob