import dgl
import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.superpixels_graph_classification.load_net import gnn_model 
from data.data import LoadData # import dataset
from train.train_superpixels_graph_classification import train_epoch

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, args):

    DATASET_NAME = dataset.name
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()
            
    trainset = dataset.train
    valset = dataset.val
    testset = dataset.test

    device = net_params['device']
    # setting seeds
    random.seed(params['seed'])#设置随机生成器的种子；传入的数值用于指定随机数生成时所用算法开始时所选定的整数值，如果使用相同的seed()值，则每次生成的随机数都相同；如果不设置这个值，则系统会根据时间来自己选择这个值，此时每次生成的随机数会因时间的差异而有所不同
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])#设置CPU生成随机数的种子

    if device == 'cuda':
        torch.cuda.manual_seed(params['seed']) #设置GPU生成随机数的种子

    output_path = './001_contrastive_models'
    save_model_dir0 = os.path.join(output_path, DATASET_NAME) #用于路径拼接文件路径
    save_model_dir1 = os.path.join(save_model_dir0, args.aug)
    if args.head:
        save_model_dir1 += "_head"
    else:
        save_model_dir1 += "_no_head"
    save_model_dir2 = os.path.join(save_model_dir1, MODEL_NAME)

    
    print('-'*40 + "Training Option" + '-'*40)
    print("Data  Name:     [{}]".format(DATASET_NAME))
    print("Model Name:     [{}]".format(MODEL_NAME))
    print("Training Graphs:[{}]".format(len(trainset)))
    print("Batch Size:     [{}]".format(net_params['batch_size']))
    print("Learning Rate:  [{}]".format(params['init_lr']))
    print("Epoch To Train: [{}]".format(args.epochs))
    print("Model Save Dir: [{}]".format(save_model_dir2))
    print('-'*40 + "Contrastive Option" + '-'*40)
    print("Aug Type:       [{}]".format(args.aug))
    print("Projection head:[{}]".format(args.head))
    print("Drop Proportion:[{}]".format(args.drop_percent))
    print("Temperature:    [{}]".format(args.temp))
    print('-'*100)
    
    model = gnn_model(MODEL_NAME, net_params)
    start_epoch = 0
    # 一个模型(torch.nn.Module)的可学习参数(也就是权重weight和偏置值bias)是包含在模型参数(model.parameters())中的
    # 模型的状态字典state_dict（本质上是一个字典），其键值对是每个网络层和其对应的参数张量，状态字典只包含带有可学习参数的网络层（比如卷积层、全连接层等）
    # 优化器也是有一个状态字典的，包含的优化器状态的信息以及使用的超参数
    # 保存状态字典  一般用.pth文件后缀保存模型  并没有保存图结构，只是保存了模型参数
    # torch.save(model.state_dict(), PATH)

    # 加载状态字典  state_dict是不带模型结构的模型参数，加载的时候要先初始化模型结构
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH)) load_state_dict() 方法必须传入一个字典对象，而不是对象的保存路径，也就是说必须先反序列化字典对象（例子中torch.load()），然后再调用该方法
    # model.eval()
    # 一般用以上来预测模型，如果想恢复训练，只有模型的状态字典是不够的

    # 保存整个模型   这里.pth将包含了整个模型，不仅有参数，还有模型结构
    # torch.save(model, PATH)

    # 加载整个模型  注意，这个模型类必须得在某个地方被定义过
    # model = torch.load(PATH)
    # model.eval()
    #
    # 保存一个通用的检查点(Checkpoint)  想继续训练，只有模型状态字典不够，还应有其他额外信息，比如优化器的状态字典（它包含了用于模型训练时需要更新的参数和缓存信息）、训练中断的批次、最后一次训练的loss损失等等
    # torch.save({
    #    'epoch': epoch,
    #    'model_state_dict': model.state_dict(),
    #    'optimizer_state_dict': optimizer.state_dict(),
    #    'loss': loss,
    #    ...
    #    }, PATH)
    # 保存这么多种信息，通过用一个字典来进行组织，然后继续调用 torch.save 方法，一般保存的文件后缀名是 .tar

    # 加载一个通用的检查点(Checkpoint)
    # model = TheModelClass(*args, **kwargs)
    # optimizer = TheOptimizerClass(*args, **kwargs)
    # checkpoint = torch.load(PATH)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    # model.eval() 加载完后，根据后续步骤，调用 model.eval() 用于预测，model.train() 用于恢复训练
    # - or -
    # model.train()
    # model.eval() 方法来将 dropout 和 batch normalization 层设置为验证模型。
    # model.train()的作用是启用 Batch Normalization 和 Dropout
    if args.resume:
        print("Resume ...")
        load_file_name = glob.glob(save_model_dir2 + '/*.pkl')[-1]  #glob.glob()返回所有匹配的文件路径列表
        epoch_nb = load_file_name.split('_')[-1]
        start_epoch = int(epoch_nb.split('.')[0]) + 1
        print("Success Resume At Epoch  : [{}]".format(start_epoch))
        checkpoint = torch.load(load_file_name)
        model_dict = model.state_dict()
        state_dict = {k:v for k,v in checkpoint.items() if k in model_dict.keys()}
        model.load_state_dict(state_dict)    
        print('Success load Resume Model: [{}]'.format(load_file_name))
        print('-'*100)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',  #调整学习率
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    # batching exception for Diffpool
    drop_last = True if MODEL_NAME == 'DiffPool' else False

    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)#DataLoader处理模型输入数据的一个工具类
    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)
    
    run_time = 0
    # batchsize批大小，就是每次调整参数前所选取的样本（称为mini-batch或batch）数量：
    # 训练中还有一个概念epoch，每学一遍数据集就称为1个epoch。比如数据集中有1000个样本，批大小为10，则将全部样本训练一遍后（也就是1个epoch），网络会调整1000/10=100次，但这并不意味着模型训练好了，接下来还要训练第2 3 4个epoch
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        
        epoch_train_loss, optimizer = train_epoch(model, optimizer, device, train_loader, epoch, 
                                                    drop_percent=args.drop_percent, 
                                                    temp=args.temp,
                                                    aug_type=args.aug,
                                                    head=args.head)
   
        epoch_time = time.time() - t0
        run_time += epoch_time
        
        scheduler.step(epoch_train_loss)  #更新学习率
        print('-'*120)
        print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
                'Epoch [{:>2d}]: Loss [{:.4f}]  Epoch Time: [{:.2f} min]   Run Total Time: [{:.2f} min]'
                .format(epoch + 1, epoch_train_loss,  epoch_time / 60, run_time / 60))
        print('-'*120)
        
        '''
        './001_contrastive_models/DATASET_NAME/nn/MODEL_NAME/*.pkl'
        '''
        if not args.debug:
            output_path = './001_contrastive_models'
            save_model_dir0 = os.path.join(output_path, DATASET_NAME)
            save_model_dir1 = os.path.join(save_model_dir0, args.aug)
            if args.head:
                save_model_dir1 += "_head"
            else:
                save_model_dir1 += "_no_head"
            save_model_dir2 = os.path.join(save_model_dir1, MODEL_NAME)
           
            if not os.path.exists(save_model_dir2):
                os.makedirs(save_model_dir2)

            save_ckpt_path = '{}.pkl'.format(save_model_dir2 + "/" + "epoch_" + str(epoch))
            torch.save(model.state_dict(),  save_ckpt_path)

            files = glob.glob(save_model_dir2  + '/*.pkl')
            for file in files:
                epoch_nb = file.split('_')[-1]
                epoch_nb = int(epoch_nb.split('.')[0])
                if epoch_nb < epoch-1:
                    os.remove(file)  #删除文件或一个空目录

    
def main():    
    
    config_path = ['configs/superpixels_graph_classification_GCN_MNIST.json',
                   'configs/superpixels_graph_classification_GIN_MNIST.json',
                   'configs/superpixels_graph_classification_GAT_MNIST.json',
                   
                   'configs/superpixels_graph_classification_GCN_CIFAR10.json',
                   'configs/superpixels_graph_classification_GIN_CIFAR10.json',
                   'configs/superpixels_graph_classification_GAT_CIFAR10.json']
    # argparse是python中内置的命令行解析模块，这个库可以让我们直接在命令行中就可以向程序中传入参数并让程序运行
    parser = argparse.ArgumentParser()  #创建解析器
    parser.add_argument('--config', type=int, default=0,   #添加参数
        help="Please give a config.json file with training/model/data/param details")

    parser.add_argument('--debug', action='store_true', default=False,
        help="Please give a value for gpu id")

    parser.add_argument('--resume', action='store_true', default=False,
        help="Please give a value for gpu id")

    parser.add_argument('--head', action='store_true', default=False,
        help="use head or not")

    parser.add_argument('--aug', type=str, default='nn',
        help="Please give a value for gpu id")

    parser.add_argument('--temp', type=float, default=0.5,
        help="Please give a value for gpu id")

    parser.add_argument('--drop_percent', type=float, default=0.2,
        help="Please give a value for gpu id")  
        
    parser.add_argument('--seed', default=41,
        help="Please give a value for seed")

    parser.add_argument('--gpu_id', default=0,
        help="Please give a value for gpu id")

    parser.add_argument('--epochs', type=int,  default=80, help="Please give a value for epochs")
    parser.add_argument('--decreasing_lr', default='50, 60', help='decreasing strategy')
    parser.add_argument('--init_lr',  help="Please give a value for init_lr")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay',  type=float, default=1e-6,  help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")    
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--graph_norm', help="Please give a value for graph_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    args = parser.parse_args()  #解析参数  得到的args，其类型可以认为是一种类似字典的数据类型  即可以利用args.参数名来提取参数
    with open(config_path[args.config]) as f:
        config = json.load(f)

    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)   
    if args.residual is not None:
        net_params['residual'] = True if args.residual=='True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated=='True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.graph_norm is not None:
        net_params['graph_norm'] = True if args.graph_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred=='True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat=='True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop=='True' else False
        
    # Superpixels

    net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].size(0)
    net_params['in_dim_edge'] = dataset.train[0][0].edata['feat'][0].size(0)
    num_classes = len(np.unique(np.array(dataset.train[:][1])))
    net_params['n_classes'] = num_classes

    if MODEL_NAME == 'DiffPool':
        # calculate assignment dimension: pool_ratio * largest graph's maximum
        # number of nodes  in the dataset
        max_num_nodes_train = max([dataset.train[i][0].number_of_nodes() for i in range(len(dataset.train))])
        max_num_nodes_test = max([dataset.test[i][0].number_of_nodes() for i in range(len(dataset.test))])
        max_num_node = max(max_num_nodes_train, max_num_nodes_test)
        net_params['assign_dim'] = int(max_num_node * net_params['pool_ratio']) * net_params['batch_size']
    
    
    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, args)


    
"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #os.environ用于获得一些有关系统的各种信息,比如os.environ['HOMEPATH']：当前用户主目录
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:  #cuda实际上提供了gpu编程的接口，不过近似认为cuda就代表gpu就行
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))  #np.prod()计算所有元素的乘积
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


if __name__ == "__main__":
    main()
    











