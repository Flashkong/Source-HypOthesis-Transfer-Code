import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
import loss
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pickle
from usps import *


# 这个函数在loss.py里面也有
def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy


# 这里是论文里面stage1那里使用的新的交叉熵函数，这个东西在loss.py里面也有
class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss


def digit_load(args):
    train_bs = args.batch_size
    if args.dset == 's2m':
        # 意思是源域数据是svhn，目标域数据是mnist
        train_source = torchvision.datasets.SVHN('./data/digit/svhn/', split='train', download=True,
                                                 transform=transforms.Compose([
                                                     transforms.Resize(32),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                 ]))
        test_source = torchvision.datasets.SVHN('./data/digit/svhn/', split='test', download=True,
                                                transform=transforms.Compose([
                                                    transforms.Resize(32),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ]))
        train_target = torchvision.datasets.MNIST('./data/digit/mnist/', train=True, download=True,
                                                  transform=transforms.Compose([
                                                      transforms.Resize(32),
                                                      transforms.Lambda(lambda x: x.convert("RGB")),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                  ]))
        test_target = torchvision.datasets.MNIST('./data/digit/mnist/', train=False, download=True,
                                                 transform=transforms.Compose([
                                                     transforms.Resize(32),
                                                     transforms.Lambda(lambda x: x.convert("RGB")),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                 ]))
    elif args.dset == 'u2m':
        # 意思是源域数据是usps，目标域数据是mnist
        train_source = USPS('./data/digit/usps/', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.RandomCrop(28, padding=4),
                                transforms.RandomRotation(10),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                            ]))
        test_source = USPS('./data/digit/usps/', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.RandomCrop(28, padding=4),
                               transforms.RandomRotation(10),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))
                           ]))
        train_target = torchvision.datasets.MNIST('./data/digit/mnist/', train=True, download=True,
                                                  transform=transforms.Compose([
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5,), (0.5,))
                                                  ]))
        test_target = torchvision.datasets.MNIST('./data/digit/mnist/', train=False, download=True,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5,), (0.5,))
                                                 ]))
    elif args.dset == 'm2u':
        # 意思是源域数据是mnist，目标域数据是usps
        train_source = torchvision.datasets.MNIST('./data/digit/mnist/', train=True, download=True,
                                                  transform=transforms.Compose([
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5,), (0.5,))
                                                  ]))
        test_source = torchvision.datasets.MNIST('./data/digit/mnist/', train=False, download=True,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5,), (0.5,))
                                                 ]))

        train_target = USPS('./data/digit/usps/', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                            ]))
        test_target = USPS('./data/digit/usps/', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))
                           ]))

    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(train_source, batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    # test的batch size是train的两倍
    dset_loaders["source_te"] = DataLoader(test_source, batch_size=train_bs * 2, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dset_loaders["target"] = DataLoader(train_target, batch_size=train_bs, shuffle=True,
                                        num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(test_target, batch_size=train_bs * 2, shuffle=True,
                                      num_workers=args.worker, drop_last=False)
    return dset_loaders


def cal_acc(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy, mean_ent


def train_source(args):
    """
    这里应该是训练源域的模型
        base使用的是LeNet居然是那么古老的模型，1994年诞生，很简单的一个模型
    """
    # 加载DataLoader用来加载数据
    dset_loaders = digit_load(args)
    # set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()
    # 这个args.bottleneck决定了bottleneck那个线性层的输出维度
    # 这个args.classifier决定了bottleneck线性层之后是否需要进行batchNorm和Dropout
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    # 这个args.layer参数决定了是否需要使用torch.nn.utils.weight_norm
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    # 居然又是和faster r-cnn代码一样手动更新参数？？？
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    # 使用SGD算法进行优化，同时还使用了momentum
    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4, nesterov=True)

    acc_init = 0
    for epoch in tqdm(range(args.max_epoch), leave=False):
        # scheduler.step()
        # 全部设置为训练模式
        netF.train()
        netB.train()
        netC.train()
        # 加载源域的训练数据
        iter_source = iter(dset_loaders["source_tr"])
        # 数据训练一遍
        for _, (inputs_source, labels_source) in tqdm(enumerate(iter_source), leave=False):
            if inputs_source.size(0) == 1:
                continue
            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
            # 简单粗暴
            outputs_source = netC(netB(netF(inputs_source)))

            classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth) \
                (outputs_source, labels_source)
            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()
        # 数据每次训练了一遍就调整到测试模式
        netF.eval()
        netB.eval()
        netC.eval()
        # 计算精度，在训练集的精度和在测试集的精度
        acc_s_tr, _ = cal_acc(dset_loaders['source_tr'], netF, netB, netC)
        acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC)
        # 将精度结果写入log文件
        log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%/ {:.2f}%'.format(args.dset, epoch + 1, args.max_epoch,
                                                                             acc_s_tr * 100, acc_s_te * 100)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str + '\n')

        if acc_s_te >= acc_init:
            # 如果这个epoch的结果比之前的所有的都好，就记录精度和模型的参数
            acc_init = acc_s_te
            best_netF = netF.state_dict()
            best_netB = netB.state_dict()
            best_netC = netC.state_dict()
    # 将最终的模型参数存储下来
    torch.save(best_netF, osp.join(args.output_dir, "source_F_val.pt"))
    torch.save(best_netB, osp.join(args.output_dir, "source_B_val.pt"))
    torch.save(best_netC, osp.join(args.output_dir, "source_C_val.pt"))

    return netF, netB, netC


# todo li 需要注意到的是，在训练的时候，此模型的lr都没改变，没有进行decay

def test_target(args):
    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    # 加载进去训练好的模型，这个代码比较好理解
    args.modelpath = args.output_dir + '/source_F_val.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_B_val.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C_val.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
    log_str = 'Task: {}, Accuracy = {:.2f}%'.format(args.dset, acc * 100)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def train_target(args):
    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir + '/source_F_val.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_B_val.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C_val.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    # 只设置netC为测试模式，也就是只设置判别器为测试模型，这样在训练整个模型的时候，netC的参数就不会变化，这样，就保证了F=g.h中的h不变了
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in tqdm(range(args.max_epoch), leave=False):
        # 设置前面的为训练模式
        netF.train()
        netB.train()
        iter_test = iter(dset_loaders["target"])

        # 在这里还保存了源域的g模型，并且设置为测试模式
        # 注意,这里拷贝的是上次epoch的模型,而不是源域模型不变
        prev_F = copy.deepcopy(netF)
        prev_B = copy.deepcopy(netB)
        prev_F.eval()
        prev_B.eval()

        # 获取质心，对应论文里面的第一个公式
        center = obtain_center(dset_loaders['target'], prev_F, prev_B, netC, args)

        for _, (inputs_test, _) in tqdm(enumerate(iter_test), leave=False):
            if inputs_test.size(0) == 1:
                continue
            inputs_test = inputs_test.cuda()
            with torch.no_grad():
                # 注意，是每进行一个数据batch的iteration就预测label一次，
                # 另外，无论iteration多少次，他们预测label使用的模型都是上次epoch使用的模型
                # 下面这两句对应论文里面的第二个公式
                # todo li 论文里面还有第三个和第四个公式，怎么没看到在哪啊。
                features_test = prev_B(prev_F(inputs_test))
                pred = obtain_label(features_test, center)

            # 这里是正常的数据经过网络
            features_test = netB(netF(inputs_test))
            outputs_test = netC(features_test)
            # 计算损失
            classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0)(outputs_test, pred)

            # 这里计算IM loss
            # 这里计算的是softmax的输出,对dim=1进行softmax
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            # 这个entropy计算的是-sum(softmax_out*log(softmax_out),dim=1),得到每个batch的概率之和
            # 然后再进行一个mean,相当于是计算出来了概率之和相对于每个batch的平均值
            # 这个im_loss对应论文里面的Lent
            im_loss = torch.mean(Entropy(softmax_out))
            # msoftmax计算出来了每个类别的概率,batch的平均值,这个对应论文里面的p^k
            msoftmax = softmax_out.mean(dim=0)
            # 这里的这个-=配合sum里面的负号,就是+=. 这里是求K个的平均值,对应论文里面的Ldiv
            im_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
            # args.par在这里用到了，是权衡IM loss和classifier_loss的超参数
            total_loss = im_loss + args.par * classifier_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        netF.eval()
        netB.eval()
        acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
        log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.dset, epoch + 1, args.max_epoch, acc * 100)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str + '\n')

    # torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F.pt"))
    # torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B.pt"))
    # torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C.pt"))
    return netF, netB, netC


def obtain_center(loader, netF, netB, netC, args, c=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    aff = np.eye(K)[pred_label]
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    center = torch.from_numpy(initc).cuda()

    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')
    return center


def obtain_label(features_target, center):
    features_target = torch.cat((features_target, torch.ones(features_target.size(0), 1).cuda()), 1)
    fea = features_target.float().detach().cpu().numpy()
    center = center.float().detach().cpu().numpy()
    dis = cdist(fea, center, 'cosine') + 1
    pred = np.argmin(dis, axis=1)
    pred = torch.from_numpy(pred).cuda()
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="maximum epoch")
    parser.add_argument('--batch_size', type=int, default=128, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    # 这里代表了源域数据和目标域数据分别是什么
    #   s代表SVHN，m代表MNIST，u代表USPS
    parser.add_argument('--dset', type=str, default='s2m', choices=['u2m', 'm2u', 's2m'])
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--par', type=float, default=0.1)
    # 这个args.bottleneck决定了bottleneck那个线性层的输出维度
    # 这个args.classifier决定了bottleneck线性层之后是否需要进行batchNorm和Dropout
    # 这个args.layer决定了最后的fc层（分类器）是否需要使用torch.nn.utils.weight_norm
    # wn和bn分别对应了论文中的weightNorm和batchNorm
    parser.add_argument('--bottleneck', type=int, default=256)
    # todo li 这个wn和bn是什么东西？？
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()
    args.class_num = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # 设置随机种子
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    current_folder = "./"
    # 这个output_dir的值为'./seed2020\\s2m'
    args.output_dir = osp.join(current_folder, args.output, 'seed' + str(args.seed), args.dset)
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # 检测训练好的源域模型是否存在
    if not osp.exists(osp.join(args.output_dir + '/source_F_val.pt')):
        # 输出的log文件为'./seed2020\\s2m\\log_src_val.txt'
        args.out_file = open(osp.join(args.output_dir, 'log_src_val.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_source(args)
        test_target(args)
    # 输出的log文件为'./seed2020\\s2m\\log_tar_val_0.1.txt'
    args.out_file = open(osp.join(args.output_dir, 'log_tar_val_' + str(args.par) + '.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    train_target(args)
