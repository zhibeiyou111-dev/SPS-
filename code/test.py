import os
import argparse
import torch
from utils import util
from utils.util import *
from model import ResNet_cifar
from model import Resnet_LT
from imbalance_data import cifar10Imbanlance,cifar100Imbanlance,dataset_lt_data
from sklearn.metrics import confusion_matrix

def validate(model,val_loader,args, cls_num_list):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    confidence = np.array([])
    pred_class = np.array([])
    true_class = np.array([])
    all_preds = []
    all_targets = []
    eps = np.finfo(np.float64).eps
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target,_) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            # compute output
            output = model(input, train=False)
            # measure accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))
            output = 'Testing:  ' + str(i) + ' Prec@1:  ' + str(top1.val) + ' Prec@5:  ' + str(top5.val)
            print(output, end="\r")
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt    
        many_shot = cls_num_list > 100
        medium_shot = (cls_num_list <= 100) & (cls_num_list > 20)
        few_shot = cls_num_list <= 20
        print("many avg, med avg, few avg",
                float(sum(cls_acc[many_shot]) * 100 / (sum(many_shot) + eps)),
                float(sum(cls_acc[medium_shot]) * 100 / (sum(medium_shot) + eps)),
                float(sum(cls_acc[few_shot]) * 100 / (sum(few_shot) + eps))
                )
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(flag='val', top1=top1, top5=top5))
        print(output)

def get_model(args):
    if args.dataset == "ImageNet-LT" or args.dataset == "iNaturelist2018":
        net = Resnet_LT.resnext50_32x4d(num_classes=args.num_classes)
        print("=> creating model '{}'".format('resnext50_32x4d'))
    else:
        if args.arch == 'resnet50':
            net = ResNet_cifar.resnet50(num_class=args.num_classes)
        elif args.arch == 'resnet18':
            net = ResNet_cifar.resnet18(num_class=args.num_classes)
        elif args.arch == 'resnet32':
            net = ResNet_cifar.resnet32(num_class=args.num_classes)
        elif args.arch == 'resnet34':
            net = ResNet_cifar.resnet34(num_class=args.num_classes)
        print("=> creating model '{}'".format(args.arch))
    return net

def get_dataset(args):
    _,transform_val = util.get_transform(args.dataset)
    if args.dataset == 'cifar10':
        testset = cifar10Imbanlance.Cifar10Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False, transform=transform_val,file_path=os.path.join(args.root,'cifar-10-batches-py/'))
        print("load cifar10")
        return testset

    if args.dataset == 'cifar100':
        trainset = cifar100Imbanlance.Cifar100Imbanlance(transform=None,imbanlance_rate=args.imbanlance_rate, train=True,file_path=os.path.join(args.root,'cifar-100-python/'))

        testset = cifar100Imbanlance.Cifar100Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False, transform=transform_val,file_path=os.path.join(args.root,'cifar-100-python/'))
        print("load cifar100")
        return trainset,testset

    if args.dataset == 'ImageNet-LT':
        testset = dataset_lt_data.LT_Dataset(args.root, args.dir_test_txt, transform_val)
        print("load ImageNet-LT")
        return testset

    if args.dataset == 'iNaturelist2018':
        testset = dataset_lt_data.LT_Dataset(args.root, args.dir_test_txt,transform_val)
        print("load iNaturelist2018")
        return testset

def main():
    args = parser.parse_args()
    global train_cls_num_list
    if args.gpu is not None:
        print("Use GPU: {} for testing".format(args.gpu))
    # create model
    model = get_model(args)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # test from a checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cuda:0')
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.resume))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    train_dataset ,val_dataset = get_dataset(args)
    cls_num_list = [0] * args.num_classes
    # train_dataset = get_dataset(args)
    for label in train_dataset.targets:
        cls_num_list[label] += 1
    train_cls_num_list = np.array(cls_num_list)
    
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),num_workers=args.workers, persistent_workers=True,pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers, pin_memory=True)


    print("Testing started!")
    # switch to evaluate mode
    model.eval()
    validate(model, val_loader, args,train_cls_num_list)

if __name__ == '__main__':
    # test set
    parser = argparse.ArgumentParser(description="Global and Local Mixture Consistency Cumulative Learning")
    parser.add_argument('--dataset', type=str, default='cifar100',help="cifar10,cifar100,ImageNet-LT,iNaturelist2018")
    # parser.add_argument('--root', type=str, default='/data/',help="dataset setting")
    parser.add_argument('--root', type=str, default='/home/liuyang/project/long-tail/GLMC-CVPR2023/data', help="dataset setting")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',choices=('resnet18', 'resnet34','resnet32', 'resnet50', 'resnext50_32x4d'))
    parser.add_argument('--imbanlance_rate', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes ',choices=('10', '100', '1000', '8142'))
    parser.add_argument('-b', '--batch_size', default=64, type=int,metavar='N', help='mini-batch size')
    # etc.
    parser.add_argument('-p', '--print_freq', default=100, type=int, metavar='N',help='print frequency (default: 100)')
    parser.add_argument('--gpu', default=None, type=int,help='GPU id to use.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default='/home/liuyang/project/long-tail/GLMC-CVPR2023/GLMC-CVPR2023/output/dataset: cifar100#arch: resnet34#imbanlance_rate: 0.01#2025-06-27 15:37:41/ckpt.best.pth.tar', type=str, metavar='PATH',help='path to latest checkpoint')
    parser.add_argument('--root_model', type=str, default='GLMC-CVPR2023/output/')
    parser.add_argument('--store_name', type=str, default='GLMC-CVPR2023/output/')
    parser.add_argument('--dir_train_txt', type=str,default="GLMC-CVPR2023/data/data_txt/iNaturalist18_train.txt")
    parser.add_argument('--dir_test_txt', type=str,default="GLMC-CVPR2023/data/data_txt/iNaturalist18_val.txt")
    main()
