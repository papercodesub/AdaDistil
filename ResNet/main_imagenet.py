import math
import time
import torch.cuda
import utils.common as utils
from utils.common import *
from utils.conv_type import *
from data import imagenet
import random
if args.debug:
    pass

import models

visible_gpus_str = ','.join(str(i) for i in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpus_str
checkpoint = utils.checkpoint(args)
args.gpus = [i for i in range(len(args.gpus))]
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
logger = utils.get_logger(os.path.join(args.job_dir, 'logger' + now + '.log'))
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'

## define loss function 
kl_loss = nn.KLDivLoss(reduction="batchmean",log_target=True).cuda()
mse_loss = nn.MSELoss().cuda()
cos_similarity = nn.CosineSimilarity(dim=2, eps=1e-6)

## distillation parameters
Temperature = args.Tau

if args.label_smoothing is None:
    loss_func = nn.CrossEntropyLoss().cuda()
else:
    loss_func = LabelSmoothing(smoothing=args.label_smoothing)

# load training data
print('==> Preparing data..')

data_tmp = imagenet.Data(args)
train_loader = data_tmp.trainLoader
val_loader = data_tmp.testLoader


def train(epoch, train_loader, model, teacher_model, criterion, optimizer):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.train()
    teacher_model.eval()
    end = time.time()

    num_iter = len(train_loader)

    print_freq = num_iter // 10
    i = 0

    for batch_idx, (images, targets) in enumerate(train_loader):
        if args.debug:
            if i > 5:
                break
            i += 1
        images = images.cuda()
        targets = targets.cuda()
        data_time.update(time.time() - end)

        adjust_learning_rate(optimizer, epoch, batch_idx, num_iter)

        # compute output
        output, student_feature_map = model(images)
        teacher_output, teacher_feature_map = teacher_model(images)

        # prepare for soft label distillation
        softed_output = nn.functional.log_softmax(output / Temperature, dim=-1)
        softed_teacher_output = nn.functional.log_softmax(teacher_output / Temperature, dim=-1)

        # prepare for feature map distillation
        sf = student_feature_map.reshape(student_feature_map.shape[0], student_feature_map.shape[1], -1)
        tf = teacher_feature_map.reshape(teacher_feature_map.shape[0], teacher_feature_map.shape[1], -1)

        diff  = (sf - tf).abs()
        value, index = torch.topk(diff, k=int((1 - cos_similarity(sf,tf).mean()) * tf.shape[-1]), dim=-1, largest=True)
        mask = torch.zeros_like(sf)
        mask.scatter_(2, index, 1)
        mask = mask.reshape_as(teacher_feature_map).to(device)

        mix_feature_map = (student_feature_map * (1 - mask) + mask * teacher_feature_map).float()

        loss = args.a * loss_func(output, targets) + args.b * Temperature * Temperature * kl_loss(softed_output,softed_teacher_output) + args.c * mse_loss(mix_feature_map,teacher_feature_map)
           
        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(output, targets, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)  # accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % print_freq == 0 and batch_idx != 0:
            logger.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, batch_idx, num_iter, loss=losses,
                    top1=top1, top5=top5))
            
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    num_iter = len(val_loader)

    model.eval()
    with torch.no_grad():
        end = time.time()
        i = 0
        for batch_idx, (images, targets) in enumerate(val_loader):
            if args.debug:
                if i > 5:
                    break
                i += 1
            images = images.cuda()
            targets = targets.cuda()

            # compute output
            output, _ = model(images)
            loss = criterion(output, targets)

            # measure accuracy and record loss
            pred1, pred5 = utils.accuracy(output, targets, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def get_model(args):
    model = models.__dict__[args.arch]().to(device)
    model = model.to(device)
    return model


def main():
    # set seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        
    start_epoch = 0
    best_acc = 0.0
    best_acc_top1 = 0.0
    model = get_model(args)
    optimizer = get_optimizer(args, model)
    if args.resume:
        start_epoch, best_acc_top1 = resume(args, model, optimizer)
        model = nn.DataParallel(model, device_ids=args.gpus)
    else: 
        # load student model
        if args.pretrained_model:
            ckpt = torch.load(args.pretrained_model)
            model.load_state_dict(ckpt["state_dict"], strict = False)
            model = nn.DataParallel(model, device_ids=args.gpus)
            logger.info("Pretrained model's Accuracy")
            validate(val_loader, model, loss_func, args)
            if args.test_only:
                return
        else:
            logger.info(f"Please set pretrained model path")
        
        if args.usenm:
            logger.info("Begin Pruning")
            state_dict = model.module.state_dict()
            with torch.no_grad():
                for n, mask in model.module.named_parameters():
                    if 'mask' in n:
                        weights = copy.deepcopy(state_dict[n[:-4] + 'weight'])
                        shape = weights.shape
                        weights = weights.permute(2,3,0,1).reshape([-1,args.M])
                        mask = copy.deepcopy(state_dict[n]).reshape([-1,args.M])
                        value, index = torch.topk(weights.abs(), k=args.M - args.N, dim=1, largest=False)
                        mask.scatter_(1, index, 0)
                        mask = mask.view(shape[2], shape[3], shape[0], shape[1]).permute(2,3,0,1).contiguous()
                        state_dict[n] = mask
            model.module.load_state_dict(state_dict)
            logger.info("Pruning Over")
            logger.info("Model's Accuracy after pruning")
            validate(val_loader, model, loss_func, args)

    print("=> Creating teacher model")
    teacher_model = models.__dict__[args.arch]().to(device)
    teacher_model = teacher_model.to(device)
    
    # load teacher model
    if args.teahcher_model:
        if os.path.exists(args.teahcher_model):
            print(f"=> Loading teacher checkpoint ")
            ckpt = torch.load(args.teahcher_model)
            teacher_model.load_state_dict(ckpt["state_dict"], strict = False)
            teacher_model = nn.DataParallel(teacher_model, device_ids=args.gpus)
            logger.info("Teacher model's Accuracy")
            validate(val_loader, teacher_model, loss_func, args)
        else:
            print(f"=> No checkpoint found at '{args.pretrained_model}'")
    else:
        print(f"=> No teahcher_model")

    for epoch in range(start_epoch, args.num_epochs):
        train_obj, train_acc_top1, train_acc = train(epoch, train_loader, model, teacher_model, loss_func, optimizer)
        valid_obj, test_acc_top1, test_acc = validate(val_loader, model, loss_func, args)

        is_best = best_acc_top1 < test_acc_top1
        best_acc_top1 = max(best_acc_top1, test_acc_top1)
        best_acc = max(best_acc, test_acc)

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_acc_top1': best_acc_top1,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }
        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best accurary(top5): {:.3f} (top1): {:.3f}'.format(float(best_acc), float(best_acc_top1)))


def resume(args, model, optimizer):
    if os.path.exists(args.job_dir + '/checkpoint/model_last.pt'):
        print(f"=> Loading checkpoint ")

        checkpoint = torch.load(args.job_dir + '/checkpoint/model_last.pt')

        start_epoch = checkpoint["epoch"]

        best_acc_top1 = checkpoint["best_acc_top1"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint (epoch) {checkpoint['epoch']})")

        return start_epoch, best_acc_top1
    else:
        print(f"=> No checkpoint found at '{args.job_dir}' '/checkpoint/")


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    # Warmup
    if args.lr_policy == 'step':
        factor = epoch // 30
        if epoch >= 80:
            factor = factor + 1
        lr = args.lr * (0.1 ** factor)
    elif args.lr_policy == 'cos':  # cos with warm-up
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch - 5) / (args.num_epochs - 5)))
    elif args.lr_policy == 'exp':
        step = 1
        decay = 0.96
        lr = args.lr * (decay ** (epoch // step))
    elif args.lr_policy == 'fixed':
        lr = args.lr
    else:
        raise NotImplementedError
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    if step == 0:
        print('current learning rate:{0}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_optimizer(args, model):
    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and ("alpha" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0,
                },
                {
                    "params": rest_params, 
                    "weight_decay": args.weight_decay
                },
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )
    else:
        print("please choose sgd or adam")
        raise
    return optimizer


if __name__ == '__main__':
    main()
