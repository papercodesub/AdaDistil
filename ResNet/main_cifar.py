import random
import time
import torch
import torch.optim as optim
import models
import utils.common as utils
from data import cifar10
from utils.common import *
from utils.conv_type import *
from utils.options import args
from itertools import combinations

visible_gpus_str = ','.join(str(i) for i in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpus_str
checkpoint = utils.checkpoint(args)
args.gpus = [i for i in range(len(args.gpus))]
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
logger = utils.get_logger(os.path.join(args.job_dir, 'logger.log'))
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'

if args.label_smoothing is None:
    loss_func = nn.CrossEntropyLoss().cuda()
else:
    loss_func = LabelSmoothing(smoothing=args.label_smoothing)

# Data
print('==> Loading Data..')
if args.data_set == 'cifar10':
    loader = cifar10.Data(args)

kl_loss = nn.KLDivLoss(reduction="batchmean",log_target=True).cuda()
mse_loss = nn.MSELoss().cuda() 
cos_similarity = nn.CosineSimilarity(dim=2, eps=1e-6).cuda()
Temperature = args.Tau

def train(model, teacher_model ,optimizer, trainLoader, args, epoch):
    teacher_model.eval()
    model.train()
    losses = utils.AverageMeter(':.4e')
    accurary = utils.AverageMeter(':6.3f')
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 2
    start_time = time.time()

    for batch, (inputs, targets) in enumerate(trainLoader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output, student_feature_map = model(inputs)
        teacher_output, teacher_feature_map = teacher_model(inputs)

        softed_output = nn.functional.log_softmax(output / Temperature, dim=-1)
        softed_teacher_output = nn.functional.log_softmax(teacher_output / Temperature, dim=-1)

        sf = student_feature_map.reshape(student_feature_map.shape[0], student_feature_map.shape[1], -1)
        tf = teacher_feature_map.reshape(teacher_feature_map.shape[0], teacher_feature_map.shape[1], -1)
  
        diff  = (sf - tf).abs()
        value, index = torch.topk(diff, k=int((1 - cos_similarity(sf,tf).mean()) * tf.shape[-1]), dim=-1, largest=True)
        mask = torch.zeros_like(sf)
        mask.scatter_(2,index,1)
        mask = mask.reshape_as(teacher_feature_map).to(device)

        mix_feature_map = (student_feature_map * (1 - mask) + mask * teacher_feature_map).float()

        loss = args.b * Temperature * Temperature * kl_loss(softed_output, softed_teacher_output) + args.c * mse_loss(mix_feature_map,teacher_feature_map)
     
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(output, targets)
        accurary.update(prec1[0], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t'
                'Accurary {:.2f}%\t\t'
                'Time {:.2f}s'.format(
                    epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                    float(losses.avg), float(accurary.avg), cost_time
                )
            )
            start_time = current_time


def validate(model, testLoader):
    global best_acc
    model.eval()

    losses = utils.AverageMeter(':.4e')
    accurary = utils.AverageMeter(':6.3f')

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets)
            accurary.update(predicted[0], inputs.size(0))

        current_time = time.time()
        logger.info(
            'Test Loss {:.4f}\tAccurary {:.2f}%\t\tTime {:.2f}s\n'
                .format(float(losses.avg), float(accurary.avg), (current_time - start_time))
        )
    return accurary.avg




def main():
    # set seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    start_epoch = 0
    best_acc = 0.0

    model = models.__dict__[args.arch]().to(device)
    model = model.to(device)
    optimizer = get_optimizer(args, model)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)
    model = nn.DataParallel(model, device_ids=args.gpus)
    
    # load student model
    if args.pretrained_model:
        ckpt = torch.load(args.pretrained_model)
        model.load_state_dict(ckpt["state_dict"], strict = False)
        validate(model, loader.testLoader)
        if args.test_only:
            return
    else:
        print(f"Please set pretrained model path")

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
        validate(model, loader.testLoader)

    # load teacher model
    teacher_model = models.__dict__[args.arch]().to(device)
    teacher_model = teacher_model.to(device)
    teacher_model = nn.DataParallel(teacher_model, device_ids=args.gpus)

    if args.teahcher_model:
        if os.path.exists(args.teahcher_model):
            ckpt = torch.load(args.teahcher_model)
            teacher_model.load_state_dict(ckpt["state_dict"], strict = False)
        else:
            print(f"=> No checkpoint found at '{args.pretrained_model}'")
    else:
        print(f"=> No teahcher_model")

    for epoch in range(start_epoch, args.num_epochs):
        train(model, teacher_model, optimizer, loader.trainLoader, args, epoch)
 
        test_acc = validate(model, loader.testLoader)
        scheduler.step()

        is_best = best_acc < test_acc
        best_acc = max(best_acc, test_acc)

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
        }

        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best accurary: {:.3f}'.format(float(best_acc)))

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
