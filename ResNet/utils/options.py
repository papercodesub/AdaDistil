import argparse

parser = argparse.ArgumentParser(description='NM Pruning')

parser.add_argument(
    "--seed", default=42, type=int, help="seed for initializing training. "
)

parser.add_argument(
    "--config",
    help="Config file to use (see configs dir)",
    default=None
)

parser.add_argument(
    "--label-smoothing",
    type=float,
    help="Label smoothing to use, default 0.0",
    default=0.0
)

parser.add_argument(
    "--warmup_length",
    default=5,
    type=int,
    help="Number of warmup iterations"
)

parser.add_argument(
    '--gpus',
    type=int,
    nargs='+',
    default=[0,1],
    help='Select gpu_id to use. default:[0]',
)

parser.add_argument(
    '--pretrained_model',
    type=str,
    default=None,
    help='Path of the pretrained_model',
)

parser.add_argument(
    '--teahcher_model',
    type=str,
    default=None,
    help='Path of the teahcher_model',
)

parser.add_argument(
    '--data_set',
    type=str,
    default='cifar10',
    help='Select dataset to train. default:cifar10',
)

parser.add_argument(
    '--data_path',
    type=str,
    default='/Imagenets',
    help='The dictionary where the input is stored. default:',
)

parser.add_argument(
    '--job_dir',
    type=str,
    default='experiments/Imagenet',
    help='The directory where the summaries will be stored. default:./experiments'
)

parser.add_argument(
    '--resume',
    action='store_true',
    help='Load the model from the specified checkpoint.'
)

## Training
parser.add_argument(
    '--arch',
    type=str,
    default='resnet50',
    help='Architecture of model. default:resnet32_cifar10, resnet 50, resnet18'
)

parser.add_argument(
    '--num_epochs',
    type=int,
    default=120,
    help='The num of epochs to train. default:180'
)

parser.add_argument(
    '--train_batch_size',
    type=int,
    default=256,
    help='Batch size for training. default:256'
)

parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=256,
    help='Batch size for validation. default:256'
)

parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='Momentum for MomentumOptimizer. default:0.9'
)

parser.add_argument(
    '--lr',
    type=float,
    default=0.1,
    help='Learning rate for train. default:0.1'
)

parser.add_argument(
    "--optimizer",
    help="Which optimizer to use",
    default="sgd"
)

parser.add_argument(
    "--lr_policy",
    default="cos",
    help="Policy for the learning rate."
)

parser.add_argument(
    '--lr_gamma',
    type=float,
    default=0.1,
    help='gamma for StepLR'
)

parser.add_argument(
    '--lr_step_size',
    type=int,
    default=30,
    help='step_size for StepLR'
)

parser.add_argument(
    '--weight_decay',
    type=float,
    default=1e-4,
    help='The weight decay of loss. default:1e-4'
)

parser.add_argument(
    "--nesterov",
    default=False,
    action="store_true",
    help="Whether or not to use nesterov for SGD",
)

parser.add_argument(
    "--first-layer-type",
    type=str,
    default=None,
    help="Conv type of first layer"
)

parser.add_argument(
    "--conv_type",
    type=str,
    default='NMConv',
    help="Conv type of conv layer. Default: NMConv"
)

parser.add_argument(
    "--bn_type",
    default='LearnedBatchNorm',
    help="BatchNorm type"
)

parser.add_argument(
    "--init",
    default="kaiming_normal",
    help="Weight initialization modifications"
)

parser.add_argument(
    "--mode",
    default="fan_in",
    help="Weight initialization mode"
)

parser.add_argument(
    "--nonlinearity",
    default="relu",
    help="Nonlinearity used by initialization"
)

parser.add_argument(
    '--debug',
    action='store_true',
    help='input to open debug state'
)

parser.add_argument(
    "--full",
    action="store_true",
    help="prune full-connect layer"
)
## NM sparsity
parser.add_argument(
    "--N",
    default=2,
    type=int,
    help="N:M's N"
)
parser.add_argument(
    "--M",
    default=4,
    type=int,
    help="N:M's M"
)

parser.add_argument(
    "--permution",
    action="store_true",
    default=False,
    help="use permution?"
)

parser.add_argument(
    "--scale-fan",
    action="store_true",
    default=False,
    help="scale fan"
)

parser.add_argument(
    "--usenm",
    action="store_true",
    default=False,
    help="use nm sparsity"
)

parser.add_argument(
    "--Tau",
    default=1.0,
    type=float,
    help="Distillaiton Temperature"
)


parser.add_argument(
    "--a", 
    type=float, 
    default=1, 
    help="Distillation Hyperparameter"
)

parser.add_argument(
    "--b", 
    type=float, 
    default=0.5, 
    help="Distillation Hyperparameter"
)

parser.add_argument(
    "--c", 
    type=float, 
    default=0.5, 
    help="Distillation Hyperparameter"
)

parser.add_argument(
    "--test-only",
    action="store_true",
    default=False,
    help="test mode"
)

args = parser.parse_args()
