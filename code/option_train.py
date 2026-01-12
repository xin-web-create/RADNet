
import torch, os, sys, torchvision, argparse
import warnings
import json

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

# 设备配置
parser.add_argument('--device', type=str, default='Automatic detection')

# 训练参数
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--iters_per_epoch', type=int, default=13)
parser.add_argument('--finer_eval_step', type=int, default=1400000)
parser.add_argument('--bs', type=int, default=4, help='batch size')
parser.add_argument('--start_lr', default=0.001, type=float)
parser.add_argument('--end_lr', default=0.000001, type=float)
parser.add_argument('--no_lr_sche', action='store_true', help='no lr cos schedule')
parser.add_argument('--use_warm_up', type=bool, default=True)

parser.add_argument('--w_loss_L1', default=1, type=float)
parser.add_argument('--w_loss_CR', default=0.1, type=float)


parser.add_argument('--exp_dir', type=str, default='../experiment')
parser.add_argument('--dataset', type=str, default='DHaze')
parser.add_argument('--model_name', type=str, default='MDCTDN')
parser.add_argument('--best_model_dir', type=str, default='best_model')
parser.add_argument('--saved_data_dir', type=str, default='saved_data')
parser.add_argument('--saved_plot_dir', type=str, default='saved_plot')
parser.add_argument('--saved_infer_dir', type=str, default='saved_infer')

# 恢复训练参数
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--pre_trained_model', type=str, default='null')
parser.add_argument('--best_model_prefix', type=str, default='Dhaze_best_model')

opt = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'


dataset_dir = os.path.join(opt.exp_dir, opt.dataset)
import time
timestamp = time.strftime("%Y%m%d_%H%M%S")
model_dir = os.path.join(dataset_dir, f"{opt.model_name}_{timestamp}")

# 创建主目录
if not os.path.exists(opt.exp_dir):
    os.makedirs(opt.exp_dir, exist_ok=True)
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)


if not os.path.exists(model_dir):
    os.makedirs(model_dir, exist_ok=True)

    opt.best_model_dir = os.path.join(model_dir, 'best_model')
    opt.saved_data_dir = os.path.join(model_dir, 'saved_data')
    opt.saved_plot_dir = os.path.join(model_dir, 'saved_plot')
    opt.saved_infer_dir = os.path.join(model_dir, 'saved_infer')

    for dir_path in [opt.best_model_dir, opt.saved_data_dir,
                    opt.saved_plot_dir, opt.saved_infer_dir]:
        os.makedirs(dir_path, exist_ok=True)
else:
    print(f'警告：{model_dir} 已存在，将复用该目录')
    opt.best_model_dir = os.path.join(model_dir, 'best_model')
    opt.saved_data_dir = os.path.join(model_dir, 'saved_data')
    opt.saved_plot_dir = os.path.join(model_dir, 'saved_plot')
    opt.saved_infer_dir = os.path.join(model_dir, 'saved_infer')

    for dir_path in [opt.best_model_dir, opt.saved_data_dir,
                    opt.saved_plot_dir, opt.saved_infer_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)


print("DHaze数据集配置:")
print(opt)
print('model_dir:', model_dir)
print('best_model_dir:', opt.best_model_dir)


with open(os.path.join(model_dir, 'Dhaze_args.txt'), 'w') as f:
    json.dump(opt.__dict__, f, indent=2)
