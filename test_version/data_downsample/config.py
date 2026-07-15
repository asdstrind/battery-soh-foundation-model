#%%
import os
import torch
from torch import nn
import numpy as np
from scipy.stats import pearsonr
from torch.autograd import Function
import torch.nn.functional as F
#from e_basic_setting import basic_setting
from sklearn.metrics import mean_absolute_error, r2_score


class Config:                                       # 创建config类，用于存放超参数、参数和路径
    def __init__(self,test_battery_id=1):
        self.Transfer = True                        # 是否迁移（为了与不迁移做对比）
        self.Run_in_server = False
        # 所有数据集以及方法
        self.All_Dataset = ['CALCE','HNEL','IECON','NASA','Oxford','SNL_LFP','SNL_NCA','SNL_NMC','TongJi','Toyota_MIT','XJTU_battery']
        self.All_normalized_type = ['minmax', 'standard']
        self.All_minmax_range = [(0, 1), (-1, 1)]
        self.ALL_quantization = ['random_quantization', 'no_quantization', 'more random', 'fix_V_range', 'IC',
                                 'IC+slope', 'slope', 'generate_more_random_data', 'raw']
        self.ALL_Method = ['Common', 'DANN', 'MMD', 'Common + Person_loss', 'DANN + Person_loss', 'MMD + Person_loss']
        self.ALL_Net = ['ResNet', 'Transformer', 'Dropout+6ResNet', 'Person_Loss', 'MMDnet', 'CNN_MMD', 'Attention', 'AE']
        # 选择使用数据集以及方法
        self.select_dataset = self.All_Dataset[3]                                                                 # 选择使用数据集
        #self.select_data_type = self.All_Data_type[0]                                                             # 选择的数据类型
        self.select_normalized_type = self.All_normalized_type[0]                                                 # 选择的归一化方法
        self.select_minmax_range = self.All_minmax_range[0]       # 最大最小归一化方法种类
        self.select_quantization = self.ALL_quantization[2]
        self.select_method = self.ALL_Method[1]
        self.select_Net = self.ALL_Net[0]

        self.start_V = 3.4
        self.end_V = 3.9

        # 划分数据集以及神经网络学习率以及损失函数
        self.random_seed = 2025  # 随机数，确保划分的训练集、验证集相同
        self.Batch_size = 256  # 设置每次iter的训练样本数目，越大越好，越大则每次梯度更新涉及的样本类别和数目都会更多，但是太大可能对算力要求较高，权衡利弊后取折中值256（默认128或256）
        self.Learning_rate = 1e-4  # 设置学习率，学习率通常来说越小越好，但是太小可能导致模型收敛速度很慢，太大可能不收敛（默认1e-4或1e-5）
        self.Loss = RMSELoss()  # 设置损失函数，对于分类问题一般是选择交叉熵函数，因为它更针对真实标签的预测情况，但是其他场合可能也会选择均方差函数
        self.delta_V = 0.4

        self.L_sample = 256
        self.Depth = 3  # 设置编码器深度                   (4)
        self.N_heads = 8  # 设置多头自注意力机制头数           (2)
        self.Dropout = 0.5  # 设置Dropout的概率p              (0)
        self.Mlp_ratio = 1  # 设置MLP层神经元相对与输入维度的比例  ()
        self.Patch_size = 32  # 设置每个Patch块的大小            (16)
        self.Embed_dim = 256  # 设置编码器隐藏层维度              (128)
        self.N_feature_1 = 256  # 全连接层维度                   (4)
        self.N_epoch = 600
        self.new_length = 256
        self.ratio_V = 0.3
        self.weight_MMD = 0.6
        self.weight_person_loss = 0.3
        self.test_battery_id =test_battery_id  # 选择电池包中的某个电池作为测试集
        self.out_feature = 256
        self.loss_epoch = self.N_epoch //5
        self.Result_path = 'D:\pycharm\Py_Projects/battery_SOH_predict\second\我的迁移学习\data/result'  # 输出信息文件汇总文件夹
        self.Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                                                            # 如无Cuda则使用Cpu训练



        #  根据不同的训练集选择不同保存位置
        self.select_dataset == 'partical_XJTU_data'
        self.dataset_path = r'E:\Battery_data\Data\Convert_XJTU_data\partical'
        self.max_capacity = 2  # 最大容量
        self.root_dataset_path = r'E:\Battery_data\Data\Convert_XJTU_data\partical'
        self.num_battery = 8


    def update(self, params):
        self.weight_MMD = params[0]
        self.Learning_rate = params[1]


    def inf_print_and_make_dirs(self, train_dataloader, test_dataloader):   # 基本参数输出及创建对应文件夹（无需改动）
        #basic_setting()
        result_path = self.Result_path
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        path = ["Loss", "Models", "pred_SOH"]
        for i, value in enumerate(path):
            i = os.path.join(result_path, path[i])
            if not os.path.exists(i):
                os.makedirs(i)
        model_path = os.path.join(self.Result_path, "models\\model.pt")
        CUDA = torch.cuda.is_available()  # 查询CUDA是否可用
        cuDNN = torch.backends.cudnn.is_available()  # 查询cuDNN是否可用
        N_rjust = 150
        print(f"\nResults will be saved in file:{self.Result_path} ")
        print(f"1. Basic information is as follows:")
        print(f"\t|".ljust(N_rjust, '-'), end='|\n')
        print(f"\t|  CUDA_is_available:{CUDA}          cuDNN_is_available:{cuDNN}".ljust(N_rjust, ' '), end='|\n')
        print(f"\t|  Train Epoch:{self.N_epoch}     Learning_rate:{self.Learning_rate}     ", end='|\n')
        print(f"\t|  Dataset:{self.select_dataset}     Method:{self.select_method}".ljust(N_rjust, ' '), end='|\n')
        print(f"\t|  Signal type:Battery_data".ljust(N_rjust, ' '), end='|\n')
        print(f"\t|  N_train_sample:{len(train_dataloader.dataset)}      "
              f"L_train_sample:{len(train_dataloader.dataset[0][0])}".ljust(N_rjust, ' '), end='|\n')
        print(f"\t|  N_test_sample:{len(test_dataloader.dataset)}       "
              f"L_test_sample:{len(test_dataloader.dataset[0][0])}".ljust(N_rjust, ' '), end='|\n')
        print(f"\t|".ljust(N_rjust, '-'), end='|\n')
        print(f"2. Queries whether the model is trained:")
        if os.path.exists(model_path):
            print('\tThere have a model trained:')
        else:
            print('\tThere have no model trained, begin to train:')


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


class Person_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = x.view(-1)
        y = y.view(-1)
        # 计算均值
        mean1 = torch.mean(x)
        mean2 = torch.mean(y)
        # 计算分子和分母
        numerator = torch.sum((x - mean1) * (y - mean2))
        denominator = torch.sqrt(torch.sum((x - mean1) ** 2) * torch.sum((y - mean2) ** 2))
        # 计算皮尔逊相关系数
        correlation = numerator / (denominator + 1e-8)  # 加一个小数以防除以零
        # 损失函数为 1 - abs(correlation)
        loss = 0.5*(1 - correlation)
        return loss


class Cosine_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        # 计算x和y的余弦相似度
        cosine_similarity = F.cosine_similarity(x, y, dim=-1)
        # 计算损失，通常是1减去余弦相似度
        loss = 1 - cosine_similarity.mean()
        return loss


class feature_Person_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        y = y.view(-1)  # 将 (128, 1) 转换为 (128,)
        mean_y = torch.mean(y)
        # 中心化目标变量
        y_centered = y - mean_y
        # 计算每列的均值并进行中心化
        mean_x = torch.mean(x, dim=0, keepdim=True)  # 保持维度以便进行广播
        x_centered = x - mean_x  # 在列上中心化特征矩阵
        # 计算相关性
        numerator = torch.sum(x_centered * y_centered.unsqueeze(1), dim=0)  # (N,)
        denominator = torch.sqrt(torch.sum(x_centered ** 2, dim=0) * torch.sum(y_centered ** 2))  # (N,)
        correlation = numerator / (denominator + 1e-8)  # (N,)
        # 计算损失
        loss = torch.mean(0.5*(1 - correlation))  # 计算平均相关系数
        return loss


def calculate_loss(pred, real):
    # pred_np = pred.detach().cpu().numpy()
    # real_np = real.detach().cpu().numpy()
    # 计算 MSE
    mse = torch.mean((pred - real) ** 2)
    # 计算 MAE
    mae = torch.mean(torch.abs(pred - real))
    # 计算 R²
    ss_res = torch.sum((real - pred) ** 2)
    ss_tot = torch.sum((real - torch.mean(real)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))  # 加小常数以防除以零
    return mse, mae, r2

class MMD_loss(nn.Module):                          # 设置MMDloss
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                           fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

class ReverseLayerF(Function):                      # 梯度反转层所需
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None