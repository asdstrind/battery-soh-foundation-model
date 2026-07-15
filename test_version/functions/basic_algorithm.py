import os
import torch
import itertools
import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
from torch import nn
from sklearn.manifold import TSNE
from torch.autograd import Function
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from .utils import plot

plt.rcParams['font.sans-serif'] = ['Times New Roman']        # 设置字体（为了matplotlib正常显示）
matplotlib.rcParams['axes.unicode_minus'] = False   # 为了matplotlib正常显示负号


# 设置全局样式
matplotlib.rcParams.update({
    'axes.grid': False,          # 关闭网格
    # 新增坐标轴边框配置
    'axes.spines.left': True,    # 开启左侧边框
    'axes.spines.right': True,   # 开启右侧边框
    'axes.spines.top': True,     # 开启顶部边框
    'axes.spines.bottom': True,  # 开启底部边框
    'axes.edgecolor': 'black',   # 边框颜色
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',  # 图形背景颜色
    'axes.facecolor': 'white'
})

# 生成示例图形
"""fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([1, 2, 3], [4, 5, 2])
plt.show()"""


class ReverseLayerF(Function):                      # 梯度反转层所需

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


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


def generate_loss_acc(train_loss, test_loss, train_acc, test_acc, result_path):  # 生成训练和测试过程loss&acc图像
    train_loss, test_loss= np.loadtxt(train_loss), np.loadtxt(test_loss)
    train_acc, test_acc = np.loadtxt(train_acc), np.loadtxt(test_acc)

    save_path = os.path.join(result_path, f"loss_acc")
    x_num = len(train_loss)

    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)                                                # 生成loss图像
    plt.title('loss', fontdict={'fontsize': 12, 'color': 'green'})
    plt.xlim(0, x_num)
    plt.plot(range(x_num), train_loss, label='train_loss')
    plt.plot(range(x_num), test_loss, label='test_loss')
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.title('acc', fontdict={'fontsize': 12, 'color': 'green'})       # 生成acc图像
    plt.xlim(0, x_num)
    plt.plot(range(x_num), train_acc, label='train_acc')
    plt.plot(range(x_num), test_acc, label='test_acc')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig(f'{save_path}\\loss_acc.png', dpi=600)
    plt.show()


def plot_confusion_matrix(y_real, y_pred, labels, cmap=plt.cm.Blues):  # 生成混淆矩阵
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    cm = confusion_matrix(y_real, y_pred)
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # cm = torch.tensor(cm) * 100
    # cm = cm.numpy()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=0, fontsize=14)
    plt.yticks(xlocations, labels, fontsize=14)
    plt.ylabel('Ground Truth', fontsize=14)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.margins(0, 0)
    plt.tick_params(which="minor", bottom=False, left=False)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=16)
    np.set_printoptions(precision=2)


def plt_tsne(test_path,x, y, config,with_T=True, train="train"):                                # 生成t-SNE聚类图像
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    #print(f"Plotting T-SNE:")
    title = f"T-SNE({train})"
    save_path = os.path.join(test_path,"t_SNE")
    os.makedirs(save_path, exist_ok=True)
    tsne_path = f'{save_path}\\{title}.png'
    plot_test, plot_testy, col, label,legends = [], [], [], [],[]
    n_test = int(0.25*len(y))
    if hasattr(config, "Transfer"):
        n_test = len(y)
    for j in range(n_test):
        plot_test.append(x[j])
        plot_testy.append(int(y[j]))
    data = np.array(x[:n_test])
    tsne = TSNE(n_components=2, init='pca').fit_transform(data)
    if hasattr(config, "Transfer"):
        cm = ['royalblue', 'red', 'green', 'orange', 'purple', 'olive', 'linen', 'orchid', 'pink', 'grey','yellow', 'cyan', 'magenta', 'brown', 'teal', 'navy', 'lime', 'maroon', 'aqua', 'fuchsia', 'silver']
        labels=config.All_Dataset
        for i in range(len(plot_testy)):
            col.append(cm[plot_testy[i]])
            legends.append(labels[plot_testy[i]])
        plt.scatter(tsne[:, 0], tsne[:, 1], alpha=0.5, s=2, facecolors="none", c=col, marker='o')
        #plot(x=tsne,y=legends,title=title)
    else:
        cm = ['royalblue', 'red', 'green', 'orange', 'purple', 'olive', 'linen', 'orchid', 'pink', 'grey', 'yellow', 'cyan', 'magenta', 'brown', 'teal', 'navy', 'lime', 'maroon', 'aqua', 'fuchsia', 'silver']
        for i in range(len(plot_testy)):
            col.append(cm[plot_testy[i]])
        plt.scatter(tsne[:, 0], tsne[:, 1], alpha=0.5, s=2, facecolors="none", c=col, marker='o')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.tight_layout()
    plt.savefig(tsne_path, dpi=600)
    #plt.show()

def plt_density(test_path,x,y,config,with_T=True, train="train"):
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    save_path = os.path.join(test_path,"t_SNE")
    os.makedirs(save_path, exist_ok=True)
    density_path = f'{save_path}\\dencity_fig_{train}.png'
    cm = ['royalblue', 'red', 'green', 'orange', 'purple', 'olive', 'linen', 'orchid', 'pink', 'grey','yellow', 'cyan', 'magenta', 'brown', 'teal', 'navy', 'lime', 'maroon', 'aqua', 'fuchsia', 'silver']
    data = np.array(x)
    y = y.flatten().astype(int)
    col_num=np.max(y)
    X_tsne = TSNE(n_components=2, init='pca').fit_transform(data)
    X_s={}
    for i in range(col_num):
        X_s[i]= X_tsne[y == (i+1)]
    X_t = X_tsne[y == 0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i in range(col_num):
        sns.kdeplot(X_s[i][:, 0], label=f'Source Domain {i+1}', ax=axes[0], fill=True, color=cm[i+1], alpha=0.4)
    sns.kdeplot(X_t[:, 0], label='Target Domain', ax=axes[0], fill=True, color=cm[0], alpha=0.4)
    axes[0].set_xlabel('First t-SNE Feature')
    axes[0].set_ylabel('Density')
    axes[0].legend(loc='upper right')

    for i in range(col_num):
        sns.kdeplot(X_s[i][:, 1], label=f'Source Domain {i+1}', ax=axes[1], fill=True, color=cm[i+1], alpha=0.4)
    sns.kdeplot(X_t[:, 1], label='Target Domain', ax=axes[1], fill=True, color=cm[0], alpha=0.4)
    axes[1].set_xlabel('Second t-SNE Feature')
    axes[1].set_ylabel('Density')
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(density_path, dpi=600)
    plt.close()

def plt_density_2(test_path,x,y,config,with_T=True, train="train"):
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    save_path = os.path.join(test_path,"t_SNE")

    os.makedirs(save_path, exist_ok=True)
    density_path = f'{save_path}\\dencity_fig_{train}.png'
    cm = ['olive', 'orchid', 'pink', 'grey', ]
    data = np.array(x)
    y = y.flatten().astype(int)
    X_tsne = TSNE(n_components=2, init='pca').fit_transform(data)
    X_s1 = X_tsne[y == 1]
    X_t = X_tsne[y == 0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.kdeplot(X_s1[:, 0], label='Target Domain All', ax=axes[0], fill=True, color=cm[1], alpha=0.4)
    sns.kdeplot(X_t[:, 0], label='Target Domain Train', ax=axes[0], fill=True, color=cm[0], alpha=0.4)
    axes[0].set_xlabel('First t-SNE Feature')
    axes[0].set_ylabel('Density')
    axes[0].legend(loc='upper right')

    sns.kdeplot(X_s1[:, 1], label='Target Domain All', ax=axes[1], fill=True, color=cm[1], alpha=0.4)
    sns.kdeplot(X_t[:, 1], label='Target Domain Train', ax=axes[1], fill=True, color=cm[0], alpha=0.4)
    axes[1].set_xlabel('Second t-SNE Feature')
    axes[1].set_ylabel('Density')
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(density_path, dpi=600)
    plt.close()

def ordered_predict(model, data_dict, config, S_soh_dict):
    """按原始顺序生成预测结果"""
    model.eval()
    all_pred = []
    all_real = []
    with torch.no_grad():
        for domain in data_dict.keys():
            # 创建未打乱的DataLoader
            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(data_dict[domain]),
                torch.from_numpy(S_soh_dict[domain])
            )
            loader = DataLoader(dataset,
                                batch_size=config.Batch_size,
                                shuffle=False)  # 关键：保持原始顺序

            for X, y in loader:
                X = X.to(config.Device)
                _,pred = model(X.float())
                pred = pred.detach().cpu().numpy()
                all_pred.extend(pred)
                all_real.extend(y.numpy())
    return np.array(all_pred), np.array(all_real)


def plt_pred_SOH(pred, real, save_path, epoch):
    """绘制预测值与真实值对比图"""
    plt.figure(figsize=(5, 4))
    os.makedirs(f"{save_path}/prediction_plots", exist_ok=True)
    plt.plot(real, label='Real Value', alpha=0.6)
    plt.plot(pred, label='Predicted Value', alpha=0.6)
    plt.xlabel('Sample Index')
    plt.ylabel('SOH Value')
    plt.title(f'SOH Prediction Comparison (Epoch {epoch+1})')
    plt.legend()
    # 自动创建目录
    #os.makedirs(f"{save_path}/prediction_plots", exist_ok=True)
    plt.savefig(f"{save_path}/prediction_plots/epoch_{epoch+1}_comparison.png")
    #plt.show()

def plt_pred_SOH_smooth(pred, real, save_path, epoch):
    """绘制平滑后的预测对比图"""
    pred_smooth=gaussian_filter1d(pred, sigma=5)
    plt.figure(figsize=(5, 4))
    os.makedirs(f"{save_path}/smoothed_plots", exist_ok=True)
    plt.plot(real, label='Real Value', alpha=0.6)
    plt.plot(pred_smooth, label='Smoothed Prediction', alpha=0.6)
    plt.xlabel('Sample Index')
    plt.ylabel('SOH Value')
    plt.title(f'Smoothed SOH Prediction (Epoch {epoch+1})')
    plt.legend()
    plt.savefig(f"{save_path}/smoothed_plots/epoch_{epoch+1}_smoothed.png")
    #plt.show()


def generate_loss(data_path, result_path):
    """生成损失曲线和最终预测图"""
    # 加载数据
    train_loss = np.loadtxt(os.path.join(data_path, "train_loss.txt"))
    # 损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label='Training Loss')
    test_loss = np.loadtxt(os.path.join(data_path, "test_loss.txt"))
    plt.plot(test_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Process')
    plt.legend()
    plt.savefig(f"{result_path}/loss_curve.png")
    plt.close()
    # 最终预测效果
    pred = np.loadtxt(os.path.join(data_path, "pred_test.txt"))
    real = np.loadtxt(os.path.join(data_path, "real_test.txt"))
    plt.figure(figsize=(12, 6))
    plt.scatter(real, pred, alpha=0.6)
    plt.plot([real.min(), real.max()], [real.min(), real.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Final Prediction vs Ground Truth')
    plt.savefig(f"{result_path}/final_prediction.png")
    plt.close()
    #训练的mmd损失
    mmd_loss = np.loadtxt(os.path.join(data_path, "mmd_loss.txt"))
    plt.figure(figsize=(12, 6))
    plt.plot(mmd_loss, label='MMD Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MMD Loss')
    plt.title('Training Process')
    plt.legend()
    plt.savefig(f"{result_path}/mmd_loss.png")
    plt.close()
    #训练与测试的rmse_loss
    train_rmse_loss = np.loadtxt(os.path.join(data_path, "train_rmse_loss.txt"))
    test_rmse_loss = np.loadtxt(os.path.join(data_path, "test_rmse_loss.txt"))
    plt.figure(figsize=(12, 6))
    plt.plot(train_rmse_loss, label='Training RMSE Loss')
    plt.plot(test_rmse_loss, label='Validation RMSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE Loss')
    plt.title('Training Process')
    plt.legend()
    plt.savefig(f"{result_path}/rmse_loss.png")
    plt.close()


def plt_train_and_test_soh_fig(model,S_data_dict,S_soh_dict,dict_S_cell_cycles,T_cell_cycles,y_pred_test,y_real_test,dict_pred_S,dict_real_S,config,_):
    ordered_pred_train, ordered_real_train = ordered_predict(model, S_data_dict, config, S_soh_dict)

    for S_domain in S_data_dict.keys():
        start=0
        for cell in range(len(dict_S_cell_cycles[S_domain])):
            end = start + dict_S_cell_cycles[S_domain][cell]
            Result_path = (
                f"{config.Result_path}/test_{config.test_id}/without_T_train/soh_curve/S_domain_{S_domain}/Trainid_{cell + 1}_B{config.Batch_size}_E{config.N_epoch}_L{config.new_length}")
            filter_dir = os.path.join(Result_path, "soh_pred_fig")
            #os.makedirs(filter_dir, exist_ok=True)
            save_path = os.path.join(filter_dir, f"soh_pred_fig_{_ + 1}.png")
            plt_pred_SOH_smooth(dict_pred_S[S_domain][start: end], dict_real_S[S_domain][start: end], Result_path, _)
            #plt_pred_SOH(dict_pred_S[S_domain][start: end], dict_real_S[S_domain][start: end], Result_path, _)
            # 在已创建的 filter_dir 目录下保存图片
            #plt.savefig(save_path,dpi=300,bbox_inches='tight',format='png',transparent=False)
            plt.close()  # 释放内存
            start=end
    start = 0
    for cell in range(len(T_cell_cycles)):
        end = start + T_cell_cycles[cell]
        Result_path = (
            f"{config.Result_path}/test_{config.test_id}/without_T_train/soh_curve/T_domain/Testid_{cell + 1}_B{config.Batch_size}_E{config.N_epoch}_L{config.new_length}")
        filter_dir = os.path.join(Result_path, "soh_pred_fig")
        #os.makedirs(filter_dir, exist_ok=True)  # 确保 soh_pred_fig 子目录存在
        save_path = os.path.join(filter_dir, f"soh_pred_fig_{_ + 1}.png")
        plt_pred_SOH_smooth(y_pred_test[start: end], y_real_test[start: end], Result_path, _)
        # 在已创建的 filter_dir 目录下保存图片
        #plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png', transparent=False)
        plt.close()  # 释放内存
        start=end

def plt_train_and_test_soh_fig_2(model,S_data_dict,S_soh_dict,dict_S_cell_cycles,T_test_cell_cycles,T_train_cell_cycles,y_pred_test,y_real_test,dict_pred_S,dict_real_S,pred_T_train,real_T_train,config,test_path,_):
    #ordered_pred_train, ordered_real_train = ordered_predict(model, S_data_dict, config, S_soh_dict)
    for S_domain in S_data_dict.keys():
        start=0
        for cell in range(len(dict_S_cell_cycles[S_domain])):
            end = start + dict_S_cell_cycles[S_domain][cell]
            Result_path = os.path.join(test_path,
                f"soh_curve/S_domain_{S_domain}/Trainid_{cell + 1}_B{config.Batch_size}_E{config.N_epoch}_L{config.new_length}")
            filter_dir = os.path.join(Result_path, "soh_pred_fig")
            #os.makedirs(filter_dir, exist_ok=True)
            save_path = os.path.join(filter_dir, f"soh_pred_fig_{_ + 1}.png")
            plt_pred_SOH(dict_pred_S[S_domain][start: end], dict_real_S[S_domain][start: end], Result_path, _)
            #plt_pred_SOH(dict_pred_S[S_domain][start: end], dict_real_S[S_domain][start: end], Result_path, _)
            # 在已创建的 filter_dir 目录下保存图片
            #plt.savefig(save_path,dpi=300,bbox_inches='tight',format='png',transparent=False)
            plt.close()  # 释放内存
            start=end
    start = 0
    for cell in range(len(T_test_cell_cycles)):
        end = start + T_test_cell_cycles[cell]
        Result_path = os.path.join(test_path,
            f"soh_curve/T_domain/Testid_{cell + 1}_B{config.Batch_size}_E{config.N_epoch}_L{config.new_length}")
        filter_dir = os.path.join(Result_path, "soh_pred_fig")
        #os.makedirs(filter_dir, exist_ok=True)  # 确保 soh_pred_fig 子目录存在
        save_path = os.path.join(filter_dir, f"soh_pred_fig_{_ + 1}.png")
        plt_pred_SOH(y_pred_test[start: end], y_real_test[start: end], Result_path, _)
        # 在已创建的 filter_dir 目录下保存图片
        #plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png', transparent=False)
        plt.close()  # 释放内存
        start=end
    start = 0
    for cell in range(len(T_train_cell_cycles)):
        end = start + T_train_cell_cycles[cell]
        Result_path = os.path.join(test_path,
            f"soh_curve/T_domain/Trainid_{cell + 1}_B{config.Batch_size}_E{config.N_epoch}_L{config.new_length}")
        filter_dir = os.path.join(Result_path, "soh_pred_fig")
        #os.makedirs(filter_dir, exist_ok=True)  # 确保 soh_pred_fig 子目录存在
        save_path = os.path.join(filter_dir, f"soh_pred_fig_{_ + 1}.png")
        plt_pred_SOH(pred_T_train[start: end], real_T_train[start: end], Result_path, _)
        # 在已创建的 filter_dir 目录下保存图片
        #plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png', transparent=False)
        plt.close()  # 释放内存
        start=end

def plt_train_and_test_soh_fig_3(model,T_test_cell_cycles,T_train_cell_cycles,y_pred_test,y_real_test,pred_T_train,real_T_train,config,_):
    #ordered_pred_train, ordered_real_train = ordered_predict(model, S_data_dict, config, S_soh_dict)
    start = 0
    for cell in range(len(T_test_cell_cycles)):
        end = start + T_test_cell_cycles[cell]
        Result_path = (
            f"{config.Result_path}/test_{config.test_id}/without_S_train/soh_curve/T_domain/Testid_{cell + 1}_B{config.Batch_size}_E{config.N_epoch}_L{config.new_length}")
        filter_dir = os.path.join(Result_path, "soh_pred_fig")
        #os.makedirs(filter_dir, exist_ok=True)  # 确保 soh_pred_fig 子目录存在
        save_path = os.path.join(filter_dir, f"soh_pred_fig_{_ + 1}.png")
        plt_pred_SOH(y_pred_test[start: end], y_real_test[start: end], Result_path, _)
        # 在已创建的 filter_dir 目录下保存图片
        #plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png', transparent=False)
        plt.close()  # 释放内存
        start=end
    start = 0
    for cell in range(len(T_train_cell_cycles)):
        end = start + T_train_cell_cycles[cell]
        Result_path = (
            f"{config.Result_path}/test_{config.test_id}/without_S_train/soh_curve/T_domain/Trainid_{cell + 1}_B{config.Batch_size}_E{config.N_epoch}_L{config.new_length}")
        filter_dir = os.path.join(Result_path, "soh_pred_fig")
        #os.makedirs(filter_dir, exist_ok=True)  # 确保 soh_pred_fig 子目录存在
        save_path = os.path.join(filter_dir, f"soh_pred_fig_{_ + 1}.png")
        plt_pred_SOH(pred_T_train[start: end], real_T_train[start: end], Result_path, _)
        # 在已创建的 filter_dir 目录下保存图片
        #plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png', transparent=False)
        plt.close()  # 释放内存
        start=end

def generate_explanation(save_path, config):
    learning_rate = config.Learning_rate
    batch_size = config.Batch_size
    n_epoch = config.N_epoch
    weight_mmd=config.weight_MMD
    os.makedirs(os.path.join(save_path, 'description'), exist_ok=True)
    np.savetxt(os.path.join(save_path, 'description',"learning_rate.txt"),learning_rate)
    np.savetxt(os.path.join(save_path, 'description',"batch_size.txt"),batch_size)
    np.savetxt(os.path.join(save_path, 'description',"n_epoch.txt"),n_epoch)
    np.savetxt(os.path.join(save_path, 'description',"weight_mmd.txt"),weight_mmd)

