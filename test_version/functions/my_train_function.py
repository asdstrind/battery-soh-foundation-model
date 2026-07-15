import os
import torch
import numpy as np
from config import Config
from tqdm import *
from itertools import cycle
from itertools import chain
from scipy.ndimage import gaussian_filter1d
from .config import calculate_loss, Person_Loss, feature_Person_Loss, MMD_loss, Cosine_Loss
from .config import MMD_loss_2 as MMD
#from e_basic_setting import generate_loss, plt_pred_SOH, plt_pred_test_smooth_SOH, plt_pred_SOH_smooth
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader, ConcatDataset
#from openTSNE import TSNE
#from tsnecuda import TSNE
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from .utils import plot
import time
from .load_data_function import generate_dataset,load_data
import itertools
from torch.utils.data import ChainDataset


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
    plt.figure(figsize=(12, 6))
    plt.plot(real, label='Real Value', alpha=0.6)
    plt.plot(pred, label='Predicted Value', alpha=0.6)
    plt.xlabel('Sample Index')
    plt.ylabel('SOH Value')
    plt.title(f'SOH Prediction Comparison (Epoch {epoch+1})')
    plt.legend()
    # 自动创建目录
    #os.makedirs(f"{save_path}/prediction_plots", exist_ok=True)
    #plt.savefig(f"{save_path}/prediction_plots/epoch_{epoch+1}_comparison.png")
    #plt.show()

def plt_pred_SOH_smooth(pred_smooth, real, save_path, epoch):
    """绘制平滑后的预测对比图"""
    plt.figure(figsize=(12, 6))
    plt.plot(real, label='Real Value', alpha=0.6)
    plt.plot(pred_smooth, label='Smoothed Prediction', alpha=0.6)
    plt.xlabel('Sample Index')
    plt.ylabel('SOH Value')
    plt.title(f'Smoothed SOH Prediction (Epoch {epoch+1})')
    plt.legend()
    #os.makedirs(f"{save_path}/smoothed_plots", exist_ok=True)
    #plt.savefig(f"{save_path}/smoothed_plots/epoch_{epoch+1}_smoothed.png")
    #plt.show()

def generate_loss( pred_path, real_path, result_path,train_loss_path, test_loss_path=None):
    """生成损失曲线和最终预测图"""
    # 加载数据
    train_loss = np.loadtxt(train_loss_path)



    # 损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label='Training Loss')

    test_loss = np.loadtxt(test_loss_path)
    plt.plot(test_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Process')
    plt.legend()
    plt.savefig(f"{result_path}/loss_curve.png")
    plt.close()

    # 最终预测效果
    pred = np.loadtxt(pred_path)
    real = np.loadtxt(real_path)
    plt.figure(figsize=(12, 6))
    plt.scatter(real, pred, alpha=0.6)
    plt.plot([real.min(), real.max()], [real.min(), real.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Final Prediction vs Ground Truth')
    plt.savefig(f"{result_path}/final_prediction.png")
    plt.close()


def train(config,dict_S_loader,  model, optimizer,T_test_loader):
    """训练模型"""
    model.train()
    MMD=MMD_loss()
    data_T_iter=iter(T_test_loader)
    dict_S_iter={}
    for S_domain, loader in dict_S_loader.items():
        dict_S_iter[S_domain] = iter(loader)
    len_dataloader=max(len(loader) for loader in dict_S_iter.values())
    for S_domain, iter in dict_S_iter.items():
        dict_S_iter[S_domain] = cycle(iter)
    data_T_iter= cycle(data_T_iter)
    for i in range(len_dataloader):
        X_t, _ = next(data_T_iter)
        X_t = X_t.to(config.Device)
        for S_domain, iter in dict_S_iter.items():
            X_s, y_s = next(iter)
            X_s = X_s.to(config.Device)
            y_s = y_s.to(config.Device)
