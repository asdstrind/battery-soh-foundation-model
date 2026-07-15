import os
import torch
import numpy as np
from tqdm import *
from itertools import cycle
from itertools import chain
from scipy.ndimage import gaussian_filter1d
from .config import calculate_loss, Person_Loss, feature_Person_Loss, MMD_loss, Cosine_Loss
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from .utils import plot

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


def train(config, S_loader,  T_loader,  model, optimizer): #模型训练算法
    model.train()
    MMD = MMD_loss()
    person_loss_fn = Person_Loss()
    cosine_Loss = Cosine_Loss()
    feature_Person_Loss_fn = feature_Person_Loss()
    y_pred_S, y_real_S, Feature_S, Feature_T, train_loss, rmse_train_loss, mse_train_loss, mae_train_loss, r2_train_loss,mmd_loss = [], [], [], [], 0, 0, 0, 0, 0,0
    data_s_iter = iter(S_loader)                 # 将源域与目标域数据转换为可迭代对象
    data_t_iter = cycle(T_loader)#创建无限循环的迭代器，当T_dataloader遍历完后自动从头开始保证在源域数据遍历过程中，目标域数据能持续供应适用于源域和目标域数据量不一致的情况
    len_dataloader = len(S_loader)
    for i in range(len_dataloader):
        loss_MMD=0
        loss=0# 遍历训练集与测试集
        X_s, Y_s = next(data_s_iter)                                                            # 获取源域数据的数据（X_s）与标签（y_s）                                   # 获取源域数据的数据（X_s）与标签（y_s）
        X_t, _ = next(data_t_iter)                                                              # 获取源域数据的数据（X_t）
        y_real_S = np.append(y_real_S, Y_s)                                                     # labels用于存放源域数据的标签
        X_s, X_t, Y_s = X_s.to(config.Device), X_t.to(config.Device), Y_s.to(config.Device)     # 将源域与目标域放入device中
        # 将源域和目标域放入模型中进行预测， 并将获得的特征以及预测值进行保存。
        feature_S, pred_S = model(X_s.float())
        feature_T, pred_T = model(X_t.float())

        assert feature_S.requires_grad, "feature_S未保留梯度！"
        assert feature_T.requires_grad, "feature_T未保留梯度！"
        # 计算损失
        rmse_train = config.Loss(pred_S, Y_s)
        # 这里的loss是最终的损失函数，可以根据需要添加其他损失函数

        #loss = rmse_train
        if config.Transfer is True:
            loss_MMD = MMD(feature_T, feature_S)
        loss = config.weight_MMD * loss_MMD+rmse_train
        # 优化器梯度清零、反向传播、更新参数

        optimizer.zero_grad(),
        loss.backward(),
        """
        # 全局梯度检查    
        # ---- 全局梯度检查 ----
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"⚠️ 参数 {name} 未接收梯度")
            else:
                grad_mean = param.grad.abs().mean().item()
                print(f"✅ {name.ljust(30)} 梯度均值: {grad_mean:.6f}")
        """
        optimizer.step()

        mse_train, mae_train, r2_train = calculate_loss(pred_S, Y_s)
        with torch.no_grad():
            y_pred_S = np.append(y_pred_S, pred_S.detach().cpu().numpy())
        # 需要改进
            Feature_S = np.append(Feature_S, feature_S.detach().cpu().numpy())
            Feature_T = np.append(Feature_T, feature_T.detach().cpu().numpy())
        train_loss += loss.detach().cpu().numpy().mean()  # 用train_loss存储总损失（每次iter计算平均）
        mmd_loss += loss_MMD.detach().cpu().numpy().mean()
        rmse_train_loss += rmse_train.detach().cpu().numpy().mean()  # 用train_loss存储总损失（每次iter计算平均）
        mse_train_loss += mse_train.detach().cpu().numpy().mean()  # 用train_loss存储总损失（每次iter计算平均）
        mae_train_loss += mae_train.detach().cpu().numpy().mean()  # 用train_loss存储总损失（每次iter计算平均）
        r2_train_loss += r2_train.detach().cpu().numpy().mean()  # 用train_loss存储总损失（每次iter计算平均）

    train_loss /= (i + 1)
    mmd_loss/= (i + 1)
    rmse_train_loss /= (i + 1)
    mse_train_loss /= (i + 1)
    mae_train_loss /= (i + 1)
    r2_train_loss /= (i + 1)
    return y_pred_S, y_real_S, Feature_S.reshape(-1, config.out_feature), Feature_T.reshape(-1, config.out_feature), train_loss, rmse_train_loss, mse_train_loss, mae_train_loss, r2_train_loss,mmd_loss


def test(config, T_loader, model):
    model.train(False)  # 模型测试
    person_loss_fn = Person_Loss()
    feature_Person_Loss_fn = feature_Person_Loss()
    y_real, y_pred, Feature_T, test_loss, rmse_test_loss, mse_test_loss, mae_test_loss, r2_test_loss = [], [], [], 0, 0, 0, 0, 0    # 设置基本变量存储部分数据，用于后续的结果输出
    with torch.no_grad():
        for i, (X, y) in enumerate(T_loader):  # 遍历测试loader
            y_real = np.append(y_real, y)
            X, y = X.to(config.Device).float(), y.to(config.Device)  # 将测试数据（X）、标签（y）放入device
            feature, pred = model(X)  # 数据放入模型中，输出预测结果(pred)
            y_pred = np.append(y_pred, pred.detach().cpu().numpy())
            Feature_T = np.append(Feature_T, feature.detach().cpu().numpy())  # 用features存储特征
            rmse_test = config.Loss(pred, y)
            person_loss = person_loss_fn(pred, y)
            feature_person_Loss = feature_Person_Loss_fn(feature, y)
            # L1 正则化损失
            # l1_reg = sum(torch.sum(torch.abs(param)) for param in model.parameters())
            if config.select_method == 'Person_Loss':
                loss = 0.4 * rmse_test + 0.3 * person_loss + 0.3 * feature_person_Loss
            else:
                loss = 1 * rmse_test
            mse_test, mae_test, r2_test = calculate_loss(pred, y)
            rmse_test_loss += rmse_test.detach().cpu().numpy().mean()  # 用train_loss存储总损失（每次iter计算平均）
            mse_test_loss += mse_test.detach().cpu().numpy().mean()  # 用train_loss存储总损失（每次iter计算平均）
            mae_test_loss += mae_test.detach().cpu().numpy().mean()  # 用train_loss存储总损失（每次iter计算平均）
            r2_test_loss += r2_test.detach().cpu().numpy().mean()  # 用train_loss存储总损失（每次iter计算平均）
            test_loss += loss.detach().cpu().numpy().mean()  # 计算预测损失loss
        test_loss /= (i + 1)
        rmse_test_loss /= (i + 1)
        mse_test_loss /= (i + 1)
        mae_test_loss /= (i + 1)
        r2_test_loss /= (i + 1)
    return y_pred, y_real, Feature_T.reshape(-1, config.out_feature), test_loss, rmse_test_loss, mse_test_loss, mae_test_loss, r2_test_loss   # 3_test_unknown_data()函数返回平均损失、准确率、模型提取到的特征、真实标签，用于后续进一步说明


# 模型训练、测试结合算法
def model_train(config, S_loader,  T_loader, model, optimizer, scheduler, T_cell_cycles,S_data, T_data, test_id):  # 模型总算法，包含训练及测试
    model_dir = os.path.join(config.Result_path, f'test_{test_id}', 'Models')
    os.makedirs(model_dir, exist_ok=True)  # exist_ok=True 防止目录已存在时报错
    model_path = os.path.join(model_dir, f"model.pt")  # 保存模型参数路径
    loss_path = os.path.join(config.Result_path, f'test_{test_id}', 'Loss')
    os.makedirs(loss_path, exist_ok=True)  # exist_ok=True 防止目录已存在时报错
    print(f"Model will be saved to {model_path}")
    if os.path.exists(model_path):  # 如已有参数相同训练模型，则直接进行测试环节，不再进行训练
        checkpoint = torch.load(model_path, map_location=torch.device(config.Device))  # 选择模型，用于后续导入参数
        model.load_state_dict(checkpoint['net_state_dict'])  # 模型导入参数
        model.eval()  # 用于测试阶段关闭BN和Dropout，以免影响结果
        y_pred_test, y_real_test, feature_T_test, loss_test, rmse_test, mse_test, mae_test, r2_test = test(config, T_loader, model)  # 获取到测试的返回值
        print(f'Test Error:\n\tTest_loss: {loss_test:>0.4f}')  # 输出基础信息（测试损失值）
    else:  # 没有已训练模型，则进行训练
        train_loss, test_loss, pred_train, real_train, pred_test, real_test = [], [], [], [], [], []
        RMSE_train, MSE_train, MAE_train, R2_train, RMSE_test, MSE_test, MAE_test, R2_test = [], [], [], [], [], [], [], []
        MMD_train=[]
        Soure_Feature, Target_Feature = [], []
        with trange(config.N_epoch) as t:  # 使用trange模块，使输出界面更易读
            for _ in t:
                torch.autograd.set_detect_anomaly(True)
                y_pred_train, y_real_train, feature_S_train, feature_T_train, loss_train, rmse_train, mse_train, mae_train, r2_train,mmd_train = train(config, S_loader,  T_loader, model, optimizer)
                y_pred_test, y_real_test, feature_T_test, loss_test, rmse_test, mse_test, mae_test, r2_test = test(config, T_loader, model)
                # 学习率调度
                scheduler.step()
                # 保存MSE、MAE、R2训练损失
                RMSE_train = np.append(RMSE_train, rmse_train)
                MSE_train = np.append(MSE_train, mse_train)
                MAE_train = np.append(MAE_train, mae_train)
                R2_train = np.append(R2_train, r2_train)
                RMSE_test = np.append(RMSE_test, rmse_test)
                MSE_test = np.append(MSE_test, mse_test)
                MAE_test = np.append(MAE_test, mae_test)
                R2_test = np.append(R2_test, r2_test)
                MMD_train = np.append(MMD_train, mmd_train)
                # 保存RMSE训练损失
                train_loss = np.append(train_loss, loss_train)
                pred_train = np.append(pred_train, y_pred_train)
                real_train = np.append(real_train, y_real_train)
                test_loss = np.append(test_loss, loss_test)
                pred_test = np.append(pred_test, y_pred_test)
                real_test = np.append(real_test, y_real_test)
                t.set_description(f"Train")
                t.set_postfix(lr=f'lr:{scheduler.get_last_lr()[0]}', Loss=f"[{train_loss.mean():>0.5f}\t{test_loss.mean():>0.5f}]")
                if (_+1) % (config.N_epoch//4) == 0:
                    start = 0
                    for cell in range(len(T_cell_cycles)):
                        end = start + T_cell_cycles[cell]
                        Result_path = (f"{config.Result_path}/test_{test_id}/train_soh_curve/Testid_{cell+1}_B{config.Batch_size}_E{config.N_epoch}_L{config.new_length}")
                        filter_dir = os.path.join(Result_path, "soh_pred_fig")
                        os.makedirs(filter_dir, exist_ok=True)  # 确保 soh_pred_fig 子目录存在
                        save_path = os.path.join(filter_dir, f"soh_pred_fig_{_ + 1}.png")
                        plt_pred_SOH(y_pred_test[start: end], y_real_test[start: end], Result_path, _)
                        plt.savefig(
                            save_path,
                            dpi=300,  # 提高分辨率
                            bbox_inches='tight',  # 去除多余白边
                            format='png',  # 指定保存格式
                            transparent=False  # 背景不透明
                        )
                        plt.close()
                        pred_test_1 = gaussian_filter1d(y_pred_test[start: end], sigma=1)
                        plt_pred_SOH_smooth(pred_test_1, y_real_test[start: end], Result_path, _)
                        start = end
                if (_+1)%(config.N_epoch//4)==0 or _==0:

                    X_combined = []
                    domain_labels = []
                    Result_path = (f"{config.Result_path}/test_{test_id}/tsne_result")
                    x = S_data
                    X_combined.append(x)  # 将所有源域数据合并
                    domain_labels.extend(['Source'] * len(x))
                    x=T_data
                    X_combined.append(x)  # 将目标域数据合并
                    domain_labels.extend(['Target'] * len(x))
                    X_combined = np.vstack(X_combined)
                    batch_size=config.Batch_size
                    features=[]
                    preds=[]
                    model.eval()
                    with torch.no_grad():
                        for i in range(0, len(X_combined), batch_size):
                            batch =torch.from_numpy( X_combined[i:i + batch_size]).to(config.Device)
                            #print(batch.shape)
                            f, p = model(batch)
                            features.append(f.cpu())
                            preds.append(p.cpu())# 移回CPU
                            del batch, f, p
                            torch.cuda.empty_cache()
                    features=torch.cat(features,dim=0)
                    preds=torch.cat(preds,dim=0)
                    preds=preds.numpy()
                    features=features.numpy()

                    #print(features.shape)

                    #X_combined=torch.from_numpy(X_combined).to(config.Device)
                    #feature,pred=model(X_combined)

                    X_scaled = StandardScaler().fit_transform(features)
                    tsne = TSNE(
                        perplexity=40,
                        metric="euclidean",
                        init="pca",
                        early_exaggeration=24,
                        n_jobs=-1
                    )
                    embedding = tsne.fit_transform(X_scaled)
                    plot(embedding, domain_labels)
                    filter_dir = os.path.join(Result_path, "tsne_fig")
                    os.makedirs(filter_dir, exist_ok=True)  # 确保 tsne_fig 子目录存在
                    # 在已创建的 filter_dir 目录下保存图片
                    save_path = os.path.join(filter_dir, f"tsne_{_+1}.png")
                    plt.savefig(
                        save_path,
                        dpi=300,  # 提高分辨率
                        bbox_inches='tight',  # 去除多余白边
                        format='png',  # 指定保存格式
                        transparent=False  # 背景不透明
                    )
                    plt.close()  # 释放内存
                if (_ +1) == config.N_epoch:
                    Soure_Feature = np.append(Soure_Feature, feature_S_train)
                    Target_Feature = np.append(Target_Feature, feature_T_train)
        # 计算训练产生的各个损失的最后30轮损失的平均值
        train_loss_last_30_avg = np.mean(np.array(train_loss[-config.loss_epoch:], dtype=np.float32))
        RMSE_train_loss_last_30_avg = np.mean(np.array(RMSE_train[-config.loss_epoch:], dtype=np.float32))
        MSE_train_loss_last_30_avg = np.mean(np.array(MSE_train[-config.loss_epoch:], dtype=np.float32))
        MAE_train_loss_last_30_avg = np.mean(np.array(MAE_train[-config.loss_epoch:], dtype=np.float32))
        R2_train_loss_last_30_avg = np.mean(np.array(R2_train[-config.loss_epoch:], dtype=np.float32))
        # 计算测试产生的各个损失的最后30轮损失的平均值
        test_loss_last_30_avg = np.mean(np.array(test_loss[-config.loss_epoch:], dtype=np.float32))
        RMSE_test_loss_last_30_avg = np.mean(np.array(RMSE_test[-config.loss_epoch:], dtype=np.float32))
        MSE_test_loss_last_30_avg = np.mean(np.array(MSE_test[-config.loss_epoch:], dtype=np.float32))
        MAE_test_loss_last_30_avg = np.mean(np.array(MAE_test[-config.loss_epoch:], dtype=np.float32))
        R2_test_loss_last_30_avg = np.mean(np.array(R2_test[-config.loss_epoch:], dtype=np.float32))
        # 保存训练损失和训练标签
        np.savetxt(f"{loss_path}/source_feature.txt", Soure_Feature)  # 保存训练损失
        np.savetxt(f"{loss_path}/train_loss.txt", train_loss)  # 保存训练损失
        np.savetxt(f"{loss_path}/RMSE_train_loss.txt", RMSE_train)  # 保存训练RMSE损失
        np.savetxt(f"{loss_path}/MMD_train_loss.txt", MMD_train)  # 保存训练MMD损失
        np.savetxt(f"{loss_path}/MSE_train_loss.txt", MSE_train)  # 保存训练MSE损失
        np.savetxt(f"{loss_path}/MAE_train_loss.txt", MAE_train)  # 保存训练MAE损失
        np.savetxt(f"{loss_path}/R2_train_loss.txt", R2_train)  # 保存训练R2损失
        np.savetxt(f"{loss_path}/pred_train.txt", pred_train)  # 保存训练预测标签
        np.savetxt(f"{loss_path}/real_train.txt", real_train)  # 保存训练真实标签
        # 保存测试损失及测试标签
        np.savetxt(f"{loss_path}/target_feature.txt", Target_Feature)  # 保存测试损失
        np.savetxt(f"{loss_path}/test_loss.txt", test_loss)  # 保存测试损失
        np.savetxt(f"{loss_path}/RMSE_test_loss.txt", RMSE_test)  # 保存测试RMSE损失
        np.savetxt(f"{loss_path}/MSE_test_loss.txt", MSE_test)  # 保存测试MSE损失
        np.savetxt(f"{loss_path}/MAE_test_loss.txt", MAE_test)  # 保存测试MAE损失
        np.savetxt(f"{loss_path}/R2_test_loss.txt", R2_test)  # 保存测试R2损失
        np.savetxt(f"{loss_path}/pred_test.txt", pred_test)  # 保存测试预测标签
        np.savetxt(f"{loss_path}/real_test.txt", real_test)  # 保存测试真实标签
        generate_loss(f"{loss_path}/pred_test.txt", f"{loss_path}/real_test.txt", config.Result_path,
                      f"{loss_path}/train_loss.txt", f"{loss_path}/test_loss.txt")  # 生成训练过程损失和准确率图
        mmd_train_loss = np.loadtxt(os.path.join(loss_path, 'mmd_train_loss.txt'))
        plt.figure(figsize=(12, 6))
        plt.plot(mmd_train_loss, label='MMD Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Process')
        plt.legend()
        plt.savefig(f"{config.Result_path}/loss_curve.png")
        plt.close()
        # plt_pred_test_smooth_SOH(f"{loss_path}/pred_test.txt", f"{loss_path}/real_test.txt", config.N_epoch)

        train_loss = np.loadtxt(os.path.join(loss_path, 'train_loss.txt'))
        test_loss = np.loadtxt(os.path.join(loss_path, 'test_loss.txt'))
        RMSE_train_loss = np.loadtxt(os.path.join(loss_path, 'RMSE_train_loss.txt'))
        mmd_train_loss = train_loss - RMSE_train_loss
        plt.figure(figsize=(12, 6))
        plt.plot(mmd_train_loss, label='MMD Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Process')
        plt.legend()
        plt.savefig(f"{config.Result_path}/test_{test_id}/MMD_loss_curve.png")
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(train_loss, label='Training Loss')
        plt.plot(test_loss, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Process')
        plt.legend()
        plt.savefig(f"{config.Result_path}/test_{test_id}/loss_curve.png")
        plt.close()

        RMSE_train_loss = np.loadtxt(os.path.join(loss_path, 'RMSE_train_loss.txt'))
        RMSE_test_loss = np.loadtxt(os.path.join(loss_path, 'RMSE_test_loss.txt'))
        plt.figure(figsize=(12, 6))
        plt.plot(RMSE_train_loss, label='RMSE Training Loss')
        plt.plot(RMSE_test_loss, label='RMSE Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Process')
        plt.legend()
        plt.savefig(f"{config.Result_path}/test_{test_id}/RMSE_loss_curve.png")
        plt.close()

        #plt_pred_test_smooth_SOH(f"{loss_path}/pred_test.txt", f"{loss_path}/real_test.txt", config.N_epoch)
        # 训练集损失
        with open(f"{loss_path}/train_loss.txt", 'a') as f:
            f.write(f"Train Loss Last {config.loss_epoch} Avg: {train_loss_last_30_avg:.6f}\n")
        with open(f"{loss_path}/RMSE_train_loss.txt", 'a') as f:
            f.write(f"Train Loss Last {config.loss_epoch} Avg: {RMSE_train_loss_last_30_avg:.6f}\n")
        with open(f"{loss_path}/MSE_train_loss.txt", 'a') as f:
            f.write(f"Train Loss Last {config.loss_epoch} Avg: {MSE_train_loss_last_30_avg:.6f}\n")
        with open(f"{loss_path}/MAE_train_loss.txt", 'a') as f:
            f.write(f"Train Loss Last {config.loss_epoch} Avg: {MAE_train_loss_last_30_avg:.6f}\n")
        with open(f"{loss_path}/R2_train_loss.txt", 'a') as f:
            f.write(f"Train Loss Last {config.loss_epoch} Avg: {R2_train_loss_last_30_avg:.6f}\n")
        # 测试集损失
        with open(f"{loss_path}/test_loss.txt", 'a') as f:
            f.write(f"Test Loss Last {config.loss_epoch} Avg: {test_loss_last_30_avg:.6f}\n")
        with open(f"{loss_path}/RMSE_test_loss.txt", 'a') as f:
            f.write(f"Test Loss Last {config.loss_epoch} Avg: {RMSE_test_loss_last_30_avg:.6f}\n")
        with open(f"{loss_path}/MSE_test_loss.txt", 'a') as f:
            f.write(f"Test Loss Last {config.loss_epoch} Avg: {MSE_test_loss_last_30_avg:.6f}\n")
        with open(f"{loss_path}/MAE_test_loss.txt", 'a') as f:
            f.write(f"Test Loss Last {config.loss_epoch} Avg: {MAE_test_loss_last_30_avg:.6f}\n")
        with open(f"{loss_path}/R2_test_loss.txt", 'a') as f:
            f.write(f"Test Loss Last {config.loss_epoch} Avg: {R2_test_loss_last_30_avg:.6f}\n")
        torch.save({'net_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                   model_path)  # 训练结束后保存模型
        print(f'Finish {config.N_epoch} epoch train')



