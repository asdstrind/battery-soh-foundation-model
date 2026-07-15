import numpy as np
from tqdm import *
from .basic_algorithm import *
import torch
from itertools import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(config, dict_S_loader,indexed_T_train_loader,indexed_T_test_loader, model, optimizer):  # 模型训练算法
    model.train()
    MMD=MMD_loss()# 实例化MMD_Loss类，用于后续计算
    train_loss,mmd_loss,rmse_loss ,features_s,features_t, labels_s =  0,0,0, [],[],[]  # 设置基本变量存储部分数据，用于后续的结果输出
    data_t_train_iter=cycle(indexed_T_train_loader)
    indexed_T_loader=chain(indexed_T_train_loader,indexed_T_test_loader)
    data_t_iter=cycle(indexed_T_loader)
    data_s_iter = {
        S_domain: cycle(iter(loader))
        for S_domain, loader in dict_S_loader.items()
    }
    len_dataloader = len(indexed_T_train_loader)

    real_T_train=np.full(len(indexed_T_train_loader.dataset), np.nan)
    pred_T_train=np.full(len(indexed_T_train_loader.dataset), np.nan)

    dict_real_S = {
        S_domain: np.full(len(loader.dataset), np.nan)
        for S_domain, loader in dict_S_loader.items()
    }
    dict_pred_S = {
        S_domain: np.full(len(loader.dataset), np.nan)
        for S_domain, loader in dict_S_loader.items()
    }

    for i in range(len_dataloader):
        loss_mmd=0
        dict_feature_S={}

        X_t_train, Y_t_train, batch_indices = next(data_t_train_iter)
        X_t_train = X_t_train.to(device)
        Y_t_train = Y_t_train.to(device)
        batch_indices = batch_indices.cpu().numpy().astype(int)
        _, pred_t_train = model(X_t_train)
        real_T_train[batch_indices] = Y_t_train.detach().cpu().numpy().flatten()
        pred_T_train[batch_indices] = pred_t_train.detach().cpu().numpy().flatten()
        RMSE_T_loss = config.Loss(pred_t_train.to(device), Y_t_train)
        X_t,Y_t,_=next(data_t_iter)
        X_t = X_t.to(device)
        Y_t = Y_t.to(device)
        feature_fc_t,pred_t=model(X_t)

        for idx, (S_domain, loader) in enumerate(dict_S_loader.items()):
            X_s, Y_s, batch_indices_2 = next(data_s_iter[S_domain])
            batch_indices_2 = batch_indices_2.cpu().numpy().astype(int)
            labels_s = np.append(labels_s, int(idx + 1) * np.ones(len(X_s)))  # labels用于存放源域数据的标签
            X_s, Y_s = X_s.to(device), Y_s.to(device)  # 将源域与目标域放入device中
            bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
            for m in bn_layers:
                m.eval()
            feature_fc_s, pred_s = model(X_s)
            # 不要忘了将 BN 恢复到 train（如果你需要继续训练）
            for m in bn_layers:
                m.train()
            dict_real_S[S_domain][batch_indices_2] = Y_s.detach().cpu().numpy().flatten()
            dict_pred_S[S_domain][batch_indices_2] = pred_s.detach().cpu().numpy().flatten()
            dict_feature_S[S_domain] = feature_fc_s  # 将源域数据放入模型中，输出预测标签及指定位置学习到的获取到的特征，用于计算MMD_Loss                             # 将目标域数据放入模型中，输出指定位置学习到的特征，用于计算MMD_Loss
            rmse_train = config.Loss(pred_s.to(device), Y_s)
            #length_rmse += 1
            #RMSE_loss = rmse_train + RMSE_loss  # 如果采用迁移学习方法，则加入MMD_Loss，否则loss仅为标签损失
            mmd = MMD(feature_fc_s, feature_fc_t)
            #length_mmd += 1  # 计算源域与目标域从卷积层出来的第二个全连接特征MMD_Loss
            loss_mmd = mmd + loss_mmd




        loss = RMSE_T_loss                                                    # 计算总损失
        train_loss += loss.detach().cpu().numpy().mean()
        rmse_loss+=(RMSE_T_loss).detach().cpu().numpy().mean()
        optimizer.zero_grad(), loss.backward(), optimizer.step()                                # 梯度清零、反向传播、梯度更新经典三步骤                     # 计算分类准确率：如果perd和y一致的,则corret+1
    train_loss /= (i + 1)
    rmse_loss/= (i + 1)                                                                                  # 计算平均损失（除以iter的次数即可获得平均损失）
    return train_loss,rmse_loss,pred_T_train,real_T_train               # train()函数返回平均损失、准确率、模型提取到的特征、真实标签，用于后续进一步说明



def test(config, test_loader, model, epoch):    # 模型测试算法
    model.train(False)                                                                          # 模型测试
    y_real, y_pred, features, test_loss, correct, batches, size = [], [], [], 0, 0, len(test_loader), len(test_loader.dataset)  # 设置基本变量存储部分数据，用于后续的结果输出
    with torch.no_grad():
        for batch, (X, y, _) in enumerate(test_loader):                                            # 遍历测试集
            X, y = X.to(device), y.to(device)                                                   # 将测试数据（X）、标签（y）放入device
            feature_2,pred = model(X)                              # 将测试数据放入模型中，输出预测标签及指定位置的学习到的特征
            features = np.append(features, feature_2.detach().cpu().numpy())                    # 用features存储特征
            y_pred = np.append(y_pred, pred.cpu().numpy())                            # y_pred存储预测标签
            y_real = np.append(y_real, y.cpu().numpy())                                         # y_real存储真实标签                                                # 转换数据类型到int
            test_loss += config.Loss(pred.to(device), y.to(device))   # 计算测试分类损失损失
                # 计算分类准确率：如果perd和y一致的,则corret+1
    return test_loss / batches,  features.reshape(-1, 256), y_pred,y_real           # test()函数返回平均损失、准确率、模型提取到的特征、真实标签，用于后续进一步说明


def model_train_with_S(config, dict_S_loader,indexed_T_train_loader,indexed_T_test_loader, model,T_test_cell_cycles,T_train_cell_cycles,optimizer,scheduler=None):                # 模型总算法，包含训练及测试
    test_path=os.path.join(config.Result_path,f'test_{config.test_id}','with_S_train')
    os.makedirs(test_path, exist_ok=True)
    model_save_path=os.path.join(test_path,'Models')
    os.makedirs(model_save_path, exist_ok=True)
    model_path = os.path.join(test_path, f'Models\\model.pt')
    # 设置模型路径
    if os.path.exists(model_path):                                                              # 如已有参数相同训练模型，则直接进行测试环节，不再进行训练
        checkpoint = torch.load(model_path, map_location=torch.device(device))                  # 选择模型，用于后续导入参数
        model.load_state_dict(checkpoint['net_state_dict'])                                     # 模型导入参数
        model.eval()                                                                            # 用于测试阶段关闭BN和Dropout，以免影响结果
        loss_test, features_test, labels_test,labels_test_real = test(config, indexed_T_test_loader, model, 0, )               # 获取到测试的返回值
        loss_train,train_rmse_loss,pred_T_train,real_T_train = train(config, dict_S_loader, indexed_T_train_loader,indexed_T_test_loader, model, optimizer)  # 获取到训练的返回值
        print(f'3. Test Error:\n\t'
              f'Train_loss:{loss_train:>0.4f}\n\t'
              f'Test_loss:{loss_test:>0.4f}')     # 输出基础信息（测试准确率及损失值）
    else:       # 没有已训练模型，则进行训练
        i, train_loss, test_loss, train_acc, test_acc = 0, [], [], [], []
        RMSE_train,RMSE_test,train_loss,test_loss,MMD_train,pred_test,real_test=[],[],[],[],[],[],[]      # 设置基本变量，用于后续数据的存储
        with trange(config.N_epoch) as t:                                       # 使用trange模块，使输出界面更易读
            for _ in t:
                loss_train,train_rmse_loss,pred_T_train,real_T_train = train(config, dict_S_loader, indexed_T_train_loader,indexed_T_test_loader, model, optimizer) # 模型训练
                #print(features_train_s.shape,labels_train.shape)
                loss_test, features_test, labels_test_pred,labels_test_real = test(config, indexed_T_test_loader, model, i)
                if scheduler is not None:
                    scheduler.step()# 模型测试
                train_loss = np.append(train_loss, loss_train)
                test_loss = np.append(test_loss, loss_test.detach().cpu().numpy())
                t.set_description(f"\tTrain")
                t.set_postfix( Loss_train=train_loss.mean(), Loss_test=test_loss.mean()) # 显示部分信息
                i += 1

                RMSE_train = np.append(RMSE_train, train_rmse_loss)
                RMSE_test = np.append(RMSE_test, loss_test.detach().cpu().numpy().mean())
                pred_test = np.append(pred_test, labels_test_pred)
                real_test = np.append(real_test, labels_test_real)
                if (_+1)%(config.N_epoch//4)==0 or _==0:
                    plt_train_and_test_soh_fig_3(model,T_test_cell_cycles,T_train_cell_cycles,labels_test_pred,labels_test_real,pred_T_train,real_T_train,config,_)



        if config.save_model:
            torch.save({'net_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, model_path)          # 训练结束后保存模型
        print(f'3. Finish {config.N_epoch} epoch train')

        loss_path = os.path.join(test_path, 'Loss')
        os.makedirs(loss_path, exist_ok=True)
        np.savetxt(os.path.join(loss_path, 'train_loss.txt'), train_loss)
        np.savetxt(os.path.join(loss_path, 'test_loss.txt'), test_loss)
        np.savetxt(os.path.join(loss_path, 'train_rmse_loss.txt'), RMSE_train)
        np.savetxt(os.path.join(loss_path, 'test_rmse_loss.txt'), RMSE_test)
        np.savetxt(os.path.join(loss_path, 'mmd_loss.txt'), MMD_train)
        np.savetxt(os.path.join(loss_path, 'pred_test.txt'), pred_test)
        np.savetxt(os.path.join(loss_path,'real_test.txt'), real_test)

        generate_loss(loss_path,test_path)


