import numpy as np
from tqdm import *
from .basic_algorithm import *
import torch
from itertools import *
from .config import calculate_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(config,dict_source_dataloader, indexed_T_train_loader,T_train_loader,T_test_loader, model, optimizer):  # 模型训练算法
    model.train()                                                                 # 实例化MMD_Loss类，用于后续计算
    train_loss,mmd_loss,rmse_loss ,features_s,features_t,features_t_2, labels_s =  0,0,0, [],[],[],[]  # 设置基本变量存储部分数据，用于后续的结果输出
    data_s_iter = {
        S_domain: cycle(iter(loader))
        for S_domain, loader in dict_source_dataloader.items()
    }
    data_t_train_iter=cycle(indexed_T_train_loader)
    target_dataloader = chain(T_train_loader, T_test_loader)
    data_t_iter = cycle(target_dataloader)
    len_dataloader = len(indexed_T_train_loader)
    real_T_train=np.full(len(T_train_loader.dataset), np.nan)
    pred_T_train=np.full(len(T_train_loader.dataset), np.nan)
    for i in range(len_dataloader):
        X_t_train,Y_t_train,batch_indices=next(data_t_train_iter)
        X_t_train = X_t_train.to(device)
        Y_t_train = Y_t_train.to(device)
        batch_indices = batch_indices.cpu().numpy().astype(int)
        feature_fc_t_2,pred_t_train=model(X_t_train)
        real_T_train[batch_indices] = Y_t_train.detach().cpu().numpy().flatten()
        pred_T_train[batch_indices] = pred_t_train.detach().cpu().numpy().flatten()
        features_t_2=np.append(features_t_2,feature_fc_t_2.detach().cpu().numpy())
        RMSE_T_loss=config.Loss(pred_t_train.to(device),Y_t_train)
        loss = RMSE_T_loss

        X_t, _ = next(data_t_iter)
        X_t = X_t.to(device)
        model.eval()
        feature_fc_t, pred_t = model(X_t)
        model.train()

        for idx,(S_domain, loader) in enumerate(dict_source_dataloader.items()):
            X_s, Y_s,batch_indices = next(data_s_iter[S_domain])
            batch_indices = batch_indices.cpu().numpy().astype(int)
            #dict_real_S[S_domain]=np.append(dict_real_S[S_domain],Y_s.numpy())
            labels_s = np.append(labels_s, int(idx+1)*np.ones(len(X_s)))                                                     # labels用于存放源域数据的标签
            X_s, Y_s = X_s.to(device),  Y_s.to(device)   # 将源域与目标域放入device中
            model.eval()
            feature_fc_s, pred_s = model(X_s)
            model.train()
            features_s = np.append(features_s, feature_fc_s.detach().cpu().numpy())
            features_t = np.append(features_t, feature_fc_t.detach().cpu().numpy())
        # 计算总损失
        train_loss += loss.detach().cpu().numpy().mean()
        rmse_loss+=(RMSE_T_loss).detach().cpu().numpy().mean()
        optimizer.zero_grad(), loss.backward(), optimizer.step()                                # 梯度清零、反向传播、梯度更新经典三步骤                     # 计算分类准确率：如果perd和y一致的,则corret+1
    train_loss /= (i + 1)
    rmse_loss/= (i + 1)                                                                                  # 计算平均损失（除以iter的次数即可获得平均损失）
    return train_loss,rmse_loss,pred_T_train,real_T_train,features_s.reshape(-1, 256), features_t.reshape(-1, 256),features_t_2.reshape(-1, 256), labels_s.reshape(-1, 1)               # train()函数返回平均损失、准确率、模型提取到的特征、真实标签，用于后续进一步说明



def test(config, test_loader, model, epoch):    # 模型测试算法
    model.train(False)                                                                          # 模型测试
    y_real, y_pred, features, test_loss, correct, batches, size = [], [], [], 0, 0, len(test_loader), len(test_loader.dataset)  # 设置基本变量存储部分数据，用于后续的结果输出
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_loader):                                            # 遍历测试集
            X, y = X.to(device), y.to(device)                                                   # 将测试数据（X）、标签（y）放入device
            feature_2,pred = model(X)                              # 将测试数据放入模型中，输出预测标签及指定位置的学习到的特征
            features = np.append(features, feature_2.detach().cpu().numpy())                    # 用features存储特征
            y_pred = np.append(y_pred, pred.cpu().numpy())                            # y_pred存储预测标签
            y_real = np.append(y_real, y.cpu().numpy())                                         # y_real存储真实标签                                                # 转换数据类型到int
            test_loss += config.Loss(pred.to(device), y.to(device))   # 计算测试分类损失损失
                # 计算分类准确率：如果perd和y一致的,则corret+1
    return test_loss / batches,  features.reshape(-1, 256), y_pred,y_real           # test()函数返回平均损失、准确率、模型提取到的特征、真实标签，用于后续进一步说明


def model_train_without_S(config, dict_source_dataloader,indexed_T_train_loader,T_train_loader,T_test_loader, model,T_test_cell_cycles,T_train_cell_cycles,optimizer,scheduler=None):                # 模型总算法，包含训练及测试
    test_path=os.path.join(config.Result_path,f'test_{config.test_id}','without_S_train')
    os.makedirs(test_path, exist_ok=True)
    model_save_path=os.path.join(test_path,'Models')
    os.makedirs(model_save_path, exist_ok=True)
    model_path = os.path.join(test_path, f'Models\\model.pt')
    # 设置模型路径
    if os.path.exists(model_path):                                                              # 如已有参数相同训练模型，则直接进行测试环节，不再进行训练
        checkpoint = torch.load(model_path, map_location=torch.device(device))                  # 选择模型，用于后续导入参数
        model.load_state_dict(checkpoint['net_state_dict'])                                     # 模型导入参数
        model.eval()                                                                            # 用于测试阶段关闭BN和Dropout，以免影响结果
        loss_test, features_test, labels_test,labels_test_real = test(config, T_test_loader, model, 0, )               # 获取到测试的返回值
        loss_train,train_rmse_loss,pred_T_train,real_T_train,features_train_s,features_train_t,features_train_t_2, labels_train = train(config, dict_source_dataloader, indexed_T_train_loader,T_train_loader,T_test_loader, model, optimizer)  # 获取到训练的返回值
        mse, mae,_=calculate_loss(torch.tensor(labels_test_real), torch.tensor(labels_test))
        print(f'3. Test Error:\n\t'
              f'Test_loss:{loss_test:>0.4f}\n\t'
              f'MSE:{mse:>0.4f}\n\t'
              f'MAE:{mae:>0.4f}\n\t')  # 输出基础信息（测试准确率及损失值）
    else:       # 没有已训练模型，则进行训练
        i, train_loss, test_loss, train_acc, test_acc = 0, [], [], [], []
        RMSE_train,RMSE_test,train_loss,test_loss,MMD_train,pred_test,real_test=[],[],[],[],[],[],[]      # 设置基本变量，用于后续数据的存储
        with trange(config.N_epoch) as t:                                       # 使用trange模块，使输出界面更易读
            for _ in t:
                loss_train,train_rmse_loss,pred_T_train,real_T_train,features_train_s,features_train_t, features_train_t_2,labels_train = train(config, dict_source_dataloader, indexed_T_train_loader,T_train_loader,T_test_loader, model, optimizer) # 模型训练
                #print(features_train_s.shape,labels_train.shape)
                loss_test, features_test, labels_test_pred,labels_test_real = test(config, T_test_loader, model, i)
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
                if (_+1)%(config.N_epoch//2)==0 or _==0:
                    plt_train_and_test_soh_fig_3(model,T_test_cell_cycles,T_train_cell_cycles,labels_test_pred,labels_test_real,pred_T_train,real_T_train,config,_)
                    combined_features = np.concatenate([features_train_s, features_train_t])
                    domain_labels = np.concatenate([labels_train, np.zeros((len(features_train_t), 1))])
                    #plt_tsne(test_path,x=combined_features, y=domain_labels, config=config, with_T=False,train=f"domain_fusion_epoch{_ + 1}")
                    #plt_density(test_path,x=combined_features, y=domain_labels, config=config, with_T=False, train=f"{_ + 1}")
                    #plt_density_2(test_path,x=np.concatenate([features_train_t_2, features_train_t]), y=np.concatenate([np.zeros((len(features_train_t_2), 1)), np.ones((len(features_train_t), 1))]), config=config,with_T=False, train=f"target_{_ + 1}")

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
