import pickle

import numpy as np
from matplotlib import pyplot as plt


#  生成数据集代码，可直接调用
def generate_UCSD_Nissan_dataset(dataset_path, new_length, config, delta_V, package):
    """
    该函数主要用于将已经保存后的所有电池数据划分为训练集和测试集
    :param dataset_path: E:\Battery_data\Data\Converted_UCSD Nissan
    :param new_length: 降采样后的数据长度，主要用于经过降采样后的文件名
    :param config: 基本配置，用于选取测试电池id和训练电池id
    :param threshold: 主要用于获取满足电压区间的的文件名
    :return: train_data, train_data_soh, test_data, test_data_soh
    """
    #threshold = config.ratio_V * (4.2 - 2.6)
    pkl_path = f'{dataset_path}\\{package}\\random_{delta_V}_{new_length}_all_battery_id_data.pkl'
    soh_path = f'{dataset_path}\\{package}\\random_{delta_V}_all_battery_id_SOH.pkl'
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)
        keys = list(data.keys())
        all_battery_ids = np.arange(len(keys)) + 1
        train_battery_ids = np.setdiff1d(all_battery_ids, config.test_battery_id)
        train_data = []
        test_data = []
        for i in train_battery_ids:
            i_battery_data = np.array(data[f'battery_{i}'])
            print(f'train_battery_id: {i}, train_data_shape: {i_battery_data.shape}', end='\n')
            train_data.append(i_battery_data)
        train_data = np.concatenate(train_data, axis=0)
        for j in [config.test_battery_id]:
            j_battery_data = np.array(data[f'battery_{j}'])
            print(f'test_battery_id: {j}, test_data_shape: {j_battery_data.shape}', end='\n')
            test_data.append(j_battery_data)
        test_data = np.concatenate(test_data, axis=0)
    train_data[:, 0] = (train_data[:, 0] - 2.6)/(4.2-2.6)
    test_data[:, 0] = (test_data[:, 0] - 2.6) / (4.2-2.6)
    with open(soh_path, 'rb') as file:
        soh_data = pickle.load(file)
        keys = list(soh_data.keys())
        all_battery_ids = np.arange(len(keys)) + 1
        train_battery_ids = np.setdiff1d(all_battery_ids, config.test_battery_id)
        train_data_soh = []
        test_data_soh = []
        for i in train_battery_ids:
            i_battery_data_soh = np.array(soh_data[f'battery_{i}']).reshape(-1, 1)
            print(f'train_battery_id: {i}, train_data_soh_shape: {i_battery_data_soh.shape}', end='\n')
            train_data_soh.append(i_battery_data_soh)
        train_data_soh = np.concatenate(train_data_soh, axis=0)
        for j in [config.test_battery_id]:
            j_battery_data_soh = np.array(soh_data[f'battery_{j}']).reshape(-1, 1)
            print(f'test_battery_id: {j}, test_data_soh_shape: {j_battery_data_soh.shape}', end='\n')
            test_data_soh.append(j_battery_data_soh)
        test_data_soh = np.concatenate(test_data_soh, axis=0)
    train_data = train_data.astype(np.float32)
    train_data_soh = train_data_soh.astype(np.float32)
    test_data = test_data.astype(np.float32)
    test_data_soh = test_data_soh.astype(np.float32)
    return train_data, train_data_soh, test_data, test_data_soh


# 生成随机电池数据，delta——V为随机充放电数据的电压变化范围
def get_random_battery_cycle_and_soh_same_distribution(dataset_path, threshold, package, delta_V):
    all_battery_id_data_path = f'{dataset_path}\\{package}\\all_battery_id_data.pkl'
    all_battery_id_SOH_path = f'{dataset_path}\\{package}\\all_battery_id_SOH.pkl'
    random_all_battery_id_data_path = f'{dataset_path}\\{package}\\random_{delta_V}_all_battery_id_data.pkl'
    random_all_battery_id_SOH_path = f'{dataset_path}\\{package}\\random_{delta_V}_all_battery_id_SOH.pkl'
    with open(all_battery_id_SOH_path, 'rb') as file:
        SOH_data = pickle.load(file)
    with open(all_battery_id_data_path, 'rb') as file:
        data = pickle.load(file)
        new_data = {}
        new_SOH = {}
        for battery_id in list(data.keys()):
            new_data[battery_id] = []
            new_SOH[battery_id] = []
            for cycle_i in range(len(data[f'{battery_id}'])):
                cycle_data = data[f'{battery_id}'][cycle_i]
                # 将时间第一个数改为0，所有数均减去第一个数
                cycle_data[2] = cycle_data[2] - cycle_data[2][0]
                if max(cycle_data[0]) - min(cycle_data[0]) > delta_V and max(cycle_data[0]) < 3.83:
                    voltage_array = cycle_data[0][:, np.newaxis]  # 将一维数组转为二维数组
                    voltage_change_matrix = voltage_array - voltage_array.T  # 计算电压变化矩阵
                    # 获取所有满足电压变化条件的起始和结束索引
                    valid_indices = np.argwhere(voltage_change_matrix > delta_V)
                    random_index = np.random.randint(0, len(valid_indices))
                    start_index, end_index = valid_indices[random_index]
                    if start_index > end_index:
                        start_index, end_index = end_index, start_index
                    random_cycle_data = cycle_data[:, start_index:end_index]
                    random_cycle_data[2] = random_cycle_data[2] - random_cycle_data[2][0]
                    random_cycle_soh = SOH_data[f'{battery_id}'][cycle_i]
                    # fig, ax = plt.subplots()
                    # ax.plot(np.array(random_cycle_data[0]), label='voltage')
                    # ax.legend()
                    # os.makedirs(f'{dataset_path}\\{package}\\charge_cycles_png\\{threshold:.2f}\\{delta_V}\\{battery_id}', exist_ok=True)
                    # plt.savefig(f'{dataset_path}\\{package}\\charge_cycles_png\\{threshold:.2f}\\{delta_V}\\{battery_id}\\{cycle_i}.png')
                    # plt.close()
                    new_data[battery_id].append(random_cycle_data)
                    new_SOH[battery_id].append(random_cycle_soh)
            print(f'电池-{battery_id}已处理完成')
        print(f'Nissan数据已处理完成')
    with open(random_all_battery_id_data_path, 'wb') as file:
        pickle.dump(new_data, file)
    with open(random_all_battery_id_SOH_path, 'wb') as file:
        pickle.dump(new_SOH, file)


# 用于对数据进行降采样
def downsample_all_battery_cycle(dataset_path, new_length, package, delta_V):
    """
    该函数主要用于对每个循环进行降采样，将其变为同一长度，便于后续输入网络
    :param dataset_path: E:\Battery_data\Data\Converted_UCSD Nissan
    :param new_length: 降采样后的长度
    :param threshold: 满足循环所设置的电压区间，用于保存文件名
    :return:
    """
    all_battery_id_data_path = f'{dataset_path}\\{package}\\random_{delta_V}_all_battery_id_data.pkl'
    downsample_all_battery_id_data_path = f'{dataset_path}\\{package}\\random_{delta_V}_{new_length}_all_battery_id_data.pkl'
    with open(all_battery_id_data_path, 'rb') as file:
        data = pickle.load(file)
        new_data = {}
        for battery_id in list(data.keys()):
            new_data[battery_id] = []
            for cycle_i in range(len(data[f'{battery_id}'])):
                cycle_data = data[f'{battery_id}'][cycle_i]
                # 将时间第一个数改为0，所有数均减去第一个数
                cycle_data[2] = cycle_data[2] - cycle_data[2][0]
                # 原始数据的长度
                original_length = cycle_data.shape[1]
                # 原始数据的索引
                original_indices = np.linspace(0, original_length - 1, original_length)
                # 目标降采样后的索引
                new_indices = np.linspace(0, original_length - 1, new_length)
                # 对每一行进行线性插值降采样
                cycle_data_resampled = np.zeros((cycle_data.shape[0], new_length))
                for i in range(cycle_data.shape[0]):
                    cycle_data_resampled[i] = np.interp(new_indices, original_indices, cycle_data[i])
                # 将降采样后的数据保存回新字典
                new_data[battery_id].append(cycle_data_resampled)
            print(f'电池-{battery_id}已处理完成')
        print(f'电池组-{package}已处理完成')
    with open(downsample_all_battery_id_data_path, 'wb') as file:
        pickle.dump(new_data, file)


def fig_plot(y, y_hat=None,ub=None,lb=None):
    plt.figure(figsize=(10, 5))
    plt.plot(y, label='True Values', color='b')
    if y_hat is not None:
        plt.plot(y_hat, label='Predicted Values', color='r', linestyle='--')
        plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('True vs Predicted Values')
    plt.legend()
    if ub is not None and lb is not None:
        plt.ylim(ub, lb)
    plt.show()

def save_data(data, save_path):
    with open(save_path, 'wb') as file:
        pickle.dump(data, file)

def load_data(load_path):
    with open(load_path, 'rb') as file:
        data = pickle.load(file)
    return data

def battery_soh_plot(soh_data,battery_list,package):
    plt.figure(figsize=(10, 5))
    for battery_id in battery_list:
        plt.plot(soh_data[package][battery_id], label=f'Battery_{battery_id}')
    plt.xlabel('Cycle Index')
    plt.legend()
    plt.ylabel('SOH')

def smooth_soh(soh_data,method, sigma=1):
    """
    对SOH数据进行平滑处理
    :param soh_data: SOH数据
    :param method: 平滑方法，'gaussian'或'moving_average'
    :param sigma: 高斯平滑的sigma值
    :return: 平滑后的SOH数据
    """
    if method == 'gaussian':
        from scipy.ndimage import gaussian_filter1d
        for package in soh_data.keys():
            for key in soh_data[package].keys():
                for i in range(len(soh_data[package][key])):
                    if soh_data[package][key][i] > 1:
                        soh_data[package][key][i] = 1
                soh_data[package][key] = gaussian_filter1d(soh_data[package][key], sigma=sigma)
    elif method == 'moving_average':
        for package in soh_data.keys():
            for key in soh_data[package].keys():
                for i in range(len(soh_data[package][key])):
                    if soh_data[package][key][i] > 1:
                        soh_data[package][key][i] = 1
                soh_data[package][key] = np.convolve(soh_data[package][key], np.ones((sigma,))/sigma, mode='valid')
    return soh_data

