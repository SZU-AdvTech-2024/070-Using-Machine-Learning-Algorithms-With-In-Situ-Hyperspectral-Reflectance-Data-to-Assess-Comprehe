import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import DenseNet_1DCNN
import OneDCNN_Att2


parser = argparse.ArgumentParser(description='args')
parser.add_argument('--train_batch_size', type=int, default=128, help='train_batch_size')
parser.add_argument('--test_batch_size', type=int, default=32, help='test_batch_size')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')
parser.add_argument('--gpu', type=str, default='cpu', help='gpu_id')
parser.add_argument('--element', type=int, default=1, help='element id')
parser.add_argument('--excel_path', type=str, default=r'./data/GLORIA_Chla.xlsx')
parser.add_argument('--save_model_path', type=str, default='./model/', help='path to save model')
parser.add_argument('--pretrained', type=bool, default=True, help='use pretrained model')
parser.add_argument('--pretrained_path', type=str, default='./model/', help='pretrained model path')
args = parser.parse_args()


plt.rcParams.update({
    'font.size': 19,
    'font.weight': 'bold',  # 全局字体加粗
    'font.family': 'Microsoft YaHei'
})


def compute_gradients(image):
    derivative = np.gradient(image, axis=0)
    return derivative


class RiverDataset(Dataset):
    def __init__(self, excel_dir):
        self.excel_dir = excel_dir
        self.data = pd.read_excel(self.excel_dir)

        self.spectral_list = self.data.iloc[:, 2:404]
        self.label_list = self.data.iloc[:, args.element]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # 读取数据
        spectral = self.spectral_list.iloc[idx, :].values
        spectral = compute_gradients(spectral)
        label = self.label_list.iloc[idx]
        # 将数据转换为Tensor
        spectral_tensor = torch.tensor(spectral)
        label_tensor = torch.tensor(label)
        return spectral_tensor, label_tensor


def get_name(element):
    if element == 1:
        element_id = 'Chla'
    elif element == 9:
        element_id = 'TP'
    elif element == 10:
        element_id = 'TN'
    elif element == 228:
        element_id = 'WQI'
    else:
        element_id = 'NH3'
    return element_id

def validate_model(model_v, val_loader1, train_loader1, criterion):
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    model_v.eval()
    # test
    val_loss = 0.0
    rmse = 0.0
    mae = 0.0
    mape = 0.0
    r2 = 0.0
    total_samples = 0
    # train
    train_loss = 0.0
    rmse_train = 0.0
    mae_train = 0.0
    mape_train = 0.0
    r2_train = 0.0
    total_samples_train = 0
    with torch.no_grad():
        for val_inputs, val_labels in val_loader1:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_inputs = val_inputs.to(torch.float32)
            val_labels = val_labels.to(torch.float32)
            val_inputs = val_inputs.unsqueeze(1)
            val_outputs = model_v(val_inputs)
            val_outputs = val_outputs.squeeze(1)
            total_samples += val_inputs.size(0)
            val_loss += criterion(val_outputs.to(torch.float32), val_labels.to(torch.float32)).item() * val_inputs.size(0)
            predict_y = val_outputs.cpu().numpy()
            true_y = val_labels.cpu().numpy()
            rmse = rmse + np.sqrt(mean_squared_error(true_y, predict_y)) * val_inputs.size(0)
            mae = mae + mean_absolute_error(true_y, predict_y) * val_inputs.size(0)
            mape = mape + np.mean(np.abs((true_y - predict_y) / true_y)) * 100 * val_inputs.size(0)
            r2 = r2 + r2_score(true_y, predict_y) * val_inputs.size(0)
    avg_val_loss = val_loss / total_samples
    avg_val_rmse = rmse / total_samples
    avg_val_mae = mae / total_samples
    avg_val_mape = mape / total_samples
    avg_val_r2 = r2 / total_samples
    with torch.no_grad():
        for train_inputs, train_labels in train_loader1:
            train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
            train_inputs = train_inputs.to(torch.float32)
            train_labels = train_labels.to(torch.float32)
            train_inputs = train_inputs.unsqueeze(1)
            train_outputs = model_v(train_inputs)
            train_outputs = train_outputs.squeeze(1)
            total_samples_train += train_loader1.batch_size
            train_loss += criterion(train_outputs.to(torch.float32), train_labels.to(torch.float32)).item() * train_inputs.size(0)
            predict_y_train = train_outputs.cpu().numpy()
            true_y_train = train_labels.cpu().numpy()
            rmse_train = rmse_train + np.sqrt(mean_squared_error(true_y_train, predict_y_train)) * train_inputs.size(0)
            mae_train = mae_train + mean_absolute_error(true_y_train, predict_y_train) * train_inputs.size(0)
            mape_train = mape_train + np.mean(np.abs((true_y_train-predict_y_train)/true_y_train)) * 100 * train_inputs.size(0)
            r2_train = r2_train + r2_score(true_y_train, predict_y_train) * train_inputs.size(0)
    avg_train_loss = train_loss / total_samples_train
    avg_train_rmse = rmse_train / total_samples_train
    avg_train_mae = mae_train / total_samples_train
    avg_train_mape = mape_train / total_samples_train
    avg_train_r2 = r2_train / total_samples_train
    # 绘制散点图
    min_val = min(min(true_y_train), min(true_y), min(predict_y_train), min(predict_y))
    max_val = max(max(true_y_train), max(true_y), min(predict_y_train), min(predict_y))
    plt.scatter(true_y_train, predict_y_train, color='blue', label='Training Set')
    plt.scatter(true_y, predict_y, color='red', label='Test Set', marker='x')
    # 绘制1:1线
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='1:1 Line')
    # 设置图表标题和坐标轴标签
    plt.xlabel(get_name(args.element) +'真实值(mg/L)', fontweight='bold')
    plt.ylabel(get_name(args.element) +'预测值(mg/L)', fontweight='bold')
    # 在右下角添加文本框
    textstr = f"R$^{2}$={avg_val_r2:.4f}\nRMSE={avg_val_rmse:.4f}\nMAE={avg_val_mae:.4f}\nMAPE={avg_val_mape:.4f}"
    props = dict(boxstyle='round', facecolor='w', alpha=0, edgecolor="none")
    plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes,
             verticalalignment='bottom', horizontalalignment='right', bbox=props, fontsize=12)
    # 添加图例
    plt.legend()
    plt.legend(loc="upper left", edgecolor="none")  # 显示图标解释，loc表示放置的位置
    plt.gca().set_aspect('equal', adjustable='box')  # 将画布设置为正方形
    plt.savefig(fr"./results/all_data/{get_name(args.element)}/" + get_name(args.element) + "散点图.png")
    plt.show()
    return ([avg_train_loss, avg_train_rmse, avg_train_mae, avg_train_r2, avg_train_mape],
            [avg_val_loss, avg_val_rmse, avg_val_mae, avg_val_r2, avg_val_mape])



if __name__ == '__main__':
    # 设置matplotlib fontsize
    plt.rcParams.update({'font.size': 13})
    river_dataset = RiverDataset(excel_dir=args.excel_path)
    train_size = int(0.75 * len(river_dataset))
    test_size = len(river_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(river_dataset, [train_size, test_size])
    # train_indexes = train_dataset.indices
    # test_indexes = test_dataset.indices
    # train_values = []
    # test_values = []
    # for train_idx in train_indexes:
    #     train_values.append(river_dataset.label_list[train_idx].item())
    # for test_idx in test_indexes:
    #     test_values.append(river_dataset.label_list[test_idx].item())
    # 创建DataLoader
    # train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # 创建模型实例
    model = DenseNet_1DCNN.DenseNet121(64, 32)
    # model = OneDCNN_Att2.OneDCNN_Att2()
    epochs_start = 0
    loss_start = 0.5
    if args.pretrained:
        # state_dict = torch.load(f'{args.pretrained_path}DenseNet/{get_name(args.element)}/1DCNN_best.pth')
        state_dict = torch.load(f'{args.pretrained_path}{get_name(args.element)}/1DCNN_best.pth')
        model.load_state_dict(state_dict['model'], strict=False)
        loss_start = state_dict['loss']

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    train_parameter, test_parameter = validate_model(model, test_loader, train_loader, criterion=nn.HuberLoss(delta=1))
    print(f'Training Set:\nLoss:{train_parameter[0]} RMSE:{train_parameter[1]} MAE:{train_parameter[2]} R2:{train_parameter[3]} MAPE:{train_parameter[4]}')
    print(f'Test Set:\nLoss:{test_parameter[0]} RMSE:{test_parameter[1]} MAE:{test_parameter[2]} R2:{test_parameter[3]} MAPE:{test_parameter[4]}')





