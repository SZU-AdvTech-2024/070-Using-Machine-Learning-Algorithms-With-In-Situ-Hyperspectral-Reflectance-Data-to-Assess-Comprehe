import argparse
import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import DenseNet_1DCNN
import OneDCNN_Att2

plt.rcParams.update({
    'font.size': 10,
    'font.sans-serif': 'Microsoft YaHei',
    'axes.unicode_minus': False
})

parser = argparse.ArgumentParser(description='OneDCNN_args')
parser.add_argument('--train_batch_size', type=int, default=128, help='train_batch_size')
parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs for training')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')
parser.add_argument('--gpu', type=str, default='cuda:1', help='gpu_id')
parser.add_argument('--element', type=int, default=1, help='element id')
parser.add_argument('--excel_path', type=str, default=r'./data/GLORIA_Chla.xlsx')
parser.add_argument('--save_model_path', type=str, default='./model/', help='path to save model')
parser.add_argument('--pretrained', type=bool, default=False, help='use pretrained model')
parser.add_argument('--pretrained_path', type=str, default='./model/', help='pretrained model path')
args = parser.parse_args()


def train_model(model, train_loader, val_loader, epochs_start, loss_start, num_epochs, learning_rate):
    criterion = nn.HuberLoss(delta=2)  # HuberLoss delta=1时为平滑L1
    # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=0.01)
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_loss = loss_start  # 初始化一个损失，用于保存模型时记录最优损失
    for epoch in range(num_epochs - epochs_start):  # 从预训练模型的epoch开始训练
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.to(torch.float32)
            labels = labels.to(torch.float32)
            inputs = inputs.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs.to(torch.float32), labels.to(torch.float32))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)


        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + epochs_start + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            all_value_train = validate_model(model, train_loader, criterion)
            all_value_test = validate_model(model, val_loader, criterion)
            print(f'Train_Data:\nAll:{all_value_train}')
            print(f'Val_Data:\nAll:{all_value_test}')
            print(f"Epoch {epoch + epochs_start + 1} save model! Loss={epoch_loss:.4f}")
            torch.save({'model': model.state_dict(), 'loss': epoch_loss},
                       f"{args.save_model_path}{get_name(args.element)}/1DCNN_best.pth")

        if (epoch + epochs_start + 1) % 10 == 0:
            torch.save({'model': model.state_dict(), 'loss': epoch_loss},
                       f"{args.save_model_path}{get_name(args.element)}/1DCNN_last.pth")
            all_value_train = validate_model(model, train_loader, criterion)
            all_value_test = validate_model(model, val_loader, criterion)
            print(f'Train_Data:\nAll:{all_value_train}')
            print(f'Val_Data:\nAll:{all_value_test}')


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



def compute_gradients(image):
    # 一阶微分
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


def validate_model(model_v, val_loader, criterion):
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    model_v.eval()
    val_loss = 0.0
    rmse = 0.0
    mae = 0.0
    mape = 0.0
    r2 = 0.0
    total_samples = 0
    with (torch.no_grad()):
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_inputs = val_inputs.to(torch.float32)
            val_labels = val_labels.to(torch.float32)
            val_inputs = val_inputs.unsqueeze(1)  # 增加1个维度
            val_outputs = model_v(val_inputs)
            val_outputs = val_outputs.squeeze(1)
            total_samples += val_inputs.size(0)
            val_loss += criterion(val_outputs.to(torch.float32), val_labels.to(torch.float32)).item() * val_inputs.size(0)
            predict_y = val_outputs.cpu().numpy() #从GPU移至CPU，张量转换成数组
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
    all_value = [avg_val_loss, avg_val_rmse, avg_val_mae, avg_val_r2, avg_val_mape]
    return all_value


# def get_element_name(element_id):  # 获取元素名的函数
#     if element_id == 4:
#         element_name = 'Chl-a'
#     elif element_id == 5:
#         element_name = 'TSS'
#     return element_name

# def draw_scatter(pred, true, rmse, mae, r2):
#     plt.subplot(2, 1, 2)
#     #     plt.plot(np.arange(len(result)), y_test, "go-", label="True value")
#     #     plt.plot(np.arange(len(result)), result, "ro-", label="Predict value")
#
#     plt.scatter(np.arange(len(pred)), pred, c="blue", label="True value")
#     plt.scatter(np.arange(len(true)), true, c="orange", label="Predict value")
#     plt.title(f"element:{args.element}\nRMSE:{rmse}---MAE:{mae}\nR²:{r2}")
#     plt.legend(loc="best")
#
#     plt.tight_layout()
#     plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
#
#     plt.savefig(
#         fr"./结果集/反演结果2/{args.element}/" + args.element + "反演散点图.png")  # 保存图片，必须写在show()前，否则无法保存。
#     plt.show()
#     plt.close()
def draw_boxplot(train, test):
    plt.boxplot(train, positions=[0], patch_artist=True, boxprops=dict(facecolor='lightgreen'),
                medianprops={'linewidth': 1, 'color': 'black'})
    plt.boxplot(test, positions=[1], patch_artist=True, boxprops=dict(facecolor='lightblue'),
                medianprops={'linewidth': 1, 'color': 'black'})
    plt.xticks([0, 1], ['Training Set', 'Test Set'])
    if args.element == 1:
        plt.ylabel(f'叶绿素a(mg/L)')
    elif args.element == 2:
        plt.ylabel(f'总悬浮物(mg/L)')
    elif args.element == 10:
        plt.ylabel(f'总氮(mg/L)')
    elif args.element == 11:
        plt.ylabel(f'氨氮(mg/L)')
    else:
        plt.ylabel(f'氨氮(mg/L)')
    plt.savefig(fr"./results/GLORIA/{get_name(args.element)}/" + get_name(args.element) + "数据分布图.png")
    plt.show()


if __name__ == '__main__':
    # 设置matplotlib fontsize
    plt.rcParams.update({'font.size': 13})
    river_dataset = RiverDataset(excel_dir=args.excel_path)
    train_size = int(0.75 * len(river_dataset))
    test_size = len(river_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(river_dataset, [train_size, test_size])
    # 绘制训练-测试集数据分布
    train_indexes = train_dataset.indices
    test_indexes = test_dataset.indices
    # train_indexes = list(range(len(train_dataset)))
    # test_indexes = list(range(len(test_dataset)))
    train_values = []
    test_values = []
    for train_idx in train_indexes:
        train_values.append(river_dataset.label_list[train_idx].item())
    for test_idx in test_indexes:
        test_values.append(river_dataset.label_list[test_idx].item())
    draw_boxplot(train_values, test_values)
    max_train = max(train_values)
    min_train = min(train_values)
    median_train = statistics.median(train_values)
    max_test = max(test_values)
    min_test = min(test_values)
    median_test = statistics.median(test_values)
    print(f"{get_name(args.element)}训练集 - 最大值: {max_train}, 最小值: {min_train}, 中位数: {median_train}")
    print(f"{get_name(args.element)}测试集 - 最大值: {max_test}, 最小值: {min_test}, 中位数: {median_test}")

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # 创建模型实例
    model = DenseNet_1DCNN.DenseNet121(64, 32)
    # model = OneDCNN_Att2.OneDCNN_Att2()
    epochs_start = 0
    loss_start = 20.0
    if args.pretrained:
        state_dict = torch.load(f'{args.pretrained_path}GLORIA/{get_name(args.element)}/1DCNN_best.pth')
        # state_dict = torch.load(f'{args.pretrained_path}DenseNet/{get_name(args.element)}/1DCNN_best.pth')
        model.load_state_dict(state_dict['model'])
        loss_start = state_dict['loss']
    # 调用训练函数开始训练
    train_model(model, train_loader, test_loader, epochs_start, loss_start, args.num_epochs, args.learning_rate)
