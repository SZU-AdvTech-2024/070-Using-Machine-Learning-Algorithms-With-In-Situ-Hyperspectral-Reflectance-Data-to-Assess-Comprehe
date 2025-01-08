import pandas as pd
import numpy as np

path = './data/all_data4.xlsx'
Dataset = pd.read_excel(path)
Chla = np.ravel(Dataset.loc[:, ['ChlaValue']].values)
TP = np.ravel(Dataset.loc[:, ['TPValue']].values)
TN = np.ravel(Dataset.loc[:, ['TNValue']].values)
NH3 = np.ravel(Dataset.loc[:, ['NH3Value']].values)

# 以WQI为标准排序
WQI = []
for i in range(0, 124):
    WQI.append(TP[i] / 0.3 + TN[i] / 1.5 + NH3[i] / 1.5)

Dataset['WQI'] = WQI
Dataset = Dataset.sort_values(by='WQI')
Dataset.to_excel(fr"./data/排序数据集2.xlsx")

# 4：1分配训练、验证集
train_dataset = []
val_dataset = []
for i in range(0, 120, 5):
    train_dataset.extend(Dataset.iloc[i: i + 4, :].values)
    val_dataset.append(Dataset.iloc[i + 4, :].values)
train_dataset.extend(Dataset.iloc[120:122, :].values)
val_dataset.extend(Dataset.iloc[122:124, :].values)
train_dataset = pd.DataFrame(train_dataset)
train_dataset.to_excel(fr"./data/Sorted_train_dataset2.xlsx")
val_dataset = pd.DataFrame(val_dataset)
val_dataset.to_excel(fr"./data/Sorted_val_dataset2.xlsx")