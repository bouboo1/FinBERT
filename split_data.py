import pandas as pd
from sklearn.model_selection import train_test_split
import chardet

with open('all-data.csv', 'rb') as f:
    result = chardet.detect(f.read())

data = pd.read_csv('all-data.csv', encoding=result['encoding'])

# 显示数据样本
print(data.head())

# 分离每个标签的数据
neutral_data = data[data['label'] == 'neutral']
negative_data = data[data['label'] == 'negative']
positive_data = data[data['label'] == 'positive']

# 按照比例划分训练集和测试集
train_neutral, test_neutral = train_test_split(neutral_data, test_size=0.2, random_state=42)
train_negative, test_negative = train_test_split(negative_data, test_size=0.2, random_state=42)
train_positive, test_positive = train_test_split(positive_data, test_size=0.2, random_state=42)

# 合并训练集和测试集
train_data = pd.concat([train_neutral, train_negative, train_positive])
test_data = pd.concat([test_neutral, test_negative, test_positive])

# 打乱数据
train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存划分后的数据集
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# 显示划分结果
print(f'Training set size: {len(train_data)}')
print(f'Testing set size: {len(test_data)}')
print(train_data['label'].value_counts())
print(test_data['label'].value_counts())
