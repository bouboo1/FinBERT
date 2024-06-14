import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import TextClassificationPipeline
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import Trainer, TrainingArguments
import torch

df = pd.read_csv('Train_Data.csv')
test_data = pd.read_csv('Test_Data.csv')
model_name = 'yiyanghkust/finbert-tone'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# 分类pipeline
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, framework='pt', device=0 if torch.cuda.is_available() else -1)


# 标签映射
label_mapping = {'positive': 0, 'negative': 1, 'neutral': 2}
df['label'] = df['label'].map(label_mapping)
test_data['label'] = test_data['label'].map(label_mapping)
# 对抗性训练参数
adversarial_training_args = TrainingArguments(
    output_dir='./output',
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    adam_epsilon=1e-5,
    weight_decay=0.01,
    warmup_steps=0,
    logging_dir='./adversarial_training_logs',
    save_steps=500,
    evaluation_strategy='steps',
    eval_steps=500,
    logging_steps=100
)

# 定义对抗性训练数据集
class AdversarialDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df.loc[index, 'content']
        label = self.df.loc[index, 'label']

        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label)
        }

train_dataset = AdversarialDataset(df, tokenizer)
test_dataset= AdversarialDataset(test_data, tokenizer)
# 定义对抗性训练器
trainer = Trainer(
    model=model,
    args=adversarial_training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=None,
    compute_metrics=None
)

# 开始对抗性训练
trainer.train()

inverse_label_mapping = {v: k for k, v in label_mapping.items()}

# 检查并转换 df['label'] 类型
test_data['label'] = test_data['label'].apply(lambda x: inverse_label_mapping.get(x, x))

# 重新计算评估指标
test_data['predicted_label'] = test_data['content'].apply(lambda x: pipeline(x)[0]['label'].lower())
precision = precision_score(test_data['label'], test_data['predicted_label'], average='weighted')
recall = recall_score(test_data['label'], test_data['predicted_label'], average='weighted')
f1 = f1_score(test_data['label'], test_data['predicted_label'], average='weighted')

print(f"After adversarial training:")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")



















# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification
# from transformers import TextClassificationPipeline
# import torch
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import torch.nn as nn
# import torch.optim as optim
# from transformers import Trainer, TrainingArguments


# df = pd.read_csv('Test_Data.csv')

# model_name = 'yiyanghkust/finbert-tone'
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# # 定义对抗攻击函数
# def fgsm_attack(input_data, epsilon, data_grad):
#     perturbed_data = input_data + epsilon * data_grad.sign()
#     perturbed_data = torch.clamp(perturbed_data, 0, 1)
#     return perturbed_data

# # 定义对抗性训练函数
# def adversarial_training(model, optimizer, criterion, inputs, labels, epsilon):
#     model.train()
#     optimizer.zero_grad()

#     inputs, labels = inputs.to(device), labels.to(device)
#     inputs.requires_grad = True

#     outputs = model(inputs)
#     loss = criterion(outputs.logits, labels)
#     loss.backward()

#     # 生成对抗样本
#     data_grad = inputs.grad.data
#     perturbed_inputs = fgsm_attack(inputs, epsilon, data_grad)

#     # 重新计算输出
#     adv_outputs = model(perturbed_inputs)
#     adv_loss = criterion(adv_outputs.logits, labels)

#     # 合并原始样本和对抗样本的损失
#     total_loss = loss + adv_loss
#     total_loss.backward()
#     optimizer.step()

#     return total_loss.item()

# # 定义训练参数
# epochs = 3
# epsilon = 0.05
# optimizer = optim.Adam(model.parameters(), lr=2e-5)
# criterion = nn.CrossEntropyLoss()

# # 对抗性训练循环
# for epoch in range(epochs):
#     running_loss = 0.0
#     for index, row in df.iterrows():
#         content = row['content']
#         label = row['label']
#         inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
#         labels = torch.tensor([label]).to(device)
#         loss = adversarial_training(model, optimizer, criterion, inputs, labels, epsilon)
#         running_loss += loss
#     print(f"Epoch {epoch + 1}, Loss: {running_loss / len(df)}")

# # 测试模型性能
# model.eval()
# predicted_labels = []
# true_labels = []
# with torch.no_grad():
#     for index, row in df.iterrows():
#         content = row['content']
#         label = row['label']
#         inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
#         labels = torch.tensor([label]).to(device)
#         outputs = model(inputs.to(device))
#         _, predicted = torch.max(outputs.logits, 1)
#         predicted_labels.append(predicted.item())
#         true_labels.append(label)

# # 计算评估指标
# precision = precision_score(true_labels, predicted_labels, average='weighted')
# recall = recall_score(true_labels, predicted_labels, average='weighted')
# f1 = f1_score(true_labels, predicted_labels, average='weighted')

# print(f"Precision: {precision * 100:.2f}%")
# print(f"Recall: {recall * 100:.2f}%")
# print(f"F1 Score: {f1 * 100:.2f}%")