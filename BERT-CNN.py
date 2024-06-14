import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

# 加载数据
train_df = pd.read_csv('Train_Data.csv')
test_df = pd.read_csv('Test_Data.csv')

# 编码标签
label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['label'])
test_df['label'] = label_encoder.transform(test_df['label'])

# 定义数据集类
class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.labels = df['label'].values
        self.texts = df['content'].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,  # 截断超过最大长度的部分
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BERT_CNN(nn.Module):
    def __init__(self, n_classes):
        super(BERT_CNN, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256, n_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        conv_output = self.conv1(sequence_output.permute(0, 2, 1))
        pooled_output = torch.max(conv_output, 2)[0]
        dropout_output = self.dropout(pooled_output)
        logits = self.fc1(dropout_output)
        return logits

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128
batch_size = 16
n_classes = len(train_df['label'].unique())

train_dataset = SentimentDataset(train_df, tokenizer, max_len)
test_dataset = SentimentDataset(test_df, tokenizer, max_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
model = BERT_CNN(n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

def train_model(model, data_loader, criterion, optimizer, device):
    model = model.train()
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def eval_model(model, data_loader, device):
    model = model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds)
            true_labels.extend(labels)

    return torch.tensor(predictions), torch.tensor(true_labels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

epochs = 3
for epoch in range(epochs):
    train_model(model, train_loader, criterion, optimizer, device)

y_pred, y_true = eval_model(model, test_loader, device)

# 计算评价指标
accuracy = accuracy_score(y_true.cpu(), y_pred.cpu())
precision, recall, f1, _ = precision_recall_fscore_support(y_true.cpu(), y_pred.cpu(), average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
