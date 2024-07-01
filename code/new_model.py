import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_data = pd.read_csv('Train_Data.csv')
test_data = pd.read_csv('Test_Data.csv')

# 加载FinBERT模型和Tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)


# 预处理数据
def preprocess_data(data, tokenizer, max_len=128):
    inputs = tokenizer(data['content'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=max_len)
    labels = torch.tensor(data['label'].map({'neutral': 0, 'positive': 1, 'negative': 2}).tolist())
    return TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)


# 数据加载器
train_dataset = preprocess_data(train_data, tokenizer)
test_dataset = preprocess_data(test_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class RCNN(nn.Module):
    def __init__(self, bert_model, hidden_size=768, num_labels=3, dropout=0.1):
        super(RCNN, self).__init__()
        self.bert = bert_model
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)

        self.conv1 = nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=2, padding=1)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size, num_labels)

        

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        lstm_output, _ = self.lstm(sequence_output)

        conv_output1 = self.conv1(lstm_output.permute(0, 2, 1))
        conv_output1 = torch.relu(conv_output1)
        conv_output2 = self.conv2(conv_output1)
        conv_output2 = torch.relu(conv_output2)

        pooled_output = torch.max(conv_output2, 2)[0]

        pooled_output = self.dropout(pooled_output)

        logits = self.fc(pooled_output)
        return logits


# Attack对抗训练函数
def adversarial_training(model, input_ids, attention_mask, labels, epsilon=1e-7):
    outputs = model.bert(input_ids, attention_mask=attention_mask)
    sequence_output = outputs.last_hidden_state
    sequence_output.retain_grad()

    # 对抗训练前向传播
    lstm_output, _ = model.lstm(sequence_output)

    conv_output1 = model.conv1(lstm_output.permute(0, 2, 1))
    conv_output1 = torch.relu(conv_output1)
    conv_output2 = model.conv2(conv_output1)
    conv_output2 = torch.relu(conv_output2)
    pooled_output = torch.max(conv_output2, 2)[0]
    pooled_output = model.dropout(pooled_output)
    logits = model.fc(pooled_output)
    loss = criterion(logits, labels)
    model.zero_grad()
    loss.backward(retain_graph=True)

    # 添加扰动
    perturbation = epsilon * sequence_output.grad.sign()
    sequence_output = sequence_output + perturbation

    # 对抗训练后向传播
    lstm_output, _ = model.lstm(sequence_output)
    

    conv_output1 = model.conv1(lstm_output.permute(0, 2, 1))
    conv_output1 = torch.relu(conv_output1)
    conv_output2 = model.conv2(conv_output1)
    conv_output2 = torch.relu(conv_output2)
    pooled_output = torch.max(conv_output2, 2)[0]
    pooled_output = model.dropout(pooled_output)
    logits = model.fc(pooled_output)
    return logits

model = RCNN(bert_model)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=5e-6)

model.to(device)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        optimizer.zero_grad()

        # 对抗训练
        outputs = adversarial_training(model, input_ids, attention_mask, labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f'Epoch {epoch + 1} completed with average loss: {avg_loss:.4f}')

    # 评估模型在当前epoch的表现
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating Epoch {epoch + 1}"):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f'Epoch {epoch + 1} - Precision: {precision:.4f}')
    print(f'Epoch {epoch + 1} - Recall: {recall:.4f}')
    print(f'Epoch {epoch + 1} - F1-Score: {f1:.4f}')
