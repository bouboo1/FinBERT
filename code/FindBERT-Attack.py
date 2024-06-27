# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# from transformers import BertTokenizer, BertModel, AdamW
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# from tqdm import tqdm
# from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# train_data = pd.read_csv('Train_Data.csv')
# test_data = pd.read_csv('Test_Data.csv')

# # 加载FinBERT模型和Tokenizer
# model_name = "yiyanghkust/finbert-tone"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# bert_model = BertForSequenceClassification.from_pretrained(model_name)


# # 预处理数据
# def preprocess_data(data, tokenizer, max_len=128):
#     inputs = tokenizer(data['content'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=max_len)
#     labels = torch.tensor(data['label'].map({'neutral': 0, 'positive': 1, 'negative': 2}).tolist())
#     return TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)


# # 数据加载器
# train_dataset = preprocess_data(train_data, tokenizer)
# test_dataset = preprocess_data(test_data, tokenizer)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# criterion = torch.nn.CrossEntropyLoss()

# # 对抗训练函数
# def adversarial_training(model, data_loader, epsilon, num_steps):
#     model.train()
#     for batch in data_loader:
#         input_ids, attention_mask, labels = [x.to(device) for x in batch]
        
#         # 原始前向传播和损失计算
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
        
#         # 对抗扰动
#         for _ in range(num_steps):
#             # 反向传播
#             model.zero_grad()
#             loss.backward()
            
#             # 获取梯度
#             grad_sign = torch.sign(input_ids.grad.data)
            
#             # 应用扰动
#             perturbed_input_ids = input_ids + epsilon * grad_sign
#             perturbed_input_ids = torch.clamp(perturbed_input_ids, 0, 1)  # 确保索引有效
            
#             # 第二轮前向传播
#             model.zero_grad()
#             perturbed_outputs = model(perturbed_input_ids, attention_mask=attention_mask, labels=labels)
#             perturbed_loss = perturbed_outputs.loss
            
#             # 累加损失
#             loss += perturbed_loss

#     # 返回平均损失
#     return loss / len(data_loader)
# model = bert_model

# # 损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = AdamW(model.parameters(), lr=5e-5)

# # 训练模型
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# num_epochs = 3
# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss = 0
#     for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
#         input_ids, attention_mask, labels = [x.to(device) for x in batch]
       

#         optimizer.zero_grad()

#         # 对抗训练
#         outputs = adversarial_training(model, train_loader, epsilon=1e-5, num_steps = 5)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()

#     avg_loss = epoch_loss / len(train_loader)
#     print(f'Epoch {epoch + 1} completed with average loss: {avg_loss:.4f}')

#     # 评估模型在当前epoch的表现
#     model.eval()
#     all_labels = []
#     all_predictions = []

#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc=f"Evaluating Epoch {epoch + 1}"):
#             input_ids, attention_mask, labels = [x.to(device) for x in batch]
#             outputs = model(input_ids, attention_mask=attention_mask)
#             _, predicted = torch.max(outputs, 1)
#             all_labels.extend(labels.cpu().numpy())
#             all_predictions.extend(predicted.cpu().numpy())

#     precision = precision_score(all_labels, all_predictions, average='weighted')
#     recall = recall_score(all_labels, all_predictions, average='weighted')
#     f1 = f1_score(all_labels, all_predictions, average='weighted')

#     print(f'Epoch {epoch + 1} - Precision: {precision:.4f}')
#     print(f'Epoch {epoch + 1} - Recall: {recall:.4f}')
#     print(f'Epoch {epoch + 1} - F1-Score: {f1:.4f}')
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from textattack import Attack, PGDAttack, BERTWordListGenerator

# 假设您已经有了训练数据集和验证数据集
train_dataset = ...  # 这里应该是您的训练数据集
val_dataset = ...  # 这里应该是您的验证数据集

model_name = 'yiyanghkust/finbert-tone'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

# 定义对抗攻击
word_list_generator = BERTWordListGenerator(tokenizer)
attack = PGDAttack(
    model=model,
    word_list_generator=word_list_generator,
    transformable_words=word_list_generator.transformable_words,
)

# 定义自定义 Trainer 类以集成对抗训练
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def train(self, model_path=None, dataset_name=None):
        train_dataset = self.dataset
        adversarial_dataset = self.create_adversarial_examples(train_dataset)
        self.dataset = adversarial_dataset
        return super().train()

    def create_adversarial_examples(self, dataset):
        adversarial_examples = []
        for idx, example in enumerate(dataset):
            attack = Attack(model, word_list_generator=word_list_generator)
            adversarial_input = attack.attack_dataset(example)
            if adversarial_input is not None:
                adversarial_examples.append(adversarial_input)
        return adversarial_examples

# 初始化自定义 Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# 训练模型
trainer.train()

# 评估模型
def evaluate(model, data_loader):
    model.eval()
    predictions, true_labels = [], []
    for batch in data_loader:
        with torch.no_grad():
            outputs = model(**{k: v.to(model.device) for k, v in batch.items() if k != 'idx'})
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=-1).tolist())
        true_labels.extend(batch['labels'].tolist())
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    return precision, recall, f1

precision, recall, f1 = evaluate(model, val_dataset)
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")