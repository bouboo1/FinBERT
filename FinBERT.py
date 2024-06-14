# FinBERT
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TextClassificationPipeline
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv('Test_Data.csv')

model_name = 'yiyanghkust/finbert-tone'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# 分类pipeline
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, framework='pt', device=0 if torch.cuda.is_available() else -1)

# 对内容进行情感分类并获取预测标签
df['predicted_label'] = df['content'].apply(lambda x: pipeline(x)[0]['label'].lower())

# 将实际标签转换为小写（因为原结果是大写，这会影响指标计算）
df['label'] = df['label'].apply(lambda x: x.lower())

# 计算准评估指标
precision = precision_score(df['label'], df['predicted_label'], average='weighted')
recall = recall_score(df['label'], df['predicted_label'], average='weighted')
f1 = f1_score(df['label'], df['predicted_label'], average='weighted')

print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")