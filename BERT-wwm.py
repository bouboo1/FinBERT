from transformers import BertTokenizer, TFBertForSequenceClassification
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 加载数据
train_df = pd.read_csv('Train_Data.csv')
test_df = pd.read_csv('Test_Data.csv')

# 标签编码
label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['label'])
test_df['label'] = label_encoder.transform(test_df['label'])

# 切分数据
X_train, y_train = train_df['content'].values, train_df['label'].values
X_test, y_test = test_df['content'].values, test_df['label'].values

bert_wwm_tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm')
bert_wwm_model = TFBertForSequenceClassification.from_pretrained('hfl/chinese-bert-wwm', num_labels=3)

X_train_enc_wwm = bert_wwm_tokenizer(X_train.tolist(), padding=True, truncation=True, return_tensors='tf', max_length=100)
X_test_enc_wwm = bert_wwm_tokenizer(X_test.tolist(), padding=True, truncation=True, return_tensors='tf', max_length=100)

bert_wwm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
bert_wwm_model.fit([X_train_enc_wwm['input_ids'], X_train_enc_wwm['attention_mask']], y_train, epochs=3, batch_size=16, validation_split=0.2)

y_pred_bert_wwm = bert_wwm_model.predict([X_test_enc_wwm['input_ids'], X_test_enc_wwm['attention_mask']]).logits
y_pred_classes_bert_wwm = y_pred_bert_wwm.argmax(axis=1)

def evaluate_performance(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return  precision, recall, f1

precision_bert_wwm, recall_bert_wwm, f1_bert_wwm = evaluate_performance(y_test, y_pred_classes_bert_wwm)

print(f"Precision: {precision_bert_wwm}, Recall: {recall_bert_wwm}, F1 Score: {f1_bert_wwm}")
