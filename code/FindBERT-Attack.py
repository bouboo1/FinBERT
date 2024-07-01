'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-06-27 22:32:32
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-07-01 23:35:53
FilePath: \undefinede:\大三下\社会计算\大作业选题\期末大作业\all\FinBERT\code\FindBERT-Attack.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
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
    num_train_epochs=50,
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