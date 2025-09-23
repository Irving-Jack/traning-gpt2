# traning-gpt2
traning gpt2 by The Complete Works of William Shakespeare​
# 首先安装正确的库（在Colab中运行）
!pip install transformers torch datasets

# 导入Hugging Face库，不是OpenAI库！
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
import torch
from torch.utils.data import Dataset
import requests
import re

# 1. 下载并清洗莎士比亚数据
url = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
response = requests.get(url)
raw_text = response.text

# 简单的数据清洗
def clean_shakespeare(text):
    # 找到主要内容开始位置
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    if start != -1:
        text = text[start:]
    
    # 删除法律声明等非文学内容
    lines = text.split('\n')
    cleaned_lines = []
    in_content = False
    
    for line in lines:
        if "*** START" in line:
            in_content = True
            continue
        if "*** END" in line:
            break
        if in_content and len(line.strip()) > 10:  # 只保留有内容的行
            cleaned_lines.append(line.strip())
    
    return '\n'.join(cleaned_lines)

cleaned_text = clean_shakespeare(raw_text)

# 保存清洗后的文本
with open('shakespeare.txt', 'w') as f:
    f.write(cleaned_text)

# 2. 加载GPT-2模型和分词器（本地免费）
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 设置填充token
tokenizer.pad_token = tokenizer.eos_token

# 3. 创建数据集
class ShakespeareDataset(Dataset):
    def __init__(self, txt_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 将文本分成 chunks
        self.chunks = []
        for i in range(0, len(text) - max_length + 1, max_length):
            chunk = text[i:i + max_length]
            self.chunks.append(chunk)
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        encoding = self.tokenizer(
            chunk,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

# 创建数据集实例
dataset = ShakespeareDataset('shakespeare.txt', tokenizer)

# 4. 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_dir='./logs',
)

# 5. 创建Trainer并开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=lambda data: {
        'input_ids': torch.stack([item['input_ids'] for item in data]),
        'attention_mask': torch.stack([item['attention_mask'] for item in data]),
        'labels': torch.stack([item['labels'] for item in data])
    }
)

print("开始训练...")
trainer.train()

# 6. 保存训练好的模型
trainer.save_model("./my_shakespeare_model")

# 7. 测试生成文本
def generate_text(prompt, model, tokenizer, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 测试生成
prompt = "To be, or not to be"
generated = generate_text(prompt, model, tokenizer)
print("生成的文本:")
print(generated)
