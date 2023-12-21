# %%
import os
import torch

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
print('device=', device)

# %%
# 定义数据集
from datasets import load_dataset, load_from_disk


class Dataset(torch.utils.data.Dataset):

    def __init__(self, split):
        self.all_dataset = load_dataset('csv',
                                        data_files={'train': 'category_2/cate_2_data/final_sense_train.csv',
                                                    'test': 'category_2/cate_2_data/final_sense_test.csv',
                                                    'validation': 'category_2/cate_2_data/final_sense_valid.csv'})
        self.dataset = self.all_dataset[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['Text']
        label = self.dataset[i]['Label']

        return text, label


dataset = Dataset('train')

len(dataset), dataset[0]
# %%
# 加载字典和分词工具
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    # 编码
    encode_data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                              truncation=True,
                                              padding='max_length',
                                              max_length=256,
                                              return_tensors='pt',
                                              return_attention_mask=True)

    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    input_ids = encode_data['input_ids'].to(device)
    attention_mask = encode_data['attention_mask'].to(device)
    labels = torch.LongTensor(labels).to(device)

    # print(data['length'], data['length'].max())

    return input_ids, attention_mask, labels


# %%
# 数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=8,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

for i, (input_ids, attention_mask, labels) in enumerate(loader):
    break

input_ids.shape, attention_mask.shape, labels

vali_loader = torch.utils.data.DataLoader(dataset=Dataset('validation'),
                                          batch_size=8,
                                          collate_fn=collate_fn,
                                          shuffle=True,
                                          drop_last=True)

loader_test = torch.utils.data.DataLoader(dataset=Dataset('test'),
                                          batch_size=8,
                                          collate_fn=collate_fn,
                                          shuffle=True,
                                          drop_last=True)
# %%
pretrained = BertModel.from_pretrained("bert-base-uncased")
pretrained.to(device)


# for param in pretrained.parameters():
#     param.requires_grad_(False)


# %%
# 定义下游任务模型
class MyModel(torch.nn.Module):

    def __init__(self, robert_model) -> None:
        super().__init__()
        self.robert_model = robert_model
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        output = self.robert_model(input_ids, attention_mask=attention_mask)
        output = output['pooler_output']
        output = self.fc(output)
        return output


class_model = MyModel(pretrained).to(device)

# %%
# 计算accuracy，f1-core，recall
from sklearn.metrics import accuracy_score, f1_score, recall_score


def three_metrics(all_labels_train, all_preds_train):
    accuracy_train = accuracy_score(all_labels_train, all_preds_train)
    f1_train = f1_score(all_labels_train, all_preds_train, average='weighted')
    recall_train = recall_score(all_labels_train, all_preds_train, average='weighted')
    return accuracy_train, f1_train, recall_train


def show_result(number, name, loop):
    loop.set_description("{} ".format(name))
    loop.set_postfix(acc=number)


# %%
from transformers import AdamW
from tqdm import tqdm
import csv

optimizer = AdamW(class_model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# 结果缓存
train_res_rows = []
vali_res_rows = []
test_res = []
model_names = []
loop = tqdm(range(40))
for epoch in loop:
    all_preds_train = []
    all_labels_train = []
    all_preds_vali = []
    all_labels_vali = []
    all_preds_test = []
    all_labels_test = []
    class_model.train()
    for i, (input_ids, attention_mask, labels) in enumerate(loader):
        out = class_model(input_ids=input_ids,
                          attention_mask=attention_mask)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        out = out.argmax(dim=1)
        all_preds_train.extend(out.cpu().numpy())
        all_labels_train.extend(labels.cpu().numpy())

    accuracy, f1_core, recall = three_metrics(all_labels_train, all_preds_train)
    show_result(number=accuracy, name="train", loop=loop)
    train_res_rows.append({'accuracy': accuracy, 'f1-core': f1_core, 'recall': recall})

    class_model.eval()
    for i, (input_ids, attention_mask, labels) in enumerate(vali_loader):
        with torch.no_grad():
            out = class_model(input_ids=input_ids,
                              attention_mask=attention_mask)

        out = out.argmax(dim=1)
        all_preds_vali.extend(out.cpu().numpy())
        all_labels_vali.extend(labels.cpu().numpy())
    accuracy, f1_core, recall = three_metrics(all_labels_vali, all_preds_vali)
    show_result(number=accuracy, name="validation", loop=loop)
    vali_res_rows.append({'accuracy': accuracy, 'f1-core': f1_core, 'recall': recall})

    for i, (input_ids, attention_mask, labels) in enumerate(loader_test):
        with torch.no_grad():
            out = class_model(input_ids=input_ids, attention_mask=attention_mask)
        out = out.argmax(dim=1)
        all_preds_test.extend(out.cpu().numpy())
        all_labels_test.extend(labels.cpu().numpy())
    accuracy, f1_core, recall = three_metrics(all_labels_test, all_preds_test)
    show_result(number=accuracy, name="test", loop=loop)
    test_res.append({'model': f'bert_2_epoch{epoch}.pt', 'accuracy': accuracy, 'f1-core': f1_core, 'recall': recall})

# %%
'''
写入train、validation、test结果
'''


def save_result(res, filename, fieldnames):
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        csv_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        # 写入标题行
        csv_writer.writeheader()
        csv_writer.writerows(res)


# %%
root_path = 'category_2/cate_2_res/'
train_column_names = ['accuracy', 'f1-core', 'recall']  # 标题行的列名
test_column_name = ['model', 'accuracy', 'f1-core', 'recall']
save_result(train_res_rows, root_path + 'train_bert_2.csv', train_column_names)
save_result(vali_res_rows, root_path + 'vali_bert_2.csv', train_column_names)
save_result(test_res, root_path + 'test_bert_2.csv', test_column_name)
# %%

# %%
with open('roberta_test_result.csv', 'w', newline='', encoding='utf-8') as output_file:
    fieldnames = ['model_pt_name', 'accuracy']  # 标题行的列名
    csv_writer = csv.DictWriter(output_file, fieldnames=fieldnames)

    # 写入标题行
    csv_writer.writeheader()
    csv_writer.writerows()
