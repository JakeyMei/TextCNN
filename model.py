import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):

    def __init__(self,
                filter_num,             # 卷积核的数量
                filter_sizes,           # 卷积核大小即滑动窗口的大小
                vocab_size,             # 词表大小
                embedding_dim,          # 词向量维度 
                output_dim,             # 最后输出维度，即分类数量 
                dropout=0.1,            # dropout率
                pad_idx=0):             # 补长填充
        super(TextCNN, self).__init__()

        chanel_num = 1 # 通道数，也就是说一篇文章一个样本只相当于一个feature map
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=chanel_num, out_channels=filter_num,
                    kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes)*filter_num, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # 获取词向量
        embedded = self.dropout(self.embedding(text))  # text: bs * sent_len * emb_dim
        embedded = embedded.unsqueeze(1) # conv2d需要输入的是一个四维数据
        # 对词向量进行卷积，并对此进行池化操作
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]  #squeeze(3)判断第三维是否是1，如果是则压缩，如不是则保持原样
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved] # 判断第二维是否为1，若是则压缩
        # 拼接经过卷积后的特征
        cat = self.dropout(torch.cat(pooled, dim=1))

        # 全连接层获取最后的分类概率
        return self.fc(cat)