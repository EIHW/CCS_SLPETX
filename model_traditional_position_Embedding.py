import os
import sys
sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.insert(1, '/home/yantianh/tianhao/lib/python3.7/site-packages')
import torch
import torch.nn as nn
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import pickle
import logging
import copy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import recall_score as recall
import math
from torch.autograd import Variable
import random
# from util.misc import NestedTensor
from sklearn.metrics import confusion_matrix,classification_report
# import seaborn as sns
# from torchvision import models
# from torchsummaryX import summary
from typing import Optional, List
from tqdm import tqdm
import gc

# def seed_torch(seed = 10):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    _, time, nmel = seq.size()
    subsequent_mask = (1 -
        torch.ones(1,time, nmel)).bool()
    return subsequent_mask

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=32, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        x1 = torch.mean(x, dim=1).squeeze()
        x_mask = get_subsequent_mask(x1)
        x_mask = x_mask.repeat(x.size()[0], 1, 1)

        y_embed = x_mask.cumsum(1, dtype=torch.float32).to(device)
        x_embed = x_mask.cumsum(2, dtype=torch.float32).to(device)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def load_data():
    f = open('/home/yantianh/icassp/Compare2021_s300_40fu.pkl', 'rb')
    train_data, train_label, test_data, Test_label, valid_data, Valid_label, test_label, valid_label, pernums_test, pernums_dev = pickle.load(
        f)
    return train_data, train_label, test_data, Test_label, valid_data, Valid_label, test_label, valid_label, pernums_test, pernums_dev

def compute_uar(pred, gold):
    reca = recall(gold, pred, 'macro')

    return reca


def get_CI(data, bstrap):
    """

    :param data: [pred, groundtruth]
    :param bstrap:
    :return:
    """

    uars = []
    for _ in range(bstrap):
        idx = np.random.choice(range(len(data)), len(data), replace=True)
        samples = [data[i] for i in idx]
        sample_pred = [x[0] for x in samples]
        sample_groundtruth = [x[1] for x in samples]
        # sample_pred, sample_groundtruth = [data[i] for i in idx]
        uar = compute_uar(sample_pred, sample_groundtruth)
        uars.append(uar)

    lower_boundary_uar = pd.DataFrame(np.array(uars)).quantile(0.025)[0]
    higher_boundary_uar = pd.DataFrame(np.array(uars)).quantile(1 - 0.025)[0]

    return (higher_boundary_uar - lower_boundary_uar) / 2


class Attention(nn.Module):
    def __init__(self, units=128):
        super(Attention, self).__init__()
        self.units = units
        # self.fc1 = nn.Linear(640, 640)
        self.softmax = nn.Softmax(dim=1)
        self.fc2 = nn.Linear(256, self.units)

    def forward(self, inputs):
        hidden_states = inputs  # b_s, time_step,features      #shape=(?, ?, 128), dtype=float32)
        # hidden_states = torch.Tensor(hidden_states)
        # print(hidden_states.shape())
        # hidden_size = torch.Tensor(hidden_states.size(2))  # features    #128
        hidden_size = torch.einsum("ntf, ntf->f", [hidden_states, hidden_states])   # features    #128
        # score_first_part = F.relu(self.fc1(hidden_states))  # b_s, time_step, features
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = torch.einsum("ntf, f->ntf", [hidden_states, hidden_size])  # b_s, time_step, features   # shape=(?, 150, 128), dtype=float32)
        h_t = hidden_states[:, -1, :]  # b_s, features                                         #shape=(?, 128), dtype=float32)
        score = torch.einsum("nf, ntf->nt", [h_t, score_first_part])                       #shape=(?, 150), dtype=float32)
        attention_weights = self.softmax(score)  # b_s, t                            #shape=(?, 150), dtype=float32)
        context_vector = torch.einsum("ntf, nt->nf", [hidden_states, attention_weights])  # b_s, features  #shape=(?, 128), dtype=float32)
        pre_activation = torch.cat((context_vector, h_t), dim=1)  # b_s, 256            #shape=(?, 256), dtype=float32)
        attention_vector = F.tanh(self.fc2(pre_activation))  # b_s,128
        return attention_vector


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout, max_len=640):
#         """
#         位置编码器类的初始化函数
#
#         共有三个参数，分别是
#         d_model：词嵌入维度
#         dropout: dropout触发比率
#         max_len：每个句子的最大长度
#         """
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         # Compute the positional encodings
#         # 注意下面代码的计算方式与公式中给出的是不同的，但是是等价的，你可以尝试简单推导证明一下。
#         # 这样计算是为了避免中间的数值计算结果超出float的范围，
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) *
#                              -(math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
#         return self.dropout(x)


# class PositionEmbeddingSine(nn.Module):
#     """
#     This is a more standard version of the position embedding, very similar to the one
#     used by the Attention is all you need paper, generalized to work on images.
#     """
#     def __init__(self, num_pos_feats=32, temperature=10000, normalize=False, scale=None):
#         super().__init__()
#         self.num_pos_feats = num_pos_feats
#         self.temperature = temperature
#         self.normalize = normalize
#         if scale is not None and normalize is False:
#             raise ValueError("normalize should be True if scale is passed")
#         if scale is None:
#             scale = 2 * math.pi
#         self.scale = scale
#
#     def forward(self, x):
#         x1 = torch.mean(x, dim=1).squeeze()
#         x_mask = get_subsequent_mask(x1)
#         x_mask = x_mask.repeat(x.size()[0], 1, 1)
#
#         y_embed = x_mask.cumsum(1, dtype=torch.float32).to(device)
#         x_embed = x_mask.cumsum(2, dtype=torch.float32).to(device)
#
#         # if self.normalize:
#         #     eps = 1e-6
#         #     y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
#         #     x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
#
#         dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
#         dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
#
#         pos_x = x_embed[:, :, :, None] / dim_t
#         pos_y = y_embed[:, :, :, None] / dim_t
#         pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
#         pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
#         pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
#         return pos

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, num_pos_row_feats=32, num_pos_col_feats=32):
        super().__init__()
        # 这里使用了nn.Embedding，这是一个矩阵类，里面初始化了一个随机矩阵，矩阵的长是字典的大小，宽是用来表示字典中每个元素的属性向量，
        # 向量的维度根据你想要表示的元素的复杂度而定。类实例化之后可以根据字典中元素的下标来查找元素对应的向量。输入下标0，输出就是embeds矩阵中第0行。
        self.row_embed = nn.Embedding(150, num_pos_row_feats)
        self.col_embed = nn.Embedding(10, num_pos_col_feats)
        self.reset_parameters()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=1, padding=0)

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    # 输入依旧是NestedTensor
    def forward(self, x):
        # x = tensor_list.tensors  h:time_step  w:n_mels
        h, w = x.shape[-2:]              #h:150   w:10
        i = torch.arange(w, device=x.device)  #0~9
        j = torch.arange(h, device=x.device)  #0~149

        # x_emb：(w, 8)
        # y_emb：(h, 56)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        pos = torch.cat([x_emb.unsqueeze(0).repeat(h, 1, 1),  # (1,w,8) → (h,w,8)
                         y_emb.unsqueeze(1).repeat(1, w, 1),  # (h,1,56) → (h,w,56)
                         ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        pos = self.conv1(pos)
        # (h,w,64) → (64,h,w) → (1,64,h,w) → (b,64,h,w)
        return pos

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model=640, nhead = 8, dim_feedforward=1024, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward,d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos:Optional[Tensor]):
        return tensor if pos is None else tensor+pos

    def forward_post(self, src, src_mask:Optional[Tensor]=None, src_key_padding_mask:Optional[Tensor]=None, pos:Optional[Tensor]=None):
        # 和标准做法有点不一样，src加上位置编码得到q和k，但是v依然还是src，
        # 也就是v和qk不一样
        q = k = self.with_pos_embed(src,pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        # Add and Norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # Add and Norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask:Optional[Tensor] = None,
                    src_key_padding_mask:Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2,pos)
        src2 = self.self_attn(q,k,value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask:Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor]=None,
                pos:Optional[Tensor]=None):
        if self.normalize_before:
            return self.forward_pre(src,src_mask,src_key_padding_mask,pos)
        return self.forward_post(src, src_mask,src_key_padding_mask,pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # 编码器copy6份
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask:Optional[Tensor] = None, src_key_padding_mask:Optional[Tensor] = None, pos:Optional[Tensor] = None):
        # 内部包括6个编码器，顺序运行
        # src是图像特征输入，shape=hxw,b,256    #src是语音频谱特征输入， shape=t,b,c*n_nel
        output = src
        for layer in self.layers:
            # 每个编码器都需要加入pos位置编码
            # 第一个编码器输入来自图像特征，后面的编码器输入来自前一个编码器输出
            output = layer(output, src_mask=mask, src_key_padding_mask = src_key_padding_mask, pos=pos)
        return output

class Transformer(nn.Module):
    def __init__(self, d_model=640, nhead=8, num_encoder_layers=1,
                 dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model,nhead,dim_feedforward,dropout,activation,normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos_embed, mask = None):
        #flatten NxCxHxW to HWxNxC -> NxCxHxW to HxNxCW
        # bs, c, h, w = src.shape
        src = src.permute(2,0,1,3).flatten(2)
        # src = src.flatten(2).permute(2,0,1)
        # pos_embed = pos_embed.permute(2,0,1,3).flatten(2)
        # pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # query_embed = query_embed.unsqueeze(1).repeat(1,bs,1)
        # mask = mask.flatten(1)

        # tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask = mask, pos=pos_embed)
        return memory.permute(1,0,2)


class AttentionConvLSTM(nn.Module):
    def __init__(self,):
        super(AttentionConvLSTM,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=1, padding=2)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        # self.batchnorm1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1, padding=2)
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        # self.batchnorm2 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        # self.batchnorm3 = nn.BatchNorm2d(64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))

        # Position_Encoding
        # self.position_encoding = PositionEmbeddingLearned(num_pos_row_feats=32, num_pos_col_feats=32)
        self.position_encoding = PositionalEncoding(d_model = 640)

        # Transformer Encoder
        self.transformer = Transformer(d_model=640, nhead=8, num_encoder_layers=1, dim_feedforward=512, dropout=0.1,
                                       activation="relu", normalize_before=False)
        # LSTM block
        # hidden_size = 128
        # self.lstm = nn.LSTM(input_size=640, hidden_size=128, bidirectional= True, batch_first=True)
        # # self.attention =
        # self.fc1 = nn.Linear(640, 256)
        # self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(640, 64)
        self.drop_out = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64, 2)
        self.out_softmax = nn.Softmax(dim=1)
        self.attention_linear = nn.Linear(640,1)
        # self.out_linear = nn.Linear(hidden_size + 128, 2)
        # self.attention = Attention(units=128)

    def forward(self, x):      #(b_s, 3, 300, 40)
        x7 = F.relu(self.conv1(x)) # 3-> 32
        # x = F.relu(self.conv2(x))  # 32->32
        x1 = self.maxpool1(x7)

        x2 = F.relu(self.conv3(x1))   #32->64
        # x2 = F.relu(self.conv4(x2))   #64->64
        x2 = self.maxpool2(x2)
        #
        x3 = F.relu(self.conv5(x7))   #32->64
        x3 = self.maxpool3(x3)

        # x4 = x2 + x3  # b_s, channel, t, n_mels    #(b_s, 64, 150, 10)

        x4 = x2 + x3  # b_s, channel, t, n_mels    #(b_s, 64, 150, 10)
        x8 = x4.permute(0, 2, 3, 1)
        x8 = torch.flatten(x8, start_dim=2)
        x8 = x8.permute(1,0,2)
        x8 = self.position_encoding(x8)
        out = self.transformer(x4,x8)               #(b_s, 150, 640)

        out = torch.mean(out, dim=[-2])
        out = self.drop_out(F.relu(self.fc3(out)))
        out = F.relu(self.fc4(out))
        return out,out


def loss_fnc(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions, target=targets)


def make_train_step(model, loss_fnc, optimizer):
    def train_step(X, Y):
        # set model to train mode
        model.train()
        # forward pass
        output_logits, output_softmax = model(X)
        predictions = torch.argmax(output_softmax, dim=1)
        accuracy = torch.sum(Y == predictions) / float(len(Y))
        # compute loss
        loss = loss_fnc(output_logits, Y)
        # compute gradients
        loss.backward()
        # update parameters and zero gradients
        optimizer.step()
        optimizer.zero_grad()
        return loss.item(), accuracy * 100

    return train_step


def make_validate_fnc(model, loss_fnc):
    def validate(X, Y):
        with torch.no_grad():
            model.eval()
            output_logits, output_softmax = model(X)
            predictions = torch.argmax(output_softmax, dim=1)
            accuracy = torch.sum(Y == predictions) / float(len(Y))
            loss = loss_fnc(output_logits, Y)
        return loss.item(), accuracy * 100, predictions

    return validate


if __name__ == '__main__':

    train_data, train_label, test_data, Test_label, valid_data, Valid_label, test_label, valid_label, pernums_test, pernums_dev = load_data()
    # _, _, test_data, Test_label, _, _, test_label, _, pernums_test,_ = load_data()


    train_data = np.transpose(train_data, (0, 3, 1, 2))
    test_data = np.transpose(test_data, (0, 3, 1, 2))
    valid_data = np.transpose(valid_data, (0, 3, 1, 2))

    index = np.arange(len(train_data))
    np.random.shuffle(index)

    train_data = train_data[index]
    train_label = train_label[index]

    #...................................
    # Valid_label = dense_to_one_hot(Valid_label, 2)
    # Test_label = dense_to_one_hot(Test_label, 2)
    # train_label = dense_to_one_hot(train_label, 2)

    # train_data = torch.Tensor(train_data)
    # train_label = torch.Tensor(train_label)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = AttentionConvLSTM().to(device)
    # model = AttentionConvLSTM()
    # print(model)
    # inputs = torch.zeros(1,3,150,40)
    # summary(model,inputs)
    print('Number of trainable params: ', sum(p.numel() for p in model.parameters()))
    OPTIMIZER = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-3, momentum=0.8)
    # OPTIMIZER = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    train_step = make_train_step(model, loss_fnc, optimizer=OPTIMIZER)
    validate = make_validate_fnc(model, loss_fnc)
    losses = []
    val_losses = []
    EPOCHS = 50
    DATASET_SIZE = train_data.shape[0]
    BATCH_SIZE = 8
    best_acc = 0
    criterion = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        ind = np.random.permutation(DATASET_SIZE)
        train_data = train_data[ind, :, :, :]
        train_label = train_label[ind]
        epoch_acc = 0
        epoch_loss = 0
        iters = int(DATASET_SIZE / BATCH_SIZE)
        for i in range(iters):
            batch_start = i * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, DATASET_SIZE)
            actual_batch_size = batch_end - batch_start
            X = train_data[batch_start:batch_end, :, :, :]
            Y = train_label[batch_start:batch_end]
            X_tensor = torch.tensor(X, device=device).float()
            Y_tensor = torch.tensor(Y, dtype=torch.long, device=device)
            loss, acc = train_step(X_tensor, Y_tensor)
            epoch_acc += acc * actual_batch_size / DATASET_SIZE
            epoch_loss += loss * actual_batch_size / DATASET_SIZE
            print(f"\r Epoch {epoch}: iteration {i}/{iters}", end='')
        X_val_tensor = torch.tensor(valid_data, device=device).float()
        Y_val_tensor = torch.tensor(Valid_label, dtype=torch.long, device=device)
        val_loss, val_acc, predictions = validate(X_val_tensor, Y_val_tensor)
        losses.append(epoch_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch} --> loss:{epoch_loss:.4f},acc:{epoch_acc:.2f}%, val_loss:{val_loss:.4f}, val_acc:{val_acc:.2f}%")
        logging.info(f"Epoch {epoch} --> loss:{epoch_loss:.4f},acc:{epoch_acc:.2f}%, val_loss:{val_loss:.4f}, val_acc:{val_acc:.2f}%")
    #
        if val_acc > best_acc:
            best_acc = val_acc
            SAVE_PATH = os.path.join(os.getcwd(), "models")
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'covid19_pyconvsin66_model' + str(epoch) + '.pt'))
            print('Model is saved to {}'.format(os.path.join(SAVE_PATH, 'covid19_pyconvsin66_model' + str(epoch) + '.pt')))
            logging.info('Model is saved to {}'.format(os.path.join(SAVE_PATH, 'covid19_pyconvsin66_model' + str(epoch) + '.pt')))

    loss_values = losses
    val_loss_values = val_losses
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# .......................................................................................................
#     LOAD_PATH = os.path.join(os.getcwd(), 'models')
#     # model = AttentionConvLSTM()
#     model.load_state_dict(torch.load(os.path.join(LOAD_PATH, 'covid19_pyconv_model13.pt')))
#     print('Model is loaded from {}'.format(os.path.join(LOAD_PATH, 'covid19_pyconv_model13.pt')))
#
#     X_test_tensor = torch.tensor(test_data, device=device).float()
#     Y_test_tensor = torch.tensor(Test_label, dtype=torch.long, device=device)
#     test_loss, test_acc, predictions = validate(X_test_tensor,Y_test_tensor)
#     print(f'Test loss is {test_loss:.3f}')
#     print(f'Test accuracy is {test_acc:.2f}%')
#
#     predictions = predictions.cpu().numpy()
#     test_acc_uw1 = recall(Test_label, predictions, average='macro')
#     print('test_acc_uw1', test_acc_uw1)
#
#     index = 0
#     pre_results = []
#     true_results = []
#     for idx, per_i in enumerate(pernums_test):
#         pred_ss = predictions[index:index + per_i]
#         if float(sum(pred_ss)) / len(pred_ss) >= 0.5:
#             pred_trunk = 1  # test_label
#         else:
#             pred_trunk = 0  # predict
#         # sample_results.append([pred_trunk,test_label[idx]])
#         pre_results.append(pred_trunk)
#         true_results.append(test_label[idx])
#         index = index + per_i
#     #
#     test_acc_uw1 = recall(true_results, pre_results, average='macro')
#     print('test_acc_uw1', test_acc_uw1)
# #.......................................................................................................
#
#
#
#     con_mat = confusion_matrix(Test_label.reshape(-1), predictions)
#     con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
#     con_mat_norm = np.around(con_mat_norm, decimals=2)
#
#
#
#
#     print(classification_report(true_results, pre_results))
#
#     cm1 = confusion_matrix(true_results, pre_results)
#     print('Confusion Matrix : \n', cm1)
#
#     total1 = sum(sum(cm1))
#     accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
#     print('Accuracy : ', accuracy1)
#
#     sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
#     print('Sensitivity:', sensitivity1)
#
#     specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
#     print('Specificity : ', specificity1)
#
#     data = [[p, g] for p, g in zip(pre_results, true_results)]
#     # a, b , c ,d = mean_confidence_interval(prediction, confidence=0.95)
#     cis = get_CI(data, 1000)
#     print('confidence_interval', cis)
#
#    # #=== plot ===
#     plt.figure(figsize=(4, 4))
#     sns.heatmap(con_mat_norm, annot=True, cmap='Blues')
#
#     plt.ylim(0, 4)
#     plt.xlabel('Predicted labels')
#     plt.ylabel('True labels')
#
#     plt.show()
#
#     loss_values = losses
#     val_loss_values = val_losses
#     epochs = range(1, len(loss_values) + 1)
#
#     plt.plot(epochs, loss_values, 'bo', label='Training loss')
#     plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#
#     plt.show()

