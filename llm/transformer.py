import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class LayerNorm(nn.Module):
    def __init__(self, feature, eps=1e-6):
       super(LayerNorm, self).__init__()
       self.a_2 = nn.Parameter(torch.ones(feature))
       self.b_2 = nn.Parameter(torch.zeros(feature))
       self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        return self.a_2 * (x - mean) / (std+self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    残差和layernorm一起做
    """

    def __init__(self, size, dropout = 0.1):
        super(SublayerConnection, self).__init__()
        # layernorm
        self.layer_norm = LayerNorm(size)
        # drop out
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        '''
        x: self-attention的输入
        sublayter: self-attention层
        '''

        return self.dropout(self.layer_norm(x + sublayer(x)))

def self_attention(query, key, value, dropout=None, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # mask 操作 在 Q @ K 之后
    if mask is not None:
        #mask.cuda()
        mask = mask.to(scores.device)
        scores = scores.masked_fill(mask ==0, -1e9)
    
    self_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        self_attn = dropout(self_attn)
    return torch.matmul(self_attn, value), self_attn


class MultiHeadAttention(nn.Module):

    def __init__(self, head, d_model, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert(d_model % head == 0)
        self.d_k = d_model // head
        self.head = head
        self.d_model = d_model
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None
        self.linear_out = nn.Linear(d_model, d_model)


    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 多头注意力机制
            mask = mask.unsqueeze(1)
        n_batch = query.size(0)

        query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1,2)
        key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1,2)
        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1,2)

        x, self.attn = self_attention(query, key, value, dropout=self.dropout, mask=mask)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)

        return self.linear_out(x)

class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        '''
        词向量的维度必须为偶数位
        '''
        if dim %2 != 0:
            raise ValueError("can not use sin/cos pe with odd dim (got dim={})".format(dim))

        
        '''
        位置编码公式
        PE(pos, 2i/2i+1) = sin/cos(pos/10000^{2i/d_model})
        '''

        pe = torch.zeros(max_len, dim)
        # position [max_len,1]
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float))*
                              -(math.log(10000.0)/dim))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb ,step=None):
        '''
        把词向量放大，词向量训练初期经过emb初始化层的值很小
        '''
        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:, :emb.size(1)]
        else:
            emb = emb + self.pe[:, step]
        emb = self.dropout(emb)
        return emb

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)
    
    def forward(self, x):
        # layer_norm删掉
        #inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        inter = self.dropout_1(self.relu(self.w_1(x)))
        output = self.dropout_2(self.w_2(inter))
        return output

def subsequent_mask(size):
    '''
    mask out subsequent positions
    '''
    atten_shape = (1, size, size)
    mask = np.triu(np.ones(atten_shape), k=1).astype('uint8')
    return mask

def pad_mask(src, trg, pad_idx):
    '''
    trg : 标签
    '''
    # [batch, 1, src_len]
    src_mask = (src != pad_idx).unsqueeze(1)                             
    # np.triu k=1: 1=未来位置(遮盖), 0=可看 → ==0 转成 True=可看
    subseq_mask = (torch.from_numpy(subsequent_mask(trg.size(1))) == 0)
    # [batch, trg_len, trg_len]
    trg_mask = subseq_mask.type_as(src_mask) \
               & (trg != pad_idx).unsqueeze(1)                            
    return src_mask, trg_mask

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

class EncoderLayer(nn.Module):
    def __init__(self, size, attn, feed_forward, dropout = 0.1):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), 2)
    
    def forward(self, x, mask):
        x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, mask))
        return self.sublayer_connection[1](x, self.feed_forward)

class Encoder(nn.Module):

    def __init__(self, n, encoder_layer):
        super(Encoder, self).__init__()
        self.encoder_layer = clones(encoder_layer, n)
    
    def forward(self, x, src_mask):
        for layer in self.encoder_layer:
            x = layer(x, src_mask)
        # n层编码后的layer
        return x

class DecoderLayer(nn.Module):

    def __init__(self, size, self_attn, src_attn, feed_forward, sublayer_num, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        # 这里需要再加一个attn？ 用于区分 self attn 和 cross attn 的权重
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), sublayer_num)
    
    def forward(self, x, memory, src_mask, trg_mask):
        x = self.sublayer_connection[0](x, lambda x: self.self_attn(x, x, x, trg_mask))
        x = self.sublayer_connection[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))

        return self.sublayer_connection[2](x, self.feed_forward)

class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

class FeatEmbedding(nn.Module):
    def __init__(self, d_feat, d_model, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(d_feat, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.linear(x))

class Decoder(nn.Module):
    def __init__(self, n, decoder_layer):
        super().__init__()
        self.layers = clones(decoder_layer, n)
    def forward(self, x, memory, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, trg_mask)
        return x

class ABDTransformer(nn.Module):
    def __init__(self, d_feat, trg_vocab_size, d_model, d_ff, n_heads,
                 n_layers, dropout, device='cuda'):
        super(ABDTransformer, self).__init__()
        self.device = device

        c = copy.deepcopy
        attn = MultiHeadAttention(n_heads, d_model, dropout)
        feed_forward = PositionWiseFeedForward(d_model, d_ff)

        self.src_embed = FeatEmbedding(d_feat, d_model, dropout)
        self.trg_embed = TextEmbedding(trg_vocab_size, d_model)
        self.pos_embed = PositionalEncoding(d_model, dropout)

        self.encoder = Encoder(n_layers, EncoderLayer(d_model, c(attn), c(feed_forward), dropout))

        self.decoder = Decoder(n_layers, DecoderLayer(d_model, c(attn), c(attn), c(feed_forward),
                                                      sublayer_num=3, dropout=dropout))

        self.generator = Generator(d_model, trg_vocab_size)

    def encode(self, src, src_mask):
        x1 = self.src_embed(src)
        x1 = self.pos_embed(x1)
        x1 = self.encoder(x1, src_mask)
        return x1
    
    def decode(self, trg, memory, src_mask, trg_mask):
        x1 = self.trg_embed(trg)
        x1 = self.pos_embed(x1)
        return self.decoder(x1, memory, src_mask, trg_mask)

    def forward(self, src, trg, src_mask, trg_mask):
        memory = self.encode(src, src_mask)

        dec_outputs = self.decode(trg, memory, src_mask, trg_mask)

        return self.generator(dec_outputs)
