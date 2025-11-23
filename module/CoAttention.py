import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MultiHeadCrossAttention', 'ScaledDotProductAttention']


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadCrossAttention(nn.Module):

    def __init__(self, config):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadCrossAttention, self).__init__()

        in_features = config['d_model']
        head_num = config['h']
        bias = config['bias']
        activation  = config['activation']

        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

        # Feed-forward network layers
        self.ffn_1 = nn.Linear(in_features, in_features * 4)  # First linear layer
        self.ffn_2 = nn.Linear(in_features * 4, in_features)  # Second linear layer
        self.ffn_activation = activation if activation is not None else nn.ReLU()

    def forward(self, q, k, v, mask=None):
        """多模态coattention中，q是模态1，kv是模态2，生成模态1的融合后结果"""
        res = k
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        # 残差连接
        y += res
        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        res = y
        # 两层前馈
        y = self.ffn_1(y)
        y = self.ffn_activation(y)
        y = self.ffn_2(y)
        # 残差连接
        y += res
        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )


class MultiHeadCoAttentionLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=2, dropout=0.1):
        super().__init__()
        
        self.cross_attn_1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.self_attn_1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)
        
        # Feed-Forward Networks
        self.ffn_1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.ffn_2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        # LayerNorms after FFN
        self.norm5 = nn.LayerNorm(embed_dim)
        self.norm6 = nn.LayerNorm(embed_dim)

    def forward(self, x1, x2, x1_mask=None, x2_mask=None):
        attn_weights = {}
        # ---- Cross-Attention ----
        x1_cross, cross_attn_1_weights = self.cross_attn_1(query=x1, key=x2, value=x2, key_padding_mask=x2_mask)
        x1 = self.norm1(x1 + x1_cross)
        attn_weights['cross_attn_1'] = cross_attn_1_weights  # shape: (batch*num_heads, query_len, key_len)

        x2_cross, cross_attn_2_weights = self.cross_attn_2(query=x2, key=x1, value=x1, key_padding_mask=x1_mask)
        x2 = self.norm2(x2 + x2_cross)
        attn_weights['cross_attn_2'] = cross_attn_2_weights

        # ---- Self-Attention ----
        x1_self, self_attn_1_weights = self.self_attn_1(query=x1, key=x1, value=x1, key_padding_mask=x1_mask)
        x1 = self.norm3(x1 + x1_self)
        attn_weights['self_attn_1'] = self_attn_1_weights

        x2_self, self_attn_2_weights = self.self_attn_2(query=x2, key=x2, value=x2, key_padding_mask=x2_mask)
        x2 = self.norm4(x2 + x2_self)
        attn_weights['self_attn_2'] = self_attn_2_weights

        # ---- Feed Forward Network ----
        x1_ffn = self.ffn_1(x1)
        x1 = self.norm5(x1 + x1_ffn)

        x2_ffn = self.ffn_2(x2)
        x2 = self.norm6(x2 + x2_ffn)

        return x1, x2, attn_weights
    
    
class MultiHeadCoAttentionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_layers = config['num_layers']
        embed_dim = config['d_model']
        num_heads = config['h']
        dropout = config['dropout']
        self.layers = nn.ModuleList([
            MultiHeadCoAttentionLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, x1, x2, x1_mask=None, x2_mask=None):
        all_attn_weights = {
            'cross_attn_1': [],
            'cross_attn_2': [],
            'self_attn_1': [],
            'self_attn_2': []
        }
        for layer in self.layers:
            x1, x2, attn_weights = layer(x1, x2, x1_mask, x2_mask)
            for key in all_attn_weights:
                all_attn_weights[key].append(attn_weights[key])
        # 把list转成tensor方便后续处理，shape大致是 (num_layers, batch*num_heads, query_len, key_len)
        for key in all_attn_weights:
            all_attn_weights[key] = torch.stack(all_attn_weights[key], dim=0)
        return x1, x2, all_attn_weights