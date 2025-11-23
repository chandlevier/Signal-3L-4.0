import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 每个head的维度

        # 定义线性层
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.shape

        # 计算 Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 变换维度: (B, L, D) -> (B, num_heads, L, head_dim)
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数: QK^T / sqrt(d_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 应用mask（用于屏蔽padding部分或自回归屏蔽）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算注意力加权后的输出
        attn_output = torch.matmul(attn_weights, V)

        # 变换维度回到 (B, L, D)
        # contiguous() 重新排列张量在内存中的布局
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)

        # 通过最终的线性层
        output = self.W_o(attn_output)
        return output, attn_weights

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)  # Expand维度
        self.fc2 = nn.Linear(ff_dim, embed_dim)  # 投影回原维度
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)  # GELU 激活
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力层 + 残差连接 + LayerNorm
        attn_output, attn_weights = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 前馈神经网络 + 残差连接 + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x, attn_weights

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        all_attentions = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            all_attentions.append(attn_weights)
        return x, all_attentions
