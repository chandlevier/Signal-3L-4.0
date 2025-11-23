
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import PretrainedConfig

from module.CNN import TextCNN
from module.signalp6_CRF import CRF
from module.CoAttention import MultiHeadCoAttentionEncoder
from module.TransformerEncoder import TransformerEncoder
from module.NormLayer import NormedLinear
from utils.basic_utils import clear_memory

SIGNALP_KINGDOM_DICT = {"EUKARYA": 0, "POSITIVE": 1, "NEGATIVE": 2, "ARCHAEA": 3}
SIGNALP6_GLOBAL_LABEL_DICT = {
    "NO_SP": 0,
    "SP": 1,
    "LIPO": 2,
    "TAT": 3,
    "TATLIPO": 4,
    "PILIN": 5,
}

INV_SIGNALP_KINGDOM_DICT = {v: k for k, v in SIGNALP_KINGDOM_DICT.items()}

SIGNALP6_CLASS_LABEL_MAP = [
    [0, 1, 2],
    [3, 4, 5, 6, 7, 8],
    [9, 10, 11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20, 21, 22],
    [23, 24, 25, 26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35, 36],
]



class Signal3L4Config(PretrainedConfig):

    def __init__(
        self,
        sp_region_labels: bool,
        constrain_crf: bool,
        seq_vocab_size=30,
        seq_emb_size=20,
        dssp_vocab_size=9,
        dssp_emb_size=8,
        pdb_emb_size=2,
        embedding_dim=1280,
        hidden_dim=1280,
        hidden_act="leakyrelu",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        max_position_embeddings=512,
        dropout_prob=0.1,
        crf_input_length=200,
        crf_scaling_factor=1.0,
        num_global_labels=6,
        num_crf_tags=37,
        transformer_enc_layers=8,
        **kwargs
    ):
        super().__init__(**kwargs)

        # 初始化配置的属性
        self.seq_vocab_size = seq_vocab_size
        self.seq_emb_size = seq_emb_size
        self.dssp_vocab_size = dssp_vocab_size
        self.dssp_emb_size = dssp_emb_size
        self.pdb_emb_size = pdb_emb_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.dropout_prob = dropout_prob
        self.crf_input_length = crf_input_length
        self.crf_scaling_factor = crf_scaling_factor
        self.num_global_labels = num_global_labels
        self.num_crf_tags = num_crf_tags
        self.transformer_enc_layers = transformer_enc_layers
        
        ## CRF设置
        # hardcoded for full model, 5 classes, 37 tags
        if constrain_crf and sp_region_labels:
            self.allowed_crf_transitions = [
                # NO_SP
                # 0 I, 1 M, 2 O
                # I-I, I-M, M-M, M-O, M-I, O-M, O-O
                (0, 0), (0, 1), (1, 1), (1, 2), (1, 0), (2, 1), (2, 2),
                # SPI
                # 3 N, 4 H, 5 C, 6 I, 7M, 8 O
                (3, 3), (3, 4), (4, 4), (4, 5), (5, 5), (5, 8), (8, 8), (8, 7),
                (7, 7), (7, 6), (6, 6), (6, 7), (7, 8),
                # SPII
                # 9 N, 10 H, 11 CS, 12 C1, 13 I, 14 M, 15 O
                (9, 9), (9, 10), (10, 10), (10, 11), (11, 11), (11, 12), (12, 15),
                (15, 15), (15, 14), (14, 14), (14, 13), (13, 13), (13, 14), (14, 15),
                # TAT
                # 16 N, 17 RR, 18 H, 19 C, 20 I, 21 M, 22 O
                (16, 16), (16, 17), (17, 17), (17, 16), (16, 18), (18, 18), (18, 19),
                (19, 19), (19, 22), (22, 22), (22, 21), (21, 21), (21, 20), (20, 20),
                (20, 21), (21, 22),
                # TATLIPO
                # 23 N, 24 RR, 25 H, 26 CS, 27 C1, 28 I, 29 M, 30 O
                (23, 23), (23, 24), (24, 24), (24, 23), (23, 25), (25, 25), (25, 26),
                (26, 26), (26, 27), (27, 30), (30, 30), (30, 29), (29, 29), (29, 28),
                (28, 28), (28, 29), (29, 30),
                # PILIN
                # 31 P, 32 CS, 33 H, 34 I, 35 M, 36 O
                (31, 31), (31, 32), (32, 32), (32, 33), (33, 33), (33, 36), (36, 36),
                (36, 35), (35, 35), (35, 34), (34, 34), (34, 35), (35, 36),
            ]
            self.allowed_crf_starts = [0, 2, 3, 9, 16, 23, 31]
            self.allowed_crf_ends = [0, 1, 2, 13, 14, 15, 20, 21, 22, 28, 29, 30, 34, 35, 36]
        
        ## 两层CNN
        self.cnn1_configs = [
            {
            'dropout_rate': 0.2,
            'kernel_size': 3,
            'embedding_size': 256,
            'feature_size': 512,
            'activation_function_type': 'Tanh',
        },
            {
            'dropout_rate': 0.2,
            'kernel_size': 3,
            'embedding_size': 512,
            'feature_size': 256,
            'activation_function_type': 'Tanh',
        }
        ]
        self.cnn2_configs = [
            {
            'dropout_rate': 0.2,
            'kernel_size': 5,
            'embedding_size': 256,
            'feature_size': 512,
            'activation_function_type': 'Tanh',
        },
            {
            'dropout_rate': 0.2,
            'kernel_size': 5,
            'embedding_size': 512,
            'feature_size': 256,
            'activation_function_type': 'Tanh',
        }
        ]
        self.cnn3_configs = [
            {
            'dropout_rate': 0.2,
            'kernel_size': 7,
            'embedding_size': 256,
            'feature_size': 512,
            'activation_function_type': 'Tanh',
        },
            {
            'dropout_rate': 0.2,
            'kernel_size': 7,
            'embedding_size': 512,
            'feature_size': 256,
            'activation_function_type': 'Tanh',
        }
        ]

        self.coatt_configs = {
            'num_layers': 4,
            'd_model': 256,
            'h': 2,
            'bias': False,
            'activation': nn.ReLU(),
            'dropout': 0.1,
        }


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
 
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


class Signal3L4(nn.Module):
    def __init__(self, config: Signal3L4Config):  # 1280, 1280
        super(Signal3L4, self).__init__()
        esm_emb_dim = 1280
        embedding_dim = config.embedding_dim
        class_num = config.class_num
        self.pipeline_selection = config.pipeline_selection if hasattr(config, "pipeline_selection") else None

        # 物种信息嵌入设置
        self.use_kingdom_id = (
            config.use_kingdom_id if hasattr(config, "use_kingdom_id") else False
        )
        # self.kingdom_agonistic = 'EUKARYA'/'NEGATIVE'/'POSITIVE'/'ARCHAEA', 表明将所有预测序列的物种信息全抹平为一个指定的物种
        self.kingdom_agonistic = config.kingdom_agonistic if hasattr(config, "kingdom_agonistic") else False
        ## Set up kingdom ID embedding layer if used
        self.kingdom_as_token = config.kingdom_as_token if hasattr(config, "kingdom_as_token") else False
        
        ## 氨基酸序列嵌入
        self.seq_embedding = nn.Embedding(config.seq_vocab_size, config.seq_emb_size)

        ## 嵌入后线性映射
        linear_outsize = 252 if self.kingdom_as_token else 256
        self.seq_linear = nn.Linear(config.seq_emb_size, linear_outsize)
        self.pdb_linear = nn.Linear(config.pdb_emb_size, 256)
        
        ## 嵌入物种信息后融合
        if self.kingdom_as_token:
            linear_outsize += 4
        
        ## 位置编码
        self.pe = PositionalEncoding(d_model=linear_outsize, dropout=config.dropout_prob, max_len=70)

        ## 选择序列特征提取模块
        self.cnn1 = nn.Sequential(
            TextCNN(config.cnn1_configs[0]),
            TextCNN(config.cnn1_configs[1]),
        )
        self.cnn2 = nn.Sequential(
            TextCNN(config.cnn2_configs[0]),
            TextCNN(config.cnn2_configs[1]),
        )
        self.cnn3 = nn.Sequential(
            TextCNN(config.cnn3_configs[0]),
            TextCNN(config.cnn3_configs[1]),
        )
        hidden_dim = 256

        self.seq_enc = TransformerEncoder(
            num_layers=config.transformer_enc_layers,
            embed_dim=hidden_dim,
            num_heads=8,
            ff_dim=1024,
            dropout=config.dropout_prob,
        )
        self.pdb_enc = TransformerEncoder(
            num_layers=config.transformer_enc_layers,
            embed_dim=hidden_dim,
            num_heads=8,
            ff_dim=1024,
            dropout=config.dropout_prob,
        )
        
        self.co_attention = MultiHeadCoAttentionEncoder(config.coatt_configs)
        
        self.normedlinear1 = NormedLinear(hidden_dim, hidden_dim)
        self.normedlinear2 = NormedLinear(hidden_dim, hidden_dim)
        
        ## 拼接语言模型输出droppout
        self.lm_output_dropout = nn.Dropout(p=0.1)  # for backwards compatbility

        self.crf_init(config, esm_emb_dim + hidden_dim*2)
        
        self.dropout = nn.Dropout(p=0.1)
        
    def crf_init(self, config: Signal3L4Config, input_dim):
        self.use_large_crf = True   # legacy for get_metrics, no other use.
        self.crf_input_length = config.crf_input_length
        self.crf_scaling_factor = config.crf_scaling_factor
        self.num_global_labels = config.num_global_labels
        self.num_crf_tags = config.num_crf_tags

        self.reduce2crf = nn.Linear(input_dim, self.num_crf_tags)

        ## Set up CRF
        self.class_label_mapping = (
            config.class_label_mapping if hasattr(config, "class_label_mapping") else SIGNALP6_CLASS_LABEL_MAP
        )
        assert (
            len(self.class_label_mapping) == self.num_global_labels
        ), "defined number of classes and class-label mapping do not agree."

        # 都是None
        self.allowed_crf_transitions = (
            config.allowed_crf_transitions if hasattr(config, "allowed_crf_transitions") else None
        )
        self.allowed_crf_starts = (
            config.allowed_crf_starts if hasattr(config, "allowed_crf_starts") else None
        )
        self.allowed_crf_ends = (
            config.allowed_crf_ends if hasattr(config, "allowed_crf_ends") else None
        )

        self.crf = CRF(
            num_tags=self.num_crf_tags,
            batch_first=True,
            allowed_transitions=self.allowed_crf_transitions,
            allowed_start=self.allowed_crf_starts,
            allowed_end=self.allowed_crf_ends,
        )

        # Legacy, remove this once i completely retire non-mulitstate labeling
        self.sp_region_tagging = (
            config.use_region_labels if hasattr(config, "use_region_labels") else False
        )  # use the right global prob aggregation function

        ## Loss scaling parameters
        self.crf_scaling_factor = (
            config.crf_scaling_factor if hasattr(config, "crf_scaling_factor") else 1
        )

    def forward(self, seq_tokens, kingdom_ids, embding, pdb_emb, targets, input_mask, force_states=False):
        """
        Inputs:  seq_tokens batch x (70): 蛋白质氨基酸序列数字向量
                 embding (batch, 70, 1280): ESM2embedding
                 kingdom_ids (batch): [0,1,2,3] for eukarya, gram_positive, gram_negative, archaea
                 targets (batch, 70, 37):
                 input_mask (batch, 70): binary tensor, 0 at padded positions
                 
        Outputs: (loss: torch.tensor)
                 global_probs: global label probs (batch, 6)
                 probs: model probs (batch, 70, 37)
                 pos_preds: best label sequences (batch, 70)
        """
        ## 序列embedding初始化
        seq_emb = self.seq_linear(self.seq_embedding(seq_tokens.long()))
        pdb_emb = self.pdb_linear(pdb_emb)
        # 如果采用物种信息，则embedding后拼接一下
        if self.kingdom_as_token:
            if self.kingdom_agonistic is False:
                # Ensure that kingdom_ids has the shape (batchsize, 1) before one-hot encoding
                if len(kingdom_ids.shape) == 1:
                    kingdom_ids = kingdom_ids.unsqueeze(-1)  # Shape: (batchsize, 1)
                ids_emb = F.one_hot(kingdom_ids.squeeze(-1), num_classes=len(SIGNALP_KINGDOM_DICT.keys())) # shape: (batch, 4)
        
            else:  # 屏蔽物种信息，kingdom_id统一置零
                if len(kingdom_ids.shape) == 1:
                    kingdom_ids = kingdom_ids.unsqueeze(-1)  # Shape: (batchsize, 1)
                ids_emb = F.one_hot(torch.zeros_like(kingdom_ids).squeeze(-1), num_classes=len(SIGNALP_KINGDOM_DICT.keys())) # shape: (batch, 4)
            ids_emb = ids_emb.unsqueeze(1).repeat(1, seq_tokens.size()[1], 1)  # shape: (batch, len, id_embdim)
            seq_emb = torch.cat([seq_emb, ids_emb], dim=2)
        
        ## 增加位置编码
        pdb_emb += self.pe.forward(pdb_emb)

        ## 特征提取模块
        # 3条两层1D-CNN
        seq_att_emb, seq_attns = self.seq_enc(seq_emb)
        seq_emb = self.cnn1(seq_emb) + self.cnn2(seq_emb) + self.cnn3(seq_emb) + seq_att_emb
        # 8层Transformer encoder
        pdb_emb, pdb_attns = self.pdb_enc(pdb_emb)

        ## 归一化模块
        seq_emb = self.normedlinear1(seq_emb)
        pdb_emb = self.normedlinear2(pdb_emb)
        
        ## 用VisualBERT的co-attention处理
        seq_emb, pdb_emb, all_attn_weights = self.co_attention(seq_emb, pdb_emb)
        joint_emb = torch.cat([seq_emb, pdb_emb, self.lm_output_dropout(embding)], dim=-1)
        embding = self.reduce2crf(joint_emb)
        
        clear_memory()

        if torch.all(targets == 0): # targets为全0张量，即独立测试集，targets只有初始化值没有别的值
            outputs = self.crf_pure_forward(embding, input_mask, kingdom_ids, force_states)
        else:
            outputs = self.crf_forward(embding, targets, input_mask, kingdom_ids, force_states)
        outputs = (outputs) + (seq_attns, pdb_attns, all_attn_weights, )
        return outputs
    
    def crf_forward(self, embding, targets, input_mask, kingdom_ids, force_states):
        log_likelihood = self.crf(
                emissions=embding,
                tags=None,
                tag_bitmap=targets,
                mask=input_mask.byte(),
                reduction="mean",
            )
        neg_log_likelihood = -log_likelihood * self.crf_scaling_factor
        probs = self.crf.compute_marginal_probabilities(emissions=embding, mask=input_mask.byte())  # (batch, 70, 37)

        global_probs = self.compute_global_labels_multistate(probs, input_mask)

        global_log_probs = torch.log(global_probs)

        preds = self.predict_global_labels(global_probs, kingdom_ids, weights=None)

        # TODO update init_states generation to new n,h,c states and actually start using it
        # from preds, make initial sequence label vector
        if force_states:
            init_states = self.inital_state_labels_from_global_labels(preds)
        else:
            init_states = None
        viterbi_paths = self.crf.decode(emissions=embding, mask=input_mask.byte(), init_state_vector=init_states)
        
        del embding
        clear_memory()

        # pad the viterbi paths
        max_pad_len = max([len(x) for x in viterbi_paths])
        pos_preds = [x + [-1] * (max_pad_len - len(x)) for x in viterbi_paths]
        pos_preds = torch.tensor(
            pos_preds, device=probs.device
        )  # NOTE convert to tensor just for compatibility with the else case, so always returns same type

        outputs = (global_probs, probs, pos_preds)

        # get the losses
        losses = neg_log_likelihood

        outputs = (losses, ) + outputs

        return outputs
    
    def crf_pure_forward(self, embding, input_mask, kingdom_ids, force_states):
        """纯用CRF进行decode推理，不涉及计算crf负对数loss的部分
        """
        probs = self.crf.compute_marginal_probabilities(emissions=embding, mask=input_mask.byte())  # (batch, 70, 37)

        global_probs = self.compute_global_labels_multistate(probs, input_mask)

        global_log_probs = torch.log(global_probs)

        preds = self.predict_global_labels(global_probs, kingdom_ids, weights=None)

        # TODO update init_states generation to new n,h,c states and actually start using it
        # from preds, make initial sequence label vector
        if force_states:
            init_states = self.inital_state_labels_from_global_labels(preds)
        else:
            init_states = None
        viterbi_paths = self.crf.decode(emissions=embding, mask=input_mask.byte(), init_state_vector=init_states)
        
        del embding
        clear_memory()

        # pad the viterbi paths
        max_pad_len = max([len(x) for x in viterbi_paths])
        pos_preds = [x + [-1] * (max_pad_len - len(x)) for x in viterbi_paths]
        pos_preds = torch.tensor(
            pos_preds, device=probs.device
        )  # NOTE convert to tensor just for compatibility with the else case, so always returns same type

        outputs = (global_probs, probs, pos_preds)

        return outputs

    def compute_global_labels(self, probs, mask):
        """Compute the global labels as sum over marginal probabilities, normalizing by seuqence length.
        For agrregation, the EXTENDED_VOCAB indices from signalp_dataset.py are hardcoded here.
        If num_global_labels is 2, assume we deal with the sp-no sp case.
        """
        # probs = b_size x seq_len x n_states tensor
        # Yes, each SP type will now have 4 labels in the CRF. This means that now you only optimize the CRF loss, nothing else.
        # To get the SP type prediction you have two alternatives. One is to use the Viterbi decoding,
        # if the last position is predicted as SPI-extracellular, then you know it is SPI protein.
        # The other option is what you mention, sum the marginal probabilities, divide by the sequence length and then sum
        # the probability of the labels belonging to each SP type, which will leave you with 4 probabilities.
        if mask is None:
            mask = torch.ones(probs.shape[0], probs.shape[1], device=probs.device)

        summed_probs = (probs * mask.unsqueeze(-1)).sum(
            dim=1
        )  # sum probs for each label over axis
        sequence_lengths = mask.sum(dim=1)
        global_probs = summed_probs / sequence_lengths.unsqueeze(-1)

        # aggregate
        no_sp = global_probs[:, 0:3].sum(dim=1)

        spi = global_probs[:, 3:7].sum(dim=1)

        if self.num_global_labels > 2:
            spii = global_probs[:, 7:11].sum(dim=1)
            tat = global_probs[:, 11:15].sum(dim=1)
            tat_spi = global_probs[:, 15:19].sum(dim=1)
            spiii = global_probs[:, 19:].sum(dim=1)

            return torch.stack([no_sp, spi, spii, tat, tat_spi, spiii], dim=-1)

        else:
            return torch.stack([no_sp, spi], dim=-1)

    def compute_global_labels_multistate(self, probs, mask):
        """Aggregates probabilities for region-tagging CRF output"""
        if mask is None:
            mask = torch.ones(probs.shape[0], probs.shape[1], device=probs.device)

        summed_probs = (probs * mask.unsqueeze(-1)).sum(
            dim=1
        )  # sum probs for each label over axis
        sequence_lengths = mask.sum(dim=1)
        global_probs = summed_probs / sequence_lengths.unsqueeze(-1)

        global_probs_list = []
        for class_indices in self.class_label_mapping:
            summed_probs = global_probs[:, class_indices].sum(dim=1)
            global_probs_list.append(summed_probs)

        return torch.stack(global_probs_list, dim=-1)


    def predict_global_labels(self, probs, kingdom_ids, weights=None):
        """Given probs from compute_global_labels, get prediction.
        Takes care of summing over SPII and TAT for eukarya, and allows reweighting of probabilities."""

        if self.use_kingdom_id:
            eukarya_idx = torch.where(kingdom_ids == 0)[0]
            summed_sp_probs = probs[eukarya_idx, 1:].sum(dim=1)
            # update probs for eukarya
            probs[eukarya_idx, 1] = summed_sp_probs
            probs[eukarya_idx, 2:] = 0

        # reweight
        if weights is not None:
            probs = probs * weights
        # predict
        preds = probs.argmax(dim=1)

        return preds
    
