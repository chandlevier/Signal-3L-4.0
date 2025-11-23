import os
import pickle
import torch
import torch.cuda
import args_maker
import numpy as np
from dataset import SPDataset
from torch.utils.data import DataLoader
from utils.basic_utils import save_mydict, clear_memory, setup_logger, set_randomseed
from utils.estimate_utils import tagged_seq_to_cs_multiclass
from Signal3L_4 import Signal3L4Config, Signal3L4


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

SIGNALP_KINGDOM_DICT = {"EUKARYA": 0, "POSITIVE": 1, "NEGATIVE": 2, "ARCHAEA": 3}
label_dict = {
    "NO_SP": 0,
    "SP": 1,
    "LIPO": 2,
    "TAT": 3,
    "TATLIPO": 4,
    "PILIN": 5,
}

# 翻转字典
inv_kingdom_dict = {v: k for k, v in SIGNALP_KINGDOM_DICT.items()}
inv_label_dict = {v: k for k, v in label_dict.items()}

MAX_LEN = 70       # 最长长度控制在200即可
   

from tqdm import tqdm
def predict(model, logger, bench_loader):
    """接受传来的基准测试数据(esm embedding, sp type id, kingdom id, cs index), 生成分类概率向量tag score和attention score
    基于分类结果决定是否计算CS index，再基于attention score计算CS index位置
    最终文件中存入字典变量，举例如下
    {'P15516': 
        {'seq': 'MKFFVFALILALMLSMTGADSHAKRHHGYKRKFHEKHHSHRGYRSNYLYDN',
        'kingdom': 'EUKARYA', 该标签由SignalP数据集提取出
        'sp_type': 'NO_SP',
        'sp_index': (0, 0),
        'sp_type_pred': 'SP', 该标签由原始6维P最大值索引对应的sp_type标签得出
        'cs_pred': (1, 19)
        },...
    }"""
    final_data_dict = {}
    all_cs_gdt = []
    all_cs_pred = []
    all_kingdom_ids = []

    all_glblbl_gdt = []
    all_global_probs = []
    all_pos_preds = []
    all_acc_list = []
    
    seq_attn_dict = {}
    pdb_attn_dict = {}
    co_attn_dict = {}
    
    with torch.no_grad():
        for batch in tqdm(bench_loader, desc="IndependentTesting", leave=False):
            (
                seq_tokens,
                batch_data,     # tensor型张量, padding后的esm embedding, 维度为(batchsize, 70, 1280), 默认在CPU上
                pdb_data,
                targets,        # label_matrix, (batchsize, 70, 37)的全零张量
                input_mask,     # tensor型张量, (batchsize, 70)
                acc_list,       # tuple型变量
                glb_label_ids,  # tensor型张量, 按照预先定义好的索引映射字典将batch内序列的SP种类信息映射到数字0~5上
                cs_gdts,        # tensor型张量, 存储batch内序列的SP索引对
                kingdom_ids,    # tensor型张量, 按照预先定义好的索引映射字典将batch内序列的物种信息映射到数字0~3上
            ) = batch
            seq_tokens = torch.tensor(seq_tokens).to(device)
            batch_data = batch_data.to(device)    # shape: (batch, batch_maxlen, 1280)
            pdb_data = pdb_data.to(device)
            targets_bit_map = targets.to(device)
            input_mask = input_mask.to(device)
            acc_list = list(acc_list)
            glb_label_ids = glb_label_ids.to(device)
            kingdom_ids = kingdom_ids.to(device)
            
            global_probs, _, pos_preds, seq_attns, pdb_attns, all_attn_weights = model(seq_tokens, kingdom_ids.long(), batch_data, pdb_data, targets_bit_map, input_mask)

            all_global_probs.append(global_probs.detach().cpu().numpy())
            all_glblbl_gdt.append(glb_label_ids.detach().cpu().numpy())
            all_pos_preds.append(pos_preds.detach().cpu().numpy())
            all_kingdom_ids.append(kingdom_ids.detach().cpu().numpy())
            all_cs_gdt.append(cs_gdts.detach().cpu().numpy())
            all_acc_list.extend(acc_list)
            
            for i, acc in enumerate(acc_list):
                # if acc not in ['Q50833', 'P15363']:   # 选定几个蛋白: Q50833(ARCHAEA_SP), P15363(NEGATIVE_LIPO)
                #     continue
                seq_attn_dict[acc] = []
                pdb_attn_dict[acc] = []
                co_attn_dict[acc] =  {
                    'cross_attn_1': [],
                    'cross_attn_2': [],
                    'self_attn_1': [],
                    'self_attn_2': []
                }
                for j, (seq_attn, pdb_attn) in enumerate(zip(seq_attns, pdb_attns)):
                    seq_attn_dict[acc].append(seq_attn[i].detach().cpu().numpy())
                    pdb_attn_dict[acc].append(pdb_attn[i].detach().cpu().numpy())
                    if j < 4:
                        for key in all_attn_weights.keys():
                            co_attn_dict[acc][key].append(all_attn_weights[key][j][i].detach().cpu().numpy())
            
            del batch_data, batch, targets_bit_map, input_mask, pos_preds, _, seq_tokens
            clear_memory()
            # break

    all_glblbl_gdt = np.concatenate(all_glblbl_gdt)
    all_global_probs = np.concatenate(all_global_probs)
    all_pos_preds = np.concatenate(all_pos_preds)
    all_kingdom_ids = np.concatenate(all_kingdom_ids)
    all_cs_gdt = np.concatenate(all_cs_gdt) if args.sp_region_labels else None

    # 还是得用TSignal的评估方法才能获得per kingdom per type ±3的评估结果, 所以需要保存预测结果
    sp_tokens = [5, 11, 19, 26, 31]
    all_glblbl_pred = all_global_probs.argmax(axis=1)
    all_cs_pred = tagged_seq_to_cs_multiclass(all_pos_preds, sp_tokens=sp_tokens)
    all_cs_pred[np.isnan(all_cs_pred)] = -1

    # 存储结果
    for (acc, glb_id, cs_gdt, kingdom_id, glb_pred, cs_pred) in zip(all_acc_list, all_glblbl_gdt, all_cs_gdt, all_kingdom_ids, all_glblbl_pred, all_cs_pred):
        if cs_gdt != int(cs_pred):
            continue
        result = {}
        result['kingdom'] = inv_kingdom_dict[kingdom_id]
        result['sp_type'] = inv_label_dict[glb_id]#'UNKNOWN'
        result['cs'] = cs_gdt
        result['sp_type_pred'] = inv_label_dict[glb_pred]
        result['cs_pred'] = int(cs_pred)
        final_data_dict[acc] = result
        logger.info(f'Recommended SP Segments of Sequence {acc} ({inv_label_dict[glb_id]}): type = {inv_label_dict[glb_pred]}')
        logger.info(f"Recommended SP Segments Index: {(result['cs_pred'])} (gdt: {cs_gdt})")
    clear_memory()
    
    with open('result_analysis/attention_weight/seq_attention_weights.pkl', 'wb') as f:
        pickle.dump(seq_attn_dict, f)
    with open('result_analysis/attention_weight/pdb_attention_weights.pkl', 'wb') as f:
        pickle.dump(pdb_attn_dict, f)
    with open('result_analysis/attention_weight/co_attention_weights.pkl', 'wb') as f:
        pickle.dump(co_attn_dict, f)

    return final_data_dict




def main(args):
    """修改后的DAN推荐逻辑"""
    model_path = os.path.join(args.output_dir, f'model_{epoch}.pt')
    model = Signal3L4(config).cuda()
    model.load_state_dict(torch.load(model_path, map_location=device))
    seed = set_randomseed(42)

    # 文件路径设置
    filename = 'GOSP2024'
    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)

    # 加载日志记录器
    logger = setup_logger("predict", output_path)
    logger.info("torch seed: %s", seed)
    logger.info(f"Loading test set {filename} and model \"{model_path}\"...")
    logger.info("Running model on %s, not using nvidia apex", device)
    logger.info("Prediction results saving to %s", model_path)
    logger.info("Batchsize: %s", str(BATCHSIZE))
    logger.info("Max sequence length: %s", MAX_LEN)
    logger.info("Embedding mode: %s", args.embedding_mode)

    test_data = SPDataset(
        device,
        MAX_LEN,
        "Dataset/fasta_data/GOSP2024_1.fasta",
        "Dataset/pdb_data/ECO269",
        seq_emb_mode=args.embedding_mode,
    )
    test_loader = DataLoader(test_data, batch_size=BATCHSIZE, collate_fn=test_data.collate_fn, shuffle=False)
    logger.info(f"{len(test_data)} benchmarking sequences.")
    
    # 预测
    pre_dict = predict(model, logger, test_loader)

    # 存储预测结果
    logger.info('Saving recommendation result...\n')
    save_mydict(pre_dict, f"{output_path}/{filename}")
    

args = args_maker.test_args()
device = torch.device(f"cuda:{args.cuda_index}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device.index)

EMB_DIM = args.embed_dim
BATCHSIZE = 740#args.batch_size
MAX_LEN = 70

# 模型参数设置 默认不包含kingdom信息
config = Signal3L4Config(sp_region_labels=args.sp_region_labels,
                   constrain_crf=args.constrain_crf,
                   embedding_dim=EMB_DIM,
                   hidden_dim=EMB_DIM,
                   class_num=6)

if args.kingdom_as_token:
    setattr(config, "kingdom_as_token", True)  # 指示esm生成的嵌入应该增加物种信息
setattr(config, 'embedding_mode', args.embedding_mode)


main(args)
