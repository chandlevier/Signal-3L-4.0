import os
import torch
import numpy as np
from tqdm import tqdm

from dataset import SPDataset
from utils.basic_utils import save_mydict, clear_memory, setup_logger, set_randomseed
from utils.estimate_utils import tagged_seq_to_cs_multiclass
from Signal3L_4 import Signal3L4Config, Signal3L4


SIGNALP_KINGDOM_DICT = {"EUKARYA": 0, "POSITIVE": 1, "NEGATIVE": 2, "ARCHAEA": 3}
label_dict = {
    "NO_SP": 0,
    "SP": 1,
    "LIPO": 2,
    "TAT": 3,
    "TATLIPO": 4,
    "PILIN": 5,
}
inv_kingdom_dict = {v: k for k, v in SIGNALP_KINGDOM_DICT.items()}
inv_label_dict = {v: k for k, v in label_dict.items()}
MAX_LEN = 70       # 最长长度控制在200即可
   



def predict(model, logger, bench_loader, criterion=None, split='bench'):
    """接受传来的基准测试数据(esm embedding, sp type id, kingdom id, cs index), 生成分类概率向量tag score和attention score
    基于分类结果决定是否计算CS index，再基于attention score计算CS index位置
    最终文件中存入字典变量，举例如下
    {'P15516': 
        {'seq': 'MKFFVFALILALMLSMTGADSHAKRHHGYKRKFHEKHHSHRGYRSNYLYDN',
        'kingdom': 'EUKARYA', 该标签由SignalP数据集提取出
        'sp_type': 'NO_SP',
        'cs': -1,
        'sp_type_pred': 'SP', 该标签由原始6维P最大值索引对应的sp_type标签得出
        'cs_pred': (1, 19)
        },...
    }"""
    final_data_dict = {}
    all_cs_gdt = []
    all_cs_pred = []
    batch_loss_list = []
    all_kingdom_ids = []

    all_targets = []
    all_glblbl_gdt = []
    all_global_probs = []
    all_pos_preds = []
    all_acc_list = []
    
    with torch.no_grad():
        for batch in tqdm(bench_loader, desc="Benchmarking", leave=False):
            (
                seq_tokens,
                batch_data,     # tensor型张量, padding后的esm embedding, 维度为(batchsize, 70, 1280), 默认在CPU上
                pdb_data,
                targets,        # label_matrix, (batchsize, 70, 37)
                input_mask,     # tensor型张量, (batchsize, 70)
                acc_list,       # tuple型变量
                glb_label_ids,  # tensor型张量, 按照预先定义好的索引映射字典将batch内序列的SP种类信息映射到数字0~5上
                cs_gdts,        # tensor型张量, 存储batch内序列的SP索引对
                kingdom_ids,    # tensor型张量, 按照预先定义好的索引映射字典将batch内序列的物种信息映射到数字0~3上
            ) = batch
            seq_tokens = torch.tensor(seq_tokens).to(device)
            batch_data = batch_data.to(device)    # shape: (batch, batch_maxlen, 1280)
            pdb_data = torch.stack(pdb_data).to(device) if isinstance(pdb_data, tuple) else pdb_data.to(device)
            targets_bit_map = targets.to(device)
            input_mask = input_mask.to(device)
            acc_list = list(acc_list)
            glb_label_ids = glb_label_ids.to(device)
            kingdom_ids = kingdom_ids.to(device)
            
            crfloss, global_probs, _, pos_preds, _, _, _ = model(seq_tokens, kingdom_ids.long(), batch_data, pdb_data, targets_bit_map, input_mask)
            crfloss = crfloss.mean()
            ldam_loss = criterion(global_probs, glb_label_ids.long()) if criterion is not None else None    # criterion输入维度为(b, 6)和(b)
            val_batch_loss = crfloss + ldam_loss if criterion is not None else crfloss
            batch_loss_list.append(val_batch_loss)

            all_targets.append(targets_bit_map.detach().cpu().numpy())
            all_glblbl_gdt.append(glb_label_ids.detach().cpu().numpy())
            all_global_probs.append(global_probs.detach().cpu().numpy())
            all_pos_preds.append(pos_preds.detach().cpu().numpy())
            all_kingdom_ids.append(kingdom_ids.detach().cpu().numpy())
            all_cs_gdt.append(cs_gdts.detach().cpu().numpy())
            all_acc_list.extend(acc_list)
            
            del batch_data, batch, targets_bit_map, input_mask, pos_preds, _, seq_tokens
            clear_memory()

    all_targets = np.concatenate(all_targets)
    all_glblbl_gdt = np.concatenate(all_glblbl_gdt)
    all_global_probs = np.concatenate(all_global_probs)
    all_pos_preds = np.concatenate(all_pos_preds)
    all_kingdom_ids = np.concatenate(all_kingdom_ids)
    all_cs_gdt = np.concatenate(all_cs_gdt) if args.sp_region_labels else None

    
    sp_tokens = [5, 11, 19, 26, 31]
    all_glblbl_pred = all_global_probs.argmax(axis=1)
    all_cs_pred = tagged_seq_to_cs_multiclass(all_pos_preds, sp_tokens=sp_tokens)
    all_cs_pred[np.isnan(all_cs_pred)] = -1

    
    for (acc, glb_id, cs_gdt, kingdom_id, glb_pred, cs_pred) in zip(all_acc_list, all_glblbl_gdt, all_cs_gdt, all_kingdom_ids, all_glblbl_pred, all_cs_pred):
        result = {}
        result['kingdom'] = inv_kingdom_dict[kingdom_id]
        result['sp_type'] = inv_label_dict[glb_id]
        result['cs'] = cs_gdt
        result['sp_type_pred'] = inv_label_dict[glb_pred]
        result['cs_pred'] = int(cs_pred)
        final_data_dict[acc] = result
        logger.info(f'Recommended SP Segments of Sequence {acc}({inv_label_dict[glb_id]}): type = {inv_label_dict[glb_pred]}')
        if glb_id != 0:
            logger.info(f"Recommended SP Segments Index: {(result['cs_pred'])} (gdt: {cs_gdt})")
    bench_loss = sum(batch_loss_list) / len(batch_loss_list) if len(batch_loss_list) != 0 else 0
    logger.info(f'Benchmark Loss: {bench_loss}.')
    clear_memory()

    return final_data_dict


        
# -------------------------------- 正式模型部分-模型运行 -------------------------------- #
import args_maker
from torch.utils.data import DataLoader

def main(args):
    fasta_file = os.path.join(args.data_file, args.data_fasta)
    
    model_path = args.model_path
    model = Signal3L4(config).cuda()
    model.load_state_dict(torch.load(model_path, map_location=device))
    seed = set_randomseed(42)

    # 文件路径设置
    filename = fasta_file.split('/')[-1][:-6].split('_')[0] # "signalp5"
    output_path = f'{args.output_dir}'
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

    benchmark_id = [0, 1, 2, 3, 4, 5]
    kingdoms = ["EUKARYA", "ARCHAEA", "NEGATIVE", "POSITIVE"]

    bench_data = SPDataset(
        device,
        MAX_LEN,
        args.input_sequence_file,   # data/seq_data/example.fasta
        "Dataset/pdb_data/signalp",
        partition_id=benchmark_id,
        kingdom_id=kingdoms,
        seq_emb_mode=args.embedding_mode,
    )
    bench_loader = DataLoader(bench_data, batch_size=args.batch_size, collate_fn=bench_data.collate_fn, shuffle=False)
    logger.info(f"{len(bench_data)} benchmarking sequences.")
    
    # 预测
    pre_dict = predict(model, logger, bench_loader)

    # 存储预测结果
    logger.info('Saving recommendation result...\n')
    save_mydict(pre_dict, f"{output_path}/{filename}_{epoch}")    # "SP_model/signalp_train/0912/test_0_valid_1/signalp5_benchmark_set.pkl"
    

args = args_maker.predict_args()
device = torch.device(f"cuda:{args.cuda_index}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device.index)

epoch = args.epoch
EMB_DIM = args.embed_dim
BATCHSIZE = args.batch_size
MAX_LEN = 70

# 模型参数设置 默认不包含kingdom信息
config = Signal3L4Config(
    sp_region_labels=args.sp_region_labels,
    constrain_crf=args.constrain_crf,
    embedding_dim=EMB_DIM,
    hidden_dim=EMB_DIM,
    class_num=6
)

if args.kingdom_as_token:
    setattr(config, "kingdom_as_token", True)  # 指示esm生成的嵌入应该增加物种信息
setattr(config, 'embedding_mode', args.embedding_mode)

main(args)
