import os
import torch
import args_maker
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import SPDataset, inv_kingdom_dict, inv_label_dict
from Signal3L_4 import Signal3L4Config, Signal3L4
from utils.basic_utils import save_mydict, clear_memory, setup_logger, set_randomseed
from utils.estimate_utils import tagged_seq_to_cs_multiclass



def predict(model, logger, data_loader):
    """Accepts the benchmark data (ESM embeddings, SP-type id, kingdom id, CS index) and
    generates the classification probability vector (tag scores) and attention scores.
    Based on the classification result, it decides whether to compute the CS index, and if so,
    determines the CS position from the attention scores.

    The final output file stores a dictionary with the following structure, for example:
    {
        'P15516': {
            'seq': 'MKFFVFALILALMLSMTGADSHAKRHHGYKRKFHEKHHSHRGYRSNYLYDN',
            'kingdom': 'EUKARYA',      # label extracted from the SignalP dataset
            'sp_type': 'NO_SP',
            'cs': -1,
            'sp_type_pred': 'SP',      # label obtained by taking the argmax over the original
                                    # 6-dimensional probability vector and mapping the index
                                    # to the corresponding SP-type label
            'cs_pred': (1, 19)
        },
        ...
    }
    """
    final_data_dict = {}
    all_cs_gdt = []
    all_cs_pred = []
    all_kingdom_ids = []

    all_targets = []
    all_glblbl_gdt = []
    all_global_probs = []
    all_pos_preds = []
    all_acc_list = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting", leave=False):
            (
                seq_tokens,
                batch_data,
                pdb_data,
                targets,
                input_mask,
                acc_list,
                glb_label_ids,
                cs_gdts,
                kingdom_ids,
                kindom_mask,
            ) = batch
            seq_tokens = torch.tensor(seq_tokens).to(device)
            batch_data = batch_data.to(device)
            pdb_data = torch.stack(pdb_data).to(device) if isinstance(pdb_data, tuple) else pdb_data.to(device)
            targets_bit_map = targets.to(device)
            input_mask = input_mask.to(device)
            acc_list = list(acc_list)
            glb_label_ids = glb_label_ids.to(device)
            kingdom_ids = kingdom_ids.to(device)
            kindom_mask = kindom_mask.to(device)
            
            global_probs, _, pos_preds, seq_attns, pdb_attns, all_attn_weights = model(seq_tokens, kingdom_ids.long(), batch_data, pdb_data, targets_bit_map, input_mask, kindom_mask)

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

    clear_memory()

    return final_data_dict


def main(args):
    EMB_DIM = args.embed_dim
    BATCHSIZE = args.batch_size
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

    model_path = args.model_path
    model = Signal3L4(config).cuda()
    model.load_state_dict(torch.load(model_path, map_location=device))
    seed = set_randomseed(42)

    filename = os.path.splitext(os.path.basename(args.fasta_file))[0]   # 'example'
    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)

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
        fasta_data_path=args.fasta_file,
        pdb_data_dir=args.pdb_dir,
        seq_emb_dir=args.seq_emb_dir,
        struc_emb_dir=args.struc_emb_dir,
        seq_emb_mode=args.embedding_mode,
    )
    test_loader = DataLoader(test_data, batch_size=BATCHSIZE, collate_fn=test_data.collate_fn, shuffle=False)
    logger.info(f"{len(test_data)} benchmarking sequences.")
    
    pre_dict = predict(model, logger, test_loader)

    logger.info('Saving recommendation result...\n')
    save_mydict(pre_dict, f"{output_path}/{filename}")
    

args = args_maker.predict_args()
device = torch.device(f"cuda:{args.cuda_index}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device.index)

MAX_LEN = 70
main(args)
