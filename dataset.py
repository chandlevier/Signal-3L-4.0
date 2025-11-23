import os
import torch
import numpy as np
import torch.cuda
from torch.utils.data import Dataset
from typing import Union, List, Dict, Any, Sequence
from pathlib import Path
from utils.label_processing_utils import process_SP

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# [S: Sec/SPI signal peptide | T: Tat/SPI signal peptide | L: Sec/SPII signal peptide | I: cytoplasm | M: transmembrane | O: extracellular]
SIGNALP_VOCAB = [
    "S",
    "I",
    "M",
    "O",
    "T",
    "L",
]  # NOTE eukarya only uses {'I', 'M', 'O', 'S'}
SIGNALP_GLOBAL_LABEL_DICT = {"NO_SP": 0, "SP": 1, "LIPO": 2, "TAT": 3}
SIGNALP_KINGDOM_DICT = {"EUKARYA": 0, "POSITIVE": 1, "NEGATIVE": 2, "ARCHAEA": 3}
SIGNALP6_GLOBAL_LABEL_DICT = {
    "NO_SP": 0,
    "SP": 1,
    "LIPO": 2,
    "TAT": 3,
    "TATLIPO": 4,
    "PILIN": 5,
}


def signalp_pad_sequences(sequences: Sequence, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], str):
        for i, seq in enumerate(sequences):
            if len(seq) < 70:
                sequences[i] = seq.ljust(70, constant_value) 
        return sequences

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        with torch.no_grad():
            arr[arrslice] = seq

    return array

def parse_twoline_fasta(filepath: Union[str, Path], maxlen: int = 70):
    with open(filepath, "r") as f:
        lines = f.read().splitlines()
        identifiers = lines[::2]
        sequences = lines[1::2]
    sequences = [seq[:maxlen] for seq in sequences]
    return identifiers, sequences

class SPDataset(Dataset):
    """Converts label sequences to array for DAN.
    esm_alphabet:       esm alphabet to yield esm token of sequence
    data_path:          training set 2-line fasta
    partition_id :      list of partition ids to use
    kingdom_id:         list of kingdom ids to use
    type_id :           list of type ids to use
    add_special_tokens: add cls, sep tokens to sequence, 闲置中
    """

    def __init__(
        self,
        device,
        max_len,
        fasta_data_path: Union[str, Path],   
        pdb_data_dir: Union[str, Path],
        seq_emb_dir: Union[str, Path],
        struc_emb_dir: Union[str, Path],
        partition_id: List[str] = [0, 1, 2],
        kingdom_id: List[str] = ["EUKARYA", "ARCHAEA", "NEGATIVE", "POSITIVE"],
        type_id: List[str] = ["LIPO", "NO_SP", "SP", "TAT", "TATLIPO", "PILIN"],
        esm_model=None,
        esm_alphabet=None,
        saprot_model=None,
        saprot_tokenizer=None,
        add_special_tokens=False,
        label_vocab=None,
        vary_n_region=False,
        seq_emb_mode='saprot',
    ):

        super().__init__()

        self.data_file = Path(fasta_data_path)
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)
        if 'signalp' in fasta_data_path:
            self.is_signalp = True
        else:
            self.is_signalp = False
            
        self.seq_emb_dir = seq_emb_dir
        self.struc_emb_dir = struc_emb_dir

        self.add_special_tokens = add_special_tokens
        self.device = device
        self.embedding_model = esm_model
        self.alphabet = esm_alphabet
        self.saprot_model = saprot_model
        self.saprot_tokenizer = saprot_tokenizer
        self.max_len = max_len
        self.global_label_dict = SIGNALP6_GLOBAL_LABEL_DICT
        self.kingdom_label_dict = SIGNALP_KINGDOM_DICT
        self.type_id = type_id
        self.partition_id = partition_id
        self.kingdom_id = kingdom_id
        self.label_vocab = label_vocab
        self.vary_n_region = vary_n_region
        self.pdb_dir = pdb_data_dir
        self.seq_emb_mode = seq_emb_mode

        # Load and filter the data
        self.identifiers, self.sequences = parse_twoline_fasta(self.data_file, maxlen=self.max_len)
        self.global_labels = [x.split("|")[2] for x in self.identifiers]
        self.labels = ['X' * 70 for _ in range(len(self.identifiers))]
        self.cs = list(map(int, [x.split("|")[-1] for x in self.identifiers]))
        
        self.accession = [x.split("|")[0][1:] for x in self.identifiers]
        self.kingdom_ids = [x.split("|")[1] for x in self.identifiers]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index):
        item = self.sequences[index]
        acc = self.accession[index]
        labels = self.labels[index]
        global_label = self.global_labels[index]
        
        global_label_id = self.global_label_dict[global_label]
        kingdom_id = (
            SIGNALP_KINGDOM_DICT[self.kingdom_ids[index]]
            if hasattr(self, "kingdom_ids")
            else None
        )

        trunc_seq = min(len(item), self.max_len)
        seq_token = self.trans_seq(item, self.max_len)

        with torch.no_grad():
            seq_embedding = torch.load(os.path.join(self.seq_emb_dir, f"{acc}.pt"))['representations'][33][:trunc_seq, :].to(self.device)
            pdb_embedding = torch.tensor(torch.load(os.path.join(self.struc_emb_dir, f"{acc}.pt"))).to(self.device)
            
        input_mask = np.ones_like(seq_token)
        
        # also need to return original tags or cs
        if self.is_signalp:
            if global_label == "NO_SP":
                cs = -1
            elif global_label == "SP":
                cs = (
                    labels.rfind("S") + 1
                )  # +1 for compatibility. CS reporting uses 1-based instead of 0-based indexing
            elif global_label == "LIPO":
                cs = labels.rfind("L") + 1
            elif global_label == "TAT":
                cs = labels.rfind("T") + 1
            elif global_label == "TATLIPO":
                cs = labels.rfind("T") + 1
            elif global_label == "PILIN":
                cs = labels.rfind("P") + 1
            else:
                raise NotImplementedError(f"Unknown CS defintion for {global_label}")

            label_matrix = process_SP(
                labels,
                item,
                sp_type=global_label,
                vocab=self.label_vocab,
                stochastic_n_region_len=self.vary_n_region,
            )
        
        else:
            cs = self.cs[index]
            label_matrix = np.zeros((70, 37))
            
        return_tuple = (
            seq_token,
            seq_embedding,
            pdb_embedding,
            label_matrix, 
            input_mask,
            acc,
            global_label_id,
            cs,
            kingdom_id,
        )

        # Manually clear the single_embedding after returning it
        del seq_embedding, pdb_embedding
        torch.cuda.empty_cache()

        return return_tuple

    def cls_num_count(self):
        from collections import Counter
        global_label_id = [self.global_label_dict[glb_lbl] for glb_lbl in self.global_labels]
        cls_counts = Counter(global_label_id)
        cls_num_list = [cls_counts[i] for i in range(len(cls_counts))]
        return cls_num_list

    
    
    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        # unpack the list of tuples
        seq_tokens, input_emb, pdb_emb, label_ids, mask, acc_list, global_label_ids, cleavage_sites, kingdom_ids = tuple(
            zip(*batch)
        )
        
        input_emb = signalp_pad_sequences(input_emb, 0)
        pdb_emb = signalp_pad_sequences(pdb_emb, 0)
        
        # ignore_index is -1
        targets = signalp_pad_sequences(label_ids, -1)
        targets = np.stack(targets)
        targets = torch.from_numpy(targets)
        mask = torch.from_numpy(signalp_pad_sequences(mask, 0))
        global_targets = torch.tensor(global_label_ids)
        cleavage_sites = torch.tensor(cleavage_sites)

        return_tuple = (seq_tokens, input_emb, pdb_emb, targets, mask, acc_list, global_targets, cleavage_sites)
        if hasattr(self, "sample_weights"):
            sample_weights = torch.tensor(sample_weights)
            return_tuple = return_tuple + (sample_weights,)
        if hasattr(self, "kingdom_ids"):
            kingdom_ids = torch.tensor(kingdom_ids)
            return_tuple = return_tuple + (kingdom_ids,)

        return return_tuple
    
    def trans_seq(self, seq, padding_length):
        # Translates amino acids into numbers
        a = []
        trans_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'L': 5, 'A': 6, 'G': 7, 'V': 8, 'E': 9, 'S': 10, 'I': 11, 'K': 12, 'R': 13, 'D': 14, 'T': 15, 'P': 16, 'N': 17, 'Q': 18, 'F': 19, 'Y': 20, 'M': 21, 'H': 22, 'C': 23, 'W': 24, 'X': 25, 'U': 26, 'B': 27, 'Z': 28, 'O': 29}

        for i in range(len(seq)):
            if (seq[i] in trans_dict.keys()):
                a.append(trans_dict.get(seq[i]))
            else:
                print("Unknown letter:" + str(seq[i]))
                a.append(trans_dict.get('X'))
        while(len(a) < padding_length):
            a.append(0)

        return a

