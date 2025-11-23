
import argparse

def make_baseline_args(is_parsered=True):
    parser = argparse.ArgumentParser(description="Train MultiDAN model")

    parser.add_argument("--embed_dim", type=int, default=1280, help="embedding dimension")
    parser.add_argument("-b", "--batch_size", type=int, default=256, metavar="N", help="batch size")
    parser.add_argument("--max_length", type=int, default=70, help="max length of sequence")
    parser.add_argument("--num_gpu", type=int, default=3, help="number of GPUs")
    parser.add_argument("--cuda_index", type=int, default=2, help="designated CUDA indes")
    parser.add_argument("--embedding_mode", type=str, default='esm2', help="Embedding type of sequence.")

    # 增加物种信息
    parser.add_argument("--kingdom_as_token", action="store_true", default=True, help="Kingdom ID is first token in the sequence")
    parser.add_argument("--kingdom_embed_size", type=int, default=0, help="If >0, embed kingdom ids to N and concatenate with LM hidden states before CRF.")

    # args for model architecture
    parser.add_argument(
        "--model_architecture", type=str,
        choices=["esm2_t48_15B_UR50D",
                "esm2_t36_3B_UR50D",
                "esm2_t33_650M_UR50D",
                "esm2_t30_150M_UR50D",
                "esm2_t12_35M_UR50D"],
        default="esm2_t30_650M_UR50D",
        help="which model architecture the checkpoint is for",)
    parser.add_argument("--region_regularization_alpha", type=float, default=0.5, help="multiplication factor for the region similarity regularization term")
    parser.add_argument(
        "--constrain_crf",
        default=True,   # action="store_true",
        help="Constrain the transitions of the region-tagging CRF.",
    )
    parser.add_argument("--use_cs_tag", action="store_true", help="Replace last token of SP with C for cleavage site")
    parser.add_argument("--sp_region_labels", default=True, help="Use labels for n,h,c regions of SPs.")
    parser.add_argument("--use_wandb", type=bool, default=True, help="Use wandb if neccessary.")
    parser.add_argument("--wandb_project", type=str, default="SPExplorer1.0", help="Project name of wandb.")

    if is_parsered:
        args = parser.parse_args()
        return args
    else:
        return parser


def predict_args():
    parser = make_baseline_args(is_parsered=False)
    parser.add_argument("--model_path", type=str, help="")
    parser.add_argument("--fasta_file", type=str, help="")
    parser.add_argument("--pdb_dir", type=str, help="")
    parser.add_argument("--model_path", type=str, help="")
    parser.add_argument("--model_path", type=str, help="")
    parser.add_argument("--data_file", type=str, default="Dataset/fasta_data", help="file to fasta data for prediction")
    parser.add_argument("--data_fasta", type=str, default="signalp5_benchmark_set.fasta", help="fasta file for prediction")
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="path to save logs and prediction outcome")
    parser.add_argument("--report_metrics_per_kingdom", default=True, help="Report per-kingdom-metrics on test set or not.")
    parser.add_argument("--kingdom_agonistic", default=False, help="Mask all kingdoms to one specified kingdom or not.")

    args = parser.parse_args()

    return args


