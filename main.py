import argparse
from gnn.train import start_train, start_train_cv
from gnn.eval import eval_saved_model, eval_model_inductive
import time


def get_args():
    parser = argparse.ArgumentParser(description="Model Implementation")
    parser.add_argument("--db_name", type=str, default="top10k", help="db name")
    parser.add_argument("--graph_name", type=str, default="full", help="graph name")
    parser.add_argument(
        "--g_to_merge", type=str, default=None, help="the graph needs to be merged"
    )
    parser.add_argument("--dropout", type=float, default=0, help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument(
        "--n-epochs", type=int, default=1000, help="number of training epochs"
    )
    parser.add_argument(
        "--n-hidden", type=int, default=16, help="number of hidden gcn units"
    )
    parser.add_argument(
        "--n-layers", type=int, default=1, help="number of hidden gcn layers"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight for L2 loss"
    )
    parser.add_argument("--r_train", type=float, default=0.8, help="training ratio")
    parser.add_argument("--r_test", type=float, default=0.1, help="testing ratio")
    parser.add_argument(
        "--testing_option",
        type=int,
        default=3,
        help="1:unseen edges; 2: seen edges; 3: all edges",
    )
    parser.add_argument("--r_val", type=float, default=0.1, help="validation ratio")
    parser.add_argument(
        "--model_name", type=str, default="tmp_model_name", help="model_name"
    )
    parser.add_argument("--file_to_write", type=str, default="", help="file_to_write")
    parser.add_argument("--fold", type=int, default=5, help="# of cv fold")
    args = parser.parse_args()
    return args


args = get_args()
print(args)


start_train(args)
eval_saved_model(args)
print("\ndone...")
