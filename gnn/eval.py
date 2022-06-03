import dgl
import torch as th
import torch.nn.functional as F
import numpy as np

from graph.graph import GraphLoader
from gnn.wtagnn import WTAGNN

# print out the performance related numbers
def performance(pred, labels, acc=None):
    print(
        "\n************model performance on {:d} test edges************".format(
            len(pred)
        )
    )
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(labels)):
        tp += 1 if pred[i] == 1 and labels[i] == 1 else 0
        fp += 1 if pred[i] == 1 and labels[i] == 0 else 0
        tn += 1 if pred[i] == 0 and labels[i] == 0 else 0
        fn += 1 if pred[i] == 0 and labels[i] == 1 else 0
    print("tp", tp, "fp", fp, "tn", tn, "fn", fn)

    acc = float((tp + tn) / (tp + tn + fp + fn)) if acc == None else acc
    precision = float(tp / (tp + fp))
    recall = float(tp / (tp + fn))
    tnr = float(tn / (tn + fp))
    tpr = float(tp / (tp + fn))
    f1 = 2 * float(precision * recall / (precision + recall))

    print("accuracy {:.4f}".format(acc))
    print("precision {:.4f}".format(precision))
    print("recall {:.4f}".format(recall))
    print("tnr {:.4f}".format(tnr))
    print("tpr {:.4f}".format(tpr))
    print("f1 {:.4f}".format(f1))
    print(
        "acc/pre/rec: ",
        str("{:.2f}".format(acc * 100))
        + "%/"
        + str("{:.2f}".format(precision * 100))
        + "%/"
        + str("{:.2f}".format(recall * 100))
        + "%",
    )
    return precision, recall, tnr, tpr, f1


def evaluate(model, g, nf, ef, labels, mask):
    model.eval()
    with th.no_grad():
        n_logits, e_logits = model(g, nf, ef)
        e_logits = e_logits[mask]
        labels = labels[mask]
        _, indices = th.max(e_logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels), indices, labels


def eval_saved_model(args):
    gloader = GraphLoader()
    if args.g_to_merge is not None:  # load two graph and merge together
        (
            g,
            nf,
            ef,
            e_label,
            train_mask,
            test_mask,
            val_mask,
        ) = gloader.load_and_merge_graph(args)
    else:  # eval the model on the same graph used for training
        (
            g,
            nf,
            ef,
            e_label,
            train_mask,
            test_mask,
            val_mask,
            train_idx,
            test_idex,
            val_idx,
        ) = gloader.load_graph(args)

    n_classes = 2
    input_node_feat_size, input_edge_feat_size = nf.shape[1], ef.shape[1]

    # load the pre-trained model
    best_model = WTAGNN(
        g,
        input_node_feat_size,
        input_edge_feat_size,
        args.n_hidden,
        n_classes,
        args.n_layers,
        F.relu,
        args.dropout,
    )
    best_model.load_state_dict(th.load("./output/best.model." + args.model_name))
    print("model load from: ./output/best.model." + args.model_name)

    acc, predictions, labels = evaluate(best_model, g, nf, ef, e_label, test_mask)
    precision, recall, tnr, tpr, f1 = performance(
        predictions.tolist(), labels.tolist(), acc
    )