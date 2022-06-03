import random
import time
import numpy as np
import torch as th
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
import dgl
import copy
from gnn.eval import performance, evaluate
from gnn.wtagnn import WTAGNN
from graph.graph import GraphLoader
from torch.utils.tensorboard import SummaryWriter


def get_train_mask(labels, ratio):
    shuffle_list = [i for i in range(labels.shape[0])]
    random.shuffle(shuffle_list)
    train_ct = int(len(shuffle_list) * ratio)
    train_mask = np.zeros(labels.shape[0])

    for idx in range(0, train_ct):
        train_mask[shuffle_list[idx]] = 1
    train_mask = th.BoolTensor(train_mask)

    return train_mask


def start_train(args):
    print("\n************start normal training************")
    args.model_name = (
        args.model_name + " " + time.strftime("%m-%d %H:%M:%S", time.localtime())
    )
    gloader = GraphLoader()
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
    n_edges = g.number_of_edges()
    input_node_feat_size, input_edge_feat_size = nf.shape[1], ef.shape[1]

    print("\n************initilize model************")
    # create WTAGNN model
    model = WTAGNN(
        g,
        input_node_feat_size,
        input_edge_feat_size,
        args.n_hidden,
        n_classes,
        args.n_layers,
        F.relu,
        args.dropout,
    )
    print(model)

    n_edges = g.number_of_edges()
    # sampler = dgl.dataloading.as_edge_prediction_sampler(
    #     sampler,
    #     exclude="reverse_id",
    #     reverse_eids=torch.cat(
    #         [torch.arange(n_edges // 2, n_edges), torch.arange(0, n_edges // 2)]
    #     ),
    # )
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    train_dataloader = dgl.dataloading.EdgeDataLoader(
        g,
        train_idx,
        sampler,
        batch_size=2048,
        shuffle=True,
        drop_last=False,
        num_workers=4,
    )

    print("using gpu:", args.gpu)
    th.cuda.empty_cache()
    th.cuda.set_device(args.gpu)
    # nf, ef, e_label = nf.cuda(), ef.cuda(), e_label.cuda()
    g.ndata["features"] = nf
    g.edata["features"] = ef
    g.edata["label"] = e_label
    # e_label = e_label.cuda()
    train_mask, val_mask, test_mask = (
        train_mask.cuda(),
        val_mask.cuda(),
        test_mask.cuda(),
    )
    model.cuda()

    loss_fcn = th.nn.CrossEntropyLoss()
    # use optimizer
    optimizer = th.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # start training
    print("\n************start training************")
    writer = SummaryWriter("./logs/" + args.model_name)
    dur, max_acc = [], -1
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()

        for input_nodes, edge_subgraph, blocks in train_dataloader:
            # print(edge_subgraph)
            # print("blocks:")
            # print(blocks)
            blocks = [b.to(th.device(args.gpu)) for b in blocks]
            edge_subgraph = edge_subgraph.to(th.device(args.gpu))
            nf_part = edge_subgraph.ndata["features"]
            ef_part = edge_subgraph.edata["features"]
            e_label_part = edge_subgraph.edata["label"]
            # edge_predictions = model(edge_subgraph, blocks, input_features)

            train_mask = get_train_mask(e_label_part, 0.8)
            train_mask = train_mask.cuda()
            n_logits, e_logits = model(edge_subgraph, nf_part, ef_part)
            loss = loss_fcn(e_logits[train_mask], e_label_part[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)
        model_copy = copy.deepcopy(model)
        model_copy = model_copy.to("cpu")
        acc, predictions, labels = evaluate(model_copy, g, nf, ef, e_label, val_mask)

        writer.add_scalar("acc", acc, epoch)
        writer.add_scalar("loss", loss.item(), epoch)

        # save the best model
        if acc > max_acc:
            max_acc = acc
            th.save(model.state_dict(), "./output/best.model." + args.model_name)

        print(
            "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
            "ETputs(KTEPS) {:.2f}".format(
                epoch, np.mean(dur), loss.item(), acc, n_edges / np.mean(dur) / 1000
            )
        )

    # load the best model
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

    # best_model.cuda()
    best_model.load_state_dict(th.load("./output/best.model." + args.model_name))

    acc, predictions, labels = evaluate(best_model, g, nf, ef, e_label, test_mask)
    precision, recall, tnr, tpr, f1 = performance(
        predictions.tolist(), labels.tolist(), acc
    )

 
    args.model_name = (
        args.model_name + "_cv " + time.strftime("%m-%d %H:%M:%S", time.localtime())
    )
    print("\n************start CV training************")
    gloader = GraphLoader()
    g, nf, ef, e_label, _, _, _, _, _, _ = gloader.load_graph(args)

    n_classes = 2
    n_edges = g.number_of_edges()
    input_node_feat_size, input_edge_feat_size = nf.shape[1], ef.shape[1]

    # if args.gpu < 0:
    #     cuda = False
    # else:
    #     cuda = True
    print("using gpu:", args.gpu)
    th.cuda.empty_cache()
    th.cuda.set_device(args.gpu)
    # nf, ef = nf.cuda(), ef.cuda()
    g.ndata["features"] = nf
    g.edata["features"] = ef
    g.edata["label"] = e_label

    print("\n************start training for {:d} folds************".format(args.fold))
    kf = StratifiedKFold(n_splits=args.fold, shuffle=True)
    kf.get_n_splits()
    print(kf)
    fold = 0
    total_precision = total_acc = total_recall = 0

    writer = SummaryWriter("./logs/" + args.model_name + "_cv")
    # create WTAGNN model
    model = WTAGNN(
        g,
        input_node_feat_size,
        input_edge_feat_size,
        args.n_hidden,
        n_classes,
        args.n_layers,
        F.relu,
        args.dropout,
    )
    print(model)
    model.cuda()
    EPOCH = 0
    for train_index, test_index in kf.split(e_label, e_label):
        fold += 1
        print("\nfold #: ", str(fold))
        train_mask = np.zeros(e_label.shape[0])
        train_mask[train_index] = 1
        train_mask = th.BoolTensor(train_mask)

        test_mask = np.zeros(e_label.shape[0])
        test_mask[test_index] = 1
        test_mask = th.BoolTensor(test_mask)

        g.edata["train_mask"] = train_mask

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        train_dataloader = dgl.dataloading.EdgeDataLoader(
            g,
            train_index,
            sampler,
            batch_size=2048,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )

        # apply cuda()
        # e_label = e_label.cuda()
        train_mask = train_mask.cuda()
        test_mask = test_mask.cuda()

        loss_fcn = th.nn.CrossEntropyLoss()
        # use optimizer
        optimizer = th.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        dur = []
        max_acc = -1
        for epoch in range(args.n_epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()
            # forward
            for input_nodes, edge_subgraph, blocks in train_dataloader:
                # print(edge_subgraph)
                # print("blocks:")
                # print(blocks)
                blocks = [b.to(th.device(args.gpu)) for b in blocks]
                edge_subgraph = edge_subgraph.to(th.device(args.gpu))
                nf_part = edge_subgraph.ndata["features"]
                ef_part = edge_subgraph.edata["features"]
                e_label_part = edge_subgraph.edata["label"]
                # edge_predictions = model(edge_subgraph, blocks, input_features)
                # forward
                n_logits, e_logits = model(edge_subgraph, nf_part, ef_part)
                loss = loss_fcn(
                    e_logits[edge_subgraph.edata["train_mask"]],
                    e_label_part[edge_subgraph.edata["train_mask"]],
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)
            model_copy = copy.deepcopy(model)
            model_copy = model_copy.to("cpu")
            acc, predictions, labels = evaluate(
                model_copy, g, nf, ef, e_label, test_mask
            )
            # save the best model
            if acc > max_acc:
                max_acc = acc
                th.save(
                    model.state_dict(),
                    "./output/best.model." + args.model_name + ".fold." + str(fold),
                )
            writer.add_scalar("acc", acc, EPOCH)
            writer.add_scalar("loss", loss.item(), EPOCH)
            print(
                "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                "ETputs(KTEPS) {:.2f}".format(
                    epoch, np.mean(dur), loss.item(), acc, n_edges / np.mean(dur) / 1000
                )
            )
            EPOCH = EPOCH + 1

        # load the best model
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
        # best_model.cuda()
        best_model.load_state_dict(
            th.load("./output/best.model." + args.model_name + ".fold." + str(fold))
        )

        acc, predictions, labels = evaluate(best_model, g, nf, ef, e_label, test_mask)

        precision, recall, tnr, tpr, f1 = performance(
            predictions.tolist(), labels.tolist(), acc
        )

        total_precision += precision
        total_acc += acc
        total_recall += recall
    writer.add_scalar("acc_fold", total_acc / args.fold * 100, fold)
    writer.add_scalar("precision_fold", total_precision / args.fold * 100, fold)
    writer.add_scalar("recall_fold", total_recall / args.fold * 100, fold)
    print("\n************training done! Averaged model performance************")
    print(
        "acc/pre/rec: ",
        str("{:.2f}".format(total_acc / args.fold * 100))
        + "%/"
        + str("{:.2f}".format(total_precision / args.fold * 100))
        + "%/"
        + str("{:.2f}".format(total_recall / args.fold * 100))
        + "%",
    )
