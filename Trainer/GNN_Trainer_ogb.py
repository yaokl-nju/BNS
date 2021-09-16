import torch.nn.functional as F
from nn.GNNs import *
import gc
from ogb.nodeproppred import Evaluator
import time

def get_optimizer(name, params, lr, lamda=0):
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, weight_decay=lamda, momentum=0.9)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, weight_decay=lamda)
    elif name == 'adagrad':
        return torch.optim.Adagrad(params, lr=lr, weight_decay=lamda)
    elif name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=lamda)
    elif name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=lamda)
    elif name == 'adamax':
        return torch.optim.Adamax(params, lr=lr, weight_decay=lamda)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

def get_scheduler(optimizer, name, epochs):
    if name == 'Mstep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//3*2, epochs//4*3], gamma = 0.1)
    elif name == 'Expo':
        gamma = (1e-6)**(1.0/epochs)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma, last_epoch=-1)
    elif name == 'Cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min = 1e-16, last_epoch=-1)
    else:
        raise ValueError('Wrong LR schedule!!')

class Trainer_ogb(torch.nn.Module):
    def __init__(self, args):
        super(Trainer_ogb, self).__init__()
        self.args = args
        self.gnn_net = GNN(args).to(self.args.device)
        self.optimizer = get_optimizer(
            args.optimizer,
            self.gnn_net.parameters(),
            args.lr,
            args.lamda
        )
        self.scheduler = get_scheduler(self.optimizer, args.lrschedular, args.epochs)
        self.criterion = torch.nn.CrossEntropyLoss()
        if args.cuda:
            self.criterion.cuda()
        self.epsilon = 1e-7

        self.evaluator = Evaluator(name=self.args.dataset)
        # training time: perform forward and backward calculation
        self.T2 = 0.0

    def reset_bce(self, y_weight):
        self.bcelogit = torch.nn.BCEWithLogitsLoss(pos_weight=y_weight)
        if self.args.cuda:
            self.bcelogit.cuda()

    def loss_GNN(self, labels, probs, scores, train_idx):
        if labels.dim() > 1:
            loss_sup = self.bcelogit(scores[train_idx], labels.to(torch.float))
        else:
            loss_sup = F.nll_loss(torch.log(probs[train_idx] + self.epsilon), labels)
        return loss_sup

    def update(self, dataset):
        train_idx, labels, feat, graph = dataset.sample('train')

        start_time = time.time()
        self.train()
        self.optimizer.zero_grad()
        self.zero_grad()

        probs, scores = self.gnn_net(feat, graph)
        loss = self.loss_GNN(labels, probs, scores, train_idx)
        loss.backward()
        self.optimizer.step()
        self.T2 += (time.time() - start_time)
        torch.cuda.empty_cache()
        acc = self.accuracy(probs[train_idx].data, scores[train_idx].data, labels)
        return {'loss': loss.item(), 'acc': acc.item()}

    def accuracy(self, probs, scores, labels):
        if self.args.dataset == 'ogbn-proteins':
            rocauc = self.evaluator.eval({
                'y_true': labels,
                'y_pred': scores,
            })['rocauc']
            return rocauc
        else:
            preds = probs.max(1)[1].type_as(labels)
            correct = preds.eq(labels).double()
            correct = correct.sum()
            return correct / len(labels)

    @torch.no_grad()
    def evaluation(self, dataset, phase):
        probs, labels, scores = [], [], []
        for i in range(dataset.batch[phase]):
            self.eval()
            idx_i, labels_i, feat, graph = dataset.sample(phase)
            probs_i, scores_i = self.gnn_net(feat, graph)
            probs.append(probs_i[idx_i].data)
            scores.append(scores_i[idx_i].data)
            labels.append(labels_i)
            self.zero_grad()
            torch.cuda.empty_cache()
        probs = torch.cat(probs)
        scores = torch.cat(scores)
        labels = torch.cat(labels)
        if labels.dim() > 1:
            loss = self.bcelogit(scores, labels.to(torch.float))
        else:
            loss = F.nll_loss(torch.log(probs + self.epsilon), labels)
        acc = self.accuracy(probs, scores, labels)
        return {'acc': acc.item(), 'loss': loss.item()}

    @torch.no_grad()
    def evaluate_full(self, dataset, bsize=10):
        self.eval()
        from samplers.Subgraph import subgraph
        num_nodes = dataset.num_nodes
        perm = torch.arange(num_nodes).split(bsize)
        x = dataset.data.x
        for l in range(self.args.layer_num):
            xconv = []
            for b in range(len(perm)):
                graph, ids_map = subgraph(perm[b], dataset)
                x_b = self.gnn_net.forward_layer(x[ids_map].to(self.args.device), graph.to(self.args.device), l)
                xconv.append(x_b.cpu())
                torch.cuda.empty_cache()
            x = torch.cat(xconv, dim=0)
            # del xconv
        mask = dataset.idx_dict['test']
        probs = torch.softmax(x[mask], dim=-1)
        acc = self.accuracy(probs, x[mask], dataset.data.y[mask])
        return {'acc': acc.item(), 'loss': 0.0}

    @torch.no_grad()
    def evaluate_test(self, dataset):
        if self.args.method == 'GraphSAINT':
            if self.args.dataset != 'ogbn-papers100M':
                batch = 10
                bsize = int(dataset.num_nodes * 1.0 / batch)
                return self.evaluate_full(dataset, bsize)
            else:
                return self.evaluation(dataset, 'test')
        else:
            return self.evaluation(dataset, 'test')