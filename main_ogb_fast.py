from Trainer.GNN_Trainer_ogb import *
from datasets.ogbn_dataset_new import ogbn_dataset
from datasets import _Sampler
from Trainer.early_stopping_ogb import *
import time
import gc
import itertools

# from tensorboardX import SummaryWriter
from parse_conf import *

dataset = ogbn_dataset(args, name=args.dataset, root=args.root)
args.num_features = dataset.data.x.size(1)
args.num_classes = dataset.num_classes
args.num_nodes = dataset.data.x.size(0)
args.batch_tr = dataset.batch['train']
args.batch_val = dataset.batch['valid']
args.batch_te = dataset.batch['test']
if args.model == 'PNA':
    args.deg = dataset.deg        # for PNA

lamdas = [args.lamda]
lrs = [args.lr]
drops = [args.drop]

results = []
for drop, lr, lamda in itertools.product(drops, lrs, lamdas):
    args.lamda = lamda
    args.lr = lr
    args.drop = drop
    acc_te_i = []

    for i in range(args.repeat):
        t_total = time.time()

        model = Trainer_ogb(args)
        early_stopping = EarlyStopping(model, **stopping_args)
        dataset.reset_iter()

        # for ogbn-proteins
        if args.dataset == 'ogbn-proteins':
            sampler_tr = _Sampler(dataset, phase='train', buffer_size=15)
            sampler_val = _Sampler(dataset, phase='valid', buffer_size=1)
            sampler_te = _Sampler(dataset, phase='test', buffer_size=1)
        # for ogbn-products
        if args.dataset == 'ogbn-products':
            sampler_tr = _Sampler(dataset, phase='train', buffer_size=15)
            sampler_val = _Sampler(dataset, phase='valid', buffer_size=3)
            sampler_te = _Sampler(dataset, phase='test', buffer_size=15)
        # for ogbn-papers100M
        if args.dataset == 'ogbn-papers100M':
            sampler_tr = _Sampler(dataset, phase='train', buffer_size=5)
            sampler_te = _Sampler(dataset, phase='test', buffer_size=3)
            sampler_val = _Sampler(dataset, phase='valid', buffer_size=3)
        if args.dataset == 'ogbn-arxiv':
            sampler_tr = _Sampler(dataset, phase='train', buffer_size=8)
            sampler_te = _Sampler(dataset, phase='test', buffer_size=2)
            sampler_val = _Sampler(dataset, phase='valid', buffer_size=2)
        print("lamda={}, lr={}, drop={}".format(str(lamda), str(lr), str(drop)))

        if args.dataset == 'ogbn-proteins':
            model.reset_bce(dataset.data.y_weight)

        start_time = time.time()
        train_time = []
        op_time = 0.0
        for j in range(args.epochs):
            start_time = time.time()
            loss_k, acc_k = 0., 0.
            for k in range(args.batch_tr):
                temp_lo = model.update(dataset)
                loss_k += temp_lo['loss']
                acc_k += temp_lo['acc']
                # torch.cuda.empty_cache()
            train_time.append(time.time() - start_time)

            acc_k /= args.batch_tr
            loss_k /= args.batch_tr
            temp_val = model.evaluation(dataset, 'valid')
            if j < args.epochs - 1:
                sampler_val.clear()

            log_step = args.eval_step
            if (j + 1) % log_step == 0:
                temp_te = str(round(model.evaluate_test(dataset)['acc'], 4)) \
                    if args.is_gen_curves else None
                if args.is_gen_curves:
                    sampler_te.clear()
                print('epoch :{:4d}, loss:{:.4f}, acc_tr:{:.4f}, loss_v:{:.4f}, acc_v:{:.4f}'
                      .format(j, loss_k, acc_k, temp_val['loss'], temp_val['acc']),
                      'acc_t {}'.format(temp_te),
                      'avr_time: {:.4f}s'.format(np.mean(train_time)),
                      'T2: {:.4f}s'.format(model.T2 / (j+1))
                      )

            iter_results = {'loss': temp_val['loss'], 'acc': temp_val['acc']}
            if (j + 1) >= args.early_stopping:
                stop_vars = [iter_results[key] for key in early_stopping.stop_vars]
                if early_stopping.check(stop_vars, j):
                    break
            # torch.cuda.empty_cache()
            if (j + 1) % 5 == 0:
                gc.collect()

            if args.lrscheduler:
                model.scheduler.step()

        model.load_state_dict(early_stopping.best_state)
        start_time = time.time()
        temp_te = model.evaluate_test(dataset)
        best_val_j = early_stopping.remembered_vals
        print("lamda={}, lr={}, drop={}".format(str(lamda), str(lr), str(drop)))
        print("Test results:",
              "best epoch={}".format(str(early_stopping.best_epoch)),
              "loss_v={:.4f}".format(best_val_j[0]),
              "acc_v={:.4f}".format(best_val_j[1]),
              "loss_t={:.4f}".format(temp_te['loss']),
              "acc_t={:.4f}".format(temp_te['acc']),
              'te_time:{:.4f}s '.format(time.time() - start_time)
              )

        if args.save:
            import os.path as osp
            path = osp.join(dataset.root, 'processed/{}_{}_{}.pt'.format(args.model, args.method, args.layer_num))
            torch.save(model.state_dict(), path)

        del model
        gc.collect()

        acc_te_i.append(temp_te['acc'])
        print("repeat: {:4d}".format(i + 1), "Now test_acc mean={:.4f}, std={:.4f}".
              format(np.mean(acc_te_i), np.std(acc_te_i)))
        # torch.cuda.empty_cache()

        sampler_tr.terminate()
        sampler_val.terminate()
        sampler_te.terminate()

    results.append([np.mean(acc_te_i), lamda, lr, drop, np.std(acc_te_i)])
    idx = np.argsort(np.array(results)[:, 0])[-1]
    print("\nBest acc={:.4f} std={:.6f} lamda={}, lr={}, drop={}\n".
          format(results[idx][0], results[idx][-1], results[idx][1],
                 results[idx][2], results[idx][3])
          )


print(args)
msg = "lamda={:f} test_acc={:.4f} test_std={:.6f}" \
    .format(results[0][1], results[0][0], results[0][-1])
print(msg, '\n\n')
