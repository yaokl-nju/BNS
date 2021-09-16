import threading
import time
from global_val import *
import torch
import samplers


class _Sampler(threading.Thread):
    def __init__(self, dataset, phase='train', buffer_size=20, daemon=True):
        super(_Sampler, self).__init__(daemon=daemon)
        self.__is_running = True
        self.daemon = daemon
        self.init_params(dataset, phase, buffer_size)
        self.start()

    def terminate(self):
        self.__is_running = False

    def init_params(self, dataset, phase='train', buffer_size=20):
        self.sample_idx = dataset.idx_dict[phase]
        self.b_num = dataset.batch[phase]

        self.buffer_size = buffer_size
        self.phase = phase
        self.dataset = dataset
        self.b_count = 0
        self.perm = self.iter_idx()
        self.sampler = getattr(samplers, dataset.args.method)

    def iter_idx(self):
        bsize = self.sample_idx.size(0) if self.dataset.args.bsize < 0 else self.dataset.args.bsize
        if self.phase == 'train':
            perm = torch.randperm(self.sample_idx.size(0)).split(bsize)
            perm = perm if self.sample_idx.size(0) % bsize == 0 else perm[:-1]
        else:
            perm = torch.arange(self.sample_idx.size(0)).split(bsize)
        for ids in perm:
            yield ids
        yield None

    def get_iter_idx(self):
        index = next(self.perm)
        if index is None:
            self.perm = self.iter_idx()
            index = next(self.perm)
        return self.sample_idx[index]

    def pass_to_global(self, params):
        global batch
        batch[self.phase].append(params)

    def clear(self):
        if self.phase != 'train':
            self.idx = 0
            global batch
            batch[self.phase].clear()
            self.b_count = 0
            self.perm = self.iter_idx()

    def run(self):
        while self.__is_running:
            global batch
            while len(batch[self.phase]) < self.buffer_size:
                if self.phase != 'train' and self.b_count >= self.b_num:
                    time.sleep(0.1)
                else:
                    ids_0 = self.get_iter_idx()
                    batch_data = self.sampler(ids_0, self.dataset)
                    self.pass_to_global(batch_data)
                    self.b_count += 1
            time.sleep(0.1)