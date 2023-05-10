import hashlib

import numpy as np


def prepare_experts(args, x_train, y_train):
    num_experts = round(len(x_train) / args.points_per_expert)
    train_inds = np.arange(len(x_train))
    np.random.seed(args.partition_seed)
    np.random.shuffle(train_inds)
    hsh = hashlib.md5(np.ascontiguousarray(train_inds)).hexdigest()
    expert_inds = np.array_split(train_inds, num_experts)
    experts_x = [x_train[v] for v in expert_inds]
    experts_y = [y_train[v] for v in expert_inds]
    return experts_x, experts_y, hsh
