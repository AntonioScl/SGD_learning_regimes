import argparse
import os
import pickle
import subprocess
import time

import numpy as np
from jax.config import config

from .train import execute


def main():
    git = {
        'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
        'status': subprocess.getoutput('git status -z'),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_init", default=0)
    parser.add_argument("--seed_batch", default=0)
    parser.add_argument("--seed_trainset", default=None)
    parser.add_argument("--seed_testset", default=-1)

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ptr", type=int, required=True)
    parser.add_argument("--pte", type=int, required=True)
    parser.add_argument("--d", type=int)

    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--act", type=str, default='relu')
    parser.add_argument("--act_beta", type=float, default=1.0)
    parser.add_argument("--L", type=int)
    parser.add_argument("--h", type=int, default=1)

    parser.add_argument("--loss", type=str, default="hinge")
    parser.add_argument("--alpha", type=float, required=True)

    parser.add_argument("--dynamics", type=str, required=True)
    parser.add_argument("--bs", type=int, default=None)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--temp", type=float)

    parser.add_argument("--regularizer", type=str)
    parser.add_argument("--regu_scale", type=float, default=0.0)
    parser.add_argument("--data_chi", type=float, default=0.0) # Depletion exponent for depleted_sign dataset

    parser.add_argument("--max_wall", type=float, required=True)
    parser.add_argument("--max_step", type=float, default=np.inf)
    parser.add_argument("--mind_stop", type=float, default=1.0)

    parser.add_argument("--ckpt_save_parameters", type=int, default=0)
    parser.add_argument("--ckpt_grad_stats", type=int, default=0)
    parser.add_argument("--ckpt_drift", type=int, default=0)
    parser.add_argument("--ckpt_drift_n0", type=int, default=16)
    parser.add_argument("--ckpt_kernels", type=int, default=0)
    parser.add_argument("--ckpt_save_pred", type=int, default=0)
    parser.add_argument("--ckpt_modulo", type=int, default=1)
    parser.add_argument("--ckpt_save_gradoverlap", type=int, default=0)
    parser.add_argument("--ckpt_save_mult", type=int, default=0)  # Save observables at log-spaced steps up to step ckpt_save_mult

    parser.add_argument("--dtype", type=str, default="f32")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args().__dict__

    if args['dtype'] == "f64":
        config.update("jax_enable_x64", True)
    else:
        config.update("jax_enable_x64", False)
        assert args['dtype'] == "f32"

    # if no batch size is given, then do GD
    if args['bs'] is None:
        args['bs'] = args['ptr']

    # if no seed training_set is given, then change it together with the initialization seed
    if args['seed_trainset'] is None:
        args['seed_trainset'] = - int(args['seed_init']) - 2 # -2 because this way is different from the test_set seed (which is -1)

    # if ckpt_grad_stats is -1, then use ptr
    if args['ckpt_grad_stats'] ==-1:
        args['ckpt_grad_stats'] = args['ptr']

    # dt and derivatives
    assert (args['dt'] is not None) + (args['temp'] is not None) == 1

    if args['temp'] is not None:
        args['dt'] = args['temp'] * args['h'] * args['bs']

    if args['temp'] is None:
        args['temp'] = args['dt'] / (args['h'] * args['bs'])
    # end

    if isinstance(args['seed_init'], str) and 'time' in args['seed_init']:
        args['seed_init'] = round(time.time() * int(args['seed_init'][4:]))

    if isinstance(args['seed_batch'], str) and 'time' in args['seed_batch']:
        args['seed_batch'] = round(time.time() * int(args['seed_batch'][4:]))

    if args['seed_init'] == 'seed_trainset':
        args['seed_init'] = args['seed_trainset']

    if args['seed_batch'] == 'seed_init':
        args['seed_batch'] = args['seed_init']

    args['seed_init'] = int(args['seed_init'])
    args['seed_batch'] = int(args['seed_batch'])
    args['seed_trainset'] = int(args['seed_trainset'])
    args['seed_testset'] = int(args['seed_testset'])

    with open(args['output'], 'wb') as handle:
        pickle.dump(args, handle)

    saved = False
    try:
        for data in execute(**args, yield_time=10.0):
            data['git'] = git
            data['args'] = args
            with open(args['output'], 'wb') as handle:
                pickle.dump(args, handle)
                pickle.dump(data, handle)
            saved = True
    except KeyboardInterrupt:
        if not saved:
            os.remove(args['output'])
        raise


if __name__ == "__main__":
    main()
