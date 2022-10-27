import threading
import time
from functools import partial, reduce
from itertools import count
import operator

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from .architecture import mlp, mlp_bias, mnas, linear_model, simple_diagonal_linear, simple_cnn, vgg11
from .dynamics import sgd_until, gd_sde_until
from .observable import (delta_pred, drifts, mean_relative_distance,
                         mean_var_grad,
                         diagonal_observables, linear_observables)
from .regularizer import grad_loss_regularizer, grad_net_regularizer, l2_regularizer, l1_regularizer


class PrintDotThread(threading.Thread):
    def __init__(self, event):
        threading.Thread.__init__(self)
        self.stopped = event

    def run(self):
        while not self.stopped.wait(5.0):
            print(".", end="", flush=True)


def neighbors_in_ordered_list(xs, x, vmin, vmax):
    """returns the neighbors of x in the ordered list xs

    Args:
        xs (jnp.ndarray): ordered list
        x (jnp.ndarray): x
        vmin (float): minimum value
        vmax (float): maximum value

    Returns:
        x_below (float): entry of xs that is closest to x from below, ``vmin`` if none.
        x_above (float): entry of xs that is closest to x from above, ``vmax`` if none.

    Note:
        The returned values are not necessarily in the list xs.
        If ``x == xs[i]``, then ``x_below, x_above == xs[i-1], xs[i+1]``.
    """
    if x < xs[0]:
        return vmin, xs[0]
    if xs[-1] < x:
        return xs[-1], vmax
    if x == xs[0]:
        return vmin, xs[1]
    if x == xs[-1]:
        return xs[-2], vmax
    i = jnp.searchsorted(xs, x)
    if xs[i] == x:
        return xs[i - 1], xs[i + 1]
    return xs[i - 1], xs[i]


def normalize_act(phi):
    with jax.core.eval_context():
        k = jax.random.PRNGKey(0)
        x = jax.random.normal(k, (1_000_000,))
        c = jnp.mean(phi(x)**2)**0.5

    def rho(x):
        return phi(x) / c
    return rho


def dataset(dataset, seed_trainset, seed_testset, ptr, pte, **args):
    from .dataset import (cifar_animal, mnist_parity, sign, random_sign, depleted_sign)

    if dataset in ['sign', 'random_sign', 'depleted_sign']:
        d = args['d']

        if dataset == 'sign':
            return sign(d, seed_trainset, seed_testset, ptr, pte)

        elif dataset == 'random_sign':
            return random_sign(d, seed_trainset*100, seed_trainset, seed_testset, ptr, pte)

        elif dataset == 'depleted_sign':
            return depleted_sign(args['data_chi'], d, seed_trainset, seed_testset, ptr, pte)

    if dataset == 'mnist_parity':
        return mnist_parity(seed_trainset, seed_testset, ptr, pte)

    if dataset == 'cifar_animal':
        return cifar_animal(seed_trainset, seed_testset, ptr, pte)

    raise ValueError(f"{dataset} not available. available are sign, random_sign, depleted_sign, mnist_parity, cifar_animal")


def hinge(o, y):
    return jax.nn.relu(1.0 - o * y)


def qhinge(o, y):
    return jax.nn.relu(1.0 - o * y)**2


def sus(x):
    return jnp.where(x > 0.0, jnp.exp(-1.0 / jnp.where(x > 0.0, x, 1.0)), 0.0)


def srelu(x):
    return sus(2.0 * x) * x


def shinge(o, y):
    return srelu(1.0 - o * y)

def mse(o, y):
    return 0.5 * (o - y)**2


def train(
    dyn_until, f, w0, xtr, xte, ytr, yte, bs, dt, seed_batch, alpha, loss,
    ckpt_modulo, ckpt_save_parameters,
    ckpt_grad_stats, ckpt_kernels, ckpt_drift_n0, ckpt_save_pred, ckpt_drift, ckpt_delta_pred,
    max_wall, max_step, mind_stop, **args
):
    assert 'dataset' in args
    assert 'dynamics' in args

    stopFlag = threading.Event()
    thread = PrintDotThread(stopFlag)
    thread.start()

    wall0 = time.perf_counter()
    wall_ckpt = 0
    wall_compile = 0
    wall_train = 0

    key_batch = jax.random.PRNGKey(seed_batch)

    loss_name = None
    if loss == "hinge":
        loss_fun = hinge
        loss_name = "hinge"
    elif loss == "softhinge":
        loss_fun = shinge
        loss_name = "shinge"
    elif loss == "quadhinge":
        loss_fun = qhinge
        loss_name = "qhinge"
    elif loss == "mse":
        loss_fun = mse
        loss_name = "mse"
    else:
        raise ValueError(f"loss {loss} not available. available are hinge, softhinge, quadhinge, mse")

    loss = lambda o, y: loss_fun(alpha * o, y) / alpha

    jit_dyn_until = jax.jit(partial(dyn_until, f, loss, bs, dt))
    jit_mean_var_grad = jax.jit(partial(mean_var_grad, f, loss))
    jit_drifts = jax.jit(partial(drifts, gf_tol, gf_mindt, gf_maxdt, f, loss, ckpt_drift_n0, bs, dt))
    jit_delta_pred = jax.jit(partial(delta_pred, dyn_until, f, loss, bs, dt))

    @jax.jit
    def jit_le(w, out0, x, y):
        out = f(w, x)
        pred = out - out0
        return pred, jnp.mean(loss(pred, y)), jnp.mean((pred * y <= 0 | ~jnp.isfinite(pred)))

    @jax.jit
    def jit_gradloss(w, out0, x, y):
        g = jax.grad(lambda w: jnp.mean(loss(f(w, x) - out0, y)))(w)
        return jnp.concatenate([jnp.reshape(x, (reduce(operator.mul, x.shape[0:], 1))) for x in jax.tree_util.tree_leaves(g)], 0)

    print("compute f(w0)", flush=True)
    out0tr = f(w0, xtr)
    out0te = f(w0, xte)
    pred, l0, _ = jit_le(w0, out0tr, xtr, ytr)

    print("jit dyn_until", flush=True)
    ckpt_step = 1
    tmp, *_ = jit_dyn_until(key_batch, w0, out0tr, xtr, ytr, l0, 0.0, jnp.inf, ckpt_step, gf_mindt)
    tmp.block_until_ready()
    while True:
        wall = time.perf_counter()
        tmp, _, _, n, *_ = jit_dyn_until(key_batch, w0, out0tr, xtr, ytr, l0, 0.0, jnp.inf, ckpt_step, gf_mindt)
        tmp.block_until_ready()
        if time.perf_counter() - wall > 0.5:
            print(f"measuring time to do {ckpt_step} steps", flush=True)
        if time.perf_counter() - wall > 5.0:
            break
        if n < ckpt_step:
            break
        ckpt_step *= 2

    if ckpt_grad_stats:
        print("computing intial kernels (train)", flush=True)
        _, _, _, _, kernel_tr0 = jit_mean_var_grad(w0, out0tr[:ckpt_grad_stats], xtr[:ckpt_grad_stats], ytr[:ckpt_grad_stats])
        print("(test)", flush=True)
        _, _, _, _, kernel_te0 = jit_mean_var_grad(w0, out0te[:ckpt_grad_stats], xte[:ckpt_grad_stats], yte[:ckpt_grad_stats])

    ckpt_loss_fast = 2.0**jnp.arange(-24, -9, 0.5)
    ckpt_loss_fast = jnp.concatenate([ckpt_loss_fast, jnp.arange(2**-9, 1, 2**-9), 1 - ckpt_loss_fast[::-1]])
    ckpt_loss_fast = jnp.concatenate([ckpt_loss_fast, (1.0 + 2.0**jnp.arange(-5, 10, 0.5))])
    ckpt_loss_fast = jnp.unique(ckpt_loss_fast)
    ckpt_loss_fast = jnp.sort(ckpt_loss_fast)
    ckpt_loss_fast = l0 * ckpt_loss_fast
    ckpt_loss_fast = ckpt_loss_fast[::ckpt_modulo]

    ckpt_loss_slow = 2.0**jnp.arange(-20, -2, 0.5)
    ckpt_loss_slow = jnp.concatenate([ckpt_loss_slow, jnp.arange(2**-2, 1.0, 2**-4), jnp.array([1.0 - 2**-20])])
    ckpt_loss_slow = jnp.unique(ckpt_loss_slow)
    ckpt_loss_slow = jnp.sort(ckpt_loss_slow)
    ckpt_loss_slow = l0 * ckpt_loss_slow

    current_loss = l0
    target_loss_slow_below = l0
    target_loss_slow_above = l0
    cumul_batch_g_norm = 0
    step_save_slow = 0

    dynamics = []
    w = w0
    t = 0
    step = 0
    gf_dt = gf_mindt
    grad_l_tr = jit_gradloss(w, out0tr[:ckpt_grad_stats], xtr[:ckpt_grad_stats], ytr[:ckpt_grad_stats])

    print("starting training", flush=True)

    for iter_index in count():
        start = (iter_index == 0)

        wtrain = time.perf_counter()
        postsave_step = 0
        postsave_step_ok = 0
        target_below, target_above = neighbors_in_ordered_list(ckpt_loss_fast, current_loss, 0.0, jnp.inf)

        if iter_index > 0:
            while True:
                key_batch, w, internal_loss, num_step, num_step_ok, delta_t, gf_dt, cumul_batch_g_norm = jit_dyn_until(
                    key_batch,
                    w,
                    out0tr,
                    xtr,
                    ytr,
                    current_loss,
                    target_below,
                    target_above,
                    1 if step < ckpt_save_all else (max(1,step) if step < args['ckpt_save_mult'] else ckpt_step),
                    gf_dt,
                    (jit_gradloss(w, out0tr[:ckpt_grad_stats], xtr[:ckpt_grad_stats], ytr[:ckpt_grad_stats]) if grad_l_tr is None else grad_l_tr) if args['ckpt_save_gradoverlap'] else None,
                )
                t += delta_t
                step += num_step_ok
                postsave_step += num_step
                postsave_step_ok += num_step_ok

                pred, current_loss, err = jit_le(w, out0tr, xtr, ytr)

                if not (target_below < current_loss < target_above):
                    if current_loss < target_below:
                        ckpt_loss_fast = ckpt_loss_fast[ckpt_loss_fast < target_below]
                        if len(ckpt_loss_fast) == 0:
                            ckpt_loss_fast = jnp.array([target_below])
                    break

                if delta_t == 0.0:
                    break

                if not jnp.isfinite(current_loss):
                    break

                if start:
                    break

                if (time.perf_counter() - wtrain) > 5.0 * (wall_ckpt / len(dynamics)):
                    break

                if step < ckpt_save_all:
                    break

                if step < args['ckpt_save_mult'] and 2*num_step_ok>=step:
                    break
        else:
            delta_t = 0.0
            pred, current_loss, err = jit_le(w, out0tr, xtr, ytr)

        wall_train += time.perf_counter() - wtrain
        wckpt = time.perf_counter()

        stop = False

        if current_loss == 0.0 or (loss_name=='mse' and alpha * current_loss < 1e-9):
            stop = True

        if not start and delta_t == 0.0:
            print(f"stopping because no progress in {postsave_step} steps ({postsave_step_ok} where successful) (interal loss is {internal_loss})", flush=True)
            print(f"{target_below} < {internal_loss}({current_loss}) < {target_above}", flush=True)
            stop = True

        if ckpt_step <= postsave_step and alpha * jnp.min(pred * ytr) >= mind_stop:
            stop = True

        if not jnp.isfinite(current_loss):
            stop = True

        if time.perf_counter() - wall0 > max_wall:
            stop = True

        if step > max_step:
            stop = True

        save_slow = False
        if (step > args['ckpt_save_mult'] and not (target_loss_slow_below <= current_loss <= target_loss_slow_above)) or (step <= args['ckpt_save_mult'] and step-step_save_slow>=step_save_slow):
            save_slow = True
            step_save_slow = step
            ckpt_loss_slow = ckpt_loss_slow[~((target_loss_slow_below < ckpt_loss_slow) & (ckpt_loss_slow <= target_loss_slow_above))]
            if len(ckpt_loss_slow) == 0:
                ckpt_loss_slow = jnp.array([target_loss_slow_below])
            target_loss_slow_below, target_loss_slow_above = neighbors_in_ordered_list(ckpt_loss_slow, current_loss, 0.0, jnp.inf)

        if start or stop:
            save_slow = True

        grad_f_tr, var_f, grad_l_tr, var_l, kernel, kernel_change, kernel_norm, kernel_yky, kernel_y_norm, drift = [None] * 10
        if save_slow:
            if ckpt_grad_stats:
                if start:
                    print("jit mean_var_grad", flush=True)

                grad_f_tr, var_f, grad_l_tr, var_l, kernel = jit_mean_var_grad(w, out0tr[:ckpt_grad_stats], xtr[:ckpt_grad_stats], ytr[:ckpt_grad_stats])
                kernel_change = jnp.mean((kernel - kernel_tr0)**2)
                kernel_norm = jnp.mean(kernel**2)
                kernel_yky = jnp.einsum("i,ij,j->", ytr[:ckpt_grad_stats], kernel, ytr[:ckpt_grad_stats])
                kernel_y_norm = (ytr[:ckpt_grad_stats]/len(ytr[:ckpt_grad_stats])) @ jax.scipy.linalg.solve(kernel, (ytr[:ckpt_grad_stats]/len(ytr[:ckpt_grad_stats])))

            if ckpt_drift:
                if start:
                    print("jit drifts", flush=True)

                drift = jit_drifts(key_batch, w, out0tr, xtr, ytr)

        if 'gf' not in args['dynamics'] and ckpt_delta_pred > 0:
            if start:
                print("jit delta_pred", flush=True)
            delta_pred_tr, delta_pred_te = jit_delta_pred(key_batch, w, out0tr, xtr, ytr, xte)
        else:
            delta_pred_tr, delta_pred_te = None, None

        def sensitivity_dict(pred_0, pred_1):
            return dict(
                pred=(pred_1 if stop or ckpt_save_pred else None),
                value=alpha**2 * jnp.mean((pred_1 - pred_0)**2)
            )

        if start:
            print("create state (train)", flush=True)

        train = dict(
            loss=current_loss,
            aloss=alpha * current_loss,
            err=err,
            mind=jnp.min(pred * ytr),
            nd=jnp.sum(alpha * pred * ytr < 1.0),
            g_norm=cumul_batch_g_norm,
            grad_f_norm=None if grad_f_tr is None else jnp.sum(grad_f_tr**2),
            grad_f_var=var_f,
            grad_l_norm=None if grad_l_tr is None else jnp.sum(grad_l_tr**2),
            grad_l_var=var_l,
            pred=(pred if stop or (ckpt_save_pred and save_slow) else None),
            label=(ytr if stop or (ckpt_save_pred and save_slow) else None),
            kernel=(kernel if ckpt_kernels else None),
            kernel_change=kernel_change,
            kernel_norm=kernel_norm,
            kernel_yky=kernel_yky,
            kernel_y_norm=kernel_y_norm,
            delta_pred_abs=dict(
                min=jnp.min(jnp.abs(delta_pred_tr)),
                max=jnp.max(jnp.abs(delta_pred_tr)),
                median=jnp.median(jnp.abs(delta_pred_tr)),
            ) if delta_pred_tr is not None else None,
            mean_relative_distance=mean_relative_distance(pred),
            pred_bias = jnp.mean(alpha * pred * ytr),
            pred_var = jnp.mean((alpha*pred)**2),
        )

        del err

        if start:
            print("create state (test)", flush=True)

        grad_f_te, grad_l_te = [None] * 2
        if save_slow:
            if ckpt_grad_stats:
                grad_f_te, var_f, grad_l_te, var_l, kernel = jit_mean_var_grad(w, out0te[:ckpt_grad_stats], xte[:ckpt_grad_stats], yte[:ckpt_grad_stats])
                kernel_change = jnp.mean((kernel - kernel_te0)**2)
                kernel_norm = jnp.mean(kernel**2)
                kernel_yky = jnp.einsum("i,ij,j->", yte[:ckpt_grad_stats], kernel, yte[:ckpt_grad_stats])
                kernel_y_norm = (yte[:ckpt_grad_stats]/len(yte[:ckpt_grad_stats])) @ jax.scipy.linalg.solve(kernel, (yte[:ckpt_grad_stats]/len(yte[:ckpt_grad_stats])))

        pred, test_loss, err = jit_le(w, out0te, xte, yte)

        test = dict(
            loss=test_loss,
            aloss=alpha * test_loss,
            err=err,
            mind=jnp.min(pred * yte),
            nd=jnp.sum(alpha * pred * yte < 1.0),
            grad_f_norm=None if grad_f_te is None else jnp.sum(grad_f_te**2),
            grad_f_var=var_f,
            grad_l_norm=None if grad_l_te is None else jnp.sum(grad_l_te**2),
            grad_l_var=var_l,
            pred=(pred if stop or (ckpt_save_pred and save_slow) else None),
            label=(yte if stop or (ckpt_save_pred and save_slow) else None),
            kernel=(kernel if ckpt_kernels else None),
            kernel_change=kernel_change,
            kernel_norm=kernel_norm,
            kernel_yky=kernel_yky,
            kernel_y_norm=kernel_y_norm,
            delta_pred_abs=dict(
                min=jnp.min(jnp.abs(delta_pred_te)),
                max=jnp.max(jnp.abs(delta_pred_te)),
                median=jnp.median(jnp.abs(delta_pred_te)),
            ) if delta_pred_te is not None else None,
            mean_relative_distance=mean_relative_distance(pred),
            pred_bias = jnp.mean(alpha * pred * yte),
            pred_var = jnp.mean((alpha*pred)**2),
        )

        del test_loss, err

        if start:
            print("create state", flush=True)

        state = dict(
            t=t,
            step=step,
            wall=time.perf_counter() - wall0,
            weights_norm=[jnp.sum(x**2) for x in jax.tree_util.tree_leaves(w)],
            delta_weights_norm=[jnp.sum((x - x0)**2) for x, x0 in zip(jax.tree_util.tree_leaves(w), jax.tree_util.tree_leaves(w0))],
            drift=drift,
            train=train,
            test=test,
            parameters=(w if (ckpt_save_parameters and (start or stop)) else None),
            diag_lin = diagonal_observables(w, w0, args) if args['arch'] == 'simple_diagonal_linear' else None,
            linear = linear_observables(w, w0, xtr, ytr, pred, alpha, args) if args['arch'] == 'linear' else None,
            train_test=dict(
                grad_l_overlap=None if grad_l_tr is None else jnp.sum(grad_l_tr * grad_l_te),
                grad_f_overlap=None if grad_f_tr is None else jnp.sum(grad_f_tr * grad_f_te),
            ) if save_slow else None,
        )
        dynamics.append(jax.tree_map(np.asarray, state))

        if args['ckpt_save_gradoverlap']:
            del grad_f_tr, grad_l_te, grad_f_te # Need to keep grad_l_tr for the non-linear measure
        else:
            del grad_l_tr, grad_f_tr, grad_l_te, grad_f_te

        internal = dict(
            f=f,
            w=w,
            w0=w0,
            fn_pred_loss_error=jit_le,
            train=dict(
                x=xtr,
                y=ytr,
                out0=out0tr,
            ),
            test=dict(
                x=xte,
                y=yte,
                out0=out0te,
            ),
        )

        yield internal, dynamics

        if not (target_below < current_loss < target_above):
            token = "*"
        else:
            token = ""
        if save_slow:
            token += "+"

        if 'gf' in args['dynamics']:
            dt_text = f" dt={gf_dt:.1e}"
        else:
            dt_text = ""

        if start:
            wall_compile = time.perf_counter() - wall0
        else:
            wall_ckpt += time.perf_counter() - wckpt

        print((
            f"[{step}{token} t={t:.2e}{dt_text} w={wall_compile:.0f}+{wall_train:.0f}+{wall_ckpt:.0f} s={len(dynamics)}] "
            f"[train aL={alpha * state['train']['loss']:.2e} err={state['train']['err']:.2f} mind={alpha * state['train']['mind']:.2f}] "
            f"[test aL={alpha * state['test']['loss']:.2e} err={state['test']['err']:.2f}]"
        ), flush=True)

        if start:
            stopFlag.set()

        del state

        if stop:
            return


def init(arch, h, act, seed_init, **args):
    """initialize weights and biases

    Args:
        arch (str): architecture
        h (int): hidden layer size
        act (str): activation function
        seed_init (int): random seed
        args (dict): additional arguments

    Returns:
        model (function): model function
        w (jax.tree.Tree): parameters
        xtr (jnp.ndarray): training data
        xte (jnp.ndarray): test data
        ytr (jnp.ndarray): training labels
        yte (jnp.ndarray): test labels
    """

    if act == 'silu':
        act = jax.nn.silu
    if act == 'gelu':
        act = jax.nn.gelu
    if act == 'relu':
        act = jax.nn.relu

    act = normalize_act(act)

    xtr, xte, ytr, yte = dataset(**args)
    print(f'dataset generated xtr.shape={xtr.shape} xte.shape={xte.shape}', flush=True)

    if arch == 'linear':
        model = hk.without_apply_rng(hk.transform(
            lambda x: linear_model(x)
        ))

    if arch == 'simple_diagonal_linear':
        model = hk.without_apply_rng(hk.transform(
            lambda x: simple_diagonal_linear(args.get("L"), x)
        ))

    if arch == 'mlp':
        model = hk.without_apply_rng(hk.transform(
            lambda x: mlp([h] * args.get("L"), act, x)
        ))

    if arch == 'mlp_bias':
        model = hk.without_apply_rng(hk.transform(
            lambda x: mlp_bias([h] * args.get("L"), act, x)
        ))

    if arch == 'mnas':
        model = hk.without_apply_rng(hk.transform(
            lambda x: mnas(h, act, x)
        ))

    if arch == 'simple_cnn':
        model = hk.without_apply_rng(hk.transform(
            lambda x: simple_cnn(h, act, x)
        ))

    if arch == 'vgg11':
        model = hk.without_apply_rng(hk.transform(
            lambda x: vgg11(h, act, x)
        ))

    w = model.init(jax.random.PRNGKey(seed_init), xtr)
    print('network initialized', flush=True)

    return model, w, xtr, xte, ytr, yte


def execute(regularizer, regu_scale, dynamics, yield_time=0.0, **args):
    print(f"device={jnp.ones(3).device_buffer.device()} dtype={jnp.ones(3).dtype}", flush=True)

    yield {
        'finished': False,
    }

    model, w, xtr, xte, ytr, yte = init(**args)

    @jax.jit
    def apply(w, x):
        if x.shape[0] < 1024:
            return model.apply(w, x)
        return jnp.concatenate([
            model.apply(w, x[i: i + 1024])
            for i in range(0, x.shape[0], 1024)
        ])

    darch = dict(
        params_shapes=[p.shape for p in jax.tree_util.tree_leaves(w)],
        params_sizes=[p.size for p in jax.tree_util.tree_leaves(w)],
    )

    regu = lambda net, loss, w: jax.tree_map(lambda x: jnp.zeros_like(x), w)
    if regularizer is None:
        pass
    elif regularizer == 'norm_grad_loss':
        regu = grad_loss_regularizer(regu_scale)
    elif regularizer == 'norm_grad_net':
        regu = grad_net_regularizer(regu_scale)
    elif regularizer == 'l2':
        regu = l2_regularizer(regu_scale)
    elif regularizer == 'l1':
        regu = l1_regularizer(regu_scale)
    else:
        raise ValueError(f"unknown regularizer {regularizer}. known: norm_grad_loss, norm_grad_net, l2, l1")

    if dynamics == 'sgd':
        dyn_until = partial(sgd_until, regu, 0)
    if dynamics == 'sgd_only_unfit':
        dyn_until = partial(sgd_until, regu, 1/args['alpha'])
    if dynamics == 'gd_sde':
        dyn_until = partial(gd_sde_until, regu, 0)
    if dynamics == 'gd_sde_only_unfit':
        dyn_until = partial(gd_sde_until, regu, 1/args['alpha'])

    wall_yield = 0

    for _, d in train(dyn_until, apply, w, xtr, xte, ytr, yte, dynamics=dynamics, **args):
        if time.perf_counter() - wall_yield > yield_time:
            wall_yield = time.perf_counter()

            yield {
                'arch': darch,
                dynamics: dict(dynamics=d),
                'finished': False,
            }

    yield {
        'arch': darch,
        dynamics: dict(dynamics=d),
        'finished': True,
    }
