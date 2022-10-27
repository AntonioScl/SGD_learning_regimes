import chunk
import operator
from functools import partial, reduce

import jax
import jax.numpy as jnp

import time


def drifts(tol, gf_mindt, gf_maxdt, f, loss, n, bs, dt, key, w, out0, x, y):
    from .gradientflow_loss import gf_loss_forward_by_time
    def loss_(w, out0, x, y):
        return -jnp.mean(loss(f(w, x) - out0, y))
    dw = jax.grad(loss_, 0)

    w2, _, _ = gf_loss_forward_by_time(
        lambda w: (f(w, x) - out0, y),
        loss,
        lambda w: 0.0,
        w,
        None,
        2 * dt,
        gf_mindt,
        gf_maxdt,
        net_tol=tol,
        loss_tol=tol * loss(0.0, 1.0),
    )
    dw_gf = jax.tree_map(lambda a, b: a - b, w2, w)
    del w2

    if n > 0 and n * bs <= x.shape[0]:
        i = jax.random.permutation(key, x.shape[0])
        i = i[:n * bs]
        o01 = out0[i].reshape((n, bs) + out0.shape[1:])
        x1 = x[i].reshape((n, bs) + x.shape[1:])
        y1 = y[i].reshape((n, bs) + y.shape[1:])

        g1 = jax.vmap(dw, (None, 0, 0, 0), 0)(w, o01, x1, y1)
        w1 = jax.tree_map(lambda a, b: a + dt * b, w, g1)

        g2 = jax.vmap(dw, (0, None, None, None), 0)(w1, out0, x, y)
        dw_sgd = jax.tree_map(lambda a, b: jnp.mean(dt * a + dt * b, 0), g1, g2)
        del i, o01, x1, y1, g1, g2, w1
    else:
        print("not computing SGD drift.")
        print(f"n={n} bs={bs} len(x)={x.shape[0]}")
        dw_sgd = None

    g1 = dw(w, out0, x, y)
    w1 = jax.tree_map(lambda a, b: a + dt * b, w, g1)
    g2 = dw(w1, out0, x, y)
    dw_gd = jax.tree_map(lambda a, b: dt * a + dt * b, g1, g2)
    del g1, g2, w1

    def norm_diff(t1, t2):
        if t1 is None or t2 is None:
            return None
        return jax.tree_util.tree_leaves(jax.tree_map(lambda a, b: jnp.sum((a - b)**2), t1, t2))

    return dict(
        gd_gf=norm_diff(dw_gd, dw_gf),
        sgd_gf=norm_diff(dw_sgd, dw_gf),
        gd_sgd=norm_diff(dw_gd, dw_sgd),
    )


def mean_var_grad(f, loss, w, out0, x, y):
    f1 = lambda w, x: f(w, x[None])[0]  # some models requires a batch index

    out, j = jax.vmap(jax.value_and_grad(f1, 0), (None, 0), 0)(w, x)
    j = jnp.concatenate([
        jnp.reshape(x, (x.shape[0], reduce(operator.mul, x.shape[1:], 1)))
        for x in jax.tree_util.tree_leaves(j)
    ], 1)  # [x, w]
    # j[i, j] = d f(w, x_i) / d w_j
    mean_f = jnp.mean(j, 0)
    var_f = jnp.mean(jnp.sum((j - mean_f)**2, 1))

    # kernel[mu,nu] = sum_j j[mu,j] j[nu,j]
    kernel = j @ j.T

    dl = jax.vmap(jax.grad(loss, 0), (0, 0), 0)
    lj = dl(out - out0, y)[:, None] * j
    mean_l = jnp.mean(lj, 0)
    var_l = jnp.mean(jnp.sum((lj - mean_l)**2, 1))

    return mean_f, var_f, mean_l, var_l, kernel


def delta_pred(dyn_until, f, loss, bs, dt, key, w, out0, xtr, ytr, xte):
    out0tr = f(w, xtr)
    out0te = f(w, xte)
    _, w, *_ = dyn_until(f, loss, bs, dt, key, w, out0, xtr, ytr, 100.0, 0.0, jnp.inf, 1, 1.0)
    out1tr = f(w, xtr)
    out1te = f(w, xte)
    return out1tr - out0tr, out1te - out0te


def diagonal_ampli_factor(w, L, teacher):
    ''''
    amplification = weights_teacher_direction / weigths on the other dimensions
    '''
    iterator = iter(w.keys())
    if len(w.keys())==1:
        assert next(iter(iterator)) == 'diagonal_layer'
        wplus = w['diagonal_layer']['w']
        weights = wplus**L
    else:
        assert next(iter(iterator)) == 'diagonal_layer'
        assert next(iter(iterator)) == 'diagonal_layer_1'
        wplus = w['diagonal_layer']['w']       # weights plus
        wminus = w['diagonal_layer_1']['w']    # weights minus
        weights = wplus**L - wminus**L
    d = weights.shape[0]     # input dimension
    wpara = weights @ teacher
    wperp = weights - wpara * teacher 
    return [(d - 1) * wpara**2 / (wperp**2).sum(), wpara, jnp.linalg.norm(wperp)]


def diagonal_delta_weights(w, w0, teacher):
    iterator = iter(w.keys())
    dw = jax.tree_map(lambda x,y: x-y, w, w0)
    if len(w.keys())==1:
        assert next(iter(iterator)) == 'diagonal_layer'
        dwplus = dw['diagonal_layer']['w']
        d = dwplus.shape[0]     # input dimension

        dwp_para = dwplus @ teacher
        dwp_perp = dwplus - dwp_para * teacher 

        return [dwp_para**2, sum(dwp_perp**2)/(d - 1)]
    else:
        assert next(iter(iterator)) == 'diagonal_layer'
        assert next(iter(iterator)) == 'diagonal_layer_1'

        dwplus = dw['diagonal_layer']['w']       # weights plus
        dwminus = dw['diagonal_layer_1']['w']    # weights minus
        d = dwplus.shape[0]     # input dimension

        dwp_para = dwplus @ teacher
        dwp_perp = dwplus - dwp_para * teacher 
        dwm_para = dwminus @ teacher
        dwm_perp = dwminus - dwm_para * teacher 

        return [dwp_para**2, sum(dwp_perp**2)/(d - 1), dwm_para**2, sum(dwm_perp**2)/(d - 1)]


def diagonal_observables(w, w0, args):
    if args['dataset'] == 'random_sign':
        seed_teacher = 100*args['seed_trainset']
        teacher = jax.random.normal(jax.random.PRNGKey(seed_teacher), (args['d'],))
        teacher = teacher / jnp.linalg.norm(teacher)
    else:
        teacher = jnp.zeros((args['d'],))
        teacher = teacher.at[0].set(1.0)

    return dict(
        ampli_factor=diagonal_ampli_factor(w, args['L'], teacher),
        dw =diagonal_delta_weights(w, w0, teacher),
        )


def linear_predictor_components(dw, d, x, y, teacher):
    d = dw.shape[0]     # input dimension

    x_para = (x @ teacher) * teacher
    x_perp = x - x_para

    y_wx_para = y * (x_para @ dw / d**0.5)
    y_wx_perp = y * (x_perp @ dw / d**0.5)

    return [y_wx_para, y_wx_perp.sum(-1), x_para@teacher]


def linear_observables(w, w0, x, y, pred, alpha, args):
    if args['dataset'] == 'random_sign':
        seed_teacher = 100*args['seed_trainset']
        teacher = jax.random.normal(jax.random.PRNGKey(seed_teacher), (args['d'],))
        teacher = teacher / jnp.linalg.norm(teacher)
    else:
        teacher = jnp.zeros((args['d'],))
        teacher = teacher.at[0].set(1.0)
    iterator = iter(w.keys())
    assert next(iter(iterator)) == 'linear'
    ww = w['linear']['w'].flatten()
    ww0 = w0['linear']['w'].flatten()
    d_perp = len(ww)-1
    # print(w['linear']['w'].shape)
    w_para = ww @ teacher
    w_perp = ww - w_para * teacher
    w0_para = ww0 @ teacher
    w0_perp = ww0 - w0_para * teacher 

    return dict(
        ampli_factor = w_para**2 / sum(w_perp**2)/d_perp,
        dw = [(w_para-w0_para)**2, sum((w_perp-w0_perp)**2)/d_perp],
        dw1 =  w_para-w0_para,
        wx =  None, #linear_predictor_components(w['linear']['w']-w0['linear']['w'], d_perp+1, x, y, teacher),
        mean_x1_unfit=jnp.mean(abs(x@teacher)[alpha * (x@(ww-ww0)) * y < 1.0]),
    )
