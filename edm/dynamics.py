import jax
import jax.numpy as jnp
from functools import reduce
import operator


def sgd(regu, only_unfit_marg, f, loss, bs, dt, key, w, out0, xtr, ytr, tot_grad):
    if only_unfit_marg:
        key, k = jax.random.split(key)
        perm_i = jax.random.permutation(k, xtr.shape[0])
        unfitted = jnp.argsort((f(w, xtr[perm_i]) - out0[perm_i]) * ytr[perm_i] < only_unfit_marg)
        i = perm_i[unfitted][-bs:]
        x = xtr[i]
        y = ytr[i]
        o0 = out0[i]
    elif bs < xtr.shape[0]:
        key, k = jax.random.split(key)
        i = jax.random.permutation(k, xtr.shape[0])[:bs]
        x = xtr[i]
        y = ytr[i]
        o0 = out0[i]
    else:
        x = xtr
        y = ytr
        o0 = out0

    minus_loss, g = jax.value_and_grad(lambda w: -jnp.mean(loss(f(w, x) - o0, y)))(w)
    g = jax.tree_map(lambda a, b: a + b, regu(lambda w: (f(w, x) - o0, y), loss, w), g)
    w = jax.tree_map(lambda w, g: w + dt * g, w, g)
    if tot_grad is not None:
        flatdw = dt * flatten_grad(g)
        g_overlap = tot_grad @ flatdw
    else:
        g_overlap = 0.0
    return key, w, -minus_loss, g_overlap


def sgd_until(regu, only_unfit_marg, label_noise, f, loss, bs, dt, key, w, out0, xtr, ytr, last_loss, target_below, target_above, max_step, gf_dt, loss_grad=None):
    # if loss_grad is None: loss_grad = flatten_grad(w)

    def cond(state):
        key, w, current_loss, batch_g_norm, i, t = state
        return (i == 0) | ((target_below < current_loss) & (current_loss < target_above) & (i < max_step) & jnp.isfinite(current_loss))

    def body(state):
        key, w, old_loss, cum_g_norm, i, old_t = state
        key, w, batch_loss, batch_g_norm = sgd(regu, only_unfit_marg, label_noise, f, loss, bs, dt, key, w, out0, xtr, ytr, loss_grad)
        new_loss = ((xtr.shape[0] - bs) * old_loss + bs * batch_loss) / xtr.shape[0]
        cum_g_norm += batch_g_norm
        return key, w, new_loss, cum_g_norm, i + 1, old_t + dt

    key, w, current_loss, cum_g_norm, i, t = jax.lax.while_loop(cond, body, (key, w, last_loss, 0.0, 0, 0.0))
    return key, w, current_loss, i, i, t, gf_dt, cum_g_norm


def flatten_grad(g):
    return jnp.concatenate([
            jnp.reshape(x, (reduce(operator.mul, x.shape[0:], 1)))
            for x in jax.tree_leaves(g)
        ], 0)


def gd_sde(regu, only_unfit_marg, label_noise, f, loss, bs, dt, key, w, out0, xtr, ytr, tot_grad):
    if only_unfit_marg:
        unfitted = (f(w, xtr) - out0) * ytr < only_unfit_marg
        x = xtr[unfitted]
        y = ytr[unfitted]
        o0 = out0[unfitted]
    else:
        x = xtr
        y = ytr
        o0 = out0

    if label_noise:
        key, k = jax.random.split(key)
        y = y + label_noise * (2*jax.random.bernoulli(k, shape=y.shape)-1)

    key, k = jax.random.split(key)
    noise = jax.random.normal(k, (x.shape[0],))

    minus_loss, g = jax.value_and_grad(lambda w: -jnp.mean(loss(f(w, x) - o0, y)))(w)
    g = jax.tree_map(lambda a, b: a + b, regu(lambda w: (f(w, x) - o0, y), loss, w), g)

    minus_dLoss = jax.vmap(jax.grad(loss))(f(w, x)-o0, y)
    sigma_noise = jnp.sqrt((minus_dLoss**2).sum() * (1/bs - 1/len(y)))
    noisy_features = jax.grad(lambda w: jnp.mean(f(w, x) * noise))(w)   # the mean counts for a factor 1/P
    
    g = jax.tree_map(lambda g, n: g + sigma_noise * n, g, noisy_features)
    w = jax.tree_map(lambda w, g,: w + dt * g, w, g)

    if tot_grad is not None:
        flatdw = dt * flatten_grad(g)
        g_overlap = tot_grad @ flatdw
    else:
        g_overlap = 0.0
    return key, w, -minus_loss, g_overlap


def gd_sde_until(regu, only_unfit_marg, label_noise, f, loss, bs, dt, key, w, out0, xtr, ytr, last_loss, target_below, target_above, max_step, gf_dt, loss_grad=None):

    def cond(state):
        key, w, current_loss, batch_g_norm, i, t = state
        return (i == 0) | ((target_below < current_loss) & (current_loss < target_above) & (i < max_step) & jnp.isfinite(current_loss))

    def body(state):
        key, w, old_loss, cum_g_norm, i, old_t = state
        key, w, new_loss, batch_g_norm = gd_sde(regu, only_unfit_marg, label_noise, f, loss, bs, dt, key, w, out0, xtr, ytr, loss_grad)
        cum_g_norm += batch_g_norm
        return key, w, new_loss, cum_g_norm, i + 1, old_t + dt

    key, w, current_loss, cum_g_norm, i, t = jax.lax.while_loop(cond, body, (key, w, last_loss, 0.0, 0, 0.0))
    return key, w, current_loss, i, i, t, gf_dt, cum_g_norm
