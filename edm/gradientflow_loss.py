from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.util import safe_map as map
from jax.flatten_util import ravel_pytree


def gf_loss_forward_by_time(net, loss, regu, w, g, time, min_dt, max_dt, net_tol, loss_tol):
    """gradient flow dynamics

    Args:
        net: neural network function ``f(weights) -> (logits, labels)``
        loss: loss function ``loss(logit, label)`` (without the mean)
        regu: regularization function ``regu(weights)`` (minus the gradient of it)
        w: weights
        g: minus gradient of loss (optional)
        time: evolution time
        net_tol: tolerance of output error
        loss_tol: tolerance of loss error

    Returns:
        w: updated weights
        g: new minus gradient of loss
        ok: whether the update is successful
    """
    if g is None:
        g = jax.grad(lambda w: -jnp.mean(jax.vmap(loss)(*net(w))))(w)

    w, unravel = ravel_pytree(w)
    g, _ = ravel_pytree(g)
    net = _ravel_first_arg(net, unravel)
    regu = _ravel_first_arg_and_output(regu, unravel)

    def cond_fun(state):
        w, g, t, dt = state
        return (t < time) & (dt > 0)

    def body_fun(state):
        w, g, t, dt = state
        w, g, dt, next_dt, _, _ = gf_loss_one_step(net, loss, regu, w, g, dt, min_dt, max_dt, net_tol, loss_tol)
        t = t + dt
        next_dt = jnp.minimum(next_dt, time - t)
        return w, g, t, next_dt

    w, g, t, dt = lax.while_loop(cond_fun, body_fun, (w, g, 0.0, time))

    return unravel(w), unravel(g), (t == time)


def gf_loss_one_step(net, loss, regu, w, g, dt, min_dt, max_dt, net_tol, loss_tol):
    """gradient flow dynamics

    Args:
        net: neural network function ``f(weights) -> (logits, labels)``
        loss: loss function ``loss(logit, label)`` (without the mean)
        regu: regularization function ``regu(weights)`` (minus the gradient of it)
        w: weights
        g: minus gradient of loss
        dt: time step
        min_dt: minimum time step
        max_dt: maximum time step
        net_tol: tolerance of output error
        loss_tol: tolerance of loss error

    Returns:
        w: updated weights
        g: new minus gradient of loss
        dt: time step used for the update
        next_dt: time step proposed for the next iteration
        loss: new loss value
        ok: whether the update is successful
    """
    w, unravel = ravel_pytree(w)
    g, _ = ravel_pytree(g)
    net = _ravel_first_arg(net, unravel)
    regu = _ravel_first_arg_and_output(regu, unravel)

    state = [0, 0.0, w, g, dt, jnp.nan]

    def pot(w):
        logits, labels = net(w)
        losses = jax.vmap(loss, (0, 0), 0)(logits, labels)
        return jnp.mean(losses)

    def cond_fun(state):
        i, t, _, _, dt, loss_value = state
        return (i == 0) & (dt > 0)

    def body_fun(state):
        i, last_t, last_w, last_g, dt, last_loss = state
        next_w, next_g, next_w_error, next_loss = _runge_kutta(pot, regu, last_w, last_g, dt)
        next_t = last_t + dt
        error_ratio = _mean_error_ratio(net, loss, next_w_error, net_tol, loss_tol, last_w)
        error_ratio = jnp.where(dt == min_dt, jnp.minimum(1.0, error_ratio), error_ratio)

        dt = _optimal_step_size(dt, error_ratio)
        dt = jnp.maximum(dt, min_dt)
        dt = jnp.where(jnp.isnan(dt), min_dt, dt)
        dt = jnp.minimum(dt, max_dt)

        new = [i + 1, next_t, next_w, next_g, dt, next_loss]
        old = [i + 0, last_t, last_w, last_g, dt, last_loss]
        return map(partial(jnp.where, error_ratio <= 1.), new, old)

    i, dt, w, g, next_dt, loss_value = lax.while_loop(cond_fun, body_fun, state)

    return unravel(w), unravel(g), dt, next_dt, loss_value, i


def _runge_kutta(potential, regu, w0, dw0, dt):
    """Runge Kutta

    Args:
        potential: potential function
        regu: regularization function (minus the gradient of it)
        w0: initial weights
        dw0: minus gradient of potential + regularization
        dt: time step

    Returns:
        w: new weights
        g: new gradients
        w_error: error estimate of weights
        u: new potential value
    """

    # Dopri5 Butcher tableaux
    beta = jnp.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
    ], dtype=w0.dtype)
    c_sol = jnp.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0], dtype=w0.dtype)
    c_error = jnp.array([
        35 / 384 - 1951 / 21600, 0, 500 / 1113 - 22642 / 50085, 125 / 192 - 451 / 720, -2187 / 6784 - -12231 / 42400, 11 / 84 - 649 / 6300, -1. / 60.
    ], dtype=w0.dtype)

    def body_fun(i, k):
        wi = w0 + dt * jnp.dot(beta[i, :], k)
        gt = jax.grad(potential)(wi)
        dwt = regu(wi) - gt
        return k.at[i, :].set(dwt)

    k = jnp.zeros((7, w0.shape[0]), w0.dtype).at[0, :].set(dw0)
    k = lax.fori_loop(1, 6, body_fun, k)

    w1 = w0 + dt * jnp.dot(c_sol, k)
    u1, g1 = jax.value_and_grad(potential)(w1)
    dw1 = regu(w1) - g1
    k = k.at[6, :].set(dw1)

    w1_error = dt * jnp.dot(c_error, k)
    return w1, g1, w1_error, u1


def _ravel_first_arg(f, unravel):
    """Ravel the first argument of a function.

    Args:
        f: function
        unravel: unravel function

    Returns:
        new function
    """
    def new_f(x, *args, **kwargs):
        return f(unravel(x), *args, **kwargs)
    return new_f


def _ravel_first_arg_and_output(f, unravel):
    """Ravel the first argument of a function and the output.

    Args:
        f: function
        unravel: unravel function

    Returns:
        new function
    """
    def new_f(x, *args, **kwargs):
        result, _ = ravel_pytree(f(unravel(x), *args, **kwargs))
        return result
    return new_f


def _mean_error_ratio(net, loss, w_error_estimate, net_tol, loss_tol, w0):
    (logits, labels), (err_logits, _) = jax.jvp(net, (w0,), (w_error_estimate,))
    err_losses = err_logits * jax.vmap(jax.grad(loss, 0))(logits, labels)
    return jnp.maximum(jnp.max(jnp.abs(err_logits / net_tol)), jnp.max(jnp.abs(err_losses / loss_tol)))


def _optimal_step_size(dt, error_ratio, safety=0.9, ifactor=10.0, dfactor=0.2, order=5.0):
    """Compute optimal Runge-Kutta stepsize."""
    dfactor = jnp.where(error_ratio < 1, 1.0, dfactor)

    factor = jnp.minimum(
        ifactor,
        jnp.maximum(
            (error_ratio + 1e-8)**(-1.0 / order) * safety,
            dfactor
        )
    )
    return jnp.where(error_ratio == 0, dt * ifactor, dt * factor)