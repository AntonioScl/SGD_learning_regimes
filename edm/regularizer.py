"""Regularizers

These functions return minus the gradient of the regularizer.

They take the following arguments:
    net: a function that takes a single argument, a list of parameters, and returns a tuple (list of outputs, list of labels)
    loss: a function that takes two arguments, an output and a label, and returns a scalar
    params: a list of parameters
"""

import jax
import jax.numpy as jnp


def l2_regularizer(scale):
    def regularizer(net, loss, params):
        return jax.tree_map(lambda x: -scale * x, params)
    return regularizer


def l1_regularizer(scale):
    def regularizer(net, loss, params):
        return jax.tree_map(lambda x: -scale * jnp.sign(x), params)
    return regularizer


def grad_loss_regularizer(scale):
    def regularizer(net, loss, params):
        g = jax.grad(lambda w: jnp.mean(loss(*net(w))))

        def reg(w):
            grad_norms = jax.tree_map(lambda x: jnp.sum(jnp.square(x)), g(w))
            return -scale * sum(jax.tree_leaves(grad_norms), 0.0)
        return jax.grad(reg)(params)
    return regularizer


def grad_net_regularizer(scale):
    def regularizer(net, loss, params):
        @jax.grad
        def g(w):
            pred, label = net(w)
            return jnp.mean(pred)

        def reg(w):
            grad_norms = jax.tree_map(lambda x: jnp.sum(jnp.square(x)), g(w))
            return -scale * sum(jax.tree_leaves(grad_norms), 0.0)
        return jax.grad(reg)(params)
    return regularizer