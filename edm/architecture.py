import haiku as hk
import jax.numpy as jnp


class diagonal_layer(hk.Module):
    def __init__(self, L, simple=False):
        super(diagonal_layer, self).__init__()
        self.L = L
        self.simple = simple

    def __call__(self, x):
        if self.simple:
            w = hk.get_parameter("w", [x.shape[1],], init=hk.initializers.Constant(1.0))
        else:
            w = hk.get_parameter("w", [x.shape[1],], init=hk.initializers.RandomNormal())
        return x @ w**self.L


def diagonal_linear(L, x):
    if x.ndim == 4:
        x = x.reshape((x.shape[0], -1))
    if x.ndim == 3:
        x = x.reshape((-1,))

    dplus = diagonal_layer(L)
    dminus= diagonal_layer(L)
    return (dplus(x) - dminus(x)) / x.shape[-1]**0.5


def simple_diagonal_linear(L, x):
    if x.ndim == 4:
        x = x.reshape((x.shape[0], -1))
    if x.ndim == 3:
        x = x.reshape((-1,))

    diag = diagonal_layer(L, simple=True)
    return diag(x) / x.shape[-1]**0.5


def linear_model(x):
    if x.ndim == 4:
        x = x.reshape((x.shape[0], -1))
    if x.ndim == 3:
        x = x.reshape((-1,))

    d = hk.Linear(
        1,
        with_bias=False,
        w_init=hk.initializers.RandomNormal()
    )
    x = d(x) / x.shape[-1]**0.5
    return x[..., 0]


def deep_linear(features, x):
    if x.ndim == 4:
        x = x.reshape((x.shape[0], -1))
    if x.ndim == 3:
        x = x.reshape((-1,))

    for feat in features:
        d = hk.Linear(
            feat,
            with_bias=False,
            w_init=hk.initializers.RandomNormal()
        )
        x = d(x) / x.shape[-1]**0.5

    d = hk.Linear(
        1,
        with_bias=False,
        w_init=hk.initializers.RandomNormal()
    )
    x = d(x) / x.shape[-1]
    return x[..., 0]


def mlp(features, act, x):
    if x.ndim == 4:
        x = x.reshape((x.shape[0], -1))
    if x.ndim == 3:
        x = x.reshape((-1,))

    for feat in features:
        d = hk.Linear(
            feat,
            with_bias=False,
            w_init=hk.initializers.RandomNormal()
        )
        x = act(d(x) / x.shape[-1]**0.5)

    d = hk.Linear(
        1,
        with_bias=False,
        w_init=hk.initializers.RandomNormal()
    )
    x = d(x) / x.shape[-1]
    return x[..., 0]


def mlp_bias(features, act, x):
    if x.ndim == 4:
        x = x.reshape((x.shape[0], -1))
    if x.ndim == 3:
        x = x.reshape((-1,))

    for feat in features:
        d = hk.Linear(
            feat,
            with_bias=True,
            w_init=hk.initializers.RandomNormal(),
            b_init=hk.initializers.RandomNormal()
        )
        x = act(d(x) / (x.shape[-1] + 1)**0.5)

    d = hk.Linear(
        1,
        with_bias=False,
        w_init=hk.initializers.RandomNormal()
    )
    x = d(x) / x.shape[-1]
    return x[..., 0]


def mnas(h, act, x):
    def conv2d(c, k, s, x):
        return hk.Conv2D(
            output_channels=c,
            kernel_shape=k,
            stride=s,
            with_bias=False,
            w_init=hk.initializers.RandomNormal(),
        )(x) / (k * x.shape[-1]**0.5)

    def conv2dg(k, s, x):
        return hk.Conv2D(
            output_channels=x.shape[-1],
            kernel_shape=k,
            stride=s,
            feature_group_count=x.shape[-1],
            with_bias=False,
            w_init=hk.initializers.RandomNormal(),
        )(x) / (k)

    x = act(conv2d(round(4 * h), 5, 2, x))
    x = act(conv2dg(5, 1, x))
    x = act(conv2d(round(2 * h), 1, 1, x))

    def inverted_residual(out_chs, k, s, x):
        in_chs = x.shape[-1]
        mid_chs = round(in_chs * 3.0)

        residual = x
        x = act(conv2d(mid_chs, 1, 1, x))
        x = act(conv2dg(k, s, x))
        x = conv2d(out_chs, 1, 1, x)

        if residual.shape == x.shape:
            x = (x + residual) / 2**0.5
        else:
            x = x
        return x

    x = inverted_residual(round(h), 5, 2, x)
    x = inverted_residual(round(h), 5, 1, x)
    x = inverted_residual(round(3 * h), 5, 2, x)
    x = inverted_residual(round(3 * h), 5, 1, x)

    x = act(conv2d(round(20 * h), 1, 1, x))

    x = jnp.mean(x, axis=(1, 2))
    x = hk.Linear(
        output_size=1,
        with_bias=True,
        w_init=hk.initializers.RandomNormal(),
        b_init=hk.initializers.RandomNormal(),
    )(x) / (x.shape[-1])
    return x[..., 0] #+ hk.get_parameter("bias", (), init=jnp.zeros) ##It is better to remove the bias at the output and instead put it in the last linear layer: we avoid ill-conditioned NTKs


def simple_cnn(h, act, x):
    def conv2d(c, k, s, x):
        return hk.Conv2D(
            output_channels=c,
            kernel_shape=k,
            stride=s,
            padding='SAME',
            with_bias=True,
            w_init=hk.initializers.RandomNormal(),
            b_init=hk.initializers.RandomNormal(),
        )(x) / (k**2 * x.shape[-1]+1)**0.5

    def pool(x):
        return hk.max_pool(x,
        window_shape=2,
        strides=2,
        padding='SAME')

    x = act(conv2d(h, 5, 1, x))
    x = pool(x)
    x = act(conv2d(round(2 * h), 5, 1, x))
    x = pool(x)
    x = x.reshape((x.shape[0], -1))

    x = hk.Linear(
        output_size=32*h,
        with_bias=True,
        w_init=hk.initializers.RandomNormal(),
        b_init=hk.initializers.RandomNormal(),
    )(x) / (x.shape[-1]+1)**0.5

    x = hk.Linear(
        output_size=1,
        with_bias=False,
        w_init=hk.initializers.RandomNormal(),
    )(x) / (x.shape[-1])

    return x[..., 0]


def vgg11(h, act, x):
    # The convolutional architecture is defined by a sequence of tuples (number of convolutions, output channels)
    conv_arch = [(1, h), (1, round(2*h)), (2, round(4*h)), (2, round(8*h)), (2, round(8*h))]

    def act_conv2d(c, k, s, x):
        x =  hk.Conv2D(
            output_channels=c,
            kernel_shape=k,
            stride=s,
            padding='SAME',
            with_bias=True,
            w_init=hk.initializers.RandomNormal(),
            b_init=hk.initializers.RandomNormal(),
        )(x) / (k**2 * x.shape[-1]+1)**0.5
        return act(x)

    def pool(x):
        return hk.max_pool(x, window_shape=2, strides=2, padding='SAME')

    def vgg_block(num_convs, c, x):
        for _ in range(num_convs):
            x = act_conv2d(c, 3, 1, x)
        return pool(x)

    # Convolutions block
    for num_convs, channels in conv_arch:
        x = vgg_block(num_convs, channels, x)

    # Classifier block
    x = x.reshape((x.shape[0], -1))
    x = hk.Linear(output_size=round(64*h), with_bias=True, 
        w_init=hk.initializers.RandomNormal(), b_init=hk.initializers.RandomNormal(),
        )(x) / (x.shape[-1]+1)**0.5
    x = act(x)
    x = hk.Linear(output_size=round(64*h), with_bias=True, 
        w_init=hk.initializers.RandomNormal(), b_init=hk.initializers.RandomNormal(),
        )(x) / (x.shape[-1]+1)**0.5
    x = act(x)

    # Output layer
    x = hk.Linear(output_size=1, with_bias=False, 
        w_init=hk.initializers.RandomNormal(),
        )(x) / x.shape[-1]

    return x[..., 0]
  
