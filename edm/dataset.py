import jax
import jax.numpy as jnp
import os

#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # allow to use tensorflow_datasets without saturating memory
def configure_tensorflow():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)



def _gaussian(d, x, y, seed_trainset, seed_testset, ptr, pte):
    xtr = jax.random.normal(jax.random.PRNGKey(seed_trainset), (ptr, d))
    xte = jax.random.normal(jax.random.PRNGKey(seed_testset), (pte, d))

    return x(xtr), x(xte), y(xtr), y(xte)


def _gamma_gaussian(chi, d, x, y, seed_trainset, seed_testset, ptr, pte):

    keytrG, keytrS = jax.random.split(jax.random.PRNGKey(seed_trainset))
    chitr  = jnp.ones((ptr, d)) * ((0.0+1.0)/2.0)
    chitr  = chitr.at[:,0].set((chi+1.0)/2.0)
    xtr    = jax.random.gamma(keytrG, chitr, (ptr, d))
    xtr    = jnp.sqrt(xtr*2)
    signtr = 2*jax.random.bernoulli(keytrS, p=0.5, shape=(ptr, d))-1
    xtr    = signtr * xtr

    keyteG, keyteS = jax.random.split(jax.random.PRNGKey(seed_testset))
    chite  = jnp.ones((pte, d)) * ((0.0+1.0)/2.0)
    chite  = chite.at[:,0].set((chi+1.0)/2.0)
    xte    = jax.random.gamma(keyteG, chite, (pte, d))
    xte    = jnp.sqrt(xte*2)
    signte = 2*jax.random.bernoulli(keyteS, p=0.5, shape=(pte, d))-1
    xte    = signte * xte

    return x(xtr), x(xte), y(xtr), y(xte)


def stripe(d, seed_trainset, seed_testset, ptr, pte):
    def x(x):
        return x

    def y(x):
        return 2 * (x[:, 0] > -0.3) * (x[:, 0] < 1.18549) - 1

    return _gaussian(d, x, y, seed_trainset, seed_testset, ptr, pte)


def sphere(d, seed_trainset, seed_testset, ptr, pte):
    def x(x):
        return x

    def y(x):
        r = jnp.linalg.norm(x, axis=1)
        d = x.shape[1]
        return 2 * (r**2 > (d - (2 - 2**0.5))) - 1

    return _gaussian(d, x, y, seed_trainset, seed_testset, ptr, pte)


def _ball_uniform(key, p, d):
    """generate a ball of uniform points

    Args:
        key: random key
        p: number of points
        d: dimension

    Returns:
        points in the ball
    """
    key1, key2 = jax.random.split(key)
    x = jax.random.normal(key1, (p, d))
    r = jax.random.uniform(key2, (p, 1), minval=0.0, maxval=2.0)**(1 / d)
    n = jnp.linalg.norm(x, axis=1, keepdims=True)
    return r * x / n


def sphere_unif(d, seed_trainset, seed_testset, ptr, pte):
    """create a dataset of uniform points in a sphere
    label is -1 if the point is inside the sphere, 1 otherwise

    Args:
        d: dimension
        seed_trainset: seed for the random number generator
        seed_testset: seed for the random number generator
        ptr: number of training points
        pte: number of test points

    Returns:
        x_tr, x_te, y_tr, y_te: training and test points
    """
    xtr = _ball_uniform(jax.random.PRNGKey(seed_trainset), ptr, d)
    xte = _ball_uniform(jax.random.PRNGKey(seed_testset), pte, d)

    def x(x):
        return x * (d**0.5)

    def y(x):
        r = jnp.linalg.norm(x, axis=1)
        return 2 * (r > (d**0.5)) - 1

    return x(xtr), x(xte), y(xtr), y(xte)


def sign(d, seed_trainset, seed_testset, ptr, pte):
    def x(x):
        return x

    def y(x):
        return 2 * (x[:, 0] > 0) - 1

    return _gaussian(d, x, y, seed_trainset, seed_testset, ptr, pte)


def random_sign(d, seed_teacher, seed_trainset, seed_testset, ptr, pte):
    teacher = jax.random.normal(jax.random.PRNGKey(seed_teacher), (d,))
    teacher = teacher / jnp.linalg.norm(teacher)

    def x(x):
        return x

    def y(x):
        return 2 * (x @ teacher>0) - 1

    return _gaussian(d, x, y, seed_trainset, seed_testset, ptr, pte)


def depleted_sign(chi, d, seed_trainset, seed_testset, ptr, pte):
    def x(x):
        return x

    def y(x):
        return 2 * (x[:, 0] > 0) - 1

    return _gamma_gaussian(chi, d, x, y, seed_trainset, seed_testset, ptr, pte)


def sparse_linear(d, seed_trainset, seed_testset, ptr, pte):
    def x(x):
        return x

    def y(x):
        return 1*x[:, 0]

    return _gaussian(d, x, y, seed_trainset, seed_testset, ptr, pte)


def sphere_surface(d, seed_trainset, seed_testset, ptr, pte):
    def x(x):
        return x / jnp.linalg.norm(x, axis=1)[:, None] * (d**0.5)

    def y(x):
        return jnp.ones(x.shape[0])

    return _gaussian(d, x, y, seed_trainset, seed_testset, ptr, pte)


def _extract(ds, x, y, seed_trainset, seed_testset, ptr, pte):
    ds = ds.shuffle(len(ds), seed=seed_testset, reshuffle_each_iteration=False)
    dte = ds.take(pte)

    dtr = ds.skip(pte)
    dtr = dtr.shuffle(len(dtr), seed=seed_trainset, reshuffle_each_iteration=False)
    dtr = dtr.take(ptr)

    dtr = next(dtr.batch(len(dtr)).as_numpy_iterator())
    xtr, ytr = x(dtr['image']), y(dtr['label'])

    dte = next(dte.batch(len(dte)).as_numpy_iterator())
    xte, yte = x(dte['image']), y(dte['label'])

    return xtr, xte, ytr, yte


def mnist_parity(seed_trainset, seed_testset, ptr, pte):
    import tensorflow_datasets as tfds
    configure_tensorflow()

    ds = tfds.load("mnist", split='train+test')

    def x(images):
        return (jnp.array(images).astype(jnp.float32) - 33.31002426147461) / 78.56748962402344   # Zero mean, unit variance per pixel (datum norm=d)

    def y(labels):
        return jnp.array([
            -1.0 if y % 2 == 1 else 1.0
            for y in labels
        ])

    return _extract(ds, x, y, seed_trainset, seed_testset, ptr, pte)


def cifar_animal(seed_trainset, seed_testset, ptr, pte):
    import tensorflow_datasets as tfds
    configure_tensorflow()

    ds = tfds.load("cifar10", split='train+test')

    def x(images):
        return (jnp.array(images).astype(jnp.float32) - 120.93118286132812)/ 64.14186096191406    # Zero mean, unit variance per pixel (datum norm=d)

    def y(labels):
        return jnp.array([
            -1.0 if y in [0, 1, 8, 9] else 1.0
            for y in labels
        ])

    return _extract(ds, x, y, seed_trainset, seed_testset, ptr, pte)


def noisy_dataset(data_noise, x, seed):

    key = jax.random.PRNGKey(seed)
    key, k = jax.random.split(key)
    x = x + data_noise*jax.random.normal(k, x.shape)
    
    return x


def noisy_edge_dataset(data_noise, x, seed, pix=4):

    key = jax.random.PRNGKey(seed)
    key, k = jax.random.split(key)
    
    noise = data_noise*jax.random.normal(k, x.shape)

    if len(x.shape)==2:
        len1 = x.shape[1]
        edge_noise = noise.at[:,pix:len1-pix].set(0.0)

    if len(x.shape)==4:
        len1 = x.shape[1]
        len2 = x.shape[2]
        edge_noise = noise.at[:,pix:len1-pix,pix:len2-pix,:].set(0.0)
    x = x + data_noise * edge_noise
    
    return x
