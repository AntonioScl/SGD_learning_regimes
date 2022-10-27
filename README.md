# SGD_learning_regimes
Code for reproducing the paper "How SGD noise affects performance in distinct regimes of deep learning"


What this code does:
1. Accepts many different [parameters]
2. Perform a single training of a neural network (depending on the parameters)
3. Compute and save observables during and at the end of the trianing (depending on the parameters)

The results are saved in a `pickle` format compatible with [grid](https://github.com/mariogeiger/grid) (`grid` allows to make sweeps in the paramters)

## Paramters
A list of some of the paramters:

`--arch`    architecture
`--act`    activation function
`--h`    width
```--L```   depth (for ```mlp``` architecture)
```--alpha```   alpha paramter, corresponding to initialization scale
```--dataset```   dataset
```--ptr```   number of training points
```--pte```   number of test points
```--loss```   loss function
```--dynamics```   training dynamics
```--bs```  batch size for ```sgd``` dynamics
```--dt```   learning rate
```--temp```   temperature, defined as ```dt/(bs * h)``` (it is alternative to defining the learning rate)
```--ckpt_grad_stats```     number of train (test) points to compute the Gram matrix of the neural tangent kernel
```--max_wall```     maximum wall time (in seconds)
 ```--seed_init```  initialization seed


## Tuto: execute a single training

```
python -m edm --dataset mnist_parity --ptr 1024 --pte 2048 --arch mlp --act gelu --h 64 --L 8 --dynamics sgd --alpha 1 --dt 0.1 --bs 64 --max_wall 120 --output test.pk
```

Many parameters are set by default!

Then the data can be loaded using `pickle`
```python
import pickle

with open('test.pk', 'rb') as f:
    args = pickle.load(f)  # dict with the paramters
    data = pickle.load(f)  # all measurements
    
# data['sgd']['dynamics'] is a list of dict

print("Initial train loss is", data['sgd']['dynamics'][0]['train']['loss'])
print("Final test error is", data['sgd']['dynamics'][-1]['test']['err'])
```


## Tuto: sweeping over many parameters

Install [grid](https://github.com/mariogeiger/grid) and the current repository (`experimental_dynamics_machinery`).
Execute the following line that makes a sweep along the parameter `dt`, note that `grid` accept python code to create the list of parameters to sweep along.

```
python -m grid tests "python -m edm --dataset mnist_parity --ptr 1024 --pte 2048 --arch mlp --act gelu --h 64 --L 8 --dynamics sgd --alpha 1 --bs 64 --max_wall 120" --dt "[2**i for i in range(-3, 1)]"
```

At the end of the execution, the runs are saved in the directory name `tests` (in this example) and can be loaded as follow
```python
import grid
runs = grid.load('tests')

print("values of dt for the different runs", [r['args']['dt'] for r in runs])
```

See more info on how to sweep and load runs using grid in the [readme](https://github.com/mariogeiger/grid#readme).



## 5-hidden-layer fully connected architecture

On parity MNIST
```
python -m grid mnist-FC-5L "python -m edm --arch mlp --act relu --L 5 --h 128 --alpha 32768 --dataset mnist_parity --pte 32768 --loss hinge --dynamics sgd --bs 16 --ckpt_grad_stats 128 --max_wall 10000" --seed_init "[i for i in range(5)]" --ptr "[1024, 2048, 4096, 8192, 16384]" --temp "[2**i for i in range(-13,3)]"
```

On CIFAR animal
```
python -m grid cifar-FC-5L "python -m edm --arch mlp --act relu --L 5 --h 128 --alpha 32768 --dataset cifar_animal --pte 32768 --loss hinge --dynamics sgd --bs 16 --ckpt_grad_stats 128 --max_wall 10000" --seed_init "[i for i in range(5)]" --ptr "[1024, 2048, 4096, 8192, 16384]" --temp "[2**i for i in range(-13,3)]"
```


Phase diagram (alpha, dt) on parity MNIST
```
python -m grid mnist-alpha_dt-FC-5L "python -m edm --arch mlp --act gelu --L 5 --h 128 --dataset mnist_parity --ptr 1024 --pte 32768 --loss hinge --dynamics sgd --bs 16 --ckpt_grad_stats 1024 --max_wall 14400" --seed_init "[i for i in range(9)]" --alpha "[2**i for i in range(-14,12)]" --dt "[2**i for i in range(-15,16)]"
```


## MNAS

On parity MNIST
```
python -m grid mnist-MNAS "python -m edm --arch mnas --act relu --h 64 --alpha 32768 --dataset mnist_parity --pte 32768 --loss hinge --dynamics sgd --bs 16 --ckpt_grad_stats 128 --max_wall 10000" --seed_init "[i for i in range(5)]" --ptr "[1024, 2048, 4096, 8192, 16384]" --temp "[2**i for i in range(-20,1)]"
```

On CIFAR animal
```
python -m grid cifar-MNAS "python -m edm --arch mnas --act relu --h 64 --alpha 32768 --dataset cifar_animal --pte 32768 --loss hinge --dynamics sgd --bs 16 --ckpt_grad_stats 128 --max_wall 8000" --seed_init "[i for i in range(5)]" --ptr "[1024, 2048, 4096, 8192, 16384]" --temp "[2**i for i in range(-20,1)]"
```


Phase diagram (alpha, dt) on CIFAR animal
```
python -m grid cifar-alpha_dt-MNAS "python -m edm --arch mnas --act relu --h 32 --dataset cifar_animal --ptr 1024 --pte 32768 --loss hinge --dynamics sgd --bs 64 --ckpt_grad_stats 1024 --max_wall 3600" --seed_init "[i for i in range(4)]" --alpha "[2**i for i in range(-14,12)]" --dt "[2**i for i in range(-13,11)]"
```


## Perceptron

```
python -m grid depleted_perceptron "python -m edm --arch linear --alpha 32768 --dataset depleted_sign --pte 32768 --loss hinge --dynamics sgd --bs 2 --ckpt_grad_stats 512 --max_wall 10000 --ckpt_save_mult 1000000" --seed_init "[i for i in range(5)]" --data_chi "[3.0, 1.0, 0.0]" --d "[16, 32, 64, 128]" --ptr "[1024, 2048, 4096, 8192, 16384]" --dt "[2**i for i in range(-15,3)]"
```
