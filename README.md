# SGD_learning_regimes
Code for reproducing the paper "How SGD noise affects performance in distinct regimes of deep learning"


What this code does:
1. Accepts many different [parameters]
2. Perform a single training of a neural network (depending on the parameters)
3. Compute and save observables during and at the end of the trianing (depending on the parameters)

The results are saved in a `pickle` format compatible with [grid](https://github.com/mariogeiger/grid) (`grid` allows to make sweeps in the paramters)

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

## Tuto: make a phase diagram

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
