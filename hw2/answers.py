r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.01
    lr =  0.01
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.01
    lr_vanilla = 0.015
    lr_momentum = 0.0035
    lr_rmsprop = 0.00014
    reg = 0.00001
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.01
    lr = 0.0035
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
1. The graphs we received matched our expectations. We were expecting to see better generalization in the dropout 
setting. We can see that without dropout our model overfits the training data and fails to generalize, we can see 
from the graphs  
that the training loss decreasing gradually in contrast to the test loss which starts to increase after a few 
iterations and is not stable. In the dropout setting we can see that both the training loss and the test loss are 
decreasing gradually together resulting in higher test accuracy which implies better generalizations.
2. We see that higher dropout rates result in slower learning and requires more iterations to converge.
In our case we see that lower dropout rate leads to better results in terms of speed of convergence and 
generalization meaning higher test accuracy. 
Since we perform the experiment with a small number of epochs, we can't tell if higher dropout rate will result in 
better convergence based on these results. 
"""

part2_q2 = r"""
It is possible for the test loss to increase for a few epochs while the test accuracy also increases. Since Cross 
Entropy loss takes into account the scores of all the categories, in situations that the model is suddenly less sure 
of its choice and gives the correct labels lower scores compared to other labels, the loss will increase but 
the accuracy will remain the same. Furthermore, the model could get more labels correct as a consequence of the 
effect we 
described of less sure choices and that will cause the test loss to increase while the test accuracy also increases.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
