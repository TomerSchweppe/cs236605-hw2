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
1. We observe that the deeper networks performed poorly compared to the shallower networks. Meaning, there is negative correlation between the depth of the network and the test accuracy. Furthermore, we observe that the deeper networks (8, 16) have a pattern of overfiting. We think that the shallow networks achieved higher accuracy rate because they weren't over parameterized and therefore were able to achieve better generalization and not to overfit. 
2. No, all our networks were able to train, but not all of them generalized as well.
From our analysis we can also deduce that enlarging the number of filters per layer helped the model better generalize while increasing the number layers didn't.
"""

part3_q2 = r"""
We conclude from the results of the experiment that increasing the number of filters per layer helps the model better generalize and achieve better test accuracy. This result matches the results we received in the previous experiment.
Our analysis is based on experiments with 2, 4 layers networks. The 8 layers network didn't converge and the results are not reliable for analysis.
"""

part3_q3 = r"""
We observe that there isn't much difference in the performance of the network with 1 or 2 layers per block. We think that both networks match our observations and conclusions from previous experiments that low number of layers and larger number of filters per layer is better than just increasing the number of layers.
Furthermore, we see that 3 layers per block doesn't perform as well as shallower networks and starts to overfit like we saw in the first experiment.
The network with 4 layers per block was unable to train. We assume that the train wasn't able to train because it was too deep and probably had the problem of vanishing gradients, meaning that the gradients were slowly decreasing in ampliplitude between layers in the back propagation stage and therefore those layers weren't able to train. Note this is the deepest network we tried to train throughout our experiments.
"""


part3_q4 = r"""
Our implementation is making use of Batch Normalization, Dropout and Xavier Initialization. We applied those enhancments in order to be able to train bigger networks without overfitting, those methods were developed in order to stabilize the gradients flowing through the network and eliminate the problem of vanishing gradients. From the experiments above we concluded that the main issue in training large networks is overfitting to the training data.
While previous results showed poor performance, in some cases the networks completelly diverged, when using networks containing many parameters, using our enhancements even larger networks were able to learn and even converge and generalize reaching much higher accuracy rates.
"""
# ==============
