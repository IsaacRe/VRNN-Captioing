import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss, _assert_no_grad
import torch.nn.functional as F

class CrossEntropyLoss(_WeightedLoss):
    r"""This criterion combines `LogSoftMax` and `NLLLoss` in one single class.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument `weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain scores for each class.

    `input` has to be a 2D `Tensor` of size `(minibatch, C)`.

    This criterion expects a class index (0 to C-1) as the
    `target` for each value of a 1D tensor of size `minibatch`

    The loss can be described as::

        loss(x, class) = -log(exp(x[class]) / (\sum_j exp(x[j])))
                       = -x[class] + log(\sum_j exp(x[j]))

    or in the case of the `weight` argument being specified::

        loss(x, class) = weight[class] * (-x[class] + log(\sum_j exp(x[j])))

    The losses are averaged across observations for each minibatch.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
           If given, has to be a Tensor of size "C"
        size_average (bool, optional): By default, the losses are averaged over observations for each minibatch.
           However, if the field size_average is set to ``False``, the losses are
           instead summed for each minibatch. Ignored if reduce is ``False``.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When size_average is
            ``True``, the loss is averaged over non-ignored targets.
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on size_average. When reduce
            is ``False``, returns a loss per batch element instead and ignores
            size_average. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`
        - Target: :math:`(N)` where each value is `0 <= targets[i] <= C-1`
        - Output: scalar. If reduce is ``False``, then :math:`(N)` instead.

    Examples::

        > loss = nn.CrossEntropyLoss()
        > input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
        > target = autograd.Variable(torch.LongTensor(3).random_(5))
        > output = loss(input, target)
        > output.backward()
    """

    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(CrossEntropyLoss, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.cross_entropy(input, target, self.weight, self.size_average,
                               self.ignore_index, self.reduce)
