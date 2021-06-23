from typing import Optional

import mindspore
import sys
import numpy as np
from mindspore import nn
from mindspore import Tensor
from mindspore import ops
from mindspore import context
import torch.nn


class MyMultiMarginLoss(nn.Cell):

    def __init__(self, margin: float = 1., weight: Optional[Tensor] = None, reduction: str = 'mean') -> None:
        super(MyMultiMarginLoss, self).__init__()
        assert weight is None or weight.dim() == 1
        self.weight = weight
        self.margin = margin
        self.reduction = reduction

    def construct(self, input: Tensor, target: Tensor) -> Tensor:
        batch_size = input.shape[0]
        split = ops.Split(0, batch_size)
        xs = split(input)
        ys = split(target)
        loss = 0
        for x, y in zip(xs, ys):
            x = x.reshape([x.shape[1]])
            tmp = self.margin - x[y] + x
            if self.weight is not None:
                tmp = tmp * self.weight[y]
            # tmp[y] = tmp[y] - self.margin
            loss = loss + ops.clip_by_value(tmp, 0, sys.float_info.max).mean() - self.margin * self.weight[y] / x.shape[
                0]
        return loss / batch_size


if __name__ == '__main__':
    weight = np.random.random([100])
    x = np.random.random([100, 100])
    y = np.random.randint(0, 99, 100)

    # mindspore
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    print(MyMultiMarginLoss(weight=mindspore.Tensor(weight, dtype=mindspore.float32))(
        mindspore.Tensor(x, dtype=mindspore.float32), mindspore.Tensor(y, dtype=mindspore.int64)))
    # pytorch
    print(torch.nn.MultiMarginLoss(weight=torch.from_numpy(weight))(torch.from_numpy(x), torch.from_numpy(y)))
