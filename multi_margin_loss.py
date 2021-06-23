import time
from typing import Optional

import mindspore
import sys
import numpy as np
from mindspore import nn
from mindspore import Tensor
from mindspore import ops
from mindspore import context


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
            x = x.squeeze(0)
            tmp = self.margin - x[y] + x
            if self.weight is not None:
                tmp = tmp * self.weight[y]
            loss = loss + ops.clip_by_value(tmp, 0, sys.float_info.max).mean() - self.margin * self.weight[y] / x.shape[
                0]
        return loss / batch_size


def test(batch_size=128, cls=20, compare_with_pytorch=False):
    weight = np.random.random([cls])
    x = np.random.random([batch_size, cls])
    y = np.random.randint(0, cls - 1, batch_size)

    # mindspore
    start_time = time.time()
    print(MyMultiMarginLoss(weight=mindspore.Tensor(weight, dtype=mindspore.float32))(
        mindspore.Tensor(x, dtype=mindspore.float32), mindspore.Tensor(y, dtype=mindspore.int64)))
    end_time = time.time()
    print("Mindspore time： ", end_time - start_time, (end_time - start_time) / batch_size)

    if compare_with_pytorch:
        import torch.nn
        start_time = time.time()
        print(torch.nn.MultiMarginLoss(weight=torch.from_numpy(weight))(torch.from_numpy(x), torch.from_numpy(y)))
        end_time = time.time()
        print("Pytorch time： ", end_time - start_time, (end_time - start_time) / batch_size)


if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test(128, 20, True)
    # context.set_context(device_id=4)
    # test(128, 20, False)
