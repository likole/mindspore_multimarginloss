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

    def __init__(self, class_num, margin: float = 1., weight: Optional[Tensor] = None) -> None:
        super(MyMultiMarginLoss, self).__init__()
        assert weight is None or weight.dim() == 1
        self.weight = weight
        self.margin = margin
        self.on_value, self.off_value = Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32)
        self.op_sum = ops.ReduceSum(keep_dims=True)
        self.onehot = nn.OneHot(depth=class_num, axis=1)

    def construct(self, input: Tensor, target: Tensor) -> Tensor:
        target = self.onehot(target)
        loss = self.margin - self.op_sum(input * target, 1) + input - target
        if self.weight is not None:
            loss = loss * self.op_sum(target * self.weight, 1)
        return ops.clip_by_value(loss, 0, sys.float_info.max).mean()


def test(batch_size=128, cls=20, times=1, compare_with_pytorch=False):
    weight = np.random.random([cls])
    criteria_mindspore = MyMultiMarginLoss(cls, weight=mindspore.Tensor(weight, dtype=mindspore.float32))
    if compare_with_pytorch:
        import torch.nn
        criteria_pytorch = torch.nn.MultiMarginLoss(weight=torch.from_numpy(weight))

    for _ in range(times):
        print("=" * 20)
        x = np.random.random([batch_size, cls])
        y = np.random.randint(0, cls - 1, batch_size)
        start_time = time.time()
        print(criteria_mindspore(mindspore.Tensor(x, dtype=mindspore.float32),
                                 mindspore.Tensor(y, dtype=mindspore.int32)))
        end_time = time.time()
        print("Mindspore time： ", end_time - start_time, (end_time - start_time) / batch_size)

        if compare_with_pytorch:
            start_time = time.time()
            print(criteria_pytorch(torch.from_numpy(x), torch.from_numpy(y)))
            end_time = time.time()
            print("Pytorch time： ", end_time - start_time, (end_time - start_time) / batch_size)


if __name__ == '__main__':
    # context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    context.set_context(device_id=5)
    test(batch_size=128, cls=20, times=100, compare_with_pytorch=True)
