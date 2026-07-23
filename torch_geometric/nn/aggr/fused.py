import math
from typing import Dict, List, Optional, Tuple, Union

from torch import Tensor

from torch_geometric.nn.aggr.base import Aggregation
from torch_geometric.nn.aggr.basic import (
    MaxAggregation,
    MeanAggregation,
    MinAggregation,
    MulAggregation,
    StdAggregation,
    SumAggregation,
    VarAggregation,
)
from torch_geometric.nn.resolver import aggregation_resolver
from torch_geometric.utils import scatter


class FusedAggregation(Aggregation):
    FUSABLE_AGGRS = {
        SumAggregation,
        MeanAggregation,
        MinAggregation,
        MaxAggregation,
        MulAggregation,
        VarAggregation,
        StdAggregation,
    }

    DEGREE_BASED_AGGRS = {
        MeanAggregation,
        VarAggregation,
        StdAggregation,
    }

    REDUCE = {
        'SumAggregation': 'sum',
        'MeanAggregation': 'sum',
        'MinAggregation': 'min',
        'MaxAggregation': 'max',
        'MulAggregation': 'mul',
        'VarAggregation': 'pow_sum',
        'StdAggregation': 'pow_sum',
    }

    def __init__(self, aggrs: List[Union[Aggregation, str]]):
        super().__init__()

        if not isinstance(aggrs, (list, tuple)):
            raise ValueError(f"'aggrs' of '{self.__class__.__name__}' should "
                             f"be a list or tuple (got '{type(aggrs)}').")

        if len(aggrs) == 0:
            raise ValueError(f"'aggrs' of '{self.__class__.__name__}' should "
                             f"not be empty.")

        aggrs = [aggregation_resolver(aggr) for aggr in aggrs]
        aggr_classes = [aggr.__class__ for aggr in aggrs]
        self.aggr_names = [cls.__name__ for cls in aggr_classes]
        self.aggr_index: Dict[str, int] = {
            name: i
            for i, name in enumerate(self.aggr_names)
        }

        for cls in aggr_classes:
            if cls not in self.FUSABLE_AGGRS:
                raise ValueError(f"Received aggregation '{cls.__name__}' in "
                                 f"'{self.__class__.__name__}' which is not "
                                 f"fusable")

        self.semi_grad = False
        for aggr in aggrs:
            if hasattr(aggr, 'semi_grad'):
                self.semi_grad = self.semi_grad or aggr.semi_grad

        self.need_degree = False
        for cls in aggr_classes:
            if cls in self.DEGREE_BASED_AGGRS:
                self.need_degree = True

        reduce_ops: List[Optional[str]] = []
        lookup_ops: List[Optional[Tuple[str, int]]] = []

        for name in self.aggr_names:
            if name == 'MeanAggregation':
                if 'SumAggregation' in self.aggr_index:
                    reduce_ops.append(None)
                    lookup_ops.append((
                        'SumAggregation',
                        self.aggr_index['SumAggregation'],
                    ))
                else:
                    reduce_ops.append(self.REDUCE[name])
                    lookup_ops.append(None)

            elif name == 'VarAggregation':
                if 'MeanAggregation' in self.aggr_index:
                    reduce_ops.append(self.REDUCE[name])
                    lookup_ops.append((
                        'MeanAggregation',
                        self.aggr_index['MeanAggregation'],
                    ))
                elif 'SumAggregation' in self.aggr_index:
                    reduce_ops.append(self.REDUCE[name])
                    lookup_ops.append((
                        'SumAggregation',
                        self.aggr_index['SumAggregation'],
                    ))
                else:
                    reduce_ops.append(self.REDUCE[name])
                    lookup_ops.append(None)

            elif name == 'StdAggregation':
                if 'VarAggregation' in self.aggr_index:
                    reduce_ops.append(None)
                    lookup_ops.append((
                        'VarAggregation',
                        self.aggr_index['VarAggregation'],
                    ))
                elif 'MeanAggregation' in self.aggr_index:
                    reduce_ops.append(self.REDUCE[name])
                    lookup_ops.append((
                        'MeanAggregation',
                        self.aggr_index['MeanAggregation'],
                    ))
                elif 'SumAggregation' in self.aggr_index:
                    reduce_ops.append(self.REDUCE[name])
                    lookup_ops.append((
                        'SumAggregation',
                        self.aggr_index['SumAggregation'],
                    ))
                else:
                    reduce_ops.append(self.REDUCE[name])
                    lookup_ops.append(None)

            else:
                reduce_ops.append(self.REDUCE[name])
                lookup_ops.append(None)

        self.reduce_ops: List[Optional[str]] = reduce_ops
        self.lookup_ops: List[Optional[Tuple[str, int]]] = lookup_ops

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> List[Tensor]:

        self.assert_index_present(index)
        self.assert_two_dimensional_input(x, dim)

        assert index is not None

        if dim_size is None:
            if ptr is not None:
                dim_size = ptr.numel() - 1
            else:
                dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

        count: Optional[Tensor] = None
        if self.need_degree:
            count = x.new_zeros(dim_size)
            count.scatter_add_(0, index, x.new_ones(x.size(0)))
            count = count.clamp_(min=1).view(-1, 1)


        outs: List[Optional[Tensor]] = []

        for reduce in self.reduce_ops:
            if reduce is None:
                outs.append(None)
                continue
            assert isinstance(reduce, str)

            if reduce == 'pow_sum':
                if self.semi_grad:
                    out = scatter(x.detach() * x.detach(), index, 0, dim_size,
                                  reduce='sum')
                else:
                    out = scatter(x * x, index, 0, dim_size, reduce='sum')
            else:
                out = scatter(x, index, 0, dim_size, reduce=reduce)

            outs.append(out)


        i = self.aggr_index.get('MeanAggregation')
        if i is not None:
            assert count is not None

            if self.lookup_ops[i] is None:
                sum_ = outs[i]
            else:
                lookup_op = self.lookup_ops[i]
                assert lookup_op is not None
                tmp_aggr, j = lookup_op
                assert tmp_aggr == 'SumAggregation'

                sum_ = outs[j]

            assert sum_ is not None
            outs[i] = sum_ / count

        if 'VarAggregation' in self.aggr_index:
            i = self.aggr_index['VarAggregation']

            assert count is not None

            if self.lookup_ops[i] is None:
                sum_ = scatter(x, index, 0, dim_size, reduce='sum')
                mean = sum_ / count
            else:
                lookup_op = self.lookup_ops[i]
                assert lookup_op is not None
                tmp_aggr, j = lookup_op

                if tmp_aggr == 'SumAggregation':
                    sum_ = outs[j]
                    assert sum_ is not None
                    mean = sum_ / count
                elif tmp_aggr == 'MeanAggregation':
                    mean = outs[j]
                else:
                    raise NotImplementedError

            pow_sum = outs[i]

            assert pow_sum is not None
            assert mean is not None
            outs[i] = (pow_sum / count) - (mean * mean)

        if 'StdAggregation' in self.aggr_index:
            i = self.aggr_index['StdAggregation']

            var: Optional[Tensor] = None
            pow_sum: Optional[Tensor] = None
            mean: Optional[Tensor] = None

            if self.lookup_ops[i] is None:
                pow_sum = outs[i]
                sum_ = scatter(x, index, 0, dim_size, reduce='sum')
                assert count is not None
                mean = sum_ / count
            else:
                lookup_op = self.lookup_ops[i]
                assert lookup_op is not None
                tmp_aggr, j = lookup_op

                if tmp_aggr == 'VarAggregation':
                    var = outs[j]
                elif tmp_aggr == 'SumAggregation':
                    pow_sum = outs[i]
                    sum_ = outs[j]
                    assert sum_ is not None
                    assert count is not None
                    mean = sum_ / count
                elif tmp_aggr == 'MeanAggregation':
                    pow_sum = outs[i]
                    mean = outs[j]
                else:
                    raise NotImplementedError

            if var is None:
                assert pow_sum is not None
                assert count is not None
                assert mean is not None
                var = (pow_sum / count) - (mean * mean)

            out = var.clamp(min=1e-5).sqrt()
            out = out.masked_fill(out <= math.sqrt(1e-5), 0.0)

            outs[i] = out


        vals: List[Tensor] = []
        for out in outs:
            assert out is not None
            vals.append(out)

        return vals
