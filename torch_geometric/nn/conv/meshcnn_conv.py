import warnings
from typing import Optional

import torch
from torch.nn import Linear, Module, ModuleList

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Tensor


class MeshCNNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 kernels: Optional[ModuleList] = None):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels

        if kernels is None:
            self.kernels = ModuleList(
                [Linear(in_channels, out_channels) for _ in range(5)])

        else:
            self._assert_kernels(kernels)
            self.kernels = kernels

    def forward(self, x: Tensor, edge_index: Tensor):
        r"""Forward pass.

        Args:
            x(torch.Tensor): :math:`X^{(k)} \in
                \mathbb{R}^{|E| \times \textit{in_channels}}`.
                The edge feature tensor returned by the prior layer
                (e.g. :math:`k`). The tensor is of shape
                :math:`|E| \times \text{Dim-Out}(k)`, or equivalently,
                :obj:`(|E|, self.in_channels)`.

            edge_index(torch.Tensor):
                :math:`A \in \{0,...,|E|-1\}^{2 \times 4*|E|}`.
                The edge adjacency tensor of the networks input mesh
                :math:`\mathcal{m} = (V, F)`. The edge adjacency tensor
                **MUST** have the following form:

                .. math::
                    &A[:,0] = (0,
                        \text{The index of the "a" edge for edge } 0) \\
                    &A[:,1] = (0,
                        \text{The index of the "b" edge for edge } 0) \\
                    &A[:,2] = (0,
                        \text{The index of the "c" edge for edge } 0) \\
                    &A[:,3] = (0,
                        \text{The index of the "d" edge for edge } 0) \\
                    \vdots \\
                    &A[:,4*|E|-4] =
                        \bigl(|E|-1,
                            a\bigl(|E|-1\bigr)\bigr) \\
                    &A[:,4*|E|-3] =
                        \bigl(|E|-1,
                            b\bigl(|E|-1\bigr)\bigr) \\
                    &A[:,4*|E|-2] =
                        \bigl(|E|-1,
                            c\bigl(|E|-1\bigr)\bigr) \\
                    &A[:,4*|E|-1] =
                        \bigl(|E|-1,
                            d\bigl(|E|-1\bigr)\bigr)

                See :obj:`MeshCNNConv` for what
                "index of the 'a'(b,c,d) edge for edge i" means, and also
                for the general definition of edge adjacency in MeshCNN.
                These definitions are also provided in the
                `paper <https://arxiv.org/abs/1809.05910>`_ itself.

        Returns:
           torch.Tensor:
           :math:`X^{(k+1)} \in \mathbb{R}^{|E| \times \textit{out_channels}}`.
           The edge feature tensor for this (e.g. the :math:`k+1` th) layer.
           The :math:`i` th row of :math:`X^{(k+1)}` is computed according
           to the formula

            .. math::
                x^{(k+1)}_i &= W^{(k+1)}_0 x^{(k)}_i \\
                &+ W^{(k+1)}_1 \bigl| x^{(k)}_{a(i)} - x^{(k)}_{c(i)} \bigr| \\
                &+ W^{(k+1)}_2 \bigl( x^{(k)}_{a(i)} + x^{(k)}_{c(i)} \bigr) \\
                &+ W^{(k+1)}_3 \bigl| x^{(k)}_{b(i)} - x^{(k)}_{d(i)} \bigr| \\
                &+ W^{(k+1)}_4 \bigl( x^{(k)}_{b(i)} + x^{(k)}_{d(i)} \bigr),

            where :math:`W_0^{(k+1)},W_1^{(k+1)},
            W_2^{(k+1)},W_3^{(k+1)}, W_4^{(k+1)}
            \in \mathbb{R}^{\text{Dim-Out}(k+1) \times \text{Dim-Out}(k)}`
            are the trainable linear functions (i.e. the trainable
            "weights") of this layer, and
            :math:`x^{(k)}_{a(i)}, x^{(k)}_{b(i)}, x^{(k)}_{c(i)}`,
            :math:`x^{(k)}_{d(i)}` are the
            :math:`\text{Dim-Out}(k)`-dimensional edge feature vectors
            computed by the prior (:math:`k` th) layer,
            that are associated with the :math:`4`
            neighboring edges of :math:`e_i`.

        """
        return self.propagate(edge_index, x=x)

    def message(self, x_j: Tensor) -> Tensor:
        r"""The messaging passing step of :obj:`MeshCNNConv`.


        Args:
          x_j: A :obj:`[4*|E|, num_node_features]` tensor.
          Its ith row holds the value
            stored by the source node in the previous layer of edge i.

        Returns:
            A :obj:`[|E|, num_node_features]` tensor,
            whose ith row will be the value
            that the target node of edge i will receive.
        """


        E4, in_channels = x_j.size()  # E4 = 4|E|, i.e. num edges in line graph
        n_a = x_j[0::4]  # shape: |E| x in_channels
        n_b = x_j[1::4]  # shape: |E| x in_channels
        n_c = x_j[2::4]  # shape: |E| x in_channels
        n_d = x_j[3::4]  # shape: |E| x in_channels
        m = torch.empty(E4, self.out_channels)
        m[0::4] = self.kernels[1].forward(torch.abs(n_a - n_c))
        m[1::4] = self.kernels[2].forward(n_a + n_c)
        m[2::4] = self.kernels[3].forward(torch.abs(n_b - n_d))
        m[3::4] = self.kernels[4].forward(n_b + n_d)
        return m


    def update(self, inputs: Tensor, x: Tensor) -> Tensor:
        r"""The UPDATE step, in reference to the UPDATE and AGGREGATE
        formulation of message passing convolution.

        Args:
           inputs(torch.Tensor): The :attr:`in_channels`-dimensional vector
            returned by aggregate.
           x(torch.Tensor): :math:`X^{(k)}`. The original inputs to this layer.

        Returns:
            torch.Tensor: :math:`X^{(k+1)}`. The output of this layer, which
            has shape :obj:`(|E|, out_channels)`.
        """
        return self.kernels[0].forward(x) + inputs

    def _assert_kernels(self, kernels: ModuleList):
        r"""Ensures that :obj:`kernels` is a list of 5 :obj:`torch.nn.Module`
        modules (i.e. networks). In addition, it also ensures that each network
        takes in input of dimension :attr:`in_channels`, and returns output
        of dimension :attr:`out_channels`.
        This method throws an error otherwise.

        .. warn::
            This method throws an error if :obj:`kernels` is
            not valid. (Otherwise this method returns nothing)

        """
        assert isinstance(kernels, ModuleList), \
            f"Parameter 'kernels' must be a \
            torch.nn.module.ModuleList with 5 members, but we got \
            {type(kernels)}."

        assert len(kernels) == 5, "Parameter 'kernels' must be a \
            torch.nn.module.ModuleList of with exactly 5 members"

        for i, network in enumerate(kernels):
            assert isinstance(network, Module), \
                f"kernels[{i}] must be torch.nn.Module, got \
                {type(network)}"
            if not hasattr(network, "in_channels") and \
                    not hasattr(network, "in_features"):
                warnings.warn(
                    f"kernel[{i}] does not have attribute 'in_channels' nor "
                    f"'out_features'. The network must take as input a "
                    f"{self.in_channels}-dimensional tensor.", stacklevel=2)
            else:
                input_dimension = getattr(network, "in_channels",
                                          network.in_features)
                assert input_dimension == self.in_channels, f"The input \
                dimension of the neural network in kernel[{i}] must \
                be \
                equal to 'in_channels', but input_dimension = \
                {input_dimension}, and \
                self.in_channels={self.in_channels}."

            if not hasattr(network, "out_channels") and \
                    not hasattr(network, "out_features"):
                warnings.warn(
                    f"kernel[{i}] does not have attribute 'in_channels' nor "
                    f"'out_features'. The network must take as input a "
                    f"{self.in_channels}-dimensional tensor.", stacklevel=2)
            else:
                output_dimension = getattr(network, "out_channels",
                                           network.out_features)
                assert output_dimension == self.out_channels, f"The output \
                    dimension of the neural network in kernel[{i}] must \
                    be \
                    equal to 'out_channels', but out_dimension = \
                    {output_dimension}, and \
                    self.out_channels={self.out_channels}."
