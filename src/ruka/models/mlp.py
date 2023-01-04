import torch
from torch import nn
from torch.nn import functional as F
from .cnn_encoders import BaseFeaturesExtractor


class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=lambda x: x,
            hidden_init=nn.init.orthogonal_,
            b_init_value=0.,
            layer_norm=False,
            layer_norm_kwargs=None,
            input_dropout = 0,
            last_dropout = 0,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        self.input_dropout = nn.Dropout(input_dropout)

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            if hidden_init is not None:
                hidden_init(fc.weight)
                fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = nn.LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_dropout = nn.Dropout(last_dropout)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_preactivations=False):
        h = input
        h = self.input_dropout(h)
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        h = self.last_dropout(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class SplitIntoManyHeads(nn.Module):
    """
           .-> head 0
          /
    input ---> head 1
          \
           '-> head 2
    """
    def __init__(
            self,
            output_sizes,
            output_activations=None,
    ):
        super().__init__()
        if output_activations is None:
            output_activations = ['identity' for _ in output_sizes]
        else:
            if len(output_activations) != len(output_sizes):
                raise ValueError("output_activation and output_sizes must have "
                                 "the same length")

        self._output_narrow_params = []
        self._output_activations = []
        for output_activation in output_activations:
            self._output_activations.append(output_activation)
        start_idx = 0
        for output_size in output_sizes:
            self._output_narrow_params.append((start_idx, output_size))
            start_idx = start_idx + output_size

    def forward(self, flat_outputs):
        pre_activation_outputs = tuple(
            flat_outputs.narrow(1, start, length)
            for start, length in self._output_narrow_params
        )
        outputs = tuple(
            activation(x)
            for activation, x in zip(
                self._output_activations, pre_activation_outputs
            )
        )
        return outputs


class MultiHeadedMlp(Mlp):
    """
                   .-> linear head 0
                  /
    input --> MLP ---> linear head 1
                  \
                   .-> linear head 2
    """
    def __init__(
            self,
            hidden_sizes,
            output_sizes,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activations=None,
            hidden_init=nn.init.orthogonal_,
            b_init_value=0.,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__(
            hidden_sizes=hidden_sizes,
            output_size=sum(output_sizes),
            input_size=input_size,
            init_w=init_w,
            hidden_activation=hidden_activation,
            hidden_init=hidden_init,
            b_init_value=b_init_value,
            layer_norm=layer_norm,
            layer_norm_kwargs=layer_norm_kwargs,
        )
        self._splitter = SplitIntoManyHeads(
            output_sizes,
            output_activations,
        )

    def forward(self, input):
        flat_outputs = super().forward(input)
        return self._splitter(flat_outputs)


class SplitIntoManyHeads(nn.Module):
    """
           .-> head 0
          /
    input ---> head 1
          \
           '-> head 2
    """
    def __init__(
            self,
            output_sizes,
            output_activations=None,
    ):
        super().__init__()
        if output_activations is None:
            output_activations = ['identity' for _ in output_sizes]
        else:
            if len(output_activations) != len(output_sizes):
                raise ValueError("output_activation and output_sizes must have "
                                 "the same length")

        self._output_narrow_params = []
        self._output_activations = []
        for output_activation in output_activations:
            self._output_activations.append(output_activation)
        start_idx = 0
        for output_size in output_sizes:
            self._output_narrow_params.append((start_idx, output_size))
            start_idx = start_idx + output_size

    def forward(self, flat_outputs):
        pre_activation_outputs = tuple(
            flat_outputs.narrow(1, start, length)
            for start, length in self._output_narrow_params
        )
        outputs = tuple(
            activation(x)
            for activation, x in zip(
                self._output_activations, pre_activation_outputs
            )
        )
        return outputs

class ConcatMlp(Mlp):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)

class ConcatEncoderMlp(ConcatMlp):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, encoder: BaseFeaturesExtractor, action_dim: int, *args, **kwargs):
        super().__init__(*args, input_size=action_dim + encoder.features_dim, **kwargs)
        self._encoder = encoder

    def forward(self, observations, *inputs, **kwargs):
        encoded_observations = self._encoder(observations)
        return super().forward(encoded_observations, *inputs, **kwargs)


class SliceEncoder(nn.Module):
    """ Slice part of embeddings """
    def __init__(self, _slice: slice):
        super().__init__()
        self._slice = _slice

    def forward(self, x):
        assert len(x.shape) == 2, x.shape
        return x[:, self._slice]