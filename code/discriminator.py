import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn.functional as F
from torch.nn import Conv1d, Linear
from torch.nn.parameter import Parameter

from typing import Sequence, Dict, List, Callable, Tuple, Optional
import math
import numpy as np

from data import tgt_vocab_size, src_vocab_size
import util

class Discriminator(nn.Module):
    def __init__(self, emb_src, emb_tgt, emb_ans, gpu=False, dropout=0.5):
        super(Discriminator, self).__init__()
        # self.hidden_dim = 256
        self.embedding_dim = 300
        self.ans_dim = 16

        # self.gpu = gpu
        self.tgt_vocab_size = tgt_vocab_size + 4
        self.src_vocab_size = src_vocab_size + 3
        self.ans_vocab_size = 6
        # self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        # self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, num_layers=2, dropout=dropout)
        # self.gru2hidden = nn.Linear(2*self.hidden_dim, self.hidden_dim)
        # self.dropout_linear = nn.Dropout(p=dropout)
        # self.hidden2out = nn.Linear(self.hidden_dim, 1)
        self.emb_ans=emb_ans

        self.kernel_num_pa = 40
        self.kernel_num_qu = 40
        self.kernel_widths = [2, 3, 4, 5]  # todo: need to change?
        self.embeddings_src = emb_src
        self.convs1 = nn.ModuleList([nn.Conv2d(1, self.kernel_num_qu, (kernel_width, self.embedding_dim))
                                     for kernel_width in self.kernel_widths])

        self.embeddings_tgt = emb_tgt
        self.convs2 = nn.ModuleList([nn.Conv2d(1, self.kernel_num_pa, (kernel_width, self.embedding_dim + self.ans_dim))
                                     for kernel_width in self.kernel_widths])
        self.dropout = nn.Dropout(0.3)
        self.bn_qu = nn.BatchNorm2d(self.kernel_num_qu)
        self.bn_pa = nn.BatchNorm2d(self.kernel_num_pa)
        self.fc1 = nn.Linear(len(self.kernel_widths)*self.kernel_num_pa + len(self.kernel_widths)*self.kernel_num_qu, 1)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def __forward(self, query, passage):
        query = query[0]
        x = self.embeddings_tgt(query)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(self.bn_qu(conv(x))).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)

        ans = passage[1]
        ans = self.emb_ans(ans)
        pa = passage[0]
        y = self.embeddings_src(pa)  # (N, W, D)
        y = torch.cat((y, ans), dim=-1)
        y = y.unsqueeze(1)  # (N, Ci, W, D)
        y = [F.relu(self.bn_qu(conv(y))).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        y = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in y]  # [(N, Co), ...]*len(Ks)
        y = torch.cat(y, 1)

        p = torch.cat((x, y), 1)
        logit = torch.sigmoid(self.fc1(p))
        return logit

    def _forward(self, input, hidden):
        """
        input: (data, lengths)
        """
        lengths = input[1].tolist()
        src_data = input[0]

        emb = self.embeddings(src_data)                            # batch_size x seq_len x embedding_dim
        emb = emb.permute(1, 0, 2)                                 # seq_len x batch_size x embedding_dim
        emb = pack(emb, lengths)
        _, hidden = self.gru(emb, hidden)                          # 4 x batch_size x hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
        out = self.gru2hidden(hidden.view(-1, 2*self.hidden_dim))  # batch_size x 4*hidden_dim
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)                                 # batch_size x 1
        out = torch.sigmoid(out)
        return out

    def forward(self, query, passage):
        query = query[0]
        x = self.embeddings_tgt(query)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(self.bn_qu(conv(x))).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)

        ans = passage[1]
        ans = self.emb_ans(ans)
        pa = passage[0]
        y = self.embeddings_src(pa)  # (N, W, D)
        y = torch.cat((y, ans), dim=-1)
        y = y.unsqueeze(1)  # (N, Ci, W, D)
        y = [F.relu(self.bn_pa(conv(y))).squeeze(3) for conv in self.convs2]  # [(N, Co, W), ...]*len(Ks)
        y = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in y]  # [(N, Co), ...]*len(Ks)
        y = torch.cat(y, 1)

        p = torch.cat((x, y), 1)
        logit = torch.sigmoid(self.fc1(p))
        return logit

    def batchClassify(self, inp, passage):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """
        # h = self.init_hidden(inp[0].size()[0])
        out = self.forward(inp, passage)
        return out.view(-1)


from bert import BidirectionalTransformerEncoder


class LinearSimilarity(nn.Module):
    """
    This similarity function performs a dot product between a vector of weights and some
    combination of the two input vectors, followed by an (optional) activation function.  The
    combination used is configurable.
    If the two vectors are ``x`` and ``y``, we allow the following kinds of combinations: ``x``,
    ``y``, ``x*y``, ``x+y``, ``x-y``, ``x/y``, where each of those binary operations is performed
    elementwise.  You can list as many combinations as you want, comma separated.  For example, you
    might give ``x,y,x*y`` as the ``combination`` parameter to this class.  The computed similarity
    function would then be ``w^T [x; y; x*y] + b``, where ``w`` is a vector of weights, ``b`` is a
    bias parameter, and ``[;]`` is vector concatenation.
    Note that if you want a bilinear similarity function with a diagonal weight matrix W, where the
    similarity function is computed as `x * w * y + b` (with `w` the diagonal of `W`), you can
    accomplish that with this class by using "x*y" for `combination`.
    Parameters
    ----------
    tensor_1_dim : ``int``
        The dimension of the first tensor, ``x``, described above.  This is ``x.size()[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    tensor_2_dim : ``int``
        The dimension of the second tensor, ``y``, described above.  This is ``y.size()[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    combination : ``str``, optional (default="x,y")
        Described above.
    activation : ``Activation``, optional (default=linear (i.e. no activation))
        An activation function applied after the ``w^T * [x;y] + b`` calculation.  Default is no
        activation.
    """
    def __init__(self,
                 tensor_1_dim: int,
                 tensor_2_dim: int,
                 combination: str = 'x,y',
                 activation = None) -> None:
        super(LinearSimilarity, self).__init__()
        self._combination = combination
        combined_dim = util.get_combined_dim(combination, [tensor_1_dim, tensor_2_dim])
        self._weight_vector = Parameter(torch.Tensor(combined_dim))
        self._bias = Parameter(torch.Tensor(1))
        # self._activation = lambda: lambda x: x()
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(6 / (self._weight_vector.size(0) + 1))
        self._weight_vector.data.uniform_(-std, std)
        self._bias.data.fill_(0)

    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        combined_tensors = util.combine_tensors(self._combination, [tensor_1, tensor_2])
        dot_product = torch.matmul(combined_tensors, self._weight_vector)
        # return self._activation(dot_product + self._bias)
        return dot_product + self._bias

class DotProductSimilarity(nn.Module):
    """
    This similarity function simply computes the dot product between each pair of vectors, with an
    optional scaling to reduce the variance of the output elements.

    Parameters
    ----------
    scale_output : ``bool``, optional
        If ``True``, we will scale the output by ``math.sqrt(tensor.size(-1))``, to reduce the
        variance in the result.
    """
    def __init__(self, scale_output: bool = False) -> None:
        super(DotProductSimilarity, self).__init__()
        self._scale_output = scale_output


    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        result = (tensor_1 * tensor_2).sum(dim=-1)
        if self._scale_output:
            result *= math.sqrt(tensor_1.size(-1))
        return result

class LegacyMatrixAttention(nn.Module):
    """
    The legacy implementation of ``MatrixAttention``.
    It should be considered deprecated as it uses much more memory than the newer specialized
    ``MatrixAttention`` modules.
    Parameters
    ----------
    similarity_function: ``SimilarityFunction``, optional (default=``DotProductSimilarity``)
        The similarity function to use when computing the attention.
    """
    def __init__(self) -> None:
        super().__init__()
        # self._similarity_function = LinearSimilarity(tensor_1_dim=600,
        #                                              tensor_2_dim=600,
        #                                              combination="x,y,x*y")
        self._similarity_function = DotProductSimilarity()

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        tiled_matrix_1 = matrix_1.unsqueeze(2).expand(matrix_1.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_1.size()[2])
        tiled_matrix_2 = matrix_2.unsqueeze(1).expand(matrix_2.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_2.size()[2])
        return self._similarity_function(tiled_matrix_1, tiled_matrix_2)

class Highway(torch.nn.Module):
    """
    A `Highway layer <https://arxiv.org/abs/1505.00387>`_ does a gated combination of a linear
    transformation and a non-linear transformation of its input.  :math:`y = g * x + (1 - g) *
    f(A(x))`, where :math:`A` is a linear transformation, :math:`f` is an element-wise
    non-linearity, and :math:`g` is an element-wise gate, computed as :math:`sigmoid(B(x))`.
    This module will apply a fixed number of highway layers to its input, returning the final
    result.
    Parameters
    ----------
    input_dim : ``int``
        The dimensionality of :math:`x`.  We assume the input has shape ``(batch_size, ...,
        input_dim)``.
    num_layers : ``int``, optional (default=``1``)
        The number of highway layers to apply to the input.
    activation : ``Callable[[torch.Tensor], torch.Tensor]``, optional (default=``torch.nn.functional.relu``)
        The non-linearity to use in the highway layers.
    """
    def __init__(self,
                 input_dim: int,
                 num_layers: int = 1,
                 activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.relu) -> None:
        super(Highway, self).__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim * 2)
                                            for _ in range(num_layers)])
        self._activation = activation
        for layer in self._layers:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, so we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            # NOTE: if you modify this, think about whether you should modify the initialization
            # above, too.
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input

class MaskedLayerNorm(torch.nn.Module):
    def __init__(self, size: int, gamma0: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(1, 1, size) * gamma0)
        self.beta = torch.nn.Parameter(torch.zeros(1, 1, size))
        self.size = size
        self.eps = eps

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        broadcast_mask = mask.unsqueeze(-1).float()
        num_elements = broadcast_mask.sum() * self.size
        mean = (tensor * broadcast_mask).sum() / num_elements
        masked_centered = (tensor - mean) * broadcast_mask
        std = torch.sqrt(
                (masked_centered * masked_centered).sum() / num_elements + self.eps
        )
        return self.gamma * (tensor - mean) / (std + self.eps) + self.beta

_VALID_PROJECTION_LOCATIONS = {'after_cnn', 'after_highway', None}
class CnnHighwayEncoder(nn.Module):
    """
    The character CNN + highway encoder from Kim et al "Character aware neural language models"
    https://arxiv.org/abs/1508.06615
    with an optional projection.
    Parameters
    ----------
    embedding_dim: int
        The dimension of the initial character embedding.
    filters: ``Sequence[Sequence[int]]``
        A sequence of pairs (filter_width, num_filters).
    num_highway: int
        The number of highway layers.
    projection_dim: int
        The output dimension of the projection layer.
    activation: str, optional (default = 'relu')
        The activation function for the convolutional layers.
    projection_location: str, optional (default = 'after_highway')
        Where to apply the projection layer. Valid values are
        'after_highway', 'after_cnn', and None.
    do_layer_norm: bool, optional (default = False)
        If True, we apply ``MaskedLayerNorm`` to the final encoded result.
    """
    def __init__(self,
                 embedding_dim: int,
                 filters: Sequence[Sequence[int]],
                 num_highway: int,
                 projection_dim: int,
                 activation: str = 'relu',
                 projection_location: str = 'after_highway',
                 do_layer_norm: bool = False) -> None:
        super().__init__()

        if projection_location not in _VALID_PROJECTION_LOCATIONS:
            raise ConfigurationError(f"unknown projection location: {projection_location}")

        self.input_dim = embedding_dim
        self.output_dim = projection_dim
        self._projection_location = projection_location

        if activation == 'tanh':
            self._activation = torch.nn.functional.tanh
        elif activation == 'relu':
            self._activation = torch.nn.functional.relu
        else:
            raise ConfigurationError(f"unknown activation {activation}")

        # Create the convolutions
        self._convolutions: List[torch.nn.Module] = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(in_channels=embedding_dim,
                                   out_channels=num,
                                   kernel_size=width,
                                   bias=True)
            conv.weight.data.uniform_(-0.05, 0.05)
            conv.bias.data.fill_(0.0)
            self.add_module(f"char_conv_{i}", conv)  # needs to match the old ELMo name
            self._convolutions.append(conv)

        # Create the highway layers
        num_filters = sum(num for _, num in filters)
        if projection_location == 'after_cnn':
            highway_dim = projection_dim
        else:
            # highway_dim is the number of cnn filters
            highway_dim = num_filters
        self._highways = Highway(highway_dim, num_highway, activation=torch.nn.functional.relu)
        for highway_layer in self._highways._layers:   # pylint: disable=protected-access
            # highway is a linear layer for each highway layer
            # with fused W and b weights
            highway_layer.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / highway_dim))
            highway_layer.bias[:highway_dim].data.fill_(0.0)
            highway_layer.bias[highway_dim:].data.fill_(2.0)

        # Projection layer: always num_filters -> projection_dim
        self._projection = torch.nn.Linear(num_filters, projection_dim, bias=True)
        self._projection.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / num_filters))
        self._projection.bias.data.fill_(0.0)

        # And add a layer norm
        if do_layer_norm:
            self._layer_norm: Callable = MaskedLayerNorm(self.output_dim, gamma0=0.1)
        else:
            self._layer_norm = lambda tensor, mask: tensor

    def forward(self,
                inputs: torch.Tensor,
                mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute context insensitive token embeddings for ELMo representations.
        Parameters
        ----------
        inputs:
            Shape ``(batch_size, num_tokens, embedding_dim)``
            of character embeddings representing the current batch.
        mask:
            Shape ``(batch_size, num_tokens)``
            mask for the current batch.
        Returns
        -------
        ``encoding``:
            Shape ``(batch_size, sequence_length, embedding_dim2)`` tensor
            with context-insensitive token representations. If bos_characters and eos_characters
            are being added, the second dimension will be ``sequence_length + 2``.
        """
        # pylint: disable=arguments-differ

        # convolutions want (batch_size, embedding_dim, num_tokens)
        inputs = inputs.transpose(1, 2)

        convolutions = []
        for i in range(len(self._convolutions)):
            char_conv_i = getattr(self, f"char_conv_{i}")
            convolved = char_conv_i(inputs)

            # (batch_size, n_filters for this width)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = self._activation(convolved)
            convolutions.append(convolved)

        # (batch_size, n_filters)
        token_embedding = torch.cat(convolutions, dim=-1)

        if self._projection_location == 'after_cnn':
            token_embedding = self._projection(token_embedding)

        # apply the highway layers (batch_size, highway_dim)
        token_embedding = self._highways(token_embedding)

        if self._projection_location == 'after_highway':
            # final projection  (batch_size, embedding_dim)
            token_embedding = self._projection(token_embedding)

        # Apply layer norm if appropriate
        token_embedding = self._layer_norm(token_embedding, mask)

        return token_embedding

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim

class CnnEncoder(nn.Module):
    """
    A ``CnnEncoder`` is a combination of multiple convolution layers and max pooling layers.  As a
    :class:`Seq2VecEncoder`, the input to this module is of shape ``(batch_size, num_tokens,
    input_dim)``, and the output is of shape ``(batch_size, output_dim)``.
    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filters. The number of times a convolution layer will be used
    is ``num_tokens - ngram_size + 1``. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max.
    This operation is repeated for every ngram size passed, and consequently the dimensionality of
    the output after maxpooling is ``len(ngram_filter_sizes) * num_filters``.  This then gets
    (optionally) projected down to a lower dimensional output, specified by ``output_dim``.
    We then use a fully connected layer to project in back to the desired output_dim.  For more
    details, refer to "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
    Networks for Sentence Classification", Zhang and Wallace 2016, particularly Figure 1.
    Parameters
    ----------
    embedding_dim : ``int``
        This is the input dimension to the encoder.  We need this because we can't do shape
        inference in pytorch, and we need to know what size filters to construct in the CNN.
    num_filters: ``int``
        This is the output dim for each convolutional layer, which is the number of "filters"
        learned by that layer.
    ngram_filter_sizes: ``Tuple[int]``, optional (default=``(2, 3, 4, 5)``)
        This specifies both the number of convolutional layers we will create and their sizes.  The
        default of ``(2, 3, 4, 5)`` will have four convolutional layers, corresponding to encoding
        ngrams of size 2 to 5 with some number of filters.
    conv_layer_activation: ``Activation``, optional (default=``torch.nn.ReLU``)
        Activation to use after the convolution layers.
    output_dim : ``Optional[int]``, optional (default=``None``)
        After doing convolutions and pooling, we'll project the collected features into a vector of
        this size.  If this value is ``None``, we will just return the result of the max pooling,
        giving an output of shape ``len(ngram_filter_sizes) * num_filters``.
    """
    def __init__(self,
                 embedding_dim: int,
                 num_filters: int,
                 ngram_filter_sizes: Tuple[int, ...] = (3,5),  # pylint: disable=bad-whitespace
                 conv_layer_activation = None,
                 output_dim: Optional[int] = None) -> None:
        super(CnnEncoder, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = nn.ReLU()
        self._output_dim = output_dim

        self._convolution_layers = [Conv1d(in_channels=self._embedding_dim,
                                           out_channels=self._num_filters,
                                           kernel_size=ngram_size)
                                    for ngram_size in self._ngram_filter_sizes]
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module('conv_layer_%d' % i, conv_layer)

        maxpool_output_dim = self._num_filters * len(self._ngram_filter_sizes)
        if self._output_dim:
            self.projection_layer = Linear(maxpool_output_dim, self._output_dim)
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    def get_input_dim(self) -> int:
        return self._embedding_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()

        # Our input is expected to have shape `(batch_size, num_tokens, embedding_dim)`.  The
        # convolution layers expect input of shape `(batch_size, in_channels, sequence_length)`,
        # where the conv layer `in_channels` is our `embedding_dim`.  We thus need to transpose the
        # tensor first.
        tokens = torch.transpose(tokens, 1, 2)
        # Each convolution layer returns output of size `(batch_size, num_filters, pool_length)`,
        # where `pool_length = num_tokens - ngram_size + 1`.  We then do an activation function,
        # then do max pooling over each filter for the whole input sequence.  Because our max
        # pooling is simple, we just use `torch.max`.  The resultant tensor of has shape
        # `(batch_size, num_conv_layers * num_filters)`, which then gets projected using the
        # projection layer, if requested.

        filter_outputs = []
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, 'conv_layer_{}'.format(i))
            filter_outputs.append(
                    self._activation(convolution_layer(tokens)).max(dim=2)[0]
            )

        # Now we have a list of `num_conv_layers` tensors of shape `(batch_size, num_filters)`.
        # Concatenating them gives us a tensor of shape `(batch_size, num_filters * num_conv_layers)`.
        maxpool_output = torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]

        if self.projection_layer:
            result = self.projection_layer(maxpool_output)
        else:
            result = maxpool_output
        return result

class PQANet(nn.Module):
    def __init__(self, emb_src, emb_tgt):
        super(PQANet, self).__init__()
        self.embeddings_src = emb_src
        self.embeddings_tgt = emb_tgt

        self.passage_encoder = BidirectionalTransformerEncoder(input_dim=300,
                                                               hidden_dim=2048,
                                                               num_layers=1)
        self.query_encoder = BidirectionalTransformerEncoder(input_dim=300,
                                                             hidden_dim=2048,
                                                             num_layers=1)
        # self.query_encoder = self.passage_encoder

        self._matrix_attention = LegacyMatrixAttention()

        # self.combine = CnnEncoder(embedding_dim=600, num_filters=100)

        # self.max_
        self.linear = nn.Linear(600, 1)
        self.sigmoid = nn.Sigmoid()

    # def forward(self, passage, query, passage_mask, query_mask):
    def forward(self, query, passage):

        # 0.
        passage, passage_length = passage
        batch_size = passage.size(0)
        passage_length = passage.size(1)
        passage_mask = passage.eq(0)
        query_mask = query.eq(0)

        # 0.1 Encoding
        embedded_query = self.embeddings_tgt(query)  # (N, W, D)
        embedded_passage = self.embeddings_src(passage)

        # 1. Separately encoding.
        encoded_passage = self.passage_encoder(embedded_passage, passage_mask)
        encoded_query = self.query_encoder(embedded_query, query_mask)
        encoding_dim = encoded_query.size(-1)

        # maxpooled_passage = F.max_pool1d(encoded_passage.transpose(1,2), encoded_passage.size(1)).squeeze(2)
        # maxpooled_query = F.max_pool1d(encoded_query.transpose(1,2), encoded_query.size(1)).squeeze(2)
        #
        # output = torch.cat((maxpooled_passage, maxpooled_query), 1)
        output = torch.mean(encoded_query, 1)
        prob = self.sigmoid(self.linear(output))
        return prob

    def __forward(self, query, passage):

        # 0.
        passage, passage_length = passage
        batch_size = passage.size(0)
        passage_length = passage.size(1)
        passage_mask = passage.eq(0)
        query_mask = query.eq(0)

        # 0.1 Encoding
        embedded_query = self.embeddings_tgt(query)  # (N, W, D)
        embedded_passage = self.embeddings_src(passage)

        # 1. Separately encoding.
        encoded_passage = self.passage_encoder(embedded_passage, passage_mask)
        encoded_query = self.query_encoder(embedded_query, query_mask)
        encoding_dim = encoded_query.size(-1)

        # 2. Interaction.

        # Shape: (batch_size, passage_length, query_length)
        passage_query_similarity = self._matrix_attention(encoded_passage, encoded_query)
        # Shape: (batch_size, passage_length, query_length)
        passage_query_attention = util.masked_softmax(passage_query_similarity, query_mask)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_query_vectors = util.weighted_sum(encoded_query, passage_query_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = util.replace_masked_values(passage_query_similarity,
                                                       query_mask.unsqueeze(1),
                                                       -1e7)
        # Shape: (batch_size, passage_length)
        query_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # Shape: (batch_size, passage_length)
        query_passage_attention = util.masked_softmax(query_passage_similarity, passage_mask)
        # Shape: (batch_size, encoding_dim)
        query_passage_vector = util.weighted_sum(encoded_passage, query_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_query_passage_vector = query_passage_vector.unsqueeze(1).expand(batch_size,
                                                                              passage_length,
                                                                              encoding_dim)

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat([encoded_passage,
                                          passage_query_vectors,
                                          encoded_passage * passage_query_vectors,
                                          encoded_passage * tiled_query_passage_vector],
                                         dim=-1)

        # 3. Compress Composition Mix ... ? or just max_pooling or mean
        # output = self.combine(final_merged_passage, passage_mask)
        output = torch.mean(final_merged_passage, 1)

        prob = self.sigmoid(self.linear(output))
        return prob

    def _forward(self, query, passage):

        # 0.
        passage, passage_length = passage
        batch_size = passage.size(0)
        passage_length = passage.size(1)
        passage_mask = passage.eq(0)
        query_mask = query.eq(0)

        # 0.1 Encoding
        embedded_query = self.embeddings_tgt(query)  # (N, W, D)

        # 1. Separately encoding.
        encoded_query = self.query_encoder(embedded_query, query_mask)
        encoding_dim = encoded_query.size(-1)


        # 3. Compress Composition Mix ... ? or just max_pooling or mean
        output = self.combine(encoded_query, query_mask)
        # output = torch.mean(encoded_query, 1)

        prob = self.sigmoid(self.linear(output))
        return prob

    def batchClassify(self, inp, passage):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """
        # h = self.init_hidden(inp[0].size()[0])
        out = self.forward(inp, passage)
        return out.view(-1)


from transformer import TransformerEncoder
class TransormerNet(nn.Module):
    def __init__(self, emb_src, emb_tgt):
        super(TransormerNet, self).__init__()
        self.embeddings_src = emb_src
        self.embeddings_tgt = emb_tgt

        self.passage_encoder = TransformerEncoder(num_layers=2, d_model=300, heads=10, d_ff=2048,
                 dropout=0.1, embeddings=emb_src)
        self.query_encoder = TransformerEncoder(num_layers=2, d_model=300, heads=10, d_ff=2048,
                 dropout=0.1, embeddings=emb_src)
        # self.query_encoder = self.passage_encoder

        self._matrix_attention = LegacyMatrixAttention()

        self.combine = CnnEncoder(embedding_dim=1200, num_filters=100)

        # self.max_
        self.linear = nn.Linear(200, 1)
        self.sigmoid = nn.Sigmoid()

    # def forward(self, passage, query, passage_mask, query_mask):
    def ___forward(self, query, passage):

        # 0.
        passage, passage_length = passage
        batch_size = passage.size(0)
        passage_length = passage.size(1)
        passage_mask = passage.eq(0)
        query_mask = query.eq(0)

        # 0.1 Encoding
        # embedded_query = self.embeddings_tgt(query)  # (N, W, D)
        # embedded_passage = self.embeddings_src(passage)

        # 1. Separately encoding.
        encoded_passage = self.passage_encoder(passage, passage_mask)
        encoded_query = self.query_encoder(query, query_mask)
        encoding_dim = encoded_query.size(-1)

        # maxpooled_passage = F.max_pool1d(encoded_passage.transpose(1,2), encoded_passage.size(1)).squeeze(2)
        # maxpooled_query = F.max_pool1d(encoded_query.transpose(1,2), encoded_query.size(1)).squeeze(2)
        # output = torch.cat((maxpooled_passage, maxpooled_query), 1)

        mean_passage = torch.mean(encoded_passage, 1)
        mean_query = torch.mean(encoded_query, 1)
        output = torch.cat((mean_passage, mean_query), 1)

        prob = self.sigmoid(self.linear(output))
        return prob

    def forward(self, query, passage):

        # 0.
        passage, passage_length = passage
        batch_size = passage.size(0)
        passage_length = passage.size(1)
        passage_mask = passage.eq(0)
        query_mask = query.eq(0)


        # 0.1 Encoding
        # embedded_query = self.embeddings_tgt(query)  # (N, W, D)
        # embedded_passage = self.embeddings_src(passage)

        # 1. Separately encoding.

        passage_mask = passage.eq(0)
        query_mask = query.eq(0)
        encoded_passage = self.passage_encoder(passage, passage_mask)
        encoded_query = self.query_encoder(query, query_mask)
        encoding_dim = encoded_query.size(-1)

        # 2. Interaction.

        # Shape: (batch_size, passage_length, query_length)
        passage_query_similarity = self._matrix_attention(encoded_passage, encoded_query)
        # Shape: (batch_size, passage_length, query_length)
        passage_query_attention = util.masked_softmax(passage_query_similarity, query_mask)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_query_vectors = util.weighted_sum(encoded_query, passage_query_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = util.replace_masked_values(passage_query_similarity,
                                                       query_mask.unsqueeze(1),
                                                       -1e7)
        # Shape: (batch_size, passage_length)
        query_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # Shape: (batch_size, passage_length)
        query_passage_attention = util.masked_softmax(query_passage_similarity, passage_mask)
        # Shape: (batch_size, encoding_dim)
        query_passage_vector = util.weighted_sum(encoded_passage, query_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_query_passage_vector = query_passage_vector.unsqueeze(1).expand(batch_size,
                                                                              passage_length,
                                                                              encoding_dim)

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat([encoded_passage,
                                          passage_query_vectors,
                                          encoded_passage * passage_query_vectors,
                                          encoded_passage * tiled_query_passage_vector],
                                         dim=-1)

        # 3. Compress Composition Mix ... ? or just max_pooling or mean
        # output = self.combine(final_merged_passage, passage_mask)
        output = torch.mean(final_merged_passage, 1)

        prob = self.sigmoid(self.linear(output))
        return prob

    def _forward(self, query, passage):

        # 0.
        passage, passage_length = passage
        batch_size = passage.size(0)
        passage_length = passage.size(1)
        passage_mask = passage.eq(0)
        query_mask = query.eq(0)

        # 0.1 Encoding
        embedded_query = self.embeddings_tgt(query)  # (N, W, D)

        # 1. Separately encoding.
        encoded_query = self.query_encoder(embedded_query, query_mask)
        encoding_dim = encoded_query.size(-1)


        # 3. Compress Composition Mix ... ? or just max_pooling or mean
        output = self.combine(encoded_query, query_mask)
        # output = torch.mean(encoded_query, 1)

        prob = self.sigmoid(self.linear(output))
        return prob

    def batchClassify(self, inp, passage):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """
        # h = self.init_hidden(inp[0].size()[0])
        out = self.forward(inp, passage)
        return out.view(-1)

class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=-1)
        return alpha

class BiLSTM(nn.Module):
    def __init__(self, emb_src, emb_tgt, emb_ans):
        super(BiLSTM, self).__init__()
        self.embeddings_src = emb_src
        self.embeddings_tgt = emb_tgt
        self.embeddings_ans = emb_ans

        self.passage_encoder = nn.LSTM(input_size=300 + 16,
                                       hidden_size=256,
                                       num_layers=2,
                                       bidirectional=True,
                                       dropout=0.3)
        self.query_encoder = nn.LSTM(input_size=300,
                                     hidden_size=256,
                                     num_layers=2,
                                     bidirectional=True,
                                     dropout=0.3)

        self._matrix_attention = LegacyMatrixAttention()

        self.LinearAttn = LinearSeqAttn(input_size=512)

        self.linear = nn.Linear(2048, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, query, passage):

        # 0.
        passage, answer, src_lens = passage
        query, tgt_lens = query
        batch_size = passage.size(0)

        # sorted by lengths
        src_index = [index for index, value in sorted(list(enumerate(src_lens)), key=lambda x: x[1], reverse=True)]
        src_index_r = [index for index, value in sorted(list(enumerate(src_index)), key=lambda x: x[1])]
        src_index = torch.LongTensor(src_index)
        src_index_r = torch.LongTensor(src_index_r)
        tgt_index = [index for index, value in sorted(list(enumerate(tgt_lens)), key=lambda x: x[1], reverse=True)]
        tgt_index_r = [index for index, value in sorted(list(enumerate(tgt_index)), key=lambda x: x[1])]
        tgt_index = torch.LongTensor(tgt_index)
        tgt_index_r = torch.LongTensor(tgt_index_r)

        passage = passage[src_index].permute(1,0)
        answer = answer[src_index].permute(1,0)
        src_lens = src_lens[src_index].tolist()
        query = query[tgt_index].permute(1,0)
        tgt_lens = tgt_lens[tgt_index].tolist()

        # 0.1 Encoding
        embedded_passage = self.embeddings_src(passage)
        embedded_query = self.embeddings_tgt(query)  # (W, N, D)
        embedded_answer = self.embeddings_ans(answer)
        embedded_pa = torch.cat((embedded_passage, embedded_answer), dim=-1)

        # pack
        pa = pack(embedded_pa, src_lens)
        qu = pack(embedded_query, tgt_lens)

        # 1. Separately encoding.
        passage_hiddens, encoded_passage = self.passage_encoder(pa)
        query_hiddens, encoded_query = self.query_encoder(qu)

        query_rep = encoded_query[0].transpose(0,1).contiguous().view(batch_size, -1)
        passage_rep = encoded_passage[0].transpose(0,1).contiguous().view(batch_size, -1)

        # recover
        query_rep = query_rep[tgt_index_r]
        passage_rep = passage_rep[src_index_r]

        output = torch.cat((query_rep, passage_rep), 1)
        # output = query_rep
        prob = self.sigmoid(self.linear(output))
        return prob

    def batchClassify(self, inp, passage):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """
        # h = self.init_hidden(inp[0].size()[0])
        out = self.forward(inp, passage)
        return out.view(-1)


class StackedCNN(nn.Module):
    def __init__(self, emb_src, emb_tgt):
        super(StackedCNN , self).__init__()
        self.embeddings_src = emb_src
        self.embeddings_tgt = emb_tgt

        self.passage_encoder = nn.LSTM(input_size=300,
                                       hidden_size=256,
                                       num_layers=2,
                                       bidirectional=True,
                                       batch_first=True,
                                       dropout=0.3)
        self.query_encoder = nn.LSTM(input_size=300,
                                     hidden_size=256,
                                     num_layers=2,
                                     bidirectional=True,
                                     batch_first=True,
                                     dropout=0.3)

        self._matrix_attention = LegacyMatrixAttention()

        self.LinearAttn = LinearSeqAttn(input_size=512)

        self.linear = nn.Linear(2048, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, query, passage):

        # 0.
        passage, answer = passage
        batch_size = passage.size(0)
        passage_length = passage.size(1)
        passage_mask = passage.eq(0)
        query_mask = query.eq(0)

        # 0.1 Encoding
        embedded_passage = self.embeddings_src(passage)
        embedded_query = self.embeddings_tgt(query)  # (N, W, D)

        # 1. Separately encoding.
        passage_hiddens, encoded_passage = self.passage_encoder(embedded_passage)
        query_hiddens, encoded_query = self.query_encoder(embedded_query)

        query_rep = encoded_query[0].transpose(0,1).contiguous().view(batch_size, -1)
        passage_rep = encoded_passage[0].transpose(0,1).contiguous().view(batch_size, -1)

        output = torch.cat((query_rep, passage_rep), 1)
        # output = query_rep
        prob = self.sigmoid(self.linear(output))
        return prob

    def batchClassify(self, inp, passage):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """
        # h = self.init_hidden(inp[0].size()[0])
        out = self.forward(inp, passage)
        return out.view(-1)