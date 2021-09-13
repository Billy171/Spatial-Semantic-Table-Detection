import copy
from typing import Optional, List
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class TableDetectionDecoder(nn.Module):
    """
    This is a module that performs table detection given the output vectors of LayoutLMv2
    """
    def __init__(self, config):
        super().__init__()

        self.num_queries = config.num_queries
        decoder_layer = TransformerDecoderLayer(config.d_model, config.nhead,
                                                config.dropout, config.dim_feedforward,
                                                config.activation)
        self.TransformerDecoder = TransformerDecoder(config.num_layers, decoder_layer)

        self.class_embed = nn.Linear(config.d_model, config.num_classes + 1)
        self.bbox_embed = MLP(config.d_model, config.d_model, 4, 3)
        self.query_embed = nn.Embedding(config.num_queries, config.d_model)

    def forward(self, memory):
        """
        :param memory: output vectors of LayoutLMv2
        :return: predicted tables and bounding boxes
        """
        bs, d1, d2 = memory.shape
        memory = torch.reshape(memory, (bs, d2, d1))
        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        h = self.TransformerDecoder(tgt,
                                    memory,
                                    query_pos=query_embed)
        outputs_class = self.class_embed(h)
        outputs_coord = self.bbox_embed(h).sigmoid()
        #outputs = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        outputs = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        return outputs


class TransformerDecoder(nn.Module):
    """
    This module is the transformer decoder for the table detection task. It combines the transformer layers
    and returns a 'decoded' output
    """
    def __init__(self, num_layers, decoder_layer):
        super().__init__()
        self.num_layers = num_layers
        self.layers = _get_clones(decoder_layer, num_layers)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """
        :param tgt: initial zero vectors of size query_embed
        :param memory: the outputs of LayoutLMv2
        :param tgt_mask: not used in this application
        :param memory_mask: not used in this application
        :param tgt_key_padding_mask: not used in this application
        :param memory_key_padding_mask: not used in this application
        :param pos: not used in this application
        :param query_pos: the learned query embedding
        :return: returns decoded vectors
        """
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
        return output


class TransformerDecoderLayer(nn.Module):
    """
    This module defines a single transformer decoder layer
    """
    def __init__(self, d_model, nhead, dropout, dim_feedforward, activation):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """
        :param tgt: the input into the layer
        :param memory: the outputs of LayoutLMv2
        :param tgt_mask: not used in this application
        :param memory_mask: not used in this application
        :param tgt_key_padding_mask: not used in this application
        :param memory_key_padding_mask: not used in this application
        :param pos: not used in this application
        :param query_pos: the learned query embedding
        :return: output vectors of the layer
        """
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class MLP(nn.Module):
    """
    Very simple multi-layer perceptron (also called FFN)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")




