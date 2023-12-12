from transformers.models.bert.modeling_bert import BertEncoder, BertLayer, BertSelfAttention, BertAttention
import torch.nn as nn
import math
import torch

class BertSelfAttentionPast(BertSelfAttention):
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            layer_past=None,
            cache_query=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if layer_past is not None:
            if cache_query:
                past_q = layer_past[2]
                query_layer = torch.cat((past_q, query_layer), dim=-2)

            past_k, past_v = layer_past[0], layer_past[1]
            key_layer = torch.cat((past_k, key_layer), dim=-2)
            value_layer = torch.cat((past_v, value_layer), dim=-2)

        if cache_query:
            present = torch.stack([key_layer, value_layer, query_layer])
        else:
            present = torch.stack([key_layer, value_layer])

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if layer_past is None and attention_mask is not None:
            attention_scores += attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs, present) if self.output_attentions else (context_layer, present)
        return outputs
    
class BertAttentionPast(BertAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = BertSelfAttentionPast(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            layer_past=None,
            cache_query=False,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states,
            encoder_attention_mask, layer_past, cache_query
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class BertLayerPast(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertAttentionPast(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            layer_past=None,
            cache_query=False
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask,
                                                head_mask, layer_past=layer_past,
                                                cache_query=cache_query)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask,
                encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

class BertEncoderPast(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.output_past = getattr(config, 'output_past', True)
        self.layer = nn.ModuleList(
            [BertLayerPast(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past=None,
            cache_query=False
    ):
        if past is None:
            past = [None] * len(self.layer)

        all_hidden_states = ()
        all_attentions = ()
        presents = ()

        for i, (layer_module, layer_past) in enumerate(zip(self.layer, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states,
                encoder_attention_mask, layer_past, cache_query
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            present = layer_outputs[-1]
            if self.output_past:
                presents = presents + (present,)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_past:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs