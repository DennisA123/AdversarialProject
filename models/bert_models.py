import torch
from transformers import BertForMaskedLM, BertModel
from models.bert_layers import BertEncoderPast

class BertAutoRegressiveModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertEncoderPast(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past=None
    ):
        if past is None:
            past_length = 0
        else:
            past_length = past[0][0].size(-2)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        batch_size, seq_length = input_shape
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_length), device=device)  # (bs, seq_length)
        seq_ids = torch.arange(past_length + seq_length, device=device)
        # add a upper triangle mask for auto-regressive language model
        causal_mask = seq_ids[None, None, :].repeat(batch_size, past_length + seq_length, 1)  <= seq_ids[None, :, None]
        causal_mask = causal_mask.to(attention_mask.dtype)
        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape
        # [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1))
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        position_ids = torch.arange(past_length, past_length + seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids,
            token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past=past
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs

class BertForLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertAutoRegressiveModel(config)
        self.start_idx = 1
        self.init_weights()

    def prepare_inputs_for_generation(self, input_ids, past):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        return {"input_ids": input_ids, "past": past}

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            one_hot_labels=None,
            past=None
    ):
        label_start_idx = 1
        if inputs_embeds is not None:
            start_embeds = self.get_input_embeddings().weight[self.start_idx]
            inputs_embeds = torch.cat([start_embeds.view(1, 1, -1), inputs_embeds], 1)
            label_start_idx = 0

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past=past
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        # Add hidden states and attention if they are here
        outputs = (prediction_scores,) + outputs[2:]

        # we are doing next-token prediction;
        # shift prediction scores and input ids by one
        if one_hot_labels is not None:
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = one_hot_labels[:, label_start_idx:, :].contiguous()
            nll = -torch.log_softmax(prediction_scores, -1)
            ltr_lm_loss = torch.sum(nll * lm_labels, -1).mean()
            outputs = (ltr_lm_loss,) + outputs
        elif labels is not None:
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = labels[:, label_start_idx:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs
        return outputs