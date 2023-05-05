import torch
from torch import nn
from models.Baselines.PSModule import PSModule

Switch_map = {
    "word_embedding": 0,
    "document_embedding": 0,
    "input_gate": 0,
    "forget_gate": 1,
    "output_gate": 0,
    "cell_state": 0
}


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, usr_dim, prd_dim, bias):
        super(LSTMCell, self).__init__()
        self.i_gate = nn.Linear(input_size+hidden_size, hidden_size, bias)
        self.f_gate = nn.Linear(input_size+hidden_size, hidden_size, bias)
        self.o_gate = nn.Linear(input_size+hidden_size, hidden_size, bias)
        self.cell   = nn.Linear(input_size+hidden_size, hidden_size, bias)

        if Switch_map["input_gate"]:
            self.PSModule_i = PSModule(usr_dim, prd_dim, hidden_size)
        else:
            self.register_buffer("PSModule_i", None)
        if Switch_map["output_gate"]:
            self.PSModule_o = PSModule(usr_dim, prd_dim, hidden_size)
        else:
            self.register_buffer("PSModule_o", None)
        if Switch_map["forget_gate"]:
            self.PSModule_f = PSModule(usr_dim, prd_dim, hidden_size)
        else:
            self.register_buffer("PSModule_f", None)
        if Switch_map["cell_state"]:
            self.PSModule_c = PSModule(usr_dim, prd_dim, hidden_size)
        else:
            self.register_buffer("PSModule_c", None)

    def forward(self, input, user_states, item_states, hx):
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        # hx (hidden_state, cell_state)
        h_, c_ = hx
        input = torch.cat((input, h_), dim=-1)

        i_state = self.i_gate(input)
        if self.PSModule_i and user_states is not None and item_states is not None:
            i_state = self.PSModule_i(i_state, user_states, item_states)
        i_state = torch.sigmoid(i_state)

        f_state = self.f_gate(input)
        if self.PSModule_f and user_states is not None and item_states is not None:
            f_state = self.PSModule_f(f_state, user_states, item_states)
        f_state = torch.sigmoid(f_state)

        o_state = self.o_gate(input)
        if self.PSModule_o and user_states is not None and item_states is not None:
            o_state = self.PSModule_o(o_state, user_states, item_states)
        o_state = torch.sigmoid(o_state)

        cell_state = self.cell(input)
        if self.PSModule_c and user_states is not None and item_states is not None:
            cell_state = self.PSModule_c(cell_state, user_states, item_states)
        cell_state = torch.tanh(cell_state)

        c = torch.mul(f_state, c_) + torch.mul(i_state, cell_state)
        output = torch.mul(torch.tanh(c), o_state)
        return (output, c)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, usr_dim, prd_dim, bidirectional=False, bias=True):
        super(LSTM, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        if bidirectional:
            self.cell_dim = self.hidden_size // 2
            self.lstm_cell_forward = LSTMCell(input_size, self.cell_dim, usr_dim, prd_dim, bias)
            self.lstm_cell_backward = LSTMCell(input_size, self.cell_dim, usr_dim, prd_dim, bias)
        else:
            self.cell_dim = self.hidden_size
            self.lstm_cell_forward = LSTMCell(input_size, hidden_size, usr_dim, prd_dim, bias)
            self.register_buffer("lstm_cell_backward", None)

    def forward(self, input, user_states, item_states, hx=None, mask=None):
        batch_size, seq_len, input_dim = input.shape
        lengths = mask.sum(-1).long() # (batch_size,)
        if self.bidirectional:
            rev_idx = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
            for i in range(lengths.shape[0]):
                rev_idx[i, :lengths[i]] = torch.arange(lengths[i] - 1, -1, -1)
            rev_idx = rev_idx.unsqueeze(2).expand_as(input)
            rev_idx = rev_idx.to(input.device)
            rev_input = input.gather(1, rev_idx)
        else:
            rev_input = None

        if hx is None:
            zeros = torch.zeros(input.size(0), self.cell_dim, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)

        forward_output, forward_hn = self._forward_rnn_no_mask(self.lstm_cell_forward, input, user_states, item_states, hx)
        if self.bidirectional:
            backward_output, backward_hn = self._forward_rnn_no_mask(self.lstm_cell_backward, rev_input, user_states, item_states, hx)
            output = torch.cat((forward_output, backward_output), dim=-1) # (bs, seq, dim)
        else:
            output = forward_output

        # exact final state index

        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).repeat(1, 1, self.hidden_size)
        idx_ = 1 - (idx > 0).long()
        idx = idx + idx_
        hn = output.gather(1, idx).squeeze(1)

        return (output, hn)


    def _forward_rnn_no_mask(self, cell, input_, user_states, item_states, hx):
        max_time = input_.size(1)
        output = []
        for time in range(max_time):
            h_next, c_next = cell(input_[:, time], user_states, item_states, hx=hx)
            hx = (h_next, c_next)
            output.append(h_next)
        output = torch.stack(output, 1)     # do this if want 3D tensor
        return output, hx


class Classifier(nn.Module):
    def __init__(self, input_dim, config):
        super(Classifier, self).__init__()
        self.pre_classifier = nn.Linear(input_dim, config.pre_classifier_dim)
        self.classifier = nn.Linear(config.pre_classifier_dim, config.num_classes)

    def forward(self, hidden):
        pre_hidden = torch.tanh(self.pre_classifier(hidden))
        logits = self.classifier(pre_hidden)
        return logits


class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_embedding = config.text_embedding
        self.usr_vocab = config.usr_vocab
        self.prd_vocab = config.prd_vocab
        self.bidirectional = config.bidirectional
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.text_embed = nn.Embedding(config.text_embedding.size(0), config.text_embedding.size(1),
                                       padding_idx=config.pad_idx)
        self.text_embed.weight.data.copy_(self.text_embedding)
        self.text_embed.weight.requires_grad = False

        self.usr_embed = nn.Embedding(len(config.usr_vocab), config.usr_dim)
        self.usr_embed.weight.data.copy_(torch.Tensor(len(config.usr_vocab), config.usr_dim).zero_())
        self.usr_embed.weight.requires_grad = True

        self.prd_embed = nn.Embedding(len(config.prd_vocab), config.prd_dim)
        self.prd_embed.weight.data.copy_(torch.Tensor(len(config.prd_vocab), config.prd_dim).zero_())
        self.prd_embed.weight.requires_grad = True

        self.lstm = LSTM(self.text_embedding.size(1), config.word_hidden_dim, config.usr_dim, config.prd_dim, bidirectional=True)
        self.classifier = Classifier(config.word_hidden_dim, config)

        if Switch_map["word_embedding"]:
            self.PSModule_word = PSModule(config.usr_dim, config.prd_dim, self.text_embedding.size(1))
        else:
            self.register_buffer("PSModule_word", None)

        if Switch_map["document_embedding"]:
            self.PSModule_doc = PSModule(config.usr_dim, config.prd_dim, config.word_hidden_dim)
        else:
            self.register_buffer("PSModule_doc", None)


    def forward(self, text, usr, prd, mask=None, agnostic=False):
        text = self.text_embed(text)  # text: (sent, batch, word, dim)
        usr = usr.squeeze(-1)  # usr: (batch, )
        prd = prd.squeeze(-1)  # prd: (batch, )
        user_states = self.usr_embed(usr)
        item_states = self.prd_embed(prd)
        if agnostic:
            user_states, item_states = None, None

        if self.PSModule_word and user_states is not None and item_states is not None:
            text = self.PSModule_word(text, user_states, item_states)
        output, hn = self.lstm(text, user_states, item_states, mask=mask)

        if self.PSModule_doc and user_states is not None and item_states is not None:
            hn = self.PSModule_doc(hn, user_states, item_states)
        logits = self.classifier(hn)

        return logits