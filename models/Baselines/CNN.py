import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Baselines.PSModule import PSModule

Switch_map = {
    "word_embedding": 1,
    "document_embedding": 0,
    "cnn": 0,
}


class Net(nn.Module):

    def __init__(self, config):
        super().__init__()
        output_channel = config.output_channel
        words_dim = config.words_dim
        # ks = 3 # There are three conv nets here

        input_channel = 1
        self.pooling = config.pooling
        self.text_embedding = config.text_embedding
        self.text_embedding = nn.Embedding.from_pretrained(self.text_embedding, freeze=False)

        # user and product
        self.usr_vocab = config.usr_vocab
        self.prd_vocab = config.prd_vocab
        self.usr_embed = nn.Embedding(len(config.usr_vocab), config.usr_dim)
        self.usr_embed.weight.data.copy_(torch.Tensor(len(config.usr_vocab), config.usr_dim).zero_())
        self.usr_embed.weight.requires_grad = True

        self.prd_embed = nn.Embedding(len(config.prd_vocab), config.prd_dim)
        self.prd_embed.weight.data.copy_(torch.Tensor(len(config.prd_vocab), config.prd_dim).zero_())
        self.prd_embed.weight.requires_grad = True

        self.conv1 = nn.Conv1d(words_dim, output_channel, (1, ))
        self.conv2 = nn.Conv1d(words_dim, output_channel, (2, ), padding=(1,))
        self.conv3 = nn.Conv1d(words_dim, output_channel, (3, ), padding=(2,))

        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(output_channel, config.num_classes)

        if Switch_map["word_embedding"]:
            self.PSModule_word = PSModule(config.usr_dim, config.prd_dim, config.words_dim)
        else:
            self.register_buffer("PSModule_word", None)

        if Switch_map["cnn"]:
            self.PSModule_cnn1 = PSModule(config.usr_dim, config.prd_dim, config.output_channel)
            self.PSModule_cnn2 = PSModule(config.usr_dim, config.prd_dim, config.output_channel)
            self.PSModule_cnn3 = PSModule(config.usr_dim, config.prd_dim, config.output_channel)
        else:
            self.register_buffer("PSModule_cnn1", None)
            self.register_buffer("PSModule_cnn2", None)
            self.register_buffer("PSModule_cnn3", None)

        if Switch_map["document_embedding"]:
            self.PSModule_doc = PSModule(config.usr_dim, config.prd_dim, config.output_channel)
        else:
            self.register_buffer("PSModule_doc", None)

    def cnn_injection(self, x, user_states, item_states):
        # (batch, channel_output, ~=sent_len) * ks
        output = []
        for x_, ps in zip(x, [self.PSModule_cnn1, self.PSModule_cnn2, self.PSModule_cnn3]):
            output.append(ps(x_.transpose(-1, -2), user_states, item_states).transpose(-1, -2))
        return output


    def forward(self, x, usr, prd, mask=None, agnostic=False):
        usr = usr.squeeze(-1)  # usr: (batch, )
        prd = prd.squeeze(-1)  # prd: (batch, )
        user_states = self.usr_embed(usr)
        item_states = self.prd_embed(prd)
        if agnostic:
            user_states, item_states = None, None
        # user_states, item_states = None, None

        non_static_input = self.text_embedding(x)
        if self.PSModule_word and user_states is not None and item_states is not None:
            non_static_input = self.PSModule_word(non_static_input, user_states, item_states)

        x = non_static_input.transpose(-2, -1)  # (batch, embed_dim, sent_len)

        x = [F.relu(self.conv1(x)), F.relu(self.conv2(x)), F.relu(self.conv3(x))]
        # (batch, channel_output, ~=sent_len) * ks

        if self.PSModule_cnn1 and user_states is not None and item_states is not None:
            x = self.cnn_injection(x, user_states, item_states)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        # (batch, channel_output) * ks
        x = torch.stack(x, 0).sum(0) # (batch, channel_output)

        if self.PSModule_doc and user_states is not None and item_states is not None:
            x = self.PSModule_doc(x, user_states, item_states)
        x = self.dropout(x)
        logit = self.fc1(x) # (batch, target_size)
        return logit