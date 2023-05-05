import torch
import torch.nn as nn
from models.Baselines.PSModule import PSModule

Switch_map = {
    "word_embedding": 0,
    "sentence_embedding": 1,
    "document_embedding": 0,
    "att_w": 0,
    "att_s": 0,
}


class Classifier(nn.Module):
    def __init__(self, input_dim, config):
        super(Classifier, self).__init__()
        self.pre_classifier = nn.Linear(input_dim, config.pre_classifier_dim)
        self.classifier = nn.Linear(config.pre_classifier_dim, config.num_classes)

    def forward(self, hidden):
        pre_hidden = torch.tanh(self.pre_classifier(hidden))
        logits = self.classifier(pre_hidden)
        return logits


class WordAttention(nn.Module):
    def __init__(self, config):
        super(WordAttention, self).__init__()
        self.pre_pooling_linear = nn.Linear(config.word_hidden_dim, config.pre_pooling_dim)
        self.pooling_linear = nn.Linear(config.pre_pooling_dim, 1)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        if Switch_map["att_w"]:
            self.PSModule = PSModule(config.usr_dim, config.prd_dim, config.pre_pooling_dim)
        else:
            self.register_buffer("PSModule", None)

    def forward(self, x, user_states=None, item_states=None, mask=None):
        states = self.pre_pooling_linear(x)
        if self.PSModule and user_states is not None and item_states is not None:
            states = self.PSModule(states, user_states, item_states)
        weights = self.pooling_linear(torch.tanh(states)).squeeze(dim=2)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        weights = nn.Softmax(dim=-1)(weights)
        weights = self.dropout(weights)
        return torch.mul(x, weights.unsqueeze(2)).sum(dim=1)
        # return weights


class SentenceAttention(nn.Module):
    def __init__(self, config):
        super(SentenceAttention, self).__init__()
        self.pre_pooling_linear = nn.Linear(config.sent_hidden_dim, config.pre_pooling_dim)
        self.pooling_linear = nn.Linear(config.pre_pooling_dim, 1)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        if Switch_map["att_s"]:
            self.PSModule = PSModule(config.usr_dim, config.prd_dim, config.pre_pooling_dim)
        else:
            self.register_buffer("PSModule", None)

    def forward(self, x, user_states=None, item_states=None, mask=None):
        states = self.pre_pooling_linear(x)
        if self.PSModule and user_states is not None and item_states is not None:
            states = self.PSModule(states, user_states, item_states)
        weights = self.pooling_linear(torch.tanh(states)).squeeze(dim=2)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        weights = nn.Softmax(dim=-1)(weights)
        weights = self.dropout(weights)
        return torch.mul(x, weights.unsqueeze(2)).sum(dim=1)


class SentLevelRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        sentence_hidden_dim = config.sent_hidden_dim
        word_hidden_dim = config.word_hidden_dim
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooling = config.pooling

        if config.bidirectional:
            self.self_gru = nn.LSTM(word_hidden_dim, sentence_hidden_dim // 2, bidirectional=True, batch_first=True)
            self.usr_gru = nn.LSTM(word_hidden_dim, sentence_hidden_dim // 2, bidirectional=True, batch_first=True)
            self.prd_gru = nn.LSTM(word_hidden_dim, sentence_hidden_dim // 2, bidirectional=True, batch_first=True)
        else:
            self.self_gru = nn.LSTM(word_hidden_dim, sentence_hidden_dim, bidirectional=False, batch_first=True)
            self.usr_gru = nn.LSTM(word_hidden_dim, sentence_hidden_dim, bidirectional=False, batch_first=True)
            self.prd_gru = nn.LSTM(word_hidden_dim, sentence_hidden_dim, bidirectional=False, batch_first=True)

        self.sen_attention = SentenceAttention(config)
        if Switch_map["sentence_embedding"]:
            self.PSModule_sent = PSModule(config.usr_dim, config.prd_dim, config.sent_hidden_dim)
        else:
            self.register_buffer("PSModule_sent", None)

    def forward(self, text, user_states, item_states, mask=None):
        # x expected to be of dimensions--> (num_words, batch_size)
        if self.PSModule_sent and user_states is not None and item_states is not None:
            text = self.PSModule_sent(text, user_states, item_states)
        if not hasattr(self, '_flattened'):
            self.self_gru.flatten_parameters()
        h_text, _ = self.self_gru(self.dropout(text))
        text_x = self.sen_attention(h_text, user_states, item_states, mask)
        return text_x, _, _


class WordLevelRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        words_dim = config.words_dim
        word_hidden_dim = config.word_hidden_dim
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooling = config.pooling

        if config.bidirectional:
            self.self_gru = nn.LSTM(words_dim, word_hidden_dim // 2, bidirectional=True, batch_first=True)
            self.usr_gru = nn.LSTM(words_dim, word_hidden_dim // 2, bidirectional=True, batch_first=True)
            self.prd_gru = nn.LSTM(words_dim, word_hidden_dim // 2, bidirectional=True, batch_first=True)
        else:
            self.self_gru = nn.LSTM(words_dim, word_hidden_dim, bidirectional=False, batch_first=True)
            self.usr_gru = nn.LSTM(words_dim, word_hidden_dim, bidirectional=False, batch_first=True)
            self.prd_gru = nn.LSTM(words_dim, word_hidden_dim, bidirectional=False, batch_first=True)

        self.word_attention = WordAttention(config)
        if Switch_map["word_embedding"]:
            self.PSModule_word = PSModule(config.usr_dim, config.prd_dim, config.words_dim)
        else:
            self.register_buffer("PSModule_word", None)

    def forward(self, text, user_states, item_states, mask=None):
        # x expected to be of dimensions--> (num_words, batch_size)
        if self.PSModule_word and user_states is not None and item_states is not None:
            text = self.PSModule_word(text, user_states, item_states)
        if not hasattr(self, '_flattened'):
            self.self_gru.flatten_parameters()
        h_text, _ = self.self_gru(self.dropout(text))
        text_x = self.word_attention(h_text, user_states, item_states, mask)
        return text_x, _, _


class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_embedding = config.text_embedding
        self.usr_vocab = config.usr_vocab
        self.prd_vocab = config.prd_vocab
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

        self.reset_attribute_embedding()

        self.word_attention_rnn = WordLevelRNN(config)
        self.sentence_attention_rnn = SentLevelRNN(config)
        self.classifier = Classifier(config.sent_hidden_dim, config)

        if Switch_map["document_embedding"]:
            self.PSModule_doc = PSModule(config.usr_dim, config.prd_dim, config.sent_hidden_dim)
        else:
            self.register_buffer("PSModule_doc", None)

    def reset_attribute_embedding(self):
        self.usr_embed.weight.data.zero_()
        self.prd_embed.weight.data.zero_()

    def forward(self, text, usr, prd, mask=None, agnostic=False):
        # text: (batch, sent, word)
        # usr: (batch, 1)
        # prd: (batch, 1)
        # mask: (batch, sent, word)
        # word embedding
        if len(text.size()) == 3:
            text = text.permute(1, 0, 2)  # text: (sent, batch, word)
            text = self.text_embed(text)  # text: (sent, batch, word, dim)
        elif len(text.size()) > 3:
            text = text.permute(1, 0, 2, 3)  # text: (sent, batch, word, dim)
        else:
            print("Error Input...")
            exit()

        usr = usr.squeeze(-1)  # usr: (batch, )
        prd = prd.squeeze(-1)  # prd: (batch, )
        user_states = self.usr_embed(usr)
        item_states = self.prd_embed(prd)
        if agnostic:
            user_states, item_states = None, None

        mask_word = None
        mask_sent = None
        if mask is not None:
            mask_word = mask.permute(1, 0, 2)  # text: (sent, batch, word)
            mask_sent = mask.long().sum(2) > 0  # (batch, sent)

        num_sentences = text.size(0)
        words_text = []
        for i in range(num_sentences):
            text_, usr_, prd_ = self.word_attention_rnn(text[i], user_states, item_states, mask=mask_word[i])
            words_text.append(text_)
        words_text = torch.stack(words_text, 1)  # (batch, sents, dim)
        sents_x, sents_usr, sents_prd = self.sentence_attention_rnn(words_text, user_states, item_states, mask=mask_sent)
        # sents = torch.cat((sents_x, sents_usr, sents_prd), dim=-1)
        sents = sents_x
        if self.PSModule_doc and user_states is not None and item_states is not None:
            sents = self.PSModule_doc(sents, user_states, item_states)
        logits = self.classifier(sents)
        doucment_representation = sents
        return logits, doucment_representation, (None, None)