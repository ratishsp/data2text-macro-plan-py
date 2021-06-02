from onmt.encoders.encoder import EncoderBase
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.modules.global_attention_context import GlobalAttentionContext
from onmt.modules.global_self_attention import GlobalSelfAttention
from onmt.utils.misc import sequence_mask
import torch
from torch.nn.utils.rnn import pad_sequence


class MacroPlanEncoder(EncoderBase):
    """ An encoder for encoding macro plans.
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None, content_selection_attn_hidden=None):
        super(MacroPlanEncoder, self).__init__()
        assert embeddings is not None
        self.rnn_encoder = RNNEncoder(rnn_type, bidirectional, num_layers, hidden_size, dropout, embeddings)
        self.attn = GlobalAttentionContext(hidden_size, attn_type="general")
        self.content_selection_attn = GlobalSelfAttention(hidden_size, attn_type="general",
                                                          attn_hidden=content_selection_attn_hidden)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    @classmethod
    def from_opt(cls, opt, embeddings=None):

        return cls(
            opt.rnn_type,
            True,  # for bidirectional
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings,
            opt.content_selection_attn_hidden)

    def forward(self, src, lengths=None, segment_count=None, padding_value=None):
        self._check_args(src, lengths)
        encoder_final, memory_bank, lengths = self.rnn_encoder(src, lengths, enforce_sorted=False)
        context = self.attn(memory_bank.transpose(0,1), memory_lengths=lengths)
        segment_count_, dim_ = context.size()
        assert dim_ == self.hidden_size
        assert segment_count.sum() == segment_count_
        batch_ = segment_count.size()
        segment_representation = context.split(segment_count.tolist())
        segment_representation_padded = pad_sequence(segment_representation, padding_value=padding_value)
        memory_bank, _ = self.content_selection_attn(segment_representation_padded.transpose(0, 1).contiguous(),
                                        segment_representation_padded.transpose(0, 1),
                                        memory_lengths=segment_count)

        _, batch, emb_dim = memory_bank.size()
        assert batch == batch_[0]
        assert emb_dim == self.hidden_size
        if segment_count is not None:
            # we avoid padding while mean pooling
            mask = sequence_mask(segment_count).float()
            mask = mask / segment_count.unsqueeze(1).float()
            mean = torch.bmm(mask.unsqueeze(1), memory_bank.transpose(0, 1)).squeeze(1)
        else:
            mean = memory_bank.mean(0)

        mean = mean.expand(self.num_layers, batch, emb_dim)
        encoder_final = (mean, mean)
        return encoder_final, memory_bank, segment_count
