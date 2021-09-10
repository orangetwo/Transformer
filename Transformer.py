# Transformer.py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Transformer
"""
from torch.nn import Transformer
https://github.com/pytorch/tutorials/blob/master/beginner_source/translation_transformer.py
https://pytorch.org/tutorials/beginner/translation_transformer.html
PyTorch官方出了一个用Transformer做 machine translation的教程
"""


class MultiHeadAttention(nn.Module):
	''' Multi-Head Attention module '''

	def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, bias=True):
		super().__init__()

		self.n_head = n_head
		self.d_k = d_k  # q, k reshape后的head dimension, 这两个维度需要设定相等
		self.d_v = d_v  # reshape 后v的head dimension, 并不需要跟qk的一样，但是通常会保持一样

		# 这里的d_model 是输入qkv的hid dim, 在Transformer的设定中,qkv的hidden state 维度一样
		# 一般来说（transformer相等）, d_model = n_head * d_k
		self.w_qs = nn.Linear(d_model, n_head * d_k, bias=bias)
		self.w_ks = nn.Linear(d_model, n_head * d_k, bias=bias)
		self.w_vs = nn.Linear(d_model, n_head * d_v, bias=bias)

		# 全连接层把 n_head * d_v 拉回 d_model
		self.fc = nn.Linear(n_head * d_v, d_model, bias=bias)

		self.dropout_fc = nn.Dropout(dropout)
		self.dropout_attn = nn.Dropout(dropout)

		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

	def forward(self, q, k, v, mask=None):
		# q = [batch size, query sequence length, q hidden dimension]
		# k = [batch size, key sequence length,   k hidden dimension]
		# v = [batch size, value sequence length, v hidden dimension]
		# key sequence length = value sequence length 原因(qk) * v

		# self attention: q=k=v
		# cross Encoder-Decoder self attention: k=v

		d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

		# batch size , query sequence length, key sequence length, value sequence length
		sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

		residual = q

		# q = [batch size, query sequence length, num heads, q head dimension]
		# k = [batch size, key sequence length,   num heads, k head dimension]
		# v = [batch size, value sequence length, num heads, v head dimension]
		# 注意reshape(view)后, q head dimension = k head dimension, 故这里为dk表示
		q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
		k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
		v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

		# Q = [batch size, num heads, query sequence length, q head dimension(d_k)]
		# K = [batch size, num heads, key sequence length,   k head dimension(d_k)]
		# V = [batch size, num heads, value sequence length, v head dimension(d_v)]
		# Transpose for attention dot product: b x n x lq x dv
		q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

		if mask is not None:
			# the input: src_mask = [batch size, 1,                   key sequence length]
			#            trg_mask = [batch size, key sequence length, key sequence length]
			# after mask.unsqueeze(1)
			# then :     src_mask = [batch size, 1, 1,                     key sequence length]
			#            trg_mask = [batch size, 1,  key sequence length,  key sequence length]
			mask = mask.unsqueeze(1)  # For head axis broadcasting.

		# energy : [batch size, num heads, query sequence length, key sequence length]
		energy = torch.matmul(q, k.transpose(2, 3)) / (self.d_k ** 0.5)

		if mask is not None:
			energy = energy.masked_fill(mask == 0, -1e10)

		attn = self.dropout_attn(F.softmax(energy, dim=-1))

		# 这里两者相乘 说明 key sequence length = value sequence length
		# v =    [batch size, num heads, value sequence length, v head dimension(d_v)]
		# attn = [batch size, num heads, query sequence length, key sequence length]
		# q =    [batch size, num heads, query sequence length, v head dimension(d_v)]
		q = torch.matmul(attn, v)

		# q =    [batch size, query sequence length, num heads * v head dimension]
		q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

		# [batch size, query sequence length, num heads * v head dimension] ->
		#                                            [batch size, query sequence length, v hidden dimension]
		# d_model = num heads * v head dimension
		q = self.dropout_fc(self.fc(q))
		q += residual

		q = self.layer_norm(q)

		return q, attn


class PositionwiseFeedForward(nn.Module):
	''' A two-feed-forward-layer module '''

	def __init__(self, d_in, d_hid, dropout=0.1):
		super().__init__()
		self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
		self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
		self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		# x = [batch size, sequence length, d_in]
		residual = x

		# [batch size, sequence length, d_in] -> [batch size, sequence length, d_hid] ->
		#                                         [batch size, sequence length, d_in]
		x = self.w_2(F.relu(self.w_1(x)))

		x = self.dropout(x)
		x += residual

		x = self.layer_norm(x)

		return x


class EncoderLayer(nn.Module):
	''' Compose with two layers '''

	def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
		super(EncoderLayer, self).__init__()
		self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
		self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

	def forward(self, enc_input, slf_attn_mask=None):
		# enc_input     = [batch size, src sequence length, hidden dimension]
		# slf_attn_mask = [batch size, 1, src sequence length]

		enc_output, enc_slf_attn = self.slf_attn(
			enc_input, enc_input, enc_input, mask=slf_attn_mask)
		enc_output = self.pos_ffn(enc_output)
		return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
	''' Compose with three layers '''

	def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
		super(DecoderLayer, self).__init__()
		self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
		self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
		self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

	def forward(
			self, dec_input, enc_output,
			slf_attn_mask=None, dec_enc_attn_mask=None):
		"""
		dec_input  =  [batch size, trg sequence length, trg hidden dimension]
		enc_input  =  [batch size, src sequence length, src hidden dimension]
		这里transformer设定 trg hidden dimension = src hidden dimension
		slf_attn_mask =     [batch size, trg sequence length, trg sequence length]
		dec_enc_attn_mask = [batch size, 1,                   src sequence length]
		"""
		dec_output, dec_slf_attn = self.slf_attn(
			dec_input, dec_input, dec_input, mask=slf_attn_mask)
		dec_output, dec_enc_attn = self.enc_attn(
			dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
		dec_output = self.pos_ffn(dec_output)
		return dec_output, dec_slf_attn, dec_enc_attn


def get_pad_mask(seq, pad_idx):
	# src_mask = [batch size, 1, src len]
	return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
	''' For masking out the subsequent info. '''
	# seq : [batch size, trg len]
	sz_b, len_s = seq.size()

	# subsequent_mask: [1, trg len, trg len]
	subsequent_mask = (1 - torch.triu(
		torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
	return subsequent_mask


class PositionalEncoding(nn.Module):

	def __init__(self, d_hid, n_position=1600):
		super(PositionalEncoding, self).__init__()

		# Not a parameter
		self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

	def _get_sinusoid_encoding_table(self, n_position, d_hid):
		''' Sinusoid position encoding table '''

		# TODO: make it with torch instead of numpy

		def get_position_angle_vec(position):
			return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

		sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
		sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
		sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

		return torch.FloatTensor(sinusoid_table).unsqueeze(0)

	def forward(self, x):
		return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
	''' A encoder model with self attention mechanism. '''

	def __init__(
			self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
			d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):

		super().__init__()

		if pad_idx is None:
			self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec)
		else:
			self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)

		self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
		self.dropout = nn.Dropout(p=dropout)
		self.layer_stack = nn.ModuleList([
			EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
		self.scale_emb = scale_emb
		self.d_model = d_model

	def forward(self, src_seq, src_mask, return_attns=False):

		enc_slf_attn_list = []

		# 对 token embedding做一个放大
		if self.scale_emb:
			print('scale embedding! -> source ')
			enc_output = (self.d_model ** 0.5) * self.src_word_emb(src_seq)
		else:
			enc_output = self.src_word_emb(src_seq)

		enc_output = self.dropout(self.position_enc(enc_output))
		enc_output = self.layer_norm(enc_output)

		for enc_layer in self.layer_stack:
			enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
			enc_slf_attn_list += [enc_slf_attn] if return_attns else []

		if return_attns:
			return enc_output, enc_slf_attn_list
		return enc_output,


class Decoder(nn.Module):
	''' A decoder model with self attention mechanism. '''

	def __init__(
			self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
			d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

		super().__init__()

		if pad_idx is not None:
			self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
		else:
			self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec)

		self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
		self.dropout = nn.Dropout(p=dropout)
		self.layer_stack = nn.ModuleList([
			DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
		self.scale_emb = scale_emb
		self.d_model = d_model

	def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

		dec_slf_attn_list, dec_enc_attn_list = [], []

		# 对 token embedding做一个放大
		if self.scale_emb:
			print('scale embedding! -> target ')
			dec_output = (self.d_model ** 0.5) * self.trg_word_emb(trg_seq)
		else:
			dec_output = self.trg_word_emb(trg_seq)

		dec_output = self.dropout(self.position_enc(dec_output))
		dec_output = self.layer_norm(dec_output)

		for dec_layer in self.layer_stack:
			dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
				dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
			dec_slf_attn_list += [dec_slf_attn] if return_attns else []
			dec_enc_attn_list += [dec_enc_attn] if return_attns else []

		if return_attns:
			return dec_output, dec_slf_attn_list, dec_enc_attn_list
		return dec_output,


class Transformer(nn.Module):
	''' A sequence to sequence model with attention mechanism. '''

	def __init__(
			self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
			d_word_vec=512, d_model=512, d_inner=2048,
			n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=512,
			trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=False,
			scale_emb_or_prj='emb'):

		super().__init__()

		self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

		# In section 3.4 of paper "Attention Is All You Need", there is such detail:
		# "In our model, we share the same weight matrix between the two
		# embedding layers and the pre-softmax linear transformation...
		# In the embedding layers, we multiply those weights by \sqrt{d_model}".
		#
		# Options here:
		#   'emb': multiply \sqrt{d_model} to embedding output
		#   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
		#   'none': no multiplication

		#  权重绑定
		assert scale_emb_or_prj in ['emb', 'prj', 'none']
		scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
		self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
		self.d_model = d_model

		self.encoder = Encoder(
			n_src_vocab=n_src_vocab, n_position=n_position,
			d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
			n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
			pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

		self.decoder = Decoder(
			n_trg_vocab=n_trg_vocab, n_position=n_position,
			d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
			n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
			pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

		self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=True)

		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

		assert d_model == d_word_vec, \
			'To facilitate the residual connections, \
		 the dimensions of all module outputs shall be the same.'

		# weight tying
		if trg_emb_prj_weight_sharing:
			# Share the weight between target word embedding & last dense layer
			print("self.trg_word_prj.weight = self.decoder.trg_word_emb.weight !!!!~~~")
			self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

		# src trg 词典一样时才会用到
		if emb_src_trg_weight_sharing:
			print("self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight")
			self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

	def forward(self, src_seq, trg_seq):
		# 在某些特殊任务下，某些字符是不存在。

		# 比如 trg vocab可能因为没有 '<pad>' 字符这里建议 将 trg_pad_idx设置为 -1
		# src_mask = [batch size, 1,                   src sequence length]
		# trg_mask = [batch size, trg sequence length, trg sequence length]
		src_mask = get_pad_mask(src_seq, self.src_pad_idx)
		trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

		# enc_output = [batch size, src len, hid dim]
		enc_output, *_ = self.encoder(src_seq, src_mask)
		# dec_output = [batch size, trg len, hid dim]
		dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)

		if self.scale_prj:
			print('scale project ! -> source ')
			seq_logit = (self.d_model ** -0.5) * self.trg_word_prj(dec_output)
		else:
			seq_logit = self.trg_word_prj(dec_output)

		# return [batch size * trg length, trg vocab]
		return seq_logit.view(-1, seq_logit.size(2)), seq_logit


if __name__ == '__main__':
	model = Transformer(10, 10, 1, 1, d_word_vec=32,d_model=32,d_inner=32*4,n_layers=2,n_head=4,d_k=8,d_v=32,)
	input4encoder = torch.tensor([[2, 2, 3, 1],
	                  [4, 1, 2, 3]], dtype=torch.long)

	input4decoder = torch.tensor([[2,2],[2,2]])

	x, y = model(input4encoder, input4decoder)
	print(y.shape)
	print(y)

	print(y[:,-1])



