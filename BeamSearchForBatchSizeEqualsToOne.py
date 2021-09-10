''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer import Transformer, get_pad_mask, get_subsequent_mask


class Translator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):

        super(Translator, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        """
        blank_seqs : (beam size, max sequence length), fill with trget padding index
        blank_seqs : 存放的是序列的 token index 
        """
        self.register_buffer(
            'blank_seqs',
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))

        # blank_seqs 第一列 为 target beginning of symbol token index
        self.blank_seqs[:, 0] = self.trg_bos_idx

        """
        len_map : (1, max sequence length)
        """
        self.register_buffer(
            'len_map',
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))

    def _model_decode(self, trg_seq, enc_output, src_mask):
        trg_mask = get_subsequent_mask(trg_seq)
        dec_output, *_ = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask)
        return F.softmax(self.model.trg_word_prj(dec_output), dim=-1)

    def _get_init_state(self, src_seq, src_mask):
        beam_size = self.beam_size

        """
        enc_output : (1, sequence length, hidden dimension)
        self.init_seq : (1, 1)
        dec_ouput  : (1, 1, vocab size)
        """
        enc_output, *_ = self.model.encoder(src_seq, src_mask)
        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)

        """
        bos -> decoder -> output(vocab) -> 取 top beam size
        dec_output[:, -1, :] : (1, vocab size)
        best_k_probs : (1, beam size)
        best_k_idx : (1, beam size)
        """
        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)

        # scores : (beam size), 注意此时 score = softmax + log
        scores = torch.log(best_k_probs).view(beam_size)
        # blank_seqs : (beam size, max sequence length), blank_seqs[:,0] = bos token index
        gen_seq = self.blank_seqs.clone().detach()
        # blank_seqs : (beam size, max sequence length), blank_seqs[:,1] = best_k_idx
        gen_seq[:, 1] = best_k_idx[0]
        # enc_ouput : (beam size, max sequence length, hidden dimension)
        enc_output = enc_output.repeat(beam_size, 1, 1)
        return enc_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1

        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        # dec_output : (beam size, vocab size)
        # best_k2_probs, best_k2_idx : (beam size, beam size)
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

        # Include the previous scores.
        # 等式 右边 第一项: (beam size, beam size), 第二项 (beam size, 1)
        # scores : (beam size , beam size), scores 中每一个tensor代表一个 序列当前的得分
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        # Get the best k candidates from k^2 candidates.
        #  best_k_idx_in_k2 指出 在  best_k2_idx 这个tensor中 哪些位置的token 的prob 比较高
        #  换句话说 他指向了 best_k2_idx 中prob高的token地址
        # scores, best_k_idx_in_k2 : (beam size)
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

        # Get the corresponding positions of the best k candidiates.
        # best_k_idx_in_k2 % beam_size 指出 是   best_k2_idx 哪一列，
        # best_k_r_idxs  // beam_size 指出 是    best_k2_idx 哪一行
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        # 复制之前的序列
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        # 把当前得到 概率最大的 top beam size 个token 的 id 放进序列
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores

    def translate_sentence(self, src_seq):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        assert src_seq.size(0) == 1

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha

        with torch.no_grad():
            src_mask = get_pad_mask(src_seq, src_pad_idx)
            enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask)

            ans_idx = 0  # default
            for step in range(2, max_seq_len):  # decode up to max length
                # 得到当前 长度下 decoder的输出
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask)
                # gen_seq当前生成序列
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()