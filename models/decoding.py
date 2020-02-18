#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 22:21 
Date: February 18, 2020	
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from dataset.iwslt_data import subsequent_mask


def greedy_decode(model, src_tokens, src_mask, start_symbol, max=100):
    memory = model.encoder(src_tokens, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src_tokens.data)
    for i in range(max):
        out = model.decoder(Variable(ys), memory, src_mask,
                            Variable(subsequent_mask(ys.size(1)).type_as(src_tokens.data)))
        prob, logit = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src_tokens.data).fill_(next_word)], dim=1)
    return ys


def batch_decode(model, src_tokens, src_mask, src_len, pad_index, sos_index, eos_index, max_len=60):
    # input batch
    bs = len(src_len)

    src_enc = model.encoder(src_tokens, src_mask)
    assert src_enc.size(0) == bs

    # generated sentences
    generated = src_len.new(max_len, bs)  # upcoming output
    generated.fill_(pad_index)  # fill upcoming ouput with <PAD>
    generated[0].fill_(sos_index)  # fill 0th index with <SOS>

    cur_len = 1
    gen_len = src_len.clone().fill_(1)
    unfinished_sents = src_len.clone().fill_(1)
    # print(unfinished_sents)

    while cur_len < max_len:
        # print(generated[:cur_len])
        # compute word scores
        tensor = model.decoder(
            tokens=Variable(generated[:cur_len].transpose(0, 1)),
            memory=src_enc,
            src_mask=src_mask,
            trg_mask=Variable(subsequent_mask(cur_len).type_as(src_tokens.data)),
        )
        # print(tensor.shape)
        tensor = tensor.data[:, -1].type_as(src_enc)  # (bs, dim)
        # print(tensor.shape)
        prob, logit = model.generator(tensor)
        # print(prob.shape, logit.shape)
        # x, next_word = torch.max(prob, dim=1)
        next_words = torch.topk(prob, 1)[1].squeeze(1)
        # print(next_words)
        # print(next_words, generated[:cur_len])

        # update generations / lengths / finished sentences / current length
        # print(next_words * unfinished_sents)
        generated[cur_len] = next_words * unfinished_sents + pad_index * (1 - unfinished_sents)
        gen_len.add_(unfinished_sents)
        unfinished_sents.mul_(next_words.ne(eos_index).long())
        cur_len = cur_len + 1

        # break
        # assert tensor.size() == (1, bs, self.dim), (cur_len, max_len,
        #                                             src_enc.size(), tensor.size(), (1, bs, self.dim))
        # tensor = tensor.data[-1, :, :].type_as(src_enc)  # (bs, dim)
        # scores = self.pred_layer.get_scores(tensor)      # (bs, n_words)
    return generated.transpose(0, 1)


def generate_beam(model, src, src_mask, src_len,
                  pad_index,
                  sos_index,
                  eos_index,
                  emb_dim,
                  beam_size=5,
                  length_penalty=False,
                  early_stopping=False,
                  max_len=200):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """

        # check inputs
        src_enc = model.encoder(src, src_mask)
        assert src_enc.size(0) == src_len.size(0)
        assert beam_size >= 1

        # batch size / number of words
        bs = len(src_len)
        n_words = 100

        # expand to beam size the source latent representations / source lengths
        src_enc = src_enc.unsqueeze(1).expand((bs, beam_size) + src_enc.shape[1:]).contiguous().view((bs * beam_size,) + src_enc.shape[1:])
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

        # generated sentences (batch with beam current hypotheses)
        generated = src_len.new(max_len, bs * beam_size)  # upcoming output
        generated.fill_(pad_index)                   # fill upcoming ouput with <PAD>
        generated[0].fill_(sos_index)                # we use <EOS> for <BOS> everywhere

        # generated hypotheses
        generated_hyps = [BeamHypotheses(beam_size, max_len, length_penalty, early_stopping) for _ in range(bs)]

        # scores for each sentence in the beam
        beam_scores = src_enc.new(bs, beam_size).fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # current position
        cur_len = 1

        # cache compute states
        cache = {'slen': 0}

        # done sentences
        done = [False for _ in range(bs)]

        def make_std_mask(tgt, pad):
            "Create a mask to hide padding and future words."
            tgt_mask = (tgt != pad).unsqueeze(-2)
            tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
            return tgt_mask

        while cur_len < max_len:
            print(generated[:cur_len].transpose(0, 1).shape)
            # compute word scores
            tensor = model.decoder(
                tokens=Variable(generated[:cur_len].transpose(0, 1)),
                memory=src_enc,
                src_mask=src_mask,
                trg_mask=Variable(make_std_mask(generated[:cur_len].transpose(0, 1), pad_index).type_as(src.data)),
            )
            tensor = tensor.transpose(0, 1)
            assert tensor.size() == (1, bs * beam_size, emb_dim)
            tensor = tensor.data[-1, :, :]               # (bs * beam_size, dim)
            scores, logit = model.generator(tensor)  # (bs * beam_size, n_words)
            scores = F.log_softmax(logit, dim=-1)  # (bs * beam_size, n_words)
            print(scores.shape)
            assert scores.size() == (bs * beam_size, n_words)

            # select next words with scores
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
            _scores = _scores.view(bs, beam_size * n_words)            # (bs, beam_size * n_words)

            next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            # next batch beam content
            # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for sent_id in range(bs):

                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
                if done[sent_id]:
                    next_batch_beam.extend([(0, pad_index, 0)] * beam_size)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = idx // n_words
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if word_id == eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(generated[:cur_len, sent_id * beam_size + beam_id].clone(), value.item())
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, pad_index, 0)] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])

            # re-order batch and internal states
            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
            for k in cache.keys():
                if k != 'slen':
                    cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(bs):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         print("%.3f " % ss + " ".join(self.dico[x] for x in ww.tolist()))
        #     print("")

        # select the best hypotheses
        tgt_len = src_len.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = src_len.new(tgt_len.max().item(), bs).fill_(pad_index)
        for i, hypo in enumerate(best):
            decoded[:tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = eos_index

        # sanity check
        assert (decoded == eos_index).sum() == 2 * bs

        return decoded, tgt_len


class BeamHypotheses(object):

    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1  # ignoring <BOS>
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_len ** self.length_penalty
