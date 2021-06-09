# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import torch
from fairseq import utils
from fairseq.data import LanguagePairDataset
from fairseq.dataclass import ChoiceEnum
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask, load_langpair_dataset
from fairseq.utils import new_arange


import logging
logger = logging.getLogger(__name__)


NOISE_CHOICES = ChoiceEnum(["random_delete", "random_mask", "no_noise", "full_mask"])

@dataclass
class TranslationLevenshteinConfig(TranslationConfig):
    noise: NOISE_CHOICES = field(
        default="random_delete",
        metadata={
            "help": "type of noise"
        },
    )

@register_task("translation_lev", dataclass=TranslationLevenshteinConfig)
class TranslationLevenshteinTask(TranslationTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    cfg: TranslationLevenshteinConfig

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            prepend_bos=True,
        )

    def inject_noise(self, target_tokens, mask_distribution=None):

        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0
            )
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True
            )

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = (
                2
                + (
                    (target_length - 2)
                    * target_score.new_zeros(target_score.size(0), 1).uniform_()
                ).long()
            )
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = (
                target_tokens.gather(1, target_rank)
                .masked_fill_(target_cutoff, pad)
                .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
            )
            prev_target_tokens = prev_target_tokens[
                :, : prev_target_tokens.ne(pad).sum(1).max()
            ]

            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = (
                target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
            )

            # convert target indices to floats to be sorted
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            _, target_rank = target_score.sort(1)

            # define mask length
            target_length = target_masks.sum(1).float()

            def min_max_method():
                """ min-max ratio method """
                nonlocal target_length
                ratios = (mask_distribution,torch.abs(mask_distribution-1))
                end_ratio = max(ratios, key=lambda p: p[0]) #target_length.clone().uniform_(0.8,1.0)
                start_ratio = min(ratios, key=lambda p: p[0])#target_length.clone().uniform_(0.,0.8)
                return map_single_segment(start_ratio,end_ratio)

            def predict_start_uniform_end():
                """ DyMask-v1 (predict start, uniform end positions) """
                nonlocal target_length
                end_ratio = target_length.clone().uniform_(0.8,1.0)
                start_ratio = mask_distribution
                return map_single_segment(start_ratio,end_ratio)

            def variable_start_fixed_end():
                """ DyMask-v2 (predict variable start, uniform end positions) """
                nonlocal target_length
                end_ratio = target_length.clone().uniform_(1.0,1.0)
                start_ratio = mask_distribution
                # logger.info(mask_distribution)
                return map_single_segment(start_ratio,end_ratio)

            def uniform_start_end():
                """ uniform start and end """
                nonlocal target_length
                end_ratio = target_length.clone().uniform_(0.8,1.0)
                start_ratio = target_length.clone().uniform_(0.,0.8)
                return map_single_segment(start_ratio,end_ratio)

            def uniform_original():
                """ uniform original """
                nonlocal target_length, target_rank
                end_ratio = target_length.clone().uniform_()
                # convert to length-wise
                target_length = target_length * end_ratio
                target_length = target_length + 1  # make sure to mask at least one token.
                target_cutoff = new_arange(target_rank) < target_length[:, None].long()
                return target_cutoff.scatter(1, target_rank, target_cutoff)

            def multi_segment():
                ### DyMask-v3: multi-segment, self-supervising masking mechanism
                nonlocal target_rank, target_length

                from fairseq import pdb; pdb.set_trace()

                seq_len = target_rank.size()[1]
                final_cutoff = target_rank.clone().type(torch.bool)
                for i in range(mask_distribution.size()[0]):
                    prob = mask_distribution[i]
                    booleans = np.where(p > np.random.rand(seq_len), 1, 0)
                    final_cutoff[i,:] = torch.from_numpy(booleans)
                return final_cutoff

            def map_single_segment(start_ratio,end_ratio):

                nonlocal target_length, target_rank

                # convert to length-wise
                start_point = target_length * start_ratio
                start_point = start_point + 1  # make sure to mask at least one token.
                target_length = target_length * end_ratio
                target_length = target_length + 1  # make sure to mask at least one token.

                # masking by checking if each index is smaller than target mask length,
                # then use scatter to reset the respective indices boolean values
                # 'target_rank' contains the sequence lengths
                # 'new_arange(target_rank)' contains the iteration of indices (zero-index)
                target_cutoff = new_arange(target_rank) < target_length[:, None].long()
                start_cutoff = new_arange(target_rank) > start_point[:, None].long()
                final_cutoff = start_cutoff & target_cutoff

                return final_cutoff.scatter(1, target_rank, final_cutoff)

            ## choose mask distribution
            mask_patterns = multi_segment()

            # masking
            prev_target_tokens = target_tokens.masked_fill(
                mask_patterns, unk
            )

            # from fairseq import pdb; pdb.set_trace()

            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = (
                target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
            )
            return target_tokens.masked_fill(~target_mask, unk)

        if self.cfg.noise == "random_delete":
            return _random_delete(target_tokens)
        elif self.cfg.noise == "random_mask":
            return _random_mask(target_tokens)
        elif self.cfg.noise == "full_mask":
            return _full_mask(target_tokens)
        elif self.cfg.noise == "no_noise":
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, models, args, **unused):
        # add models input to match the API for SequenceGenerator
        from fairseq.iterative_refinement_generator import IterativeRefinementGenerator

        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
            max_iter=getattr(args, "iter_decode_max_iter", 10),
            beam_size=getattr(args, "iter_decode_with_beam", 1),
            reranking=getattr(args, "iter_decode_with_external_reranker", False),
            decoding_format=getattr(args, "decoding_format", None),
            adaptive=not getattr(args, "iter_decode_force_max_iter", False),
            retain_history=getattr(args, "retain_iter_history", False),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError(
                "Constrained decoding with the translation_lev task is not supported"
            )

        return LanguagePairDataset(
            src_tokens, src_lengths, self.source_dictionary, append_bos=True
        )

    def get_mask_distribution(self,sample,model):
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]
        # B x T
        encoder_out = model.encode_only(
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"]
        )
        _,_,mask_distribution = model.decoder.forward_mask_prediction(encoder_out)
        return mask_distribution

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()

        mask_distribution = self.get_mask_distribution(sample,model)
        sample["prev_target"] = self.inject_noise(sample["target"],mask_distribution)
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            mask_distribution = self.get_mask_distribution(sample,model)
            sample["prev_target"] = self.inject_noise(sample["target"],mask_distribution)
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output
