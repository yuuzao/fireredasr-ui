import os
import time
import logging

import torch
import argparse

from fireredasr.data.asr_feat import ASRFeatExtractor
from fireredasr.models.fireredasr_aed import FireRedAsrAed
from fireredasr.models.fireredasr_llm import FireRedAsrLlm
from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
from fireredasr.tokenizer.llm_tokenizer import LlmTokenizerWrapper

logger = logging.getLogger(__name__)


class FireRedAsr:
    @classmethod
    def from_pretrained(cls, asr_type, model_dir):
        assert asr_type in ["aed", "llm"]
        logger.info(f"[模型加载] 开始加载 {asr_type.upper()} 模型，模型目录: {model_dir}")

        cmvn_path = os.path.join(model_dir, "cmvn.ark")
        logger.info(f"[特征提取器] 加载CMVN文件: {cmvn_path}")
        feat_extractor = ASRFeatExtractor(cmvn_path)

        if asr_type == "aed":
            model_path = os.path.join(model_dir, "model.pth.tar")
            dict_path =os.path.join(model_dir, "dict.txt")
            spm_model = os.path.join(model_dir, "train_bpe1000.model")
            logger.info(f"[AED模型] 加载模型文件: {model_path}")
            logger.info(f"[AED模型] 加载字典文件: {dict_path}")
            logger.info(f"[AED模型] 加载SPM模型: {spm_model}")
            model = load_fireredasr_aed_model(model_path)
            tokenizer = ChineseCharEnglishSpmTokenizer(dict_path, spm_model)
            logger.info(f"[AED模型] 模型和分词器加载完成")
        elif asr_type == "llm":
            model_path = os.path.join(model_dir, "model.pth.tar")
            encoder_path = os.path.join(model_dir, "asr_encoder.pth.tar")
            # 使用 Qwen2-7B-Instruct
            llm_dir = os.path.join(model_dir, "Qwen2-7B-Instruct")
            logger.info(f"[LLM模型] 加载模型文件: {model_path}")
            logger.info(f"[LLM模型] 加载编码器文件: {encoder_path}")
            logger.info(f"[LLM模型] LLM目录: {llm_dir}")
            model, tokenizer = load_firered_llm_model_and_tokenizer(
                model_path, encoder_path, llm_dir)
            logger.info(f"[LLM模型] 模型和分词器加载完成")
        model.eval()
        logger.info(f"[模型加载] {asr_type.upper()} 模型加载完成")
        return cls(asr_type, feat_extractor, model, tokenizer)

    def __init__(self, asr_type, feat_extractor, model, tokenizer):
        self.asr_type = asr_type
        self.feat_extractor = feat_extractor
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def transcribe(self, batch_uttid, batch_wav_path, args={}):
        logger.info(f"[特征提取] 开始提取特征，批次大小: {len(batch_wav_path)}")
        feat_start = time.time()
        feats, lengths, durs = self.feat_extractor(batch_wav_path)
        feat_elapsed = time.time() - feat_start
        total_dur = sum(durs)
        logger.info(f"[特征提取] 特征提取完成，耗时: {feat_elapsed:.2f}秒，总音频时长: {total_dur:.2f}秒")

        use_gpu = args.get("use_gpu", False)
        if use_gpu:
            logger.info(f"[设备] 将数据移动到GPU")
            feats, lengths = feats.cuda(), lengths.cuda()
            self.model.cuda()
        else:
            logger.info(f"[设备] 使用CPU处理")
            self.model.cpu()

        if self.asr_type == "aed":
            logger.info(f"[AED转录] 开始AED模型转录")
            start_time = time.time()

            hyps = self.model.transcribe(
                feats, lengths,
                args.get("beam_size", 1),
                args.get("nbest", 1),
                args.get("decode_max_len", 0),
                args.get("softmax_smoothing", 1.0),
                args.get("aed_length_penalty", 0.0),
                args.get("eos_penalty", 1.0)
            )

            elapsed = time.time() - start_time
            rtf= elapsed / total_dur if total_dur > 0 else 0
            logger.info(f"[AED转录] 转录完成，耗时: {elapsed:.2f}秒，RTF: {rtf:.4f}")

            results = []
            for uttid, wav, hyp in zip(batch_uttid, batch_wav_path, hyps):
                hyp = hyp[0]  # only return 1-best
                hyp_ids = [int(id) for id in hyp["yseq"].cpu()]
                text = self.tokenizer.detokenize(hyp_ids)
                logger.debug(f"[AED转录] {uttid}: {text[:50]}..." if len(text) > 50 else f"[AED转录] {uttid}: {text}")
                results.append({"uttid": uttid, "text": text, "wav": wav,
                    "rtf": f"{rtf:.4f}"})
            return results

        elif self.asr_type == "llm":
            logger.info(f"[LLM转录] 开始LLM模型转录")
            logger.info(f"[LLM转录] 预处理文本输入")
            input_ids, attention_mask, _, _ = \
                LlmTokenizerWrapper.preprocess_texts(
                    origin_texts=[""]*feats.size(0), tokenizer=self.tokenizer,
                    max_len=128, decode=True)
            if use_gpu:
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            start_time = time.time()

            generated_ids = self.model.transcribe(
                feats, lengths, input_ids, attention_mask,
                args.get("beam_size", 1),
                args.get("decode_max_len", 0),
                args.get("decode_min_len", 0),
                args.get("repetition_penalty", 1.0),
                args.get("llm_length_penalty", 0.0),
                args.get("temperature", 1.0)
            )

            elapsed = time.time() - start_time
            rtf= elapsed / total_dur if total_dur > 0 else 0
            logger.info(f"[LLM转录] 转录完成，耗时: {elapsed:.2f}秒，RTF: {rtf:.4f}")
            texts = self.tokenizer.batch_decode(generated_ids,
                                                skip_special_tokens=True)
            results = []
            for uttid, wav, text in zip(batch_uttid, batch_wav_path, texts):
                logger.debug(f"[LLM转录] {uttid}: {text[:50]}..." if len(text) > 50 else f"[LLM转录] {uttid}: {text}")
                results.append({"uttid": uttid, "text": text, "wav": wav,
                                "rtf": f"{rtf:.4f}"})
            return results



def load_fireredasr_aed_model(model_path):
    logger.info(f"[模型加载] 开始加载AED模型权重: {model_path}")
    torch.serialization.add_safe_globals([argparse.Namespace]) # 添加这一行
    load_start = time.time()
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    logger.info(f"[模型加载] 模型权重加载完成，耗时: {time.time() - load_start:.2f}秒")
    logger.info(f"[模型参数] {package['args']}")
    model = FireRedAsrAed.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=True)
    logger.info(f"[模型加载] AED模型结构加载完成")
    return model


def load_firered_llm_model_and_tokenizer(model_path, encoder_path, llm_dir):
    logger.info(f"[模型加载] 开始加载LLM模型权重: {model_path}")
    torch.serialization.add_safe_globals([argparse.Namespace]) # 添加这一行
    load_start = time.time()
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    logger.info(f"[模型加载] 模型权重加载完成，耗时: {time.time() - load_start:.2f}秒")
    package["args"].encoder_path = encoder_path
    package["args"].llm_dir = llm_dir
    logger.info(f"[模型参数] {package['args']}")
    model = FireRedAsrLlm.from_args(package["args"])

    # 加载模型权重，并记录未匹配的参数
    missing_keys, unexpected_keys = model.load_state_dict(package["model_state_dict"], strict=False)
    if missing_keys:
        logger.warning(f"[模型加载] 以下参数未找到匹配项（将使用随机初始化）: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
    if unexpected_keys:
        logger.warning(f"[模型加载] 以下参数在检查点中但不在模型中: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")

    # 检查是否有维度不匹配的参数（通常在 encoder_projector 中）
    state_dict = package["model_state_dict"]
    model_dict = model.state_dict()
    dimension_mismatch = []
    for key in state_dict.keys():
        if key in model_dict:
            if state_dict[key].shape != model_dict[key].shape:
                dimension_mismatch.append(f"{key}: checkpoint {state_dict[key].shape} vs model {model_dict[key].shape}")

    if dimension_mismatch:
        logger.warning(f"[模型加载] 检测到维度不匹配的参数（可能是LLM模型版本不同导致的）:")
        for mismatch in dimension_mismatch[:10]:  # 只显示前10个
            logger.warning(f"  - {mismatch}")
        if len(dimension_mismatch) > 10:
            logger.warning(f"  ... 还有 {len(dimension_mismatch) - 10} 个不匹配项")
        logger.warning(f"[模型加载] 这些参数将使用随机初始化，可能影响模型性能！")
        logger.warning(f"[模型加载] 建议：确保使用的LLM模型版本与训练时使用的版本一致")

    logger.info(f"[模型加载] LLM模型结构加载完成")
    logger.info(f"[分词器] 开始加载LLM分词器: {llm_dir}")
    tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(llm_dir)
    logger.info(f"[分词器] LLM分词器加载完成")
    return model, tokenizer
