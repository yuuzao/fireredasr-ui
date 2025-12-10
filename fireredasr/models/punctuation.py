"""
FireRedChat-punc 标点处理模块
使用本地 FireRedChat-punc 模型进行标点恢复
"""
import os
import logging
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# 全局变量，用于缓存模型实例
_punctuation_model = None
_punctuation_pipeline = None


def load_punctuation_model(model_path: Optional[str] = None, root_dir: Optional[str] = None):
    """
    从本地路径加载 FireRedChat-punc 标点模型

    Args:
        model_path: 本地模型路径，如果为 None 则按优先级查找：
                   1. 环境变量 PUNCTUATION_MODEL_PATH
                   2. pretrained_models/FireRedChat-punc 目录
        root_dir: 项目根目录，用于构建默认模型路径

    Returns:
        标点处理 pipeline 对象，如果模型不存在则返回 None
    """
    global _punctuation_model, _punctuation_pipeline

    if _punctuation_pipeline is not None:
        logger.info("[标点模型] 使用已缓存的标点模型")
        return _punctuation_pipeline

    # 获取模型路径（按优先级）
    if model_path is None:
        # 优先级1: 环境变量
        model_path = os.environ.get("PUNCTUATION_MODEL_PATH")

        # 优先级2: pretrained_models 目录
        if not model_path and root_dir:
            default_path = Path(root_dir) / "pretrained_models" / "FireRedChat-punc"
            if default_path.exists():
                model_path = str(default_path)
                logger.info(f"[标点模型] 使用默认路径: {model_path}")

    if not model_path:
        logger.info("[标点模型] 未找到标点模型，将使用基于规则的标点恢复方法")
        logger.info("[标点模型] 提示: 将模型放在 pretrained_models/FireRedChat-punc 目录，或设置环境变量 PUNCTUATION_MODEL_PATH")
        return None

    model_path = Path(model_path).expanduser().resolve()

    if not model_path.exists():
        logger.warning(f"[标点模型] 模型路径不存在: {model_path}")
        logger.warning("[标点模型] 将使用基于规则的标点恢复方法")
        return None

    try:
        logger.info(f"[标点模型] 开始从本地路径加载标点模型: {model_path}")
        load_start = time.time()

        model_dir = str(model_path)
        logger.info(f"[标点模型] 模型目录: {model_dir}")

        # 检查模型文件类型
        model_file = Path(model_dir) / "model.pth.tar"
        chinese_lert_base = Path(model_dir) / "chinese-lert-base"

        if model_file.exists():
            # 这是一个 PyTorch checkpoint，优先尝试使用 redpost 模块加载
            try:
                from redpost import RedPost, RedPostConfig
                import torch

                logger.info("[标点模型] 检测到 PyTorch checkpoint 格式，尝试使用 redpost 模块加载")

                # 检查是否需要 chinese-lert-base 基础模型
                if not chinese_lert_base.exists():
                    logger.warning("[标点模型] 未找到 chinese-lert-base 基础模型目录")
                    logger.warning("[标点模型] 提示: 需要先下载 hfl/chinese-lert-base 到 FireRedChat-punc/chinese-lert-base 目录")
                    logger.info("[标点模型] 将使用基于规则的标点恢复方法")
                    return None

                # 配置 RedPost
                use_gpu = torch.cuda.is_available()
                post_config = RedPostConfig(
                    use_gpu=use_gpu,
                    sentence_max_length=30
                )

                # 加载模型
                post_model = RedPost.from_pretrained(model_dir, post_config)
                _punctuation_pipeline = post_model
                logger.info(f"[标点模型] 使用 redpost 加载模型成功 (GPU: {use_gpu})")

                # 成功加载后立即返回，避免继续执行后续加载逻辑
                load_elapsed = time.time() - load_start
                logger.info(f"[标点模型] 标点模型加载完成，耗时: {load_elapsed:.2f}秒")
                return _punctuation_pipeline
            except ImportError:
                # redpost 未安装，尝试使用 transformers 直接加载 chinese-lert-base
                logger.info("[标点模型] redpost 模块未安装，尝试使用 transformers 加载 chinese-lert-base")
                try:
                    from transformers import AutoModelForTokenClassification, AutoTokenizer
                    import torch

                    # 检查基础模型是否存在
                    if not chinese_lert_base.exists():
                        logger.warning("[标点模型] 未找到 chinese-lert-base 基础模型，尝试从 HuggingFace 加载")
                        base_model_path = "hfl/chinese-lert-base"
                    else:
                        base_model_path = str(chinese_lert_base)

                    # 先加载 checkpoint 检查实际的 num_labels
                    torch.serialization.add_safe_globals([argparse.Namespace])
                    checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)

                    # 从 checkpoint 中检测 num_labels
                    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
                    if isinstance(state_dict, dict) and "classifier.weight" in state_dict:
                        num_labels = state_dict["classifier.weight"].shape[0]
                        logger.info(f"[标点模型] 从 checkpoint 检测到 num_labels: {num_labels}")
                    else:
                        num_labels = 4  # 默认值
                        logger.warning(f"[标点模型] 无法从 checkpoint 检测 num_labels，使用默认值: {num_labels}")

                    # 根据 checkpoint 中的分类器形状判断模型类型
                    # classifier.weight 形状为 [num_labels, hidden_size] 表示 SequenceClassification
                    # 但标点恢复任务通常需要 TokenClassification
                    # 由于 checkpoint 是 SequenceClassification 格式，我们需要手动转换为 TokenClassification
                    from transformers import AutoModelForTokenClassification
                    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

                    # 尝试加载为 TokenClassification（即使 checkpoint 是 SequenceClassification）
                    # 我们需要手动调整分类器权重
                    model = AutoModelForTokenClassification.from_pretrained(
                        base_model_path,
                        num_labels=num_labels,
                        trust_remote_code=True
                    )

                    # 如果 checkpoint 中的分类器是 SequenceClassification 格式，需要调整
                    # TokenClassification 的 classifier 形状应该是 [num_labels, hidden_size]
                    # 这与 SequenceClassification 相同，所以可以直接加载

                    # 加载 checkpoint 权重
                    if "model_state_dict" in checkpoint:
                        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                    elif "state_dict" in checkpoint:
                        model.load_state_dict(checkpoint["state_dict"], strict=False)
                    elif isinstance(checkpoint, dict) and any(k.startswith("bert.") or k.startswith("classifier.") for k in checkpoint.keys()):
                        # 直接是 state_dict
                        model.load_state_dict(checkpoint, strict=False)
                    else:
                        # 尝试直接加载
                        model.load_state_dict(checkpoint, strict=False)

                    model.eval()
                    if torch.cuda.is_available():
                        model = model.cuda()

                    # 保存模型和 tokenizer
                    _punctuation_model = model
                    _punctuation_pipeline = {
                        "model": model,
                        "tokenizer": tokenizer,
                        "is_sequence_classification": False,
                        "is_token_classification": True,
                        "model_type": "token_classification"
                    }
                    logger.info(f"[标点模型] 使用 transformers 加载模型成功 (GPU: {torch.cuda.is_available()})")

                    # 成功加载后立即返回，避免继续执行后续加载逻辑
                    load_elapsed = time.time() - load_start
                    logger.info(f"[标点模型] 标点模型加载完成，耗时: {load_elapsed:.2f}秒")
                    return _punctuation_pipeline
                except Exception as e2:
                    error_msg = str(e2)
                    if len(error_msg) > 200:
                        error_msg = error_msg[:200] + "..."
                    logger.warning(f"[标点模型] transformers 加载失败: {error_msg}")
                    logger.info("[标点模型] 将使用基于规则的标点恢复方法")
                    return None
            except Exception as e:
                error_msg = str(e)
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                logger.warning(f"[标点模型] redpost 加载失败: {error_msg}")
                logger.info("[标点模型] 将使用基于规则的标点恢复方法")
                return None

        # 如果没有检测到 model.pth.tar，尝试使用 transformers 直接加载（适用于标准 transformers 格式）
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            logger.info("[标点模型] 尝试使用 transformers 直接加载模型")
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None
            )
            model.eval()
            _punctuation_model = model
            _punctuation_pipeline = tokenizer
            logger.info("[标点模型] 使用 transformers 直接加载模型成功")
        except Exception as e1:
            # transformers 加载失败，静默回退（不输出详细错误，避免日志过长）
            logger.debug(f"[标点模型] transformers 加载失败")
            # 尝试使用 modelscope pipeline（如果可用）
            try:
                from modelscope.pipelines import pipeline

                logger.info("[标点模型] 尝试使用 modelscope pipeline 加载模型")
                # 尝试不同的任务类型来创建 pipeline
                task_types = ["punctuation", "text-generation", "text2text-generation", "nlp"]
                _punctuation_pipeline = None

                for task_type in task_types:
                    try:
                        _punctuation_pipeline = pipeline(
                            task=task_type,
                            model=model_dir,
                        )
                        logger.info(f"[标点模型] 使用任务类型 '{task_type}' 成功创建 pipeline")
                        break
                    except Exception:
                        continue

                if _punctuation_pipeline is None:
                    raise Exception("所有 modelscope pipeline 任务类型都失败")
            except (ImportError, Exception):
                # 所有加载方式都失败，静默回退
                logger.info("[标点模型] 无法加载标点模型，将使用基于规则的标点恢复方法")
                return None

        load_elapsed = time.time() - load_start
        logger.info(f"[标点模型] 标点模型加载完成，耗时: {load_elapsed:.2f}秒")

        return _punctuation_pipeline

    except ImportError:
        logger.info("[标点模型] modelscope 未安装，将使用基于规则的标点恢复方法")
        return None
    except Exception as e:
        # 简化错误日志，避免输出过长
        error_msg = str(e)
        if len(error_msg) > 200:
            error_msg = error_msg[:200] + "..."
        logger.info(f"[标点模型] 加载标点模型失败，将使用基于规则的标点恢复方法: {error_msg}")
        return None


def restore_punctuation_with_model(
    texts: List[str],
    pipeline_obj: Optional[object] = None
) -> List[str]:
    """
    使用 FireRedChat-punc 模型恢复标点符号

    Args:
        texts: 待处理的文本列表（无标点）
        pipeline_obj: 标点处理 pipeline 对象，如果为 None 则自动加载

    Returns:
        恢复标点后的文本列表
    """
    global _punctuation_model

    if pipeline_obj is None:
        # 尝试从默认位置加载（需要传入 root_dir）
        pipeline_obj = load_punctuation_model()

    if pipeline_obj is None:
        # 如果模型加载失败，返回原文本
        logger.warning("[标点处理] 标点模型不可用，返回原文本")
        return texts

    try:
        logger.info(f"[标点处理] 开始使用模型处理 {len(texts)} 条文本")
        process_start = time.time()

        results = []
        import re

        # 检查模型类型
        is_redpost_model = hasattr(pipeline_obj, 'process') and hasattr(pipeline_obj, '__class__') and 'RedPost' in str(type(pipeline_obj))
        is_sequence_classification = isinstance(pipeline_obj, dict) and pipeline_obj.get("is_sequence_classification", False)
        is_token_classification = isinstance(pipeline_obj, dict) and pipeline_obj.get("is_token_classification", False)
        is_transformers_model = _punctuation_model is not None and not is_sequence_classification and not is_token_classification

        # 对于 redpost 模型，可以批量处理以提高效率
        if is_redpost_model and len(texts) > 1:
            # 批量处理所有文本
            clean_texts = [re.sub(r'[，。！？、；：]', '', text.strip()) for text in texts]
            clean_texts = [t for t in clean_texts if t]  # 过滤空文本

            if clean_texts:
                try:
                    # 按照 README 示例进行批量处理
                    # process 方法接受文本列表和类型列表
                    batch_post_results = pipeline_obj.process(clean_texts, ["text"] * len(clean_texts))

                    # 处理结果
                    result_dict = {}
                    for i, result in enumerate(batch_post_results):
                        if result and "punc_text" in result:
                            punc_text = result["punc_text"]
                            # 移除 <unk> 等特殊标记
                            punc_text = re.sub(r'<unk>|<UNK>|\[unk\]|\[UNK\]', '', punc_text)
                            result_dict[i] = punc_text

                    # 将结果映射回原始文本列表
                    clean_idx = 0
                    for text in texts:
                        if not text or not text.strip():
                            results.append(text)
                        else:
                            clean_text = re.sub(r'[，。！？、；：]', '', text.strip())
                            if clean_text and clean_idx in result_dict:
                                results.append(result_dict[clean_idx])
                                clean_idx += 1
                            else:
                                results.append(text)

                    process_elapsed = time.time() - process_start
                    logger.info(f"[标点处理] 批量处理完成，耗时: {process_elapsed:.2f}秒")
                    return results
                except Exception as e:
                    logger.warning(f"[标点处理] 批量处理失败，回退到逐条处理: {str(e)}")
                    # 继续执行逐条处理

        # 逐条处理文本
        for text in texts:
            if not text or not text.strip():
                results.append(text)
                continue

            # 移除现有标点，只保留文本内容
            clean_text = re.sub(r'[，。！？、；：]', '', text.strip())

            if not clean_text:
                results.append(text)
                continue

            # 使用模型添加标点
            try:
                if is_redpost_model:
                    # 使用 redpost 模型进行推理（按照 README 中的使用方式）
                    # 参考: https://github.com/FireRedTeam/FireRedChat/tree/main/fireredasr-server/server/redpost
                    batch_post_results = pipeline_obj.process([clean_text], ["text"])
                    if batch_post_results and len(batch_post_results) > 0:
                        # 按照 README 示例，如果有多个结果需要连接
                        # 但通常单个文本输入只会返回一个结果
                        punctuated_text = "".join([r.get("punc_text", "") for r in batch_post_results])
                        # 如果结果为空，使用原文本
                        if not punctuated_text:
                            punctuated_text = clean_text
                        # 移除 <unk> 等特殊标记（按照 README 要求）
                        punctuated_text = re.sub(r'<unk>|<UNK>|\[unk\]|\[UNK\]', '', punctuated_text)
                    else:
                        punctuated_text = clean_text
                elif is_token_classification:
                    # 使用 TokenClassification 模型进行推理（基于 chinese-lert-base）
                    import torch
                    model = pipeline_obj["model"]
                    tokenizer = pipeline_obj["tokenizer"]

                    # 标点标签映射
                    num_labels = model.config.num_labels
                    if num_labels == 4:
                        punc_labels = ["", "，", "。", "？"]
                    else:
                        punc_labels = ["", "，", "。", "？", "！"]

                    # 对文本进行 tokenization 和标点预测
                    inputs = tokenizer(clean_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                        model = model.cuda()

                    with torch.no_grad():
                        outputs = model(**inputs)
                        # TokenClassification 输出形状: [batch_size, seq_len, num_labels]
                        predicted_labels = outputs.logits.argmax(dim=-1)[0].cpu().numpy()  # [seq_len]

                    # 将 token ids 转换回文本，并根据预测的标签添加标点
                    input_ids = inputs["input_ids"][0].cpu().numpy()
                    tokens = tokenizer.convert_ids_to_tokens(input_ids)
                    punctuated_tokens = []

                    for i, (token, label) in enumerate(zip(tokens, predicted_labels)):
                        if token in ["[CLS]", "[SEP]", "[PAD]"]:
                            continue
                        # 移除 ## 前缀（BERT tokenizer 的子词标记）
                        clean_token = token.replace("##", "")
                        # 获取对应的标点符号
                        punc = punc_labels[label] if label < len(punc_labels) else ""
                        punctuated_tokens.append(clean_token + punc)

                    punctuated_text = "".join(punctuated_tokens)
                elif is_sequence_classification:
                    # 使用 SequenceClassification 模型进行推理（需要特殊处理）
                    # 注意：SequenceClassification 是句子级分类，不适合直接用于标点恢复
                    # 这里可能需要将文本分段处理，或者使用其他方法
                    logger.warning("[标点处理] SequenceClassification 模型不适合标点恢复任务，回退到原文本")
                    punctuated_text = clean_text
                    # 移除 <unk> 等特殊标记
                    punctuated_text = re.sub(r'<unk>|<UNK>|\[unk\]|\[UNK\]', '', punctuated_text)
                    # 清理多余的空格
                    punctuated_text = re.sub(r'\s+', '', punctuated_text)
                elif is_transformers_model:
                    # 使用 transformers 模型进行推理
                    import torch
                    tokenizer = pipeline_obj
                    model = _punctuation_model

                    # 构建输入
                    inputs = tokenizer(clean_text, return_tensors="pt")
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}

                    # 生成标点
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=len(clean_text) * 2,
                            do_sample=False,
                            num_beams=1,
                        )

                    punctuated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    # 使用 ModelScope pipeline
                    result = pipeline_obj(clean_text)
                    # 处理返回结果（可能是字符串或字典）
                    if isinstance(result, dict):
                        punctuated_text = result.get("text", result.get("output", result.get("result", clean_text)))
                    elif isinstance(result, str):
                        punctuated_text = result
                    elif isinstance(result, list) and len(result) > 0:
                        # 可能是列表格式
                        if isinstance(result[0], dict):
                            punctuated_text = result[0].get("text", result[0].get("output", clean_text))
                        else:
                            punctuated_text = str(result[0])
                    else:
                        punctuated_text = clean_text

                # 如果结果为空，使用原文本
                if not punctuated_text or not punctuated_text.strip():
                    punctuated_text = text

                results.append(punctuated_text)
            except Exception as e:
                logger.warning(f"[标点处理] 处理文本失败: {text[:30]}... 错误: {str(e)}")
                results.append(text)  # 失败时返回原文本

        process_elapsed = time.time() - process_start
        logger.info(f"[标点处理] 模型处理完成，耗时: {process_elapsed:.2f}秒")

        return results

    except Exception as e:
        logger.error(f"[标点处理] 标点处理过程中发生错误: {str(e)}")
        import traceback
        logger.error(f"[标点处理] 错误堆栈: {traceback.format_exc()}")
        return texts  # 出错时返回原文本

