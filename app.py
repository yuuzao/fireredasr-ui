import os

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", 5078))
BATCH = int(os.environ.get("BATCH", 1))
from flask import Flask, request, jsonify, Response, render_template
from pathlib import Path
import sys
import threading
import webbrowser
import time
import datetime
import hashlib
import logging
import subprocess
import torch
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from waitress import serve
from flask_cors import CORS
from datetime import timedelta
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps
from pydub import AudioSegment
from fireredasr.models.fireredasr import FireRedAsr
from fireredasr.models.punctuation import load_punctuation_model, restore_punctuation_with_model

ROOT_DIR = Path(__file__).parent.as_posix()
if sys.platform == "win32":
    os.environ["PATH"] = ROOT_DIR + f";{ROOT_DIR}/ffmpeg;" + os.environ["PATH"]

STATIC_DIR = f"{ROOT_DIR}/static"
LOGS_DIR = f"{ROOT_DIR}/logs"
TMP_DIR = f"{STATIC_DIR}/tmp"

Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_file_handler = logging.FileHandler(
    f"{LOGS_DIR}/{datetime.datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8"
)
_file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_file_handler.setFormatter(formatter)
logger.addHandler(_file_handler)


app = Flask(__name__, template_folder=f"{ROOT_DIR}/templates")
CORS(app)


# 将字符串做 md5 hash处理
def get_md5(input_string):
    md5 = hashlib.md5()
    md5.update(input_string.encode("utf-8"))
    return md5.hexdigest()


def runffmpeg(cmd, get_duration=False):
    try:
        if cmd[0] != "ffmpeg":
            cmd.insert(0, "ffmpeg")
        # 优化FFmpeg性能：添加多线程参数
        if "-threads" not in cmd:
            # 在输入文件参数前插入线程数参数
            input_idx = cmd.index("-i") if "-i" in cmd else -1
            if input_idx > 0:
                cmd.insert(input_idx, "-threads")
                cmd.insert(input_idx + 1, "0")  # 0表示自动使用所有可用线程
        logger.info(f"[FFmpeg] 开始执行命令: {' '.join(cmd)}")
        start_time = time.time()
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            check=True,
            text=True,
            creationflags=0 if sys.platform != "win32" else subprocess.CREATE_NO_WINDOW,
        )
        elapsed = time.time() - start_time
        logger.info(f"[FFmpeg] 命令执行完成，耗时: {elapsed:.2f}秒")

        # 如果需要获取音频时长，从stderr中解析
        duration = None
        if get_duration and result.stderr:
            # 从FFmpeg输出中提取时长信息
            duration_match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2})\.(\d{2})', result.stderr)
            if duration_match:
                hours, minutes, seconds, centiseconds = map(int, duration_match.groups())
                duration = (hours * 3600 + minutes * 60 + seconds) * 1000 + centiseconds * 10
                logger.info(f"[FFmpeg] 检测到音频时长: {duration}毫秒")

        return duration if get_duration else True

    except Exception as e:
        logger.error(f"[FFmpeg] 执行失败: {str(e)}")
        raise Exception(
            str(e.stderr)
            if hasattr(e, "stderr") and e.stderr
            else f"执行Ffmpeg操作失败:{cmd=}"
        )


"""
格式化毫秒或秒为符合srt格式的 2位小时:2位分:2位秒,3位毫秒 形式
print(ms_to_time_string(ms=12030))
-> 00:00:12,030
"""


def ms_to_time_string(*, ms=0, seconds=None):
    # 计算小时、分钟、秒和毫秒
    if seconds is None:
        td = timedelta(milliseconds=ms)
    else:
        td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000

    time_string = f"{hours}:{minutes}:{seconds},{milliseconds}"
    return format_time(time_string, ",")


# 将不规范的 时:分:秒,|.毫秒格式为  aa:bb:cc,ddd形式
# eg  001:01:2,4500  01:54,14 等做处理
def format_time(s_time="", separate=","):
    if not s_time.strip():
        return f"00:00:00{separate}000"
    hou, min, sec, ms = 0, 0, 0, 0

    tmp = s_time.strip().split(":")
    if len(tmp) >= 3:
        hou, min, sec = tmp[-3].strip(), tmp[-2].strip(), tmp[-1].strip()
    elif len(tmp) == 2:
        min, sec = tmp[0].strip(), tmp[1].strip()
    elif len(tmp) == 1:
        sec = tmp[0].strip()

    if re.search(r",|\.", str(sec)):
        t = re.split(r",|\.", str(sec))
        sec = t[0].strip()
        ms = t[1].strip()
    else:
        ms = 0
    hou = f"{int(hou):02}"[-2:]
    min = f"{int(min):02}"[-2:]
    sec = f"{int(sec):02}"
    ms = f"{int(ms):03}"[-3:]
    return f"{hou}:{min}:{sec}{separate}{ms}"


def restore_punctuation(srts):
    """
    为中文ASR转录结果恢复标点符号
    优先使用 FireRedChat-punc 模型，如果模型不可用则回退到基于规则的标点恢复
    """
    if not srts or len(srts) == 0:
        return srts

    logger.info(f"[标点恢复] 开始恢复标点符号，共 {len(srts)} 个片段")
    restore_start = time.time()

    # 尝试使用 FireRedChat-punc 模型
    punctuation_pipeline = load_punctuation_model(root_dir=ROOT_DIR)

    if punctuation_pipeline is not None:
        logger.info("[标点恢复] 使用 FireRedChat-punc 模型进行标点恢复")
        try:
            # 提取所有文本
            texts = [srt["text"].strip() for srt in srts if srt.get("text", "").strip()]

            if texts:
                # 使用模型恢复标点
                punctuated_texts = restore_punctuation_with_model(texts, punctuation_pipeline)

                # 将结果写回 srts
                text_idx = 0
                for i, srt in enumerate(srts):
                    if srt.get("text", "").strip():
                        srt["text"] = punctuated_texts[text_idx]
                        text_idx += 1

                # 仍然需要根据时间间隔添加结尾标点
                _add_final_punctuation_by_gap(srts)

                restore_elapsed = time.time() - restore_start
                logger.info(f"[标点恢复] 模型标点恢复完成，耗时: {restore_elapsed:.3f}秒")
                return srts
        except Exception as e:
            logger.warning(f"[标点恢复] 模型处理失败，回退到基于规则的方法: {str(e)}")
            # 继续执行基于规则的方法

    # 回退到基于规则的标点恢复方法
    logger.info("[标点恢复] 使用基于规则的标点恢复方法")

    # 疑问词和感叹词
    question_words = ["吗", "呢", "啊", "呀", "吧", "么", "什么", "怎么", "为什么", "哪里", "哪个", "哪些"]
    exclamation_words = ["啊", "呀", "哇", "哦", "哟", "哎", "唉"]

    # 转折词和连接词（需要在这些词前面添加逗号）
    conjunction_words = ["但是", "可是", "然而", "不过", "而且", "并且", "同时", "然后", "接着",
                        "所以", "因此", "因而", "因为", "由于", "虽然", "尽管", "如果", "假如",
                        "另外", "此外", "另外", "还有", "以及", "或者", "要么", "还是"]

    # 停顿词（需要在这些词后面添加逗号）
    pause_words = ["今天", "昨天", "明天", "现在", "刚才", "之前", "之后", "首先", "其次",
                   "最后", "总之", "总的来说", "其实", "实际上", "事实上", "当然", "确实"]

    def add_internal_punctuation(text):
        """在文本内部添加逗号"""
        if not text or len(text) < 3:
            return text

        result = text
        added_count = 0
        original_result = result

        # 特殊模式：处理"来朋友们"、"来大家"等
        result = re.sub(r"(来|好|那|这)(朋友们|大家|各位|同志们)", r"\1，\2", result)
        if result != original_result:
            added_count += 1
            original_result = result

        # 如果文本中已有较多标点，减少处理
        punc_count = sum(1 for p in result if p in "，。！？")
        if punc_count >= len(result) / 10:  # 标点密度已经较高
            return result

        # 在转折词和连接词前面添加逗号（但不在开头）
        for word in conjunction_words:
            pattern = f"([^，。！？])({word})"
            if re.search(pattern, result):
                result = re.sub(pattern, r"\1，\2", result, count=1)
                added_count += 1
                break  # 每段只添加一个，避免过度添加

        # 在停顿词后面添加逗号（但不在结尾）
        for word in pause_words:
            pattern = f"({word})([^，。！？])"
            if re.search(pattern, result) and not result.endswith(word):
                result = re.sub(pattern, r"\1，\2", result, count=1)
                added_count += 1
                break

        # 处理并列结构："又...又..."、"既...又..."、"不仅...还..."
        if "又" in result and result.count("又") >= 2:
            # 在第二个"又"前面添加逗号
            pos = result.find("又", result.find("又") + 1)
            if pos > 0 and result[pos-1] not in "，。！？":
                result = result[:pos] + "，" + result[pos:]
                added_count += 1

        # 处理"而且"、"并且"等词
        for word in ["而且", "并且", "同时", "另外", "此外"]:
            if word in result:
                pattern = f"([^，。！？])({word})"
                if re.search(pattern, result):
                    result = re.sub(pattern, r"\1，\2", result, count=1)
                    added_count += 1
                    break

        # 如果文本很长（超过12个字符）且标点很少，在合适位置添加逗号
        if len(result) > 12 and punc_count < 2:
            # 尝试在"又"、"还"、"也"等词前面添加逗号
            for word in ["又", "还", "也", "并且"]:
                pattern = f"([^，。！？]{{3,}})({word})"
                if re.search(pattern, result):
                    result = re.sub(pattern, r"\1，\2", result, count=1)
                    added_count += 1
                    break

        # 如果文本很长（超过18个字符）且仍然标点很少，在中间位置添加逗号
        if len(result) > 18 and punc_count < 2:
            # 在文本中间位置寻找合适位置
            mid_pos = len(result) // 2
            # 寻找合适的断点词
            for offset in range(-5, 6):
                pos = mid_pos + offset
                if 2 < pos < len(result) - 2:
                    char = result[pos]
                    # 在"的"、"了"、"是"、"在"、"有"、"而"等词后面添加逗号
                    if char in ["的", "了", "是", "在", "有", "而", "但"]:
                        if result[pos-1] not in "，。！？" and result[pos+1] not in "，。！？":
                            result = result[:pos+1] + "，" + result[pos+1:]
                            added_count += 1
                            break
            else:
                # 如果没找到合适的词，在中间位置直接添加（如果前后都不是标点）
                if mid_pos < len(result) - 1:
                    if result[mid_pos] not in "，。！？" and result[mid_pos+1] not in "，。！？":
                        result = result[:mid_pos+1] + "，" + result[mid_pos+1:]
                        added_count += 1

        if added_count > 0:
            logger.debug(f"[标点恢复] 文本内部添加了 {added_count} 个逗号: {text[:30]}... -> {result[:40]}...")

        return result

    for i in range(len(srts)):
        text = srts[i]["text"].strip()
        if not text:
            continue

        # 先在文本内部添加标点
        text = add_internal_punctuation(text)

        # 判断当前片段结尾应该添加什么标点
        punctuation = ""

        # 检查是否以疑问词结尾
        is_question = False
        for qw in question_words:
            if text.endswith(qw):
                is_question = True
                break

        # 检查是否以感叹词结尾
        is_exclamation = False
        for ew in exclamation_words:
            if text.endswith(ew) and len(text) <= 3:  # 短句且以感叹词结尾
                is_exclamation = True
                break

        # 根据时间间隔判断标点（降低阈值，添加更多标点）
        gap = None
        if i < len(srts) - 1:
            # 计算当前片段结束到下一个片段开始的时间间隔
            current_end = srts[i]["end_time"]
            next_start = srts[i + 1]["start_time"]
            gap = next_start - current_end

            if is_question:
                punctuation = "？"
            elif is_exclamation:
                punctuation = "！"
            elif gap > 1500:  # 间隔超过1.5秒，可能是句号
                punctuation = "。"
            elif gap > 400:  # 间隔400ms-1.5秒，添加逗号
                punctuation = "，"
            elif gap > 150:  # 间隔150-400ms，也可能需要逗号（如果文本较长）
                if len(text) > 10:
                    punctuation = "，"
            # gap <= 150ms 不添加标点，可能是连续语音
        else:
            # 最后一个片段
            if is_question:
                punctuation = "？"
            elif is_exclamation:
                punctuation = "！"
            else:
                punctuation = "。"

        # 添加标点符号（如果文本末尾还没有标点）
        if punctuation and not text[-1] in "。，、；：！？…":
            text = text + punctuation

        srts[i]["text"] = text
        gap_info = f"{gap}ms" if gap is not None else "N/A"
        logger.debug(f"[标点恢复] 片段 {i+1}: 最终文本长度 {len(text)} 字符，结尾标点: '{punctuation if punctuation else '无'}' (间隔: {gap_info})")

    restore_elapsed = time.time() - restore_start
    logger.info(f"[标点恢复] 基于规则的标点恢复完成，耗时: {restore_elapsed:.3f}秒")
    return srts


def _add_final_punctuation_by_gap(srts):
    """
    根据时间间隔为片段添加结尾标点符号（作为模型处理的补充）
    """
    if not srts or len(srts) == 0:
        return

    # 疑问词和感叹词
    question_words = ["吗", "呢", "啊", "呀", "吧", "么", "什么", "怎么", "为什么", "哪里", "哪个", "哪些"]
    exclamation_words = ["啊", "呀", "哇", "哦", "哟", "哎", "唉"]

    for i in range(len(srts)):
        text = srts[i].get("text", "").strip()
        if not text:
            continue

        # 判断当前片段结尾应该添加什么标点
        punctuation = ""

        # 检查是否以疑问词结尾
        is_question = False
        for qw in question_words:
            if text.endswith(qw):
                is_question = True
                break

        # 检查是否以感叹词结尾
        is_exclamation = False
        for ew in exclamation_words:
            if text.endswith(ew) and len(text) <= 3:  # 短句且以感叹词结尾
                is_exclamation = True
                break

        # 根据时间间隔判断标点
        gap = None
        if i < len(srts) - 1:
            # 计算当前片段结束到下一个片段开始的时间间隔
            current_end = srts[i].get("end_time", 0)
            next_start = srts[i + 1].get("start_time", 0)
            gap = next_start - current_end

            if is_question:
                punctuation = "？"
            elif is_exclamation:
                punctuation = "！"
            elif gap > 1500:  # 间隔超过1.5秒，可能是句号
                punctuation = "。"
            elif gap > 400:  # 间隔400ms-1.5秒，添加逗号
                punctuation = "，"
            elif gap > 150:  # 间隔150-400ms，也可能需要逗号（如果文本较长）
                if len(text) > 10:
                    punctuation = "，"
        else:
            # 最后一个片段
            if is_question:
                punctuation = "？"
            elif is_exclamation:
                punctuation = "！"
            else:
                punctuation = "。"

        # 添加标点符号（如果文本末尾还没有标点）
        if punctuation and text and text[-1] not in "。，、；：！？…":
            srts[i]["text"] = text + punctuation


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(app.config["STATIC_FOLDER"], filename)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/v1/audio/transcriptions", methods=["POST", "GET"])
@app.route(
    "/v1/audio/translations", methods=["POST", "GET"]
)  # 别名路由，兼容 translations
def uploadfile():
    request_start_time = time.time()
    try:
        logger.info("=" * 80)
        logger.info("[请求开始] 收到音频转录请求")

        if "file" not in request.files:  # 检查是否上传了文件
            logger.error("[请求失败] 未找到文件部分")
            return jsonify({"code": 1, "error": "No file part"}), 500

        file = request.files["file"]
        if file.filename == "":  # 检查是否选择了文件
            logger.error("[请求失败] 未选择文件")
            return jsonify({"code": 1, "error": "No selected file"}), 500

        # 检查文件格式
        ext = os.path.splitext(file.filename)[1].lower()
        supported_formats = [
            # 音频格式
            '.mp3', '.m4a', '.wav', '.flac', '.aac', '.ogg', '.opus',
            '.wma', '.amr', '.m3u', '.mp2', '.ac3', '.dts',
            # 视频格式（包含音频轨道）
            '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v',
            '.wmv', '.3gp', '.3g2', '.asf', '.rm', '.rmvb', '.vob',
            '.ts', '.mts', '.m2ts', '.f4v', '.ogv'
        ]

        if ext not in supported_formats:
            logger.warning(f"[文件格式] 不常见的文件格式: {ext}，将尝试使用FFmpeg处理")
            # 不直接拒绝，让FFmpeg尝试处理

        response_format = request.form.get("response_format", "srt")
        model = request.form.get("model", "AED").upper()
        if model not in ["AED", "LLM"]:
            model = "AED"

        logger.info(f"[请求信息] 文件名: {file.filename}, 文件格式: {ext}, 模型类型: {model}, 响应格式: {response_format}")

        model_path = f"{ROOT_DIR}/pretrained_models/FireRedASR-{model}-L/model.pth.tar"
        if not Path(model_path).exists():
            logger.error(f"[请求失败] 模型文件不存在: {model_path}")
            return jsonify(
                {
                    "code": 2,
                    "error": f"请下载 {model} 模型并放入 {ROOT_DIR}/pretrained_models/FireRedASR-{model}-L/",
                }
            ), 500

        # 使用时间戳生成文件名（ext已在上面获取）
        name = f"{time.time()}"
        filename_raw = f"{TMP_DIR}/raw-{name}{ext}"
        filename_16k = f"{TMP_DIR}/16k-{name}.wav"
        # 创建目录
        target_dir = f"{TMP_DIR}/{name}"
        Path(target_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"[文件保存] 开始保存原始文件到: {filename_raw}")
        save_start = time.time()
        file.save(filename_raw)
        logger.info(f"[文件保存] 原始文件保存完成，耗时: {time.time() - save_start:.2f}秒")

        # 保存文件到 /tmp 目录
        logger.info(f"[音频转换] 开始转换音频为16kHz WAV格式，源格式: {ext}")
        convert_start = time.time()
        try:
            # 尝试从FFmpeg获取音频时长，避免重复读取文件
            file_length = runffmpeg(
                [
                    "-y",
                    "-i",
                    filename_raw,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    "-f",
                    "wav",
                    filename_16k,
                ],
                get_duration=True
            )
            logger.info(f"[音频转换] 音频转换完成，总耗时: {time.time() - convert_start:.2f}秒")
        except Exception as e:
            logger.error(f"[音频转换] 转换失败: {str(e)}")
            error_msg = f"不支持的文件格式 {ext} 或文件损坏。支持的格式: MP3, M4A, WAV, FLAC, AAC, OGG, OPUS, WMA, AMR, MP4, AVI, MOV, MKV等"
            raise Exception(error_msg)

        # 如果FFmpeg未能获取时长，则使用AudioSegment读取
        if file_length is None:
            logger.info(f"[音频分析] FFmpeg未能获取时长，使用AudioSegment分析音频文件长度")
            file_length = len(
                AudioSegment.from_file(filename_16k, format=filename_16k[-3:])
            )
        file_length_seconds = file_length / 1000.0
        logger.info(f"[音频分析] 音频长度: {file_length}毫秒 ({file_length_seconds:.2f}秒)")

        if file_length >= 30000:
            logger.info(f"[音频切割] 音频长度超过30秒，开始VAD语音活动检测和切割")
            cut_start = time.time()
            wavs = cut_audio(filename_16k, target_dir)
            logger.info(f"[音频切割] 切割完成，共生成 {len(wavs)} 个音频片段，耗时: {time.time() - cut_start:.2f}秒")
        else:
            logger.info(f"[音频处理] 音频长度小于30秒，无需切割，直接处理")
            wavs = [
                {
                    "line": 1,
                    "start_time": 0,
                    "end_time": file_length,
                    "file": filename_16k,
                    "text": "",
                    "uttid": f"0_{file_length}",
                    "startraw": "00:00:00,000",
                    "endraw": ms_to_time_string(ms=file_length),
                }
            ]

        logger.info(f"[ASR转录] 开始ASR转录，共 {len(wavs)} 个音频片段")
        asr_start = time.time()
        srts = asr_task(wavs, asr_type=model)
        asr_elapsed = time.time() - asr_start
        logger.info(f"[ASR转录] ASR转录完成，总耗时: {asr_elapsed:.2f}秒，平均RTF: {asr_elapsed / file_length_seconds:.4f}")

        # 恢复标点符号
        srts = restore_punctuation(srts)

        total_elapsed = time.time() - request_start_time
        logger.info(f"[请求完成] 请求处理完成，总耗时: {total_elapsed:.2f}秒")
        logger.info("=" * 80)

        if response_format == "text":
            # 纯文本格式：直接连接所有文本，用空格分隔
            text_result = " ".join([it["text"].strip() for it in srts if it["text"].strip()])
            logger.info(f"[响应格式] 返回纯文本格式，长度: {len(text_result)} 字符")
            return Response(text_result, mimetype="text/plain")
        if response_format == "json":
            text_result = " ".join([it["text"].strip() for it in srts if it["text"].strip()])
            return jsonify({"text": text_result})
        # SRT字幕格式（默认）
        result = [
            f"{it['line']}\n{it['startraw']} --> {it['endraw']}\n{it['text']}"
            for it in srts
        ]
        logger.info(f"[响应格式] 返回SRT字幕格式，共 {len(result)} 条字幕")
        return Response("\n\n".join(result), mimetype="text/plain")
    except Exception as e:
        total_elapsed = time.time() - request_start_time
        logger.error(f"[请求失败] 处理过程中发生错误，总耗时: {total_elapsed:.2f}秒")
        logger.error(f"[错误详情] {str(e)}")
        logger.error(f"[错误堆栈] {traceback.format_exc()}")
        return jsonify({"code": 1, "error": str(e)}), 500


def asr_task(wavs, asr_type="AED"):
    logger.info(f"[ASR初始化] 开始加载 {asr_type} 模型")
    model_load_start = time.time()
    model = FireRedAsr.from_pretrained(
        asr_type.lower(), f"{ROOT_DIR}/pretrained_models/FireRedASR-{asr_type}-L"
    )
    logger.info(f"[ASR初始化] 模型加载完成，耗时: {time.time() - model_load_start:.2f}秒")

    idxs = {}
    for i, it in enumerate(wavs):
        idxs[it["uttid"]] = i
    wavs_chunks = [wavs[i : i + BATCH] for i in range(0, len(wavs), BATCH)]
    total_chunks = len(wavs_chunks)
    logger.info(f"[ASR批处理] 共 {len(wavs)} 个音频片段，分为 {total_chunks} 个批次处理，批次大小: {BATCH}")

    use_gpu = 1 if torch.cuda.is_available() else 0
    device_info = "GPU" if use_gpu else "CPU"
    logger.info(f"[ASR配置] 使用设备: {device_info}")

    param = {
        "use_gpu": use_gpu,
        "beam_size": 1,
        "nbest": 1,
        "decode_max_len": 0,
        "softmax_smoothing": 1.0,
        "aed_length_penalty": 0.0,
        "decode_min_len": 0,
        "repetition_penalty": 1.0,
        "llm_length_penalty": 0.0,
        "eos_penalty": 1.0,
        "temperature": 1.0,
    }

    processed_count = 0
    for chunk_idx, it in enumerate(wavs_chunks, 1):
        chunk_start = time.time()
        logger.info(f"[ASR批处理] 处理批次 {chunk_idx}/{total_chunks}，包含 {len(it)} 个音频片段")
        results = model.transcribe(
            [em["uttid"] for em in it], [em["file"] for em in it], param
        )
        batch_rtf = None
        for result in results:
            wavs[idxs[result["uttid"]]]["text"] = result["text"]
            processed_count += 1
            if "rtf" in result and batch_rtf is None:
                batch_rtf = result["rtf"]
        chunk_elapsed = time.time() - chunk_start
        logger.info(f"[ASR批处理] 批次 {chunk_idx}/{total_chunks} 完成，耗时: {chunk_elapsed:.2f}秒，进度: {processed_count}/{len(wavs)} ({processed_count*100//len(wavs)}%)")
        if batch_rtf:
            logger.info(f"[ASR批处理] 批次 {chunk_idx} RTF: {batch_rtf}")

    logger.info(f"[ASR完成] 所有批次处理完成，共处理 {processed_count} 个音频片段")
    return wavs


def openurl(url):
    def op():
        time.sleep(5)
        try:
            webbrowser.open_new_tab(url)
        except:
            pass

    threading.Thread(target=op).start()


# 根据 时间开始结束点，切割音频片段,并保存为wav到临时目录，记录每个wav的绝对路径到list，然后返回该list
def cut_audio(audio_file, dir_name):
    sampling_rate = 16000
    logger.info(f"[VAD] 开始VAD语音活动检测，音频文件: {audio_file}")

    def convert_to_milliseconds(timestamps):
        milliseconds_timestamps = []
        for timestamp in timestamps:
            milliseconds_timestamps.append(
                {
                    "start": int(round(timestamp["start"] / sampling_rate * 1000)),
                    "end": int(round(timestamp["end"] / sampling_rate * 1000)),
                }
            )

        return milliseconds_timestamps

    vad_p = {
        "threshold": 0.5,
        "neg_threshold": 0.35,
        "min_speech_duration_ms": 0,
        "max_speech_duration_s": float("inf"),
        "min_silence_duration_ms": 250,
        "speech_pad_ms": 200,
    }
    logger.info(f"[VAD] VAD参数: {vad_p}")

    vad_start = time.time()
    logger.info(f"[VAD] 开始解码音频和检测语音活动")
    audio_data = decode_audio(audio_file, sampling_rate=sampling_rate)
    logger.info(f"[VAD] 音频解码完成，开始VAD检测")
    speech_chunks = get_speech_timestamps(
        audio_data,
        vad_options=VadOptions(**vad_p),
    )
    vad_elapsed = time.time() - vad_start
    logger.info(f"[VAD] VAD检测完成，检测到 {len(speech_chunks)} 个语音片段，耗时: {vad_elapsed:.2f}秒")

    speech_chunks = convert_to_milliseconds(speech_chunks)

    logger.info(f"[音频切割] 开始切割音频片段")
    cut_start = time.time()

    total_chunks = len(speech_chunks)

    # 使用FFmpeg并行切割音频片段（比pydub更快且线程安全）
    def cut_single_chunk_ffmpeg(chunk_info):
        """使用FFmpeg切割单个音频片段"""
        i, it = chunk_info
        start_ms, end_ms = it["start"], it["end"]
        chunk_duration = end_ms - start_ms
        file_name = f"{dir_name}/{start_ms}_{end_ms}.wav"

        # 计算时长（秒）
        duration_sec = chunk_duration / 1000.0
        start_sec = start_ms / 1000.0

        # 使用FFmpeg切割（不显示日志，提高速度）
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss", str(start_sec),
                    "-i", audio_file,
                    "-t", str(duration_sec),
                    "-ac", "1",
                    "-ar", "16000",
                    "-c:a", "pcm_s16le",
                    "-f", "wav",
                    file_name,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                creationflags=0 if sys.platform != "win32" else subprocess.CREATE_NO_WINDOW,
            )
        except Exception as e:
            logger.error(f"[音频切割] FFmpeg切割片段 {i} 失败: {str(e)}")
            raise

        return {
            "line": i + 1,
            "start_time": start_ms,
            "end_time": end_ms,
            "file": file_name,
            "text": "",
            "uttid": f"{start_ms}_{end_ms}",
            "startraw": ms_to_time_string(ms=start_ms),
            "endraw": ms_to_time_string(ms=end_ms),
        }

    # 使用线程池并行处理音频切割
    data = []
    max_workers = min(8, total_chunks)  # 最多使用8个线程
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_chunk = {
            executor.submit(cut_single_chunk_ffmpeg, (i, it)): (i, it)
            for i, it in enumerate(speech_chunks, 1)
        }

        # 收集结果并保持顺序
        results = [None] * total_chunks
        completed = 0
        for future in as_completed(future_to_chunk):
            i, it = future_to_chunk[future]
            try:
                result = future.result()
                results[i - 1] = result
                completed += 1
                if completed % 10 == 0 or completed == total_chunks:
                    logger.info(f"[音频切割] 进度: {completed}/{total_chunks} ({completed*100//total_chunks}%)，当前片段: {ms_to_time_string(ms=it['start'])} -> {ms_to_time_string(ms=it['end'])}，时长: {it['end']-it['start']}ms")
            except Exception as e:
                logger.error(f"[音频切割] 切割片段 {i} 时发生错误: {str(e)}")
                raise

        data = [r for r in results if r is not None]

    cut_elapsed = time.time() - cut_start
    logger.info(f"[音频切割] 所有音频片段切割完成，共 {total_chunks} 个片段，耗时: {cut_elapsed:.2f}秒")

    return data


if __name__ == "__main__":
    try:
        print(f"api接口地址  http://{HOST}:{PORT}")
        openurl(f"http://{HOST}:{PORT}")
        serve(app, host=HOST, port=PORT, threads=4)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
