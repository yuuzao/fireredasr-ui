$(document).ready(function () {
  [].forEach.call(document.querySelectorAll('[role="tooltip"]'), it => {
    new bootstrap.Tooltip(it);
  });

  // 拖拽上传区域
  const dropArea = $('#drop-area');
  const fileInput = $('#file-upload');
  const srtOutput = $('#srt-output');
  const downloadButton = $('#download-srt-button');
  const debugLogs = $('#debug-logs');
  const statusText = $('#status-text');
  const fileName = $('#file-name');
  const fileSize = $('#file-size');
  const elapsedTime = $('#elapsed-time');
  const clearLogsBtn = $('#clear-logs-btn');

  // 状态管理
  let currentFile = null;
  let startTime = null;
  let elapsedInterval = null;

  // Debug日志功能
  function addLog(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = $(`
      <div class="log-entry ${type}">
        <span class="text-muted">[${timestamp}]</span> ${message}
      </div>
    `);
    debugLogs.append(logEntry);
    // 自动滚动到底部
    debugLogs.scrollTop(debugLogs[0].scrollHeight);
    // 限制日志条数，避免内存占用过大
    const logs = debugLogs.children('.log-entry');
    if (logs.length > 100) {
      logs.first().remove();
    }
  }

  // 清空日志
  clearLogsBtn.on('click', function () {
    debugLogs.empty();
    addLog('日志已清空', 'info');
  });

  // 更新状态
  function updateStatus(status, badgeClass = 'bg-secondary') {
    statusText.text(status).removeClass().addClass(`badge ${badgeClass} status-badge`);
    addLog(`状态更新: ${status}`, 'info');
  }

  // 格式化文件大小
  function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  }

  // 开始计时
  function startTimer() {
    startTime = Date.now();
    elapsedInterval = setInterval(function () {
      if (startTime) {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        elapsedTime.text(`${minutes}分${seconds}秒`);
      }
    }, 1000);
  }

  // 停止计时
  function stopTimer() {
    if (elapsedInterval) {
      clearInterval(elapsedInterval);
      elapsedInterval = null;
    }
  }

  // 重置上传状态，允许再次上传文件
  function resetUploadState() {
    // 重置文件输入框（清空选择，允许再次选择同一文件）
    fileInput.val('');
    fileInput[0].value = '';

    // 重新启用文件输入
    fileInput.prop('disabled', false);

    // 重置文件信息
    currentFile = null;
    fileName.text('-');
    fileSize.text('-');
    elapsedTime.text('-');

    // 重置拖拽区域状态
    dropArea.removeClass('dragover');

    // 确保上传区域可点击
    dropArea.css('pointer-events', 'auto');

    addLog('状态已重置，可以上传新文件', 'info');
  }










  // 拖拽事件
  dropArea.on('dragover', function (e) {
    e.preventDefault();
    dropArea.addClass('dragover');
  });

  dropArea.on('dragleave', function () {
    dropArea.removeClass('dragover');
  });

  dropArea.on('drop', function (e) {
    e.preventDefault();
    dropArea.removeClass('dragover');
    const files = e.originalEvent.dataTransfer.files;
    if (files.length > 0) {
      addLog(`拖拽上传文件: ${files[0].name}`, 'info');
      handleFile(files[0]);
    }
  });

  // 点击上传区域
  dropArea.on('click', function (e) {
    if (e.target !== fileInput[0]) {
      fileInput.click();
    }
  });

  fileInput.on('change', function () {
    if (fileInput[0].files.length > 0) {
      addLog(`选择文件: ${fileInput[0].files[0].name}`, 'info');
      handleFile(fileInput[0].files[0]);
    }
  });

  // 模型切换日志
  $('#model').on('change', function () {
    addLog(`切换模型: ${$(this).val().toUpperCase()}`, 'info');
  });

  // 输出格式切换日志
  $('#output-format').on('change', function () {
    const format = $(this).val();
    addLog(`切换输出格式: ${format === 'text' ? '纯文本' : 'SRT字幕'}`, 'info');
  });

  async function handleFile(file) {
    // 防止重复上传
    if (currentFile && $('#zz').hasClass('d-none') === false) {
      addLog('文件正在处理中，请等待完成', 'warning');
      return;
    }

    // 检查文件格式
    const fileName_ext = file.name.toLowerCase();
    const supportedExtensions = [
      // 音频格式
      '.mp3', '.m4a', '.wav', '.flac', '.aac', '.ogg', '.opus',
      '.wma', '.amr', '.m3u', '.mp2', '.ac3', '.dts',
      // 视频格式
      '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v',
      '.wmv', '.3gp', '.3g2', '.asf', '.rm', '.rmvb', '.vob',
      '.ts', '.mts', '.m2ts', '.f4v', '.ogv'
    ];

    const fileExt = fileName_ext.substring(fileName_ext.lastIndexOf('.'));
    if (fileExt && !supportedExtensions.includes(fileExt)) {
      const errorMsg = `不支持的文件格式: ${fileExt}。支持的格式: MP3, M4A, WAV, FLAC, AAC, OGG, OPUS, WMA, AMR, MP4, AVI, MOV, MKV等`;
      addLog(errorMsg, 'error');
      alert(errorMsg);
      return;
    }

    currentFile = file;
    const modelType = $('#model').val().toUpperCase();
    const outputFormat = $('#output-format').val();

    // 更新文件信息
    fileName.text(file.name);
    fileSize.text(formatFileSize(file.size));
    elapsedTime.text('0秒');

    // 记录文件格式
    addLog(`文件格式: ${fileExt || '未知'}`, 'info');

    // 更新状态和日志
    updateStatus('已上传', 'bg-info');
    addLog(`文件上传: ${file.name} (${formatFileSize(file.size)})`, 'success');
    addLog(`选择模型: ${modelType}`, 'info');
    addLog(`输出格式: ${outputFormat === 'text' ? '纯文本' : 'SRT字幕'}`, 'info');

    $('#logs').text(file.name + ' 识别中,用时可能较久，请耐心等待...');
    $('#zz').removeClass('d-none');

    // 禁用文件输入，防止重复上传
    fileInput.prop('disabled', true);
    dropArea.css('pointer-events', 'none');

    let formData = new FormData();
    formData.append('file', file);
    formData.append('model', modelType);
    formData.append('response_format', outputFormat);

    srtOutput.val('');
    downloadButton.prop('disabled', true);

    // 开始计时
    startTimer();
    updateStatus('处理中', 'bg-warning');
    addLog('开始处理文件...', 'info');
    addLog('步骤: 文件保存中...', 'info');

    // 模拟进度更新（因为后端是同步处理，无法获取实时进度）
    let progressSteps = [
      { delay: 1000, message: '步骤: 音频格式转换中...' },
      { delay: 2000, message: '步骤: VAD语音活动检测中...' },
      { delay: 3000, message: '步骤: 音频切割处理中...' },
      { delay: 4000, message: '步骤: ASR模型加载中...' },
      { delay: 5000, message: '步骤: 语音识别处理中...' },
      { delay: 6000, message: '步骤: 标点符号恢复中...' }
    ];

    let stepIndex = 0;
    const progressInterval = setInterval(function () {
      if (stepIndex < progressSteps.length) {
        addLog(progressSteps[stepIndex].message, 'info');
        stepIndex++;
      }
    }, 2000);

    $.ajax({
      url: '/v1/audio/translations',
      type: 'POST',
      data: formData,
      timeout: 86400000,
      processData: false,
      contentType: false,
      success: function (response) {
        clearInterval(progressInterval);
        stopTimer();

        srtOutput.val(response);
        downloadButton.prop('disabled', false);
        $('#logs').text('点击或拖拽音频或视频到这里，转录为文本或字幕');
        $('#zz').addClass('d-none');

        updateStatus('处理完成', 'bg-success');
        addLog('处理完成！转录结果已生成', 'success');
        addLog(`结果长度: ${response.length} 字符`, 'info');
        addLog(`输出格式: ${outputFormat === 'text' ? '纯文本' : 'SRT字幕'}`, 'info');

        // 重置所有状态，允许再次上传
        resetUploadState();
      },
      error: function (err) {
        clearInterval(progressInterval);
        stopTimer();

        const errorMsg = err.responseJSON ? err.responseJSON['error'] : "处理失败";
        updateStatus('处理失败', 'bg-danger');
        addLog(`错误: ${errorMsg}`, 'error');
        alert(errorMsg);
        $('#zz').addClass('d-none');

        // 重置所有状态，允许再次上传
        resetUploadState();
      }
    });
  }




  function getCurrentDateTimeString() {
    const now = new Date();

    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0'); // 月份从 0 开始，所以 + 1
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');

    return `${year}-${month}-${day}-${hours}-${minutes}-${seconds}`;
  }


  // 下载结果
  downloadButton.on('click', function () {
    const outputText = srtOutput.val();
    if (outputText) {
      const outputFormat = $('#output-format').val();
      const extension = outputFormat === 'text' ? 'txt' : 'srt';
      const filename = (outputFormat === 'text' ? 'transcript-' : 'subtitle-') + getCurrentDateTimeString() + '.' + extension;
      const blob = new Blob([outputText], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const link = $('<a>').attr({
        href: url,
        download: filename
      });
      $('body').append(link);
      link[0].click();
      link.remove();
      URL.revokeObjectURL(url);
      addLog(`下载${outputFormat === 'text' ? '文本' : '字幕'}文件: ${filename}`, 'success');
    } else {
      addLog('下载失败: 没有可下载的内容', 'error');
    }
  });

  // 初始化日志
  addLog('系统初始化完成', 'success');
  addLog('等待文件上传...', 'info');

});