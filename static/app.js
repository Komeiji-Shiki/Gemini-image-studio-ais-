// --- 状态变量 ---
let selectedFiles = [];
let presets = [];
let currentApiFormat = 'gemini';
let isSoundEnabled = true;
let soundVolume = 1;
let isSelectionMode = false;
let selectedItems = new Set();
let generationController = null; // 用于中止 fetch 请求

// 分别存储不同 API 的 Base URL
let geminiBaseUrl = "https://generativelanguage.googleapis.com";
let openaiBaseUrl = "https://api.openai.com";

// 统计状态
let globalStats = {
    success: 0,
    failed: 0,
    waiting: 0,
    processing: 0,
    cost: 0.0
};

// --- API 基础配置 ---
const API_BASE = "/api";

// --- 初始化 ---
async function init() {
    await loadConfig();
    loadPromptFromLocal(); // 加载本地缓存的 prompt
    setupAutoSave();       // 绑定自动保存事件
    await loadHistory(true);
    await updateGlobalStats(); // 加载统计
    setupSound();
    setupPasteHandler();

    const volumeSlider = document.getElementById('volumeSlider');
    volumeSlider.addEventListener('input', () => {
        soundVolume = parseFloat(volumeSlider.value);
    });
    volumeSlider.addEventListener('change', () => {
        soundVolume = parseFloat(volumeSlider.value);
        playSound('type');
        triggerAutoSave();
    });

    // “加载更多”按钮
    document.getElementById('loadMoreBtn').onclick = () => loadHistory(false);
}

// DOM Ready
document.addEventListener('DOMContentLoaded', init);

// --- 粘贴图片到 Prompt 区 ---
function setupPasteHandler() {
    const promptBox = document.getElementById('prompt');
    promptBox.addEventListener('paste', (e) => {
        const items = (e.clipboardData || window.clipboardData).items;
        let hasImage = false;
        if (items) {
            for (let i = 0; i < items.length; i++) {
                if (items[i].type.indexOf("image") !== -1) {
                    const file = items[i].getAsFile();
                    if (file) {
                        if (selectedFiles.length >= 14) {
                            alert("最多只允许14张参考图");
                            return;
                        }
                        selectedFiles.push(file);
                        hasImage = true;
                    }
                }
            }
        }
        if (hasImage) {
            e.preventDefault();
            renderPreviews();
            playSound('success');

            // 视觉反馈
            const uploadArea = document.querySelector('.upload-area');
            uploadArea.style.borderColor = 'var(--primary)';
            setTimeout(() => {
                uploadArea.style.borderColor = 'var(--border)';
            }, 500);
        }
    });
}

// --- 配置管理 ---
async function loadConfig() {
    try {
        const res = await fetch(`${API_BASE}/config`);
        if (!res.ok) {
            console.error("Failed to load config:", res.status, res.statusText);
            return;
        }
        const config = await res.json();
        if (config) {
            // 基本设置
            if (config.api_key) document.getElementById('apiKey').value = config.api_key;

            // --- Base URL Loading Logic ---
            // Set defaults first, in case config is partial
            geminiBaseUrl = "https://generativelanguage.googleapis.com";
            openaiBaseUrl = "https://api.openai.com";
            
            // Load new, specific URLs if they exist, overriding defaults
            if (config.gemini_api_base_url) geminiBaseUrl = config.gemini_api_base_url;
            if (config.openai_api_base_url) openaiBaseUrl = config.openai_api_base_url;
        
            // Backward compatibility for old config files with a single api_base_url
            if (config.api_base_url && !config.gemini_api_base_url && !config.openai_api_base_url) {
                if (config.api_format === 'gemini') {
                    geminiBaseUrl = config.api_base_url;
                } else { // openai or openai_chat
                    openaiBaseUrl = config.api_base_url;
                }
            }
            // This will set the initial format and trigger the URL swap in the input field
            if (config.api_format) switchApiFormat(config.api_format, false);
            if (config.model_name) document.getElementById('modelName').value = config.model_name;
            if (config.trans_model_name) document.getElementById('transModelName').value = config.trans_model_name;

            // 生成参数
            if (config.image_size) document.getElementById('imageSize').value = config.image_size;
            if (config.image_format) {
                document.getElementById('imageFormat').value = config.image_format;
            }
            if (config.aspect_ratio) document.getElementById('aspectRatio').value = config.aspect_ratio;
            if (config.batch_size) document.getElementById('batchSize').value = config.batch_size;
            if (config.retry_count !== undefined) document.getElementById('retryCount').value = config.retry_count;
            if (config.timeout !== undefined) document.getElementById('timeout').value = config.timeout;
            if (config.generation_mode) document.getElementById('generationMode').value = config.generation_mode;
            if (config.temperature !== undefined) document.getElementById('temperature').value = config.temperature;
            if (config.top_p !== undefined) document.getElementById('topP').value = config.top_p;
            updateSliderValue('temperature');
            updateSliderValue('topP');

            // 破限模式
            document.getElementById('jailbreakToggle').checked = config.jailbreak_enabled || false;
            document.getElementById('systemPrompt').value = config.system_prompt || '';
            document.getElementById('forgedResponse').value = config.forged_response || '';
            if (config.system_instruction_method) {
                document.getElementById('systemInstructionMethod').value = config.system_instruction_method;
            }
            toggleSystemOverrideSection();

            // 高级设置
            document.getElementById('includeThoughtsToggle').checked = config.include_thoughts || false;
            
            // 思考预算：0 ~ 30000，默认 2048
            const thinkingSlider = document.getElementById('thinkingBudget');
            const tbMax = parseInt(thinkingSlider.max, 10) || 30000;
            let tbVal = config.thinking_budget !== undefined
                ? parseInt(config.thinking_budget, 10)
                : 2048;
            
            if (isNaN(tbVal)) tbVal = 2048;
            tbVal = Math.min(tbMax, Math.max(0, tbVal));
            thinkingSlider.value = tbVal;
            updateSliderValue('thinkingBudget');

            document.getElementById('includeSafetyToggle').checked = config.include_safety_settings || false;
            if (config.safety_settings) {
                document.getElementById('safeHarassment').value =
                    config.safety_settings.HARM_CATEGORY_HARASSMENT || 'BLOCK_NONE';
                document.getElementById('safeHate').value =
                    config.safety_settings.HARM_CATEGORY_HATE_SPEECH || 'BLOCK_NONE';
                document.getElementById('safeSex').value =
                    config.safety_settings.HARM_CATEGORY_SEXUALLY_EXPLICIT || 'BLOCK_NONE';
                document.getElementById('safeDanger').value =
                    config.safety_settings.HARM_CATEGORY_DANGEROUS_CONTENT || 'BLOCK_NONE';
                document.getElementById('safeCivic').value =
                    config.safety_settings.HARM_CATEGORY_CIVIC_INTEGRITY || 'BLOCK_NONE';
            }
            toggleSafetySettingsSection();

            // UI状态
            isSoundEnabled = config.sound_enabled !== false;
            soundVolume = config.sound_volume ?? 1;
            document.getElementById('volumeSlider').value = soundVolume;
            presets = config.presets || [];
            renderPresets();
            updateSoundBtn();
        }
    } catch (e) {
        console.error("Failed to load config", e);
    }
}

// --- 自动保存逻辑 ---
let autoSaveTimer = null;
const AUTO_SAVE_DELAY = 1000; // 1秒防抖

function setupAutoSave() {
    // 1. 配置项自动保存
    const inputs = document.querySelectorAll('.auto-save');
    inputs.forEach(input => {
        input.addEventListener('input', triggerAutoSave);
        input.addEventListener('change', triggerAutoSave);
    });

    // 2. Prompt 本地缓存 (不存服务器)
    const promptBox = document.getElementById('prompt');
    const savePrompt = () => {
        if (promptBox) {
            localStorage.setItem('gs_last_prompt', promptBox.value);
        }
    };

    promptBox.addEventListener('input', savePrompt);

    // 在页面关闭或刷新前再次确保保存，增强稳定性
    window.addEventListener('beforeunload', savePrompt);
}

function loadPromptFromLocal() {
    const lastPrompt = localStorage.getItem('gs_last_prompt');
    if (lastPrompt) {
        document.getElementById('prompt').value = lastPrompt;
    }
}

function triggerAutoSave() {
    const statusEl = document.getElementById('autoSaveStatus');
    statusEl.textContent = "正在同步...";
    statusEl.style.color = "#fbbf24"; // yellow

    if (autoSaveTimer) clearTimeout(autoSaveTimer);
    autoSaveTimer = setTimeout(saveConfigToServer, AUTO_SAVE_DELAY);
}

async function saveConfigToServer() {
    const statusEl = document.getElementById('autoSaveStatus');

    // Before saving, ensure the JS variables for URLs are updated with the current input value
    const currentUrlInBox = document.getElementById('baseUrl').value.trim();
    if (currentApiFormat === 'gemini') {
        geminiBaseUrl = currentUrlInBox;
    } else { // openai or openai_chat
        openaiBaseUrl = currentUrlInBox;
    }
    
    const config = {
        api_key: document.getElementById('apiKey').value,
        gemini_api_base_url: geminiBaseUrl,
        openai_api_base_url: openaiBaseUrl,
        api_format: currentApiFormat,
        model_name: document.getElementById('modelName').value,
        trans_model_name: document.getElementById('transModelName').value,
        image_format: document.getElementById('imageFormat').value,
        image_size: document.getElementById('imageSize').value,
        aspect_ratio: document.getElementById('aspectRatio').value,
        batch_size: parseInt(document.getElementById('batchSize').value, 10),
        retry_count: parseInt(document.getElementById('retryCount').value, 10),
        timeout: parseFloat(document.getElementById('timeout').value) || 120,
        generation_mode: document.getElementById('generationMode').value,
        temperature: parseFloat(document.getElementById('temperature').value),
        top_p: parseFloat(document.getElementById('topP').value),
        sound_enabled: isSoundEnabled,
        sound_volume: soundVolume,
        presets: presets,

        // Advanced settings
        include_thoughts: document.getElementById('includeThoughtsToggle').checked,
        thinking_budget: Math.min(
            30000,
            Math.max(0, parseInt(document.getElementById('thinkingBudget').value, 10) || 0)
        ),
        include_safety_settings: document.getElementById('includeSafetyToggle').checked,
        safety_settings: {
            "HARM_CATEGORY_HARASSMENT":
                document.getElementById('safeHarassment').value || 'BLOCK_NONE',
            "HARM_CATEGORY_HATE_SPEECH":
                document.getElementById('safeHate').value || 'BLOCK_NONE',
            "HARM_CATEGORY_SEXUALLY_EXPLICIT":
                document.getElementById('safeSex').value || 'BLOCK_NONE',
            "HARM_CATEGORY_DANGEROUS_CONTENT":
                document.getElementById('safeDanger').value || 'BLOCK_NONE',
            "HARM_CATEGORY_CIVIC_INTEGRITY":
                document.getElementById('safeCivic').value || 'BLOCK_NONE'
        },

        jailbreak_enabled: document.getElementById('jailbreakToggle').checked,
        system_instruction_method: document.getElementById('systemInstructionMethod').value,
        system_prompt: document.getElementById('systemPrompt').value,
        forged_response: document.getElementById('forgedResponse').value
    };

    try {
        const res = await fetch(`${API_BASE}/config`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        if (!res.ok) {
            throw new Error(`HTTP ${res.status} ${res.statusText}`);
        }
        statusEl.textContent = "已同步";
        statusEl.style.color = "#4ade80"; // green
    } catch (e) {
        console.error("Auto save failed", e);
        statusEl.textContent = "同步失败";
        statusEl.style.color = "#f87171"; // red
    }
}

// --- 数据导入 ---
function importLegacyData() {
    if (confirm("请选择之前导出的 .zip 文件 (格式如: GreySoul_All_History_*.zip)。导入可能需要一些时间，请耐心等待。")) {
        document.getElementById('legacyImportFile').click();
    }
}

async function handleLegacyImport(input) {
    const file = input.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    const btn = document.getElementById('btnImportLegacy');
    const originalText = btn.textContent;
    btn.textContent = "正在上传并导入...";
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/import_zip`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();

        if (data.success) {
            alert(`导入成功！共导入 ${data.count} 条记录。页面将刷新。`);
            playSound('success');
            loadHistory(true); // 刷新列表
        } else {
            throw new Error(data.detail || "未知错误");
        }
    } catch (e) {
        alert("导入失败: " + e.message);
        playSound('error');
    } finally {
        btn.textContent = originalText;
        btn.disabled = false;
        input.value = '';
    }
}

// --- 统计更新 ---
async function updateGlobalStats() {
    try {
        const res = await fetch(`${API_BASE}/stats`);
        if (!res.ok) {
            console.error("Failed to load stats:", res.status, res.statusText);
            return;
        }
        const data = await res.json();
        globalStats.success = data.success_count || 0;
        globalStats.failed = data.failed_count || 0;
        globalStats.cost = data.total_cost || 0.0;
        renderStats();
    } catch (e) {
        console.error("Failed to load stats", e);
    }
}

function renderStats() {
    document.getElementById('statSuccess').textContent = globalStats.success;
    document.getElementById('statFailed').textContent = globalStats.failed;
    document.getElementById('statWaiting').textContent = globalStats.waiting;
    document.getElementById('statProcessing').textContent = globalStats.processing;

    const costVal = parseFloat(globalStats.cost);
    document.getElementById('statCost').textContent = "$" + (isNaN(costVal) ? 0.0 : costVal).toFixed(4);
}

// --- 历史记录列表 (分页加载) ---
let currentOffset = 0;
const PAGE_SIZE = 10;

async function loadHistory(isReset = false) {
    if (isReset) {
        document.getElementById('gallery').innerHTML = '';
        currentOffset = 0;
    }

    const btn = document.getElementById('loadMoreBtn');
    btn.disabled = true;
    btn.textContent = "加载中...";

    try {
        const res = await fetch(`${API_BASE}/history?limit=${PAGE_SIZE}&offset=${currentOffset}`);
        if (!res.ok) {
            throw new Error(`HTTP ${res.status} ${res.statusText}`);
        }
        const items = await res.json();

        if (items.length > 0) {
            items.forEach(createResultCard);
            currentOffset += items.length;
            btn.disabled = false;
            btn.textContent = "加载更多记录...";
        } else {
            btn.textContent = "没有更多记录了";
        }
    } catch (e) {
        console.error(e);
        btn.textContent = "加载失败";
    }
}

// --- 多选与下载 ---
function toggleSelectionMode(forceOff = false) {
    isSelectionMode = forceOff ? false : !isSelectionMode;

    const selectionBtn = document.getElementById('selectionBtn');
    selectionBtn.style.display = 'flex';
    selectionBtn.style.visibility = isSelectionMode ? 'hidden' : 'visible';
    document.getElementById('downloadBtn').style.display = isSelectionMode ? 'flex' : 'none';
    document.getElementById('cancelSelectionBtn').style.display = isSelectionMode ? 'flex' : 'none';

    const cards = document.querySelectorAll('.compact-card');
    cards.forEach(card => {
        const checkboxContainer = card.querySelector('.selection-checkbox-container');
        if (checkboxContainer) {
            checkboxContainer.style.display = isSelectionMode ? 'flex' : 'none';
        }
        if (isSelectionMode) {
            card.classList.add('selection-mode');
        } else {
            card.classList.remove('selection-mode', 'selected');
            const checkbox = card.querySelector('.selection-checkbox');
            if (checkbox) checkbox.checked = false;
        }
    });

    if (!isSelectionMode) {
        selectedItems.clear();
    }
    updateSelectedCount();
}

function updateSelectionState(card) {
    const checkbox = card.querySelector('.selection-checkbox');
    const id = parseInt(card.dataset.id, 10);

    if (checkbox.checked) {
        selectedItems.add(id);
        card.classList.add('selected');
    } else {
        selectedItems.delete(id);
        card.classList.remove('selected');
    }
    updateSelectedCount();
}

function updateSelectedCount() {
    document.getElementById('selectedCount').textContent = selectedItems.size;
}

async function downloadSelected() {
    if (selectedItems.size === 0) {
        alert("请至少选择一项进行下载。");
        return;
    }

    const btn = document.getElementById('downloadBtn');
    const originalHTML = btn.innerHTML;
    btn.disabled = true;

    const zip = new JSZip();
    let processedCount = 0;
    const totalCount = selectedItems.size;
    const countSpan = document.getElementById('selectedCount');

    const textNode = btn.firstChild;
    if (textNode && textNode.nodeType === Node.TEXT_NODE) {
        textNode.nodeValue = '下载中 ';
    }
    countSpan.textContent = `0/${totalCount}`;

    const allCards = Array.from(document.querySelectorAll('.compact-card'))
        .filter(c => selectedItems.has(parseInt(c.dataset.id, 10)));

    for (const card of allCards) {
        const images = JSON.parse(card.dataset.images || '[]');

        if (images && images.length > 0) {
            for (const image of images) {
                try {
                    const response = await fetch(image.path);
                    if (!response.ok) throw new Error(`HTTP error ${response.status}`);
                    const blob = await response.blob();
                    zip.file(image.filename, blob);
                } catch (e) {
                    console.error(`Failed to download ${image.path}`, e);
                    zip.file(`${image.filename}.error.txt`, `Failed to download file. Error: ${e.message}`);
                }
            }
        }

        processedCount++;
        countSpan.textContent = `${processedCount}/${totalCount}`;
    }

    if (textNode && textNode.nodeType === Node.TEXT_NODE) {
        textNode.nodeValue = '正在生成 ZIP... ';
    }

    try {
        const content = await zip.generateAsync({ type: "blob" });
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `GreySoul_Selection_${timestamp}.zip`;

        const link = document.createElement('a');
        link.href = URL.createObjectURL(content);
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        toggleSelectionMode(true);
    } catch (err) {
        alert("创建 ZIP 文件失败: " + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalHTML;
    }
}

// --- 占位卡片（Skeleton，用于流式预览） ---
function createPlaceholderCard() {
    const gallery = document.getElementById('gallery');
    const card = document.createElement('div');
    card.className = 'compact-card placeholder-card';

    card.innerHTML = `
        <div class="placeholder-cover"></div>
        <div class="compact-overlay">
            <div class="compact-meta">
                <span>生成中...</span>
                <span>${new Date().toLocaleTimeString()}</span>
            </div>
            <div class="compact-info placeholder-text">等待模型响应...</div>
        </div>
    `;

    if (gallery.firstElementChild) {
        gallery.insertBefore(card, gallery.firstElementChild);
    } else {
        gallery.appendChild(card);
    }
    return card;
}

// --- 渲染卡片 ---
function createResultCard(data) {
    const gallery = document.getElementById('gallery');
    const card = document.createElement('div');
    card.className = 'compact-card';
    card.dataset.id = data.id;
    card.dataset.images = JSON.stringify(data.images || []);
    card.dataset.prompt = data.prompt || "";

    card.addEventListener('click', (e) => {
        if (e.target.closest('.btn-mini')) return;

        if (isSelectionMode) {
            const checkbox = card.querySelector('.selection-checkbox');
            if (e.target !== checkbox) {
                checkbox.checked = !checkbox.checked;
            }
            updateSelectionState(card);
        } else {
            loadDetail(data.id);
        }
    });

    let coverHtml = '';
    if (data.images && data.images.length > 0) {
        const thumbUrl = data.images[0].thumb || data.images[0].path;
        coverHtml = `<img src="${thumbUrl}" class="cover-img" loading="lazy" alt="Cover">`;
    } else {
        coverHtml = `<div style="height:100%; display:flex; align-items:center; justify-content:center; background:#1a202c; color:#64748b;">无图</div>`;
    }

    const promptText = data.prompt || '无描述';
    const preview = promptText.length > 60 ? promptText.substring(0, 60) + '...' : promptText;

    card.innerHTML = `
        <div class="selection-checkbox-container">
            <input type="checkbox" class="selection-checkbox" data-id="${data.id}">
        </div>
        ${coverHtml}
        <div class="compact-overlay">
            <div class="compact-meta">
                <span>ID: ${data.id}</span>
                <span>${new Date(data.timestamp).toLocaleTimeString()}</span>
            </div>
            <div class="compact-info">${escapeHtml(preview)}</div>
            <div class="compact-actions">
                <div class="btn-mini danger" onclick="deleteHistory(${data.id}, this.closest('.compact-card'), event)">🗑️ 删除</div>
            </div>
        </div>
    `;

    gallery.appendChild(card);

    if (isSelectionMode) {
        const checkboxContainer = card.querySelector('.selection-checkbox-container');
        if (checkboxContainer) checkboxContainer.style.display = 'flex';
        card.classList.add('selection-mode');
    }
}

// --- 详情查看 ---
async function loadDetail(id) {
    try {
        const res = await fetch(`${API_BASE}/history/${id}`);
        if (!res.ok) throw new Error("Failed to load detail");
        const data = await res.json();
        openDetailModal(data);
    } catch (e) {
        alert("加载详情失败: " + e.message);
    }
}

function openDetailModal(data) {
    const modal = document.getElementById('detailModal');
    const container = document.getElementById('detailContainer');

    let imagesHtml = '';
    if (data.images && data.images.length > 0) {
        data.images.forEach(img => {
            const src = img.path;
            imagesHtml += `
                <div style="margin-bottom: 30px; text-align: center;">
                    <img src="${src}" style="max-width: 100%; max-height: 80vh; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); cursor: zoom-in;" onclick="openModal('${src}')">
                    <div style="margin-top: 10px;">
                        <a href="${src}" download="${img.filename}" class="btn-secondary" style="text-decoration:none; display:inline-block; padding: 8px 20px;">
                            ⬇️ 下载此图
                        </a>
                    </div>
                </div>
            `;
        });
    } else {
        imagesHtml = '<div style="color:#666; text-align:center; margin-top: 50px;">此记录未生成图片</div>';
    }

    const uniqueId = 'detail-thought-' + data.id;

    let refImagesHtml = '';
    if (data.refImages && data.refImages.length > 0) {
        refImagesHtml += '<div class="thought-label" style="margin-top: 20px;">参考原图</div>';
        refImagesHtml += '<div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;">';
        data.refImages.forEach(refImg => {
            refImagesHtml += `
                <img src="${refImg.path}"
                     style="width: 70px; height: 70px; object-fit: cover; border-radius: 6px; cursor: zoom-in; border: 1px solid var(--border);"
                     onclick="openModal('${refImg.path}')">
            `;
        });
        refImagesHtml += '</div>';
    }

    let thoughtImagesHtml = '';
    if (data.thoughtImages && data.thoughtImages.length > 0) {
        thoughtImagesHtml += '<div class="thought-label" style="margin-top: 20px;">思维草稿</div>';
        thoughtImagesHtml += '<div class="thought-images-container">';
        data.thoughtImages.forEach(thoughtImg => {
            thoughtImagesHtml += `
                <img src="${thoughtImg.thumb || thoughtImg.path}"
                     class="thought-thumbnail"
                     onclick="openModal('${thoughtImg.path}')"
                     title="点击查看大图">
            `;
        });
        thoughtImagesHtml += '</div>';
    }

    container.innerHTML = `
        <div class="detail-left-pane">
            <div style="width: 100%; height: 100%; overflow-y: auto; padding: 20px;">
                ${imagesHtml}
            </div>
        </div>
        <div class="detail-right-pane">
            <div style="padding: 20px; border-bottom: 1px solid var(--border);">
                <div style="font-size: 1.2rem; margin-bottom: 5px;">Record #${data.id}</div>
                <div style="color: var(--text-muted); font-size: 0.85rem;">${new Date(data.timestamp).toLocaleString()} | ${escapeHtml(data.model || '')}</div>
            </div>

            <div class="detail-scroll-area">
                <div class="thought-label">Prompt</div>
                <div class="content-body">${escapeHtml(data.prompt || '')}</div>

                ${refImagesHtml}

                <div class="thought-label">Thinking Process / Output</div>
                ${thoughtImagesHtml}
                <div class="content-body" id="${uniqueId}">${escapeHtml(data.text || "(无文本输出)")}</div>

                <div class="actions" style="margin: 20px 0;">
                    <button class="btn-secondary" onclick="translateThought(document.getElementById('${uniqueId}').innerText, document.getElementById('${uniqueId}-trans'))">
                        🔄 翻译思维链
                    </button>
                </div>

                <div class="thought-label">Translation Result</div>
                <div class="content-body" id="${uniqueId}-trans" style="min-height:80px; color: #94a3b8;">...</div>
            </div>
        </div>
    `;

    modal.style.display = 'block';
    document.body.style.overflow = 'hidden';
}

function closeDetailModal() {
    document.getElementById('detailModal').style.display = 'none';
    document.body.style.overflow = '';
}

async function deleteHistory(id, element, event) {
    if (event) event.stopPropagation();
    if (!confirm('确定要删除这条记录吗？文件也将被物理删除。')) return;

    try {
        const res = await fetch(`${API_BASE}/history/${id}`, { method: 'DELETE' });
        if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
        element.remove();
        playSound('success');
    } catch (e) {
        alert("删除失败: " + e.message);
        playSound('error');
    }
}

// --- 生成逻辑 ---
async function startGeneration() {
    const apiKey = document.getElementById('apiKey').value;
    const prompt = document.getElementById('prompt').value;
    if (!apiKey || !prompt) {
        alert("请填写 API Key 和 Prompt");
        return;
    }

    const generateBtn = document.getElementById('generateBtn');
    const stopBtn = document.getElementById('stopBtn');
    const status = document.getElementById('status');
    const errorArea = document.getElementById('errorArea');

    generateBtn.style.display = 'none';
    stopBtn.style.display = 'block';
    status.textContent = "✨ 正在请求服务器进行生成...";
    errorArea.innerHTML = '';

    generationController = new AbortController();
    const signal = generationController.signal;

    try {
        // 处理图片上传
        const refImagesData = await Promise.all(selectedFiles.map(fileToBase64));

        const payload = {
            apiKey: apiKey,
            model: document.getElementById('modelName').value,
            prompt: prompt,
            api_format: currentApiFormat,
            api_base_url: document.getElementById('baseUrl').value,
            aspectRatio: document.getElementById('aspectRatio').value || undefined,
            image_format: document.getElementById('imageFormat').value,
            imageSize: document.getElementById('imageSize').value,
            batchSize: 1, // 后端目前只处理单次请求
            refImages: refImagesData,
            timeout: parseFloat(document.getElementById('timeout').value) || 120,
            temperature: parseFloat(document.getElementById('temperature').value),
            topP: parseFloat(document.getElementById('topP').value),

            // Advanced settings
            include_thoughts: document.getElementById('includeThoughtsToggle').checked,
            thinking_budget: Math.min(
                30000,
                Math.max(0, parseInt(document.getElementById('thinkingBudget').value, 10) || 0)
            ),
            include_safety_settings: document.getElementById('includeSafetyToggle').checked,
            safety_settings: {
                "HARM_CATEGORY_HARASSMENT":
                    document.getElementById('safeHarassment').value || 'BLOCK_NONE',
                "HARM_CATEGORY_HATE_SPEECH":
                    document.getElementById('safeHate').value || 'BLOCK_NONE',
                "HARM_CATEGORY_SEXUALLY_EXPLICIT":
                    document.getElementById('safeSex').value || 'BLOCK_NONE',
                "HARM_CATEGORY_DANGEROUS_CONTENT":
                    document.getElementById('safeDanger').value || 'BLOCK_NONE',
                "HARM_CATEGORY_CIVIC_INTEGRITY":
                    document.getElementById('safeCivic').value || 'BLOCK_NONE'
            },

            jailbreak_enabled: document.getElementById('jailbreakToggle').checked
        };

        if (payload.jailbreak_enabled) {
            payload.system_instruction_method = document.getElementById('systemInstructionMethod').value;
            payload.system_prompt = document.getElementById('systemPrompt').value;
            payload.forged_response = document.getElementById('forgedResponse').value;
        }

        const batchCount = parseInt(document.getElementById('batchSize').value, 10) || 1;
        const mode = document.getElementById('generationMode').value;
        let successCount = 0;

        globalStats.waiting = batchCount;
        renderStats();

        const generateOne = async () => {
            globalStats.waiting = Math.max(0, globalStats.waiting - 1);
            globalStats.processing++;
            renderStats();

            try {
                const res = await fetch(`${API_BASE}/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                    signal: signal
                });

                const resText = await res.text();
                if (!res.ok) {
                    let errorDetail = `服务器错误 (HTTP ${res.status})`;
                    try {
                        const errorJson = JSON.parse(resText);
                        if (errorJson.detail) errorDetail = errorJson.detail;
                    } catch (_) {
                        errorDetail = `${errorDetail}: ${resText}`;
                    }
                    throw new Error(errorDetail);
                }
                
                const data = JSON.parse(resText);
                if (!data.success) throw new Error(data.detail || "Unknown Error");

                globalStats.processing = Math.max(0, globalStats.processing - 1);
                globalStats.success++;
                if (data.data && typeof data.data.cost === 'number') {
                    globalStats.cost += data.data.cost;
                }
                renderStats();

                return data.data;
            } catch (err) {
                globalStats.processing = Math.max(0, globalStats.processing - 1);
                globalStats.failed++;
                renderStats();
                throw err;
            }
        };

        if (mode === 'parallel') {
            status.textContent = `⚡ 正在并发生成 ${batchCount} 张...`;
            const promises = Array(batchCount).fill(0).map(() =>
                generateOne()
                .then(resultData => {
                    createResultCard(resultData);
                    const gallery = document.getElementById('gallery');
                    if (gallery.firstElementChild && gallery.lastElementChild) {
                       gallery.insertBefore(gallery.lastElementChild, gallery.firstElementChild);
                    }
                    successCount++;
                    playSound('success');
                    status.textContent = `并发生成中... 完成 ${successCount}/${batchCount}`;
                    return { status: 'fulfilled', value: resultData };
                })
                .catch(err => {
                    return { status: 'rejected', reason: err };
                })
            );
            
            const results = await Promise.all(promises);
            const failures = results.filter(r => r.status === 'rejected');
            
            if (failures.length > 0) {
                const errorMessages = failures
                    .map(f => `- ${escapeHtml(f.reason && f.reason.message ? f.reason.message : String(f.reason))}`)
                    .join('<br>');
                errorArea.innerHTML = `<div class="error-box">并发任务中出现错误:<br>${errorMessages}</div>`;
                playSound('error');
            }

            status.textContent = `并发任务全部完成。成功 ${successCount} 张，失败 ${failures.length} 张。`;
            
        } else { // Serial mode
            for (let i = 0; i < batchCount; i++) {
                status.textContent = `正在生成第 ${i + 1}/${batchCount} 张...`;
                try {
                    const resultData = await generateOne();
                    createResultCard(resultData);
                    const gallery = document.getElementById('gallery');
                     if (gallery.firstElementChild && gallery.lastElementChild) {
                       gallery.insertBefore(gallery.lastElementChild, gallery.firstElementChild);
                    }
                    successCount++;
                    playSound('success');
                } catch(err) {
                    const msg = escapeHtml(err.message || String(err));
                    errorArea.innerHTML += `<div class="error-box">第 ${i + 1} 张生成失败: ${msg}</div>`;
                    playSound('error');
                }
            }
            status.textContent = `任务完成。成功生成 ${successCount} 张。`;
        }

        setTimeout(updateGlobalStats, 1000);

    } catch (e) {
        if (e.name !== 'AbortError') {
            console.error(e);
            const msg = escapeHtml(e.message || String(e));
            status.textContent = "生成出错";
            errorArea.innerHTML = `<div class="error-box">${msg}</div>`;
            playSound('error');
        } else {
            status.textContent = "任务已手动停止。";
        }
        setTimeout(updateGlobalStats, 1000);
    } finally {
        globalStats.waiting = 0;
        globalStats.processing = 0;
        renderStats();
        generateBtn.style.display = 'block';
        stopBtn.style.display = 'none';
        generationController = null;
    }
}


// --- 翻译思维链 ---
function stopGeneration() {
    if (generationController) {
        generationController.abort();
        generationController = null;
    }
}

async function translateThought(text, outputElem) {
    if (!text || text.trim() === '(无文本输出)') return;
    outputElem.textContent = "正在翻译...";
    try {
        const transModel = document.getElementById('transModelName').value;
        const res = await fetch(`${API_BASE}/translate_thought`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                apiKey: document.getElementById('apiKey').value,
                model: transModel
            })
        });
        if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
        const data = await res.json();
        outputElem.textContent = data.translated || "翻译失败";
    } catch (e) {
        outputElem.textContent = "翻译错误: " + e.message;
    }
}

// --- 工具函数 ---
function updateSliderValue(id) {
    const slider = document.getElementById(id);
    const display = document.getElementById(id + 'Value');
    if (slider && display) {
        display.textContent = slider.value;
    }
}

function editValue(id, max, step) {
    const display = document.getElementById(id + 'Value');
    const slider = document.getElementById(id);
    if (!slider || !display || display.isEditing) return;

    const sliderMin = slider.min !== undefined ? parseFloat(slider.min) : 0;

    const sliderMax = (typeof max === 'number')
        ? max
        : (slider.max !== undefined ? parseFloat(slider.max) : 100);

    const sliderStep = (typeof step === 'number')
        ? step
        : (slider.step ? parseFloat(slider.step) : 1);

    const originalValue = slider.value || display.textContent;

    display.isEditing = true;

    const input = document.createElement('input');
    input.type = 'number';
    input.value = originalValue;
    input.min = sliderMin;
    input.max = sliderMax;
    input.step = sliderStep;
    input.style.width = '70px';
    input.style.padding = '2px 4px';
    input.style.textAlign = 'center';
    input.style.border = '1px solid var(--primary)';
    input.style.background = 'var(--input-bg)';
    input.style.color = 'var(--text)';
    input.style.borderRadius = '6px';

    const originalTextContent = display.textContent;
    display.textContent = '';
    display.appendChild(input);
    input.focus();
    input.select();

    const finishEditing = (commit) => {
        const newValue = input.value;
        display.removeChild(input);

        let finalVal = originalValue;

        if (commit && newValue !== null && newValue.trim() !== '' && !isNaN(newValue)) {
            let v = parseFloat(newValue);
            v = Math.min(sliderMax, Math.max(sliderMin, v));

            // 小数位按 step 来：0.01 -> 2 位；128 -> 0 位
            const stepStr = String(sliderStep);
            const precision = stepStr.includes('.') ? stepStr.split('.')[1].length : 0;

            if (precision > 0) {
                finalVal = parseFloat(v.toFixed(precision));
            } else {
                finalVal = Math.round(v);
            }
        }

        slider.value = finalVal;
        display.isEditing = false;

        // 统一通过 slider 的逻辑刷新 UI + 自动保存
        updateSliderValue(id);
        triggerAutoSave();
    };

    input.addEventListener('blur', () => finishEditing(true));
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            finishEditing(true);
        } else if (e.key === 'Escape') {
            e.preventDefault();
            input.value = originalTextContent;
            finishEditing(false);
        }
    });
}

function toggleSystemOverrideSection() {
    const checkbox = document.getElementById('jailbreakToggle');
    const section = document.getElementById('jailbreakSection');
    section.style.display = checkbox.checked ? 'flex' : 'none';
}

function toggleSafetySettingsSection() {
    const checkbox = document.getElementById('includeSafetyToggle');
    const section = document.getElementById('safetySettingsContainer');
    section.style.display = checkbox.checked ? 'grid' : 'none';
}

const fileToBase64 = (file) => new Promise((resolve) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve({
        mime_type: file.type,
        data: reader.result.split(',')[1]
    });
});

function escapeHtml(text) {
    if (typeof text !== 'string') return '';
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function switchApiFormat(format, doSave = true) {
    const baseUrlInput = document.getElementById('baseUrl');
    const currentUrlInBox = baseUrlInput.value.trim();
    
    // Save current input value to its corresponding variable before switching
    if (currentApiFormat === 'gemini') {
        geminiBaseUrl = currentUrlInBox;
    } else { // openai or openai_chat
        openaiBaseUrl = currentUrlInBox;
    }

    currentApiFormat = format; // Update the format state
    
    // Update UI buttons to show the active format
    const buttons = document.querySelectorAll('#apiFormatGroup .btn-secondary');
    buttons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.value === format);
    });

    // Load the URL for the new format into the input box
    if (format === 'gemini') {
        baseUrlInput.value = geminiBaseUrl;
    } else { // openai or openai_chat
        baseUrlInput.value = openaiBaseUrl;
    }

    if (doSave) {
        triggerAutoSave();
    }
}

// --- 预设 (Presets) ---
function renderPresets() {
    const container = document.getElementById('presetContainer');
    container.innerHTML = '';
    presets.forEach((p, index) => {
        const tag = document.createElement('div');
        tag.className = 'preset-tag';
        tag.innerHTML = `
            <span class="preset-name">${escapeHtml(p.name)}</span>
            <div class="preset-actions">
                <span class="preset-action-btn preset-edit" title="编辑" onclick="editPreset(${index}, event)">✏️</span>
                <span class="preset-action-btn preset-delete" title="删除" onclick="deletePreset(${index}, event)">&times;</span>
            </div>
        `;
        tag.onclick = (e) => {
            if (e.target.classList.contains('preset-edit') || e.target.classList.contains('preset-delete')) {
                return;
            }
            document.getElementById('prompt').value = p.content;
            playSound('type');
        };
        container.appendChild(tag);
    });
}

function saveCurrentPrompt() {
    const content = document.getElementById('prompt').value;
    if (!content) return;
    const name = prompt("预设名称:", content.substring(0, 20));
    if (name) {
        presets.push({ name, content });
        renderPresets();
        triggerAutoSave();
    }
}

function editPreset(index, e) {
    e.stopPropagation();
    const preset = presets[index];

    const modal = document.getElementById('preset-editor-modal');
    const nameInput = document.getElementById('preset-name-input');
    const contentInput = document.getElementById('preset-content-input');
    const saveBtn = document.getElementById('save-preset-edit');
    const cancelBtn = document.getElementById('cancel-preset-edit');

    nameInput.value = preset.name;
    contentInput.value = preset.content;

    modal.classList.remove('hidden');

    const saveHandler = () => {
        const newName = nameInput.value.trim();
        if (!newName) {
            alert("预设名称不能为空。");
            return;
        }
        presets[index] = { name: newName, content: contentInput.value };
        renderPresets();
        triggerAutoSave();
        closeModal();
    };

    const cancelHandler = () => {
        closeModal();
    };

    const closeModal = () => {
        modal.classList.add('hidden');
        saveBtn.removeEventListener('click', saveHandler);
        cancelBtn.removeEventListener('click', cancelHandler);
    };

    saveBtn.addEventListener('click', saveHandler);
    cancelBtn.addEventListener('click', cancelHandler);
}

function deletePreset(index, e) {
    e.stopPropagation();
    if (confirm(`确定要删除预设 "${presets[index].name}" 吗？`)) {
        presets.splice(index, 1);
        renderPresets();
        triggerAutoSave();
    }
}

// --- 音效 ---
let audioCtx = null;
try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (AudioCtx) {
        audioCtx = new AudioCtx();
    }
} catch (e) {
    audioCtx = null;
}

const btnSound = document.getElementById('btnSoundToggle');

function setupSound() {
    updateSoundBtn();
}

function updateSoundBtn() {
    btnSound.textContent = isSoundEnabled ? "🔔 提示音: 开" : "🔕 提示音: 关";
    btnSound.style.opacity = isSoundEnabled ? "1" : "0.6";
    const volumeSlider = document.getElementById('volumeSlider');
    volumeSlider.disabled = !isSoundEnabled;
    volumeSlider.style.opacity = isSoundEnabled ? "1" : "0.5";
}

function toggleSound() {
    isSoundEnabled = !isSoundEnabled;
    updateSoundBtn();
    triggerAutoSave();
}

function playSound(type) {
    if (!isSoundEnabled || soundVolume === 0) return;
    if (!audioCtx) return;
    if (audioCtx.state === 'suspended') audioCtx.resume();

    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);

    if (type === 'success') {
        osc.type = 'sine';
        osc.frequency.setValueAtTime(2000, audioCtx.currentTime);
        gain.gain.setValueAtTime(0, audioCtx.currentTime);
        gain.gain.linearRampToValueAtTime(soundVolume * 6.4, audioCtx.currentTime + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 1.5);
        osc.start(); osc.stop(audioCtx.currentTime + 1.5);
    } else if (type === 'error') {
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(150, audioCtx.currentTime);
        gain.gain.setValueAtTime(soundVolume * 6.4, audioCtx.currentTime);
        gain.gain.linearRampToValueAtTime(0.001, audioCtx.currentTime + 0.5);
        osc.start(); osc.stop(audioCtx.currentTime + 0.5);
    } else {
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(600, audioCtx.currentTime);
        gain.gain.setValueAtTime(soundVolume * 1.6, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.1);
        osc.start(); osc.stop(audioCtx.currentTime + 0.1);
    }
}

// --- 文件上传UI ---
function handleFileSelect(input) {
    const files = Array.from(input.files);
    if (selectedFiles.length + files.length > 14) {
        alert("最多14张");
        return;
    }
    selectedFiles = [...selectedFiles, ...files];
    renderPreviews();
    input.value = '';
}

function renderPreviews() {
    const container = document.getElementById('filePreview');
    container.innerHTML = '';
    document.getElementById('fileCount').textContent = selectedFiles.length;
    selectedFiles.forEach((file, index) => {
        const div = document.createElement('div');
        div.className = 'preview-item';
        const url = URL.createObjectURL(file);
        div.innerHTML = `<img src="${url}" alt="ref" onclick="openModal('${url}')"><div class="preview-remove" onclick="removeFile(${index}, event)">&times;</div>`;
        container.appendChild(div);
    });
}

function removeFile(index, event) {
    event.stopPropagation();
    selectedFiles.splice(index, 1);
    renderPreviews();
}

// --- 图片模态框 (缩放拖拽) ---
const modal = document.getElementById('imageModal');
const modalImg = document.getElementById("modalImg");
let scale = 1;
let isDragging = false;
let startX, startY, translateX = 0, translateY = 0;

function openModal(src) {
    modal.style.display = "block";
    modalImg.src = src;
    scale = 1;
    translateX = 0;
    translateY = 0;
    updateTransform();
}
function closeModal() { modal.style.display = "none"; }

// Zoom with wheel (Zoom to cursor)
modal.addEventListener('wheel', (e) => {
    if (modal.style.display !== 'block') return;
    e.preventDefault();

    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    let newScale = scale * factor;

    if (newScale < 0.1) newScale = 0.1;
    if (newScale > 20) newScale = 20;

    const rect = document.getElementById('modalViewport').getBoundingClientRect();
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    const mx = e.clientX - rect.left - centerX;
    const my = e.clientY - rect.top - centerY;

    const scaleRatio = newScale / scale;
    translateX = mx - (mx - translateX) * scaleRatio;
    translateY = my - (my - translateY) * scaleRatio;

    scale = newScale;
    updateTransform();
});

// Drag
modalImg.addEventListener('mousedown', (e) => {
    if (modal.style.display !== 'block') return;
    isDragging = true;
    startX = e.clientX - translateX;
    startY = e.clientY - translateY;
    modalImg.style.cursor = 'grabbing';
    e.preventDefault();
});

window.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    e.preventDefault();
    translateX = e.clientX - startX;
    translateY = e.clientY - startY;
    updateTransform();
});

window.addEventListener('mouseup', () => {
    isDragging = false;
    modalImg.style.cursor = 'grab';
});

function updateTransform() {
    modalImg.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
}

// 导出到全局（给 inline onclick 用）
window.toggleSound = toggleSound;
window.importLegacyData = importLegacyData;
window.handleLegacyImport = handleLegacyImport;
window.switchApiFormat = switchApiFormat;
window.handleFileSelect = handleFileSelect;
window.removeFile = removeFile;
window.saveCurrentPrompt = saveCurrentPrompt;
window.editPreset = editPreset;
window.deletePreset = deletePreset;
window.startGeneration = startGeneration;
window.toggleSelectionMode = toggleSelectionMode;
window.downloadSelected = downloadSelected;
window.openModal = openModal;
window.closeModal = closeModal;
window.closeDetailModal = closeDetailModal;
window.toggleSystemOverrideSection = toggleSystemOverrideSection;
window.toggleSafetySettingsSection = toggleSafetySettingsSection;
window.updateSliderValue = updateSliderValue;
window.editValue = editValue;