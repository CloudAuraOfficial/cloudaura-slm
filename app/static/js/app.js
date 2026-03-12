const API_BASE = window.location.origin;

const MODEL_COLORS = {
    'phi3:mini': 'phi3',
    'gemma2:2b': 'gemma',
    'qwen2.5:1.5b': 'qwen',
};

const MODEL_DESCRIPTIONS = {
    'phi3:mini': {
        vendor: 'Microsoft',
        params: '3.8B',
        focus: 'Strong reasoning and instruction-following for its size. Best at structured tasks, code generation, and multi-step logic.',
    },
    'gemma2:2b': {
        vendor: 'Google',
        params: '2.6B',
        focus: 'Balanced general-purpose model. Good at summarization, Q&A, and conversational tasks with low resource usage.',
    },
    'qwen2.5:1.5b': {
        vendor: 'Alibaba',
        params: '1.5B',
        focus: 'Ultra-lightweight and fastest inference. Ideal for high-throughput, latency-sensitive applications where speed outweighs depth.',
    },
};

// ── State ──
let benchmarkData = null;
let modelsInfo = [];

// ── Init ──
document.addEventListener('DOMContentLoaded', async () => {
    await checkHealth();
    await loadModels();
    await loadLatestBenchmark();
});

// ── Health check ──
async function checkHealth() {
    try {
        const resp = await fetch(`${API_BASE}/health`);
        const data = await resp.json();
        const dot = document.getElementById('ollama-status-dot');
        const text = document.getElementById('ollama-status-text');
        if (data.ollama_connected) {
            dot.classList.remove('offline');
            text.textContent = 'Ollama Connected';
        } else {
            dot.classList.add('offline');
            text.textContent = 'Ollama Offline';
        }
        const modelCount = document.getElementById('model-count');
        if (modelCount) {
            modelCount.textContent = `${data.models_available.length} Models Loaded`;
        }
    } catch {
        const dot = document.getElementById('ollama-status-dot');
        const text = document.getElementById('ollama-status-text');
        dot.classList.add('offline');
        text.textContent = 'API Offline';
    }
}

// ── Load models ──
async function loadModels() {
    try {
        const resp = await fetch(`${API_BASE}/api/models`);
        modelsInfo = await resp.json();
        populateModelCards(modelsInfo);
        populateModelSelect(modelsInfo);
    } catch {
        console.error('Failed to load models');
    }
}

function populateModelCards(models) {
    const grid = document.getElementById('model-grid');
    if (!grid) return;
    grid.innerHTML = '';

    const summaryMap = {};
    if (benchmarkData && benchmarkData.summaries) {
        benchmarkData.summaries.forEach(s => { summaryMap[s.model] = s; });
    }

    models.forEach(m => {
        const cls = MODEL_COLORS[m.name] || 'phi3';
        const desc = MODEL_DESCRIPTIONS[m.name] || {};
        const bench = summaryMap[m.name];
        const card = document.createElement('div');
        card.className = `model-card ${cls}`;

        let benchHtml = '';
        if (bench) {
            benchHtml = `
            <div class="model-bench">
                <div class="meta-item">
                    <label>Speed</label>
                    <span>${formatNumber(bench.avg_tokens_per_second)} tok/s</span>
                </div>
                <div class="meta-item">
                    <label>TTFT</label>
                    <span>${formatNumber(bench.avg_time_to_first_token_ms)} ms</span>
                </div>
                <div class="meta-item">
                    <label>Avg Latency</label>
                    <span>${formatNumber(bench.avg_total_duration_ms)} ms</span>
                </div>
                <div class="meta-item">
                    <label>Avg Tokens</label>
                    <span>${formatNumber(bench.avg_tokens_generated)}</span>
                </div>
            </div>`;
        }

        card.innerHTML = `
            <div class="model-header">
                <div class="model-name">${m.name}</div>
                <span class="model-tag">${desc.vendor || m.family}</span>
            </div>
            <div class="model-meta">
                <div class="meta-item">
                    <label>Parameters</label>
                    <span>${m.parameter_count}</span>
                </div>
                <div class="meta-item">
                    <label>Size on Disk</label>
                    <span>${m.size_gb} GB</span>
                </div>
                <div class="meta-item">
                    <label>Quantization</label>
                    <span>${m.quantization}</span>
                </div>
                <div class="meta-item">
                    <label>Family</label>
                    <span>${m.family}</span>
                </div>
            </div>
            ${benchHtml}
            <p class="model-desc">${desc.focus || ''}</p>
        `;
        grid.appendChild(card);
    });
}

function populateModelSelect(models) {
    const select = document.getElementById('model-select');
    if (!select) return;
    select.innerHTML = '';
    models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = m.name;
        select.appendChild(opt);
    });
}

// ── Benchmark ──
async function runBenchmark() {
    const btn = document.getElementById('run-benchmark-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Running benchmark...';

    try {
        const resp = await fetch(`${API_BASE}/api/benchmark`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
        });

        if (!resp.ok) {
            const err = await resp.json();
            alert(`Benchmark failed: ${err.detail?.message || 'Unknown error'}`);
            return;
        }

        benchmarkData = await resp.json();
        renderBenchmarkResults(benchmarkData);
    } catch (e) {
        alert(`Benchmark error: ${e.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run Benchmark';
    }
}

async function loadLatestBenchmark() {
    try {
        const resp = await fetch(`${API_BASE}/api/benchmark/latest`);
        if (resp.ok) {
            benchmarkData = await resp.json();
            renderBenchmarkResults(benchmarkData);
        }
    } catch {
        // No previous results
    }
}

function renderBenchmarkResults(data) {
    if (!data || !data.summaries || data.summaries.length === 0) return;

    renderBarChart('tps-chart', data.summaries, 'avg_tokens_per_second', 'tok/s');
    renderBarChart('ttft-chart', data.summaries, 'avg_time_to_first_token_ms', 'ms');
    renderBarChart('latency-chart', data.summaries, 'avg_total_duration_ms', 'ms');
    renderBarChart('tokens-chart', data.summaries, 'avg_tokens_generated', 'tokens');
    renderResultsTable(data.results);
    renderHardwareInfo(data.hardware);

    if (modelsInfo.length > 0) {
        populateModelCards(modelsInfo);
    }

    const ts = document.getElementById('benchmark-timestamp');
    if (ts) {
        ts.textContent = `Last run: ${new Date(data.timestamp).toLocaleString()}`;
    }
}

function renderBarChart(containerId, summaries, metric, unit) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const maxVal = Math.max(...summaries.map(s => s[metric]));
    container.innerHTML = '';

    const chart = document.createElement('div');
    chart.className = 'bar-chart';

    summaries.forEach(s => {
        const pct = maxVal > 0 ? (s[metric] / maxVal) * 100 : 0;
        const cls = MODEL_COLORS[s.model] || 'phi3';
        const group = document.createElement('div');
        group.className = 'bar-group';
        group.innerHTML = `
            <div class="bar ${cls}" style="height: ${Math.max(pct, 2)}%">
                <span class="bar-value">${formatNumber(s[metric])} ${unit}</span>
            </div>
            <span class="bar-label">${s.model}</span>
        `;
        chart.appendChild(group);
    });

    container.appendChild(chart);
}

function renderResultsTable(results) {
    const wrapper = document.getElementById('results-table');
    if (!wrapper) return;

    let html = `<table class="results-table">
        <thead><tr>
            <th>Model</th>
            <th>Prompt</th>
            <th>Tokens</th>
            <th>TTFT (ms)</th>
            <th>Total (ms)</th>
            <th>Tok/s</th>
        </tr></thead><tbody>`;

    results.forEach(r => {
        html += `<tr>
            <td>${r.model}</td>
            <td>${truncate(r.prompt, 50)}</td>
            <td>${r.tokens_generated}</td>
            <td>${formatNumber(r.time_to_first_token_ms)}</td>
            <td>${formatNumber(r.total_duration_ms)}</td>
            <td>${formatNumber(r.tokens_per_second)}</td>
        </tr>`;
    });

    html += '</tbody></table>';
    wrapper.innerHTML = html;
}

function renderHardwareInfo(hw) {
    const el = document.getElementById('hardware-info');
    if (!el || !hw) return;
    el.textContent = `${hw.platform} ${hw.architecture} | ${hw.cpu_count} CPUs | ${hw.memory_gb} GB RAM | ${hw.gpu}`;
}

// ── Try it ──
async function sendPrompt() {
    const model = document.getElementById('model-select').value;
    const prompt = document.getElementById('prompt-input').value.trim();
    if (!prompt) return;

    const btn = document.getElementById('send-btn');
    const responseBox = document.getElementById('response-output');
    const metaBox = document.getElementById('response-meta');

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Generating...';
    responseBox.textContent = 'Thinking...';
    metaBox.innerHTML = '';

    try {
        const resp = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model,
                messages: [{ role: 'user', content: prompt }],
                temperature: 0.7,
                max_tokens: 512,
            }),
        });

        if (!resp.ok) {
            const err = await resp.json();
            responseBox.textContent = `Error: ${err.detail?.message || 'Unknown error'}`;
            return;
        }

        const data = await resp.json();
        responseBox.textContent = data.content;
        metaBox.innerHTML = `
            <span>Tokens: ${data.tokens_generated}</span>
            <span>Speed: ${formatNumber(data.tokens_per_second)} tok/s</span>
            <span>Duration: ${formatNumber(data.total_duration_ms)} ms</span>
        `;
    } catch (e) {
        responseBox.textContent = `Error: ${e.message}`;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Send';
    }
}

// ── Utilities ──
function formatNumber(n) {
    return typeof n === 'number' ? n.toLocaleString(undefined, { maximumFractionDigits: 1 }) : n;
}

function truncate(s, len) {
    return s.length > len ? s.substring(0, len) + '...' : s;
}
