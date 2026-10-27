// ==========================================================================
// DeepDive LangGraph State Visualizer - Frontend Engine
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
  // Application State Store
  const state = {
    original_query: '',
    sub_questions: [],
    is_sufficient: false,
    evidence_bank: [],
    audit_logs: [],
    research_loops: 0,
    final_report: '',
    isRunning: false,
    activeNode: null,
    timerInterval: null,
    startTime: null,
  };

  // DOM Elements
  const inputQuery = document.getElementById('input-query');
  const selectProvider = document.getElementById('select-model-provider');
  const btnRun = document.getElementById('btn-run');
  const btnRunIcon = document.getElementById('btn-run-icon');
  const btnRunText = document.getElementById('btn-run-text');
  const btnSample = document.getElementById('btn-sample');
  const liveTimer = document.getElementById('live-timer');
  const lblStatus = document.getElementById('lbl-status');

  // Tabs
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  // Metrics
  const metricSufficiency = document.getElementById('metric-sufficiency');
  const metricLoops = document.getElementById('metric-loops');
  const metricSubQ = document.getElementById('metric-subq-count');
  const metricEvidence = document.getElementById('metric-evidence-count');
  const lblCurrentQuery = document.getElementById('lbl-current-query');
  const lblLoopCount = document.getElementById('lbl-loop-count');
  const lblLoopStatus = document.getElementById('lbl-loop-status');
  const loopIndicator = document.getElementById('loop-indicator');

  // Badges
  const badgeSubQ = document.getElementById('badge-subq-count');
  const badgeEvidence = document.getElementById('badge-evidence-count');
  const badgeLogs = document.getElementById('badge-logs-count');

  // Content Containers
  const questionsContainer = document.getElementById('questions-container');
  const evidenceContainer = document.getElementById('evidence-container');
  const filterEvidence = document.getElementById('filter-evidence');
  const evidenceSummaryLbl = document.getElementById('evidence-summary-lbl');
  const terminalLogs = document.getElementById('terminal-logs');
  const reportRendered = document.getElementById('report-rendered');
  const jsonViewer = document.getElementById('json-viewer');

  // Node Elements Mapping
  const nodes = {
    start: document.getElementById('node-start'),
    planner: document.getElementById('node-planner'),
    researcher: document.getElementById('node-researcher'),
    verifier: document.getElementById('node-verifier'),
    synthesizer: document.getElementById('node-synthesizer'),
    end: document.getElementById('node-end'),
  };

  const edges = {
    'start-planner': document.getElementById('edge-start-planner'),
    'planner-researcher': document.getElementById('edge-planner-researcher'),
    'researcher-verifier': document.getElementById('edge-researcher-verifier'),
    'verifier-synthesizer': document.getElementById('edge-verifier-synthesizer'),
    'synthesizer-end': document.getElementById('edge-synthesizer-end'),
  };

  // --- Initial System Status Check ---
  async function checkSystemStatus() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        const dotGroq = document.getElementById('dot-groq');
        const dotGemini = document.getElementById('dot-gemini');
        const dotTavily = document.getElementById('dot-tavily');

        if (data.groq_configured) dotGroq.classList.add('active');
        if (data.google_configured) dotGemini.classList.add('active');
        if (data.tavily_configured) dotTavily.classList.add('active');

        if (data.default_provider) {
          selectProvider.value = data.default_provider;
        }
      }
    } catch (e) {
      console.warn('Backend not responding yet or status check failed', e);
    }
  }
  checkSystemStatus();

  // --- Tab Switching Logic ---
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetPane = document.getElementById(btn.dataset.tab);
      if (targetPane) targetPane.classList.add('active');
    });
  });

  // --- Chip Click Handlers ---
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      inputQuery.value = chip.dataset.query;
      inputQuery.focus();
    });
  });

  // --- Pipeline Node Animation ---
  function setActiveNode(nodeName) {
    state.activeNode = nodeName;

    // Remove active pulse from all nodes
    Object.keys(nodes).forEach(key => {
      nodes[key].classList.remove('active');
    });

    // Remove moving beam animation from ALL edges
    Object.keys(edges).forEach(key => {
      edges[key].classList.remove('animated');
    });

    if (nodeName && nodes[nodeName]) {
      nodes[nodeName].classList.add('active');
    }

    const order = ['start', 'planner', 'researcher', 'verifier', 'synthesizer', 'end'];
    const edgeOrder = [
      { to: 'planner', edge: 'start-planner' },
      { to: 'researcher', edge: 'planner-researcher' },
      { to: 'verifier', edge: 'researcher-verifier' },
      { to: 'synthesizer', edge: 'verifier-synthesizer' },
      { to: 'end', edge: 'synthesizer-end' }
    ];

    const activeIdx = order.indexOf(nodeName);

    if (activeIdx >= 0) {
      // Mark traversed nodes as completed
      for (let i = 0; i <= activeIdx; i++) {
        nodes[order[i]].classList.add('completed');
      }

      // Mark previously traversed edges as completed (solid line, NO moving beam)
      for (let i = 0; i < activeIdx; i++) {
        const edgeKey = edgeOrder[i]?.edge;
        if (edgeKey && edges[edgeKey]) {
          edges[edgeKey].classList.add('completed');
          edges[edgeKey].classList.remove('animated');
        }
      }

      // ONLY the single incoming edge for the current transition gets the moving beam during running
      if (activeIdx > 0 && nodeName !== 'end') {
        const currentEdgeKey = edgeOrder[activeIdx - 1]?.edge;
        if (currentEdgeKey && edges[currentEdgeKey]) {
          edges[currentEdgeKey].classList.add('active', 'animated');
        }
      }
    }

    updateLoopIndicator();
  }

  // Stop ALL moving animations across the entire topology when flow ends
  function completePipelineVisuals() {
    // 1. Immediately remove all moving animations and active beam particles from edges
    Object.keys(edges).forEach(key => {
      edges[key].classList.remove('animated');
      edges[key].classList.remove('active');
      edges[key].classList.add('completed');
    });

    // 2. Remove pulsing active animations from all nodes, mark all as completed
    Object.keys(nodes).forEach(key => {
      nodes[key].classList.remove('active');
      nodes[key].classList.add('completed');
    });

    // 3. Settle conditional feedback indicator
    if (loopIndicator) {
      if (state.research_loops > 0) {
        loopIndicator.className = 'conditional-edge-track passed';
        if (lblLoopCount) lblLoopCount.textContent = `Loop: ${state.research_loops} / 2`;
        if (lblLoopStatus) lblLoopStatus.textContent = 'Loop Resolved';
      } else {
        loopIndicator.className = 'conditional-edge-track passed-subtle';
        if (lblLoopCount) lblLoopCount.textContent = 'Loop: 0 / 2';
        if (lblLoopStatus) lblLoopStatus.textContent = 'Direct Path';
      }
    }
  }

  function resetPipelineVisuals() {
    Object.keys(nodes).forEach(key => {
      nodes[key].classList.remove('active', 'completed');
    });
    Object.keys(edges).forEach(key => {
      edges[key].classList.remove('active', 'animated', 'completed');
    });
    if (loopIndicator) {
      loopIndicator.className = 'conditional-edge-track idle';
    }
    if (lblLoopCount) lblLoopCount.textContent = 'Loop: 0 / 2';
    if (lblLoopStatus) lblLoopStatus.textContent = 'Dormant';
  }

  function updateLoopIndicator() {
    if (!loopIndicator) return;

    // If completely idle / reset, stay hidden so it doesn't appear over the topology
    if (!state.isRunning && state.research_loops === 0 && !state.is_sufficient) {
      loopIndicator.className = 'conditional-edge-track idle';
      if (lblLoopCount) lblLoopCount.textContent = 'Loop: 0 / 2';
      if (lblLoopStatus) lblLoopStatus.textContent = 'Dormant';
      return;
    }

    if (state.research_loops > 0 && !state.is_sufficient) {
      // Actively looping back from Verifier to Researcher
      loopIndicator.className = 'conditional-edge-track active-loop';
      if (lblLoopCount) lblLoopCount.textContent = `Loop: ${state.research_loops} / 2`;
      if (lblLoopStatus) lblLoopStatus.textContent = 'Re-Searching Gaps';
    } else if (state.is_sufficient) {
      if (state.research_loops > 0) {
        loopIndicator.className = 'conditional-edge-track passed';
        if (lblLoopCount) lblLoopCount.textContent = `Loop: ${state.research_loops} / 2`;
        if (lblLoopStatus) lblLoopStatus.textContent = 'Loop Resolved';
      } else {
        loopIndicator.className = 'conditional-edge-track passed-subtle';
        if (lblLoopCount) lblLoopCount.textContent = 'Loop: 0 / 2';
        if (lblLoopStatus) lblLoopStatus.textContent = 'Direct Path';
      }
    } else if (state.isRunning) {
      loopIndicator.className = 'conditional-edge-track waiting';
      if (lblLoopCount) lblLoopCount.textContent = 'Loop: 0 / 2';
      if (lblLoopStatus) lblLoopStatus.textContent = 'Pending Audit';
    }
  }

  // --- UI Update Functions ---
  function updateUI() {
    // 1. Overview Metrics
    lblCurrentQuery.textContent = state.original_query || 'None';
    metricLoops.textContent = `${state.research_loops} / 2`;
    lblLoopCount.textContent = `Loop: ${state.research_loops} / 2`;
    metricSubQ.textContent = state.sub_questions.length;
    metricEvidence.textContent = state.evidence_bank.length;

    badgeSubQ.textContent = state.sub_questions.length;
    badgeEvidence.textContent = state.evidence_bank.length;
    badgeLogs.textContent = state.audit_logs.length;

    if (state.is_sufficient) {
      metricSufficiency.textContent = 'Verified Sufficient';
      metricSufficiency.className = 'metric-pill success';
    } else if (state.research_loops > 0) {
      metricSufficiency.textContent = 'Needs Research Loop';
      metricSufficiency.className = 'metric-pill pending';
    } else {
      metricSufficiency.textContent = 'Pending';
      metricSufficiency.className = 'metric-pill pending';
    }

    // 2. Sub-Questions Tab
    if (state.sub_questions.length === 0) {
      questionsContainer.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">❓</div>
          <div>No sub-questions generated yet.</div>
        </div>`;
    } else {
      questionsContainer.innerHTML = state.sub_questions.map((q, idx) => `
        <div class="question-item-card">
          <div class="question-index-badge">${idx + 1}</div>
          <div class="question-text">${escapeHtml(q)}</div>
        </div>
      `).join('');
    }

    // 3. Evidence Bank Tab
    renderEvidenceCards();

    // 4. Report Tab
    if (state.final_report) {
      if (window.marked && typeof window.marked.parse === 'function') {
        reportRendered.innerHTML = marked.parse(state.final_report);
      } else {
        reportRendered.innerHTML = `<pre style="white-space: pre-wrap; font-family: inherit;">${escapeHtml(state.final_report)}</pre>`;
      }
    } else {
      reportRendered.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">📄</div>
          <div>The final report will be compiled here once the Synthesizer node completes.</div>
        </div>`;
    }

    // 5. Raw JSON Tab
    jsonViewer.textContent = JSON.stringify(state, null, 2);

    // 6. Update Conditional Feedback Edge
    updateLoopIndicator();
  }

  function renderEvidenceCards() {
    const filter = (filterEvidence.value || '').toLowerCase().trim();
    const filtered = state.evidence_bank.filter(item => {
      if (!filter) return true;
      return (
        (item.title && item.title.toLowerCase().includes(filter)) ||
        (item.content && item.content.toLowerCase().includes(filter)) ||
        (item.sub_question && item.sub_question.toLowerCase().includes(filter))
      );
    });

    evidenceSummaryLbl.textContent = `Showing ${filtered.length} of ${state.evidence_bank.length} items`;

    if (filtered.length === 0) {
      evidenceContainer.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1;">
          <div class="empty-icon">📚</div>
          <div>${state.evidence_bank.length === 0 ? 'No evidence chunks collected in state yet.' : 'No evidence matches your search keyword.'}</div>
        </div>`;
      return;
    }

    evidenceContainer.innerHTML = filtered.map(item => `
      <div class="evidence-card">
        <div class="evidence-header">
          <div class="evidence-title">${escapeHtml(item.title || 'Untitled Source')}</div>
          ${item.source_url ? `
            <a href="${escapeHtml(item.source_url)}" target="_blank" rel="noopener noreferrer" class="source-link-btn">
              🔗 Visit Source
            </a>` : ''}
        </div>
        ${item.sub_question ? `<div class="evidence-subq-tag">📌 ${escapeHtml(item.sub_question)}</div>` : ''}
        <div class="evidence-content">${escapeHtml(item.content || '')}</div>
      </div>
    `).join('');
  }

  filterEvidence.addEventListener('input', renderEvidenceCards);

  // --- Append Log Line ---
  function appendLog(logMsg) {
    const timeStr = new Date().toTimeString().split(' ')[0];
    let prefixClass = 'system';
    if (logMsg.startsWith('[Planner]')) prefixClass = 'planner';
    else if (logMsg.startsWith('[Researcher]')) prefixClass = 'researcher';
    else if (logMsg.startsWith('[Verifier]')) prefixClass = 'verifier';
    else if (logMsg.startsWith('[Synthesizer]')) prefixClass = 'synthesizer';

    const line = document.createElement('div');
    line.className = 'log-line';
    line.innerHTML = `
      <span class="log-time">[${timeStr}]</span>
      <span class="log-msg ${prefixClass}">${escapeHtml(logMsg)}</span>
    `;
    terminalLogs.appendChild(line);
    terminalLogs.scrollTop = terminalLogs.scrollHeight;
  }

  // --- Copy Buttons ---
  document.getElementById('btn-copy-logs').addEventListener('click', () => {
    navigator.clipboard.writeText(state.audit_logs.join('\n'));
    showToast('Audit logs copied to clipboard!');
  });

  document.getElementById('btn-clear-logs').addEventListener('click', () => {
    terminalLogs.innerHTML = '';
  });

  document.getElementById('btn-copy-report').addEventListener('click', () => {
    if (!state.final_report) return;
    navigator.clipboard.writeText(state.final_report);
    showToast('Final report copied as Markdown!');
  });

  document.getElementById('btn-print-report').addEventListener('click', () => {
    window.print();
  });

  document.getElementById('btn-copy-json').addEventListener('click', () => {
    navigator.clipboard.writeText(JSON.stringify(state, null, 2));
    showToast('ResearchState JSON copied!');
  });

  // --- Load Sample State ---
  btnSample.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/sample-state');
      if (res.ok) {
        const sample = await res.json();
        Object.assign(state, sample);
        inputQuery.value = state.original_query;
        resetPipelineVisuals();
        completePipelineVisuals();

        terminalLogs.innerHTML = '';
        state.audit_logs.forEach(log => appendLog(log));

        updateUI();
        showToast('Sample ResearchState loaded successfully!');
      }
    } catch (e) {
      console.error(e);
      showToast('Failed to load sample state');
    }
  });

  // --- Start Research Stream ---
  btnRun.addEventListener('click', () => {
    const query = inputQuery.value.trim();
    if (!query) {
      inputQuery.focus();
      return;
    }

    if (state.isRunning) return;
    startExecution(query);
  });

  function startExecution(query) {
    state.isRunning = true;
    state.original_query = query;
    state.sub_questions = [];
    state.is_sufficient = false;
    state.evidence_bank = [];
    state.audit_logs = [];
    state.research_loops = 0;
    state.final_report = '';

    btnRun.disabled = true;
    btnRunIcon.textContent = '⏳';
    btnRunText.textContent = 'Agent Running...';
    lblStatus.textContent = 'Executing Graph';

    resetPipelineVisuals();
    setActiveNode('start');

    terminalLogs.innerHTML = '';
    appendLog(`[System] Initializing DeepDive LangGraph with query: "${query}"`);

    // Start Timer
    state.startTime = Date.now();
    clearInterval(state.timerInterval);
    state.timerInterval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - state.startTime) / 1000);
      liveTimer.textContent = `Running: ${elapsed}s`;
    }, 1000);

    const provider = selectProvider.value;
    const url = `/api/stream?query=${encodeURIComponent(query)}&provider=${encodeURIComponent(provider)}`;

    const eventSource = new EventSource(url);

    eventSource.addEventListener('node_start', (e) => {
      const data = JSON.parse(e.data);
      setActiveNode(data.node);
      appendLog(`[Graph] Entering node: '${data.node}'`);
    });

    eventSource.addEventListener('log', (e) => {
      const data = JSON.parse(e.data);
      if (!state.audit_logs.includes(data.log)) {
        state.audit_logs.push(data.log);
      }
      appendLog(data.log);
    });

    eventSource.addEventListener('state_update', (e) => {
      const data = JSON.parse(e.data);
      if (data.state) {
        Object.assign(state, data.state);
        updateUI();
      }
    });

    eventSource.addEventListener('complete', (e) => {
      const data = JSON.parse(e.data);
      if (data.final_state) {
        Object.assign(state, data.final_state);
      }
      completePipelineVisuals();
      finishExecution();
      eventSource.close();
      appendLog('[Graph] LangGraph execution reached END state.');
      showToast('Research synthesis complete!');
    });

    eventSource.addEventListener('error', (e) => {
      console.error('SSE Error:', e);
      appendLog('[Error] Stream disconnected or error occurred.');
      completePipelineVisuals();
      finishExecution();
      eventSource.close();
    });
  }

  function finishExecution() {
    state.isRunning = false;
    btnRun.disabled = false;
    btnRunIcon.textContent = '▶';
    btnRunText.textContent = 'Run DeepDive';
    lblStatus.textContent = 'Complete';
    clearInterval(state.timerInterval);
    const total = Math.floor((Date.now() - state.startTime) / 1000);
    liveTimer.textContent = `Completed in ${total}s`;
    completePipelineVisuals();
    updateUI();
  }

  // Helper Escape HTML
  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Toast Notification
  function showToast(message) {
    const existing = document.querySelector('.toast-banner');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'toast-banner';
    toast.style.cssText = `
      position: fixed;
      bottom: 24px;
      right: 28px;
      background: rgba(15, 23, 42, 0.95);
      border: 1px solid var(--accent-primary);
      box-shadow: 0 10px 25px rgba(0,0,0,0.5), 0 0 15px var(--accent-primary-glow);
      color: #fff;
      padding: 12px 20px;
      border-radius: 10px;
      font-size: 0.88rem;
      z-index: 9999;
      animation: fadeIn 0.2s ease-out;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3200);
  }
});
