<script>
  import { tick } from 'svelte'

  // ── state ──────────────────────────────────────────────────────────────────
  let messages = []        // { role: 'user'|'assistant', text: string }
  let input    = ''
  let loading  = false
  let error    = null

  // sidebar controls
  let temperature = 0.8
  let topK        = 50
  let maxTokens   = 200

  let chatEl   // bound to the chat scroll container
  let inputEl  // bound to the textarea

  // ── helpers ────────────────────────────────────────────────────────────────
  async function scrollToBottom() {
    await tick()
    if (chatEl) chatEl.scrollTop = chatEl.scrollHeight
  }

  function handleKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  // ── send ───────────────────────────────────────────────────────────────────
  async function send() {
    const text = input.trim()
    if (!text || loading) return

    error   = null
    input   = ''
    loading = true

    messages = [...messages, { role: 'user', text }]
    messages = [...messages, { role: 'assistant', text: '' }]
    await scrollToBottom()

    try {
      const res = await fetch('/generate', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt:      text,
          temperature: parseFloat(temperature),
          top_k:       parseInt(topK),
          max_tokens:  parseInt(maxTokens),
        }),
      })

      if (!res.ok) throw new Error(`Server error: ${res.status}`)

      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let   buf     = ''
      let   finished = false

      while (!finished) {
        const { done, value } = await reader.read()
        if (done) break

        buf += decoder.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop()  // keep incomplete line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6).trim()
          if (data === '[DONE]') { finished = true; break }

          try {
            const { token } = JSON.parse(data)
            messages = messages.map((m, i) =>
              i === messages.length - 1 ? { ...m, text: m.text + token } : m
            )
            await scrollToBottom()
          } catch (_) {}
        }
      }
    } catch (err) {
      error = err.message
      messages = messages.slice(0, -1)
    } finally {
      loading = false
      await tick()
      await tick()  // double tick ensures disabled=false is flushed before focus
      inputEl?.focus()
    }
  }

  function clearChat() {
    messages = []
    error    = null
  }
</script>

<!-- ── markup ────────────────────────────────────────────────────────────── -->
<div class="shell">

  <!-- sidebar -->
  <aside class="sidebar">
    <div class="logo-block">
      <img src="/light.png" alt="Marcella" class="logo" onerror={(e) => e.currentTarget.style.display='none'} />
    </div>

    <div class="controls">
      <label class="ctrl-label" for="temp">
        Temperature
        <span class="ctrl-value">{temperature}</span>
      </label>
      <input id="temp" type="range" min="0.1" max="1.0" step="0.05"
             bind:value={temperature} class="slider" />

      <label class="ctrl-label" for="topk">
        Top-k
        <span class="ctrl-value">{topK}</span>
      </label>
      <input id="topk" type="range" min="1" max="500" step="1"
             bind:value={topK} class="slider" />

      <label class="ctrl-label" for="maxtok">
        Max tokens
        <span class="ctrl-value">{maxTokens}</span>
      </label>
      <input id="maxtok" type="range" min="16" max="1024" step="8"
             bind:value={maxTokens} class="slider" />
    </div>

    <div class="sidebar-footer">
      <button class="clear-btn" onclick={clearChat}>Clear chat</button>
      <span class="model-tag">60M · 32K vocab</span>
    </div>
  </aside>

  <!-- main area -->
  <main class="main">

    <!-- chat messages -->
    <div class="chat" bind:this={chatEl}>
      {#if messages.length === 0}
        <div class="empty">
          <p class="empty-title">Start a conversation</p>
          <p class="empty-sub">Marcella is a small language model trained from scratch.</p>
        </div>
      {/if}

      {#each messages as msg, i}
        <div class="bubble-row {msg.role}">
          <div class="bubble {msg.role}">
            {msg.text}{#if msg.role === 'assistant' && loading && i === messages.length - 1}<span class="cursor-blink">▍</span>{/if}
          </div>
        </div>
      {/each}

      {#if error}
        <div class="error-row">
          <span class="error-msg">⚠ {error}</span>
        </div>
      {/if}
    </div>

    <!-- input bar -->
    <div class="input-bar">
      <textarea
        bind:this={inputEl}
        bind:value={input}
        onkeydown={handleKeydown}
        placeholder="Send a message…"
        rows="1"
        disabled={loading}
        class="input-textarea"
      ></textarea>
      <button class="send-btn" onclick={send} disabled={loading || !input.trim()}>
        {#if loading}
          <svg class="spin" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 12a9 9 0 11-6.219-8.56"/>
          </svg>
        {:else}
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="22" y1="2" x2="11" y2="13"/>
            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        {/if}
      </button>
    </div>

  </main>
</div>

<!-- ── styles ─────────────────────────────────────────────────────────────── -->
<style>
  :global(*, *::before, *::after) { box-sizing: border-box; margin: 0; padding: 0; }
  :global(html, body) {
    height: 100%;
    background: #f5f3ef;
    color: #1a1a1a;
    font-family: 'DM Sans', sans-serif;
  }

  /* ── layout ── */
  .shell {
    display: flex;
    height: 100vh;
    overflow: hidden;
  }

  /* ── sidebar ── */
  .sidebar {
    width: 240px;
    min-width: 240px;
    background: #faf9f7;
    border-right: 1px solid #e4dfd6;
    display: flex;
    flex-direction: column;
    padding: 24px 20px;
    gap: 32px;
  }

  .logo-block {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .logo {
    width: 100%;
    max-width: 160px;
    height: auto;
    object-fit: contain;
  }

  .controls {
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .ctrl-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    font-weight: 500;
    color: #5a5550;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .ctrl-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #1a1a1a;
  }

  .slider {
    width: 100%;
    -webkit-appearance: none;
    appearance: none;
    height: 3px;
    border-radius: 2px;
    background: #ddd8d0;
    outline: none;
    cursor: pointer;
    margin-top: 4px;
  }

  .slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #1a1a1a;
    cursor: pointer;
    transition: transform 0.15s;
  }
  .slider::-webkit-slider-thumb:hover { transform: scale(1.2); }

  .sidebar-footer {
    margin-top: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .clear-btn {
    padding: 8px 12px;
    border: 1px solid #ddd8d0;
    background: transparent;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    color: #5a5550;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
  }
  .clear-btn:hover { background: #ede9e2; color: #1a1a1a; }

  .model-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #aaa49c;
    text-align: center;
  }

  /* ── main ── */
  .main {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    background: #f5f3ef;
  }

  /* ── chat area ── */
  .chat {
    flex: 1;
    overflow-y: auto;
    padding: 32px 24px 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    scroll-behavior: smooth;
  }

  .chat::-webkit-scrollbar { width: 5px; }
  .chat::-webkit-scrollbar-track { background: transparent; }
  .chat::-webkit-scrollbar-thumb { background: #ddd8d0; border-radius: 4px; }

  /* ── empty state ── */
  .empty {
    margin: auto;
    text-align: center;
    max-width: 320px;
  }
  .empty-title {
    font-family: 'Lora', serif;
    font-size: 1.3rem;
    font-weight: 500;
    color: #3a3530;
    margin-bottom: 8px;
  }
  .empty-sub {
    font-size: 0.88rem;
    color: #8a857e;
    line-height: 1.6;
  }

  /* ── bubbles ── */
  .bubble-row {
    display: flex;
    animation: fadeUp 0.2s ease both;
  }
  .bubble-row.user      { justify-content: flex-end; }
  .bubble-row.assistant { justify-content: flex-start; }

  .bubble {
    max-width: min(72%, 640px);
    padding: 12px 16px;
    border-radius: 16px;
    font-size: 0.93rem;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .bubble.user {
    background: #1a1a1a;
    color: #f5f3ef;
    border-bottom-right-radius: 4px;
    font-family: 'DM Sans', sans-serif;
  }

  .bubble.assistant {
    background: #ffffff;
    color: #1a1a1a;
    border: 1px solid #e4dfd6;
    border-bottom-left-radius: 4px;
    font-family: 'Lora', serif;
    font-size: 0.95rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  }

  /* ── streaming cursor ── */
  .cursor-blink {
    display: inline-block;
    animation: blink 0.9s step-end infinite;
    color: #8a857e;
    font-weight: 300;
  }

  @keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* ── error ── */
  .error-row {
    display: flex;
    justify-content: center;
  }
  .error-msg {
    background: #fff0f0;
    border: 1px solid #f5c6c6;
    color: #c0392b;
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 0.83rem;
  }

  /* ── input bar ── */
  .input-bar {
    padding: 16px 24px 20px;
    display: flex;
    align-items: flex-end;
    gap: 10px;
    background: #f5f3ef;
    border-top: 1px solid #e4dfd6;
  }

  .input-textarea {
    flex: 1;
    padding: 12px 16px;
    border: 1px solid #ddd8d0;
    border-radius: 12px;
    background: #ffffff;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.93rem;
    color: #1a1a1a;
    resize: none;
    outline: none;
    line-height: 1.5;
    max-height: 180px;
    overflow-y: auto;
    transition: border-color 0.15s, box-shadow 0.15s;
  }

  .input-textarea::placeholder { color: #b0aba3; }
  .input-textarea:focus {
    border-color: #aaa49c;
    box-shadow: 0 0 0 3px rgba(26,26,26,0.06);
  }
  .input-textarea:disabled { opacity: 0.6; cursor: not-allowed; }

  .send-btn {
    width: 42px;
    height: 42px;
    border-radius: 10px;
    border: none;
    background: #1a1a1a;
    color: #f5f3ef;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    flex-shrink: 0;
    transition: background 0.15s, transform 0.1s;
  }
  .send-btn:hover:not(:disabled) { background: #333; }
  .send-btn:active:not(:disabled) { transform: scale(0.95); }
  .send-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .spin {
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
</style>