// quick-screens.jsx — ConversationView (merged Chat + OpenAI), Composer, Segmented, MetricsTab
/* global React, MONO, DISP, hexA, cat, Lbl, Dot, ChipRow, Btn, Card, Field, SectionHeader */

function Segmented({ t, options, value, onPick, full, small }) {
  return (
    <div style={{ display: 'inline-flex', gap: 3, padding: 3, borderRadius: 999, background: t.surfaceContainer, border: `1px solid ${t.outlineVariant}`, width: full ? '100%' : 'auto', boxSizing: 'border-box' }}>
      {options.map(o => {
        const on = o.key === value;
        return (
          <button key={o.key} onClick={() => onPick(o.key)} style={{
            flex: full ? 1 : 'none', cursor: 'pointer', border: 'none', borderRadius: 999,
            padding: small ? '5px 11px' : '9px 14px', fontFamily: MONO, fontSize: small ? 11 : 12.5,
            fontWeight: on ? 700 : 500, whiteSpace: 'nowrap',
            background: on ? t.primary : 'transparent', color: on ? t.onPrimary : t.onSurfaceVar,
            boxShadow: on ? `0 2px 8px ${hexA(t.primary, 0.3)}` : 'none', transition: 'all .14s',
          }}>{o.label}</button>
        );
      })}
    </div>
  );
}

function buildJson(system, messages) {
  const arr = [];
  if (system && system.trim()) arr.push({ role: 'system', content: system.trim() });
  messages.forEach(m => arr.push({ role: m.role, content: m.text }));
  if (arr.length === 0) arr.push({ role: 'user', content: 'Why run an LLM locally?' });
  return JSON.stringify(arr, null, 2);
}

function AttChip({ t, att, onRemove }) {
  const glyph = att.kind === 'image' ? '▣' : att.kind === 'camera' ? '◎' : '▤';
  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: 7, padding: '5px 8px 5px 7px', borderRadius: 9, background: t.surfaceContainerHigh, border: `1px solid ${t.outlineVariant}` }}>
      <span style={{ width: 22, height: 22, borderRadius: 6, background: `linear-gradient(135deg, ${hexA(t.primary, 0.35)}, ${hexA(t.tertiary, 0.35)})`, color: t.onSurface, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: MONO, fontSize: 11 }}>{glyph}</span>
      <span style={{ fontFamily: MONO, fontSize: 11, color: t.onSurface, maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{att.name}</span>
      {onRemove && <button onClick={onRemove} style={{ border: 'none', background: 'transparent', cursor: 'pointer', color: t.onSurfaceVar, fontSize: 12, padding: 0, lineHeight: 1 }}>✕</button>}
    </div>
  );
}

function Bubble({ t, msg, streaming }) {
  const user = msg.role === 'user';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: user ? 'flex-end' : 'flex-start' }}>
      <div style={{ fontFamily: MONO, fontSize: 9, fontWeight: 700, letterSpacing: '0.1em', color: user ? t.primary : t.tertiary, marginBottom: 4 }}>
        {user ? 'USER ▸' : '◂ ASSISTANT'}
      </div>
      {user && msg.atts && msg.atts.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, justifyContent: 'flex-end', marginBottom: 5 }}>
          {msg.atts.map((a, i) => <AttChip key={i} t={t} att={a} />)}
        </div>
      )}
      <div style={{
        maxWidth: '88%', padding: '10px 13px', borderRadius: 14,
        borderTopRightRadius: user ? 4 : 14, borderTopLeftRadius: user ? 14 : 4,
        background: user ? t.primary : t.surfaceContainerHigh,
        color: user ? t.onPrimary : t.onSurface,
        border: user ? 'none' : `1px solid ${t.outlineVariant}`,
        fontFamily: MONO, fontSize: 13, lineHeight: 1.55, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
      }}>
        {msg.text || (streaming ? '' : '…')}
        {streaming && <span className="qcaret" style={{ color: user ? t.onPrimary : t.primary }}>▋</span>}
      </div>
    </div>
  );
}

function EmptyState({ t, modelId, backend, onSuggest }) {
  const suggestions = [
    'Why run an LLM locally?',
    'Summarize this file for me',
    'Stream a haiku about flash storage',
  ];
  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', padding: '24px 24px' }}>
      <div style={{
        width: 52, height: 52, borderRadius: 16, marginBottom: 16,
        background: `linear-gradient(135deg, ${t.primary}, ${t.tertiary})`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        boxShadow: `0 6px 18px ${hexA(t.primary, 0.35)}`,
      }}><span style={{ fontSize: 26 }}>☄</span></div>
      <div style={{ fontFamily: DISP, fontSize: 21, fontWeight: 700, color: t.onSurface, letterSpacing: '-0.02em' }}>Ready when you are.</div>
      <div style={{ fontFamily: MONO, fontSize: 11, color: t.onSurfaceVar, marginTop: 6 }}>on-device · {modelId} · {backend}</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 22, width: '100%', maxWidth: 300 }}>
        {suggestions.map(s => (
          <button key={s} onClick={() => onSuggest(s)} style={{
            fontFamily: MONO, fontSize: 12, color: t.onSurface, cursor: 'pointer', textAlign: 'left',
            padding: '11px 13px', borderRadius: 12, border: `1px solid ${t.outlineVariant}`, background: t.surface,
            display: 'flex', alignItems: 'center', gap: 9,
          }}>
            <span style={{ color: t.primary }}>›</span>{s}
          </button>
        ))}
      </div>
    </div>
  );
}

function Composer({ t, ctx }) {
  const { composer, setComposer, onSend, streaming, onStop, attachments, removeAttach,
    attachMenuOpen, setAttachMenuOpen, onAttach, model, loadedLabel, loadStatus, onOpenSettings } = ctx;
  const desc = cat.resolve(model.family, model.runtime, model.backend);
  const mm = cat.multimodal(desc);
  const canSend = composer.trim().length > 0 && !streaming;
  const imgWarn = attachments.some(a => a.kind !== 'file') && !mm;

  return (
    <div style={{ position: 'relative', padding: '10px 12px 12px', borderTop: `1px solid ${t.outlineVariant}`, background: t.bg }}>
      {attachments.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
          {attachments.map((a, i) => <AttChip key={i} t={t} att={a} onRemove={() => removeAttach(i)} />)}
        </div>
      )}
      <div style={{ borderRadius: 18, border: `1px solid ${t.outlineVariant}`, background: t.surface, padding: 8, boxShadow: t.shadow }}>
        <textarea rows={2} value={composer} placeholder="Message Quick.AI…"
          onChange={(e) => setComposer(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey && canSend) { e.preventDefault(); onSend(); } }}
          style={{ width: '100%', boxSizing: 'border-box', border: 'none', background: 'transparent', outline: 'none', resize: 'none', fontFamily: MONO, fontSize: 13.5, color: t.onSurface, padding: '6px 8px', lineHeight: 1.5 }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
          <div style={{ position: 'relative', flexShrink: 0 }}>
            <button onClick={() => setAttachMenuOpen(o => !o)} style={{
              width: 34, height: 34, borderRadius: 10, border: `1px solid ${t.outlineVariant}`,
              background: attachMenuOpen ? t.surfaceContainerHigh : 'transparent', color: t.onSurfaceVar,
              cursor: 'pointer', fontSize: 17, display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>＋</button>
            <AttachMenu t={t} open={attachMenuOpen} onClose={() => setAttachMenuOpen(false)} onAttach={onAttach} multimodal={mm} />
          </div>
          <button onClick={onOpenSettings} style={{
            display: 'flex', alignItems: 'center', gap: 6, minWidth: 0, maxWidth: 168,
            padding: '7px 11px', borderRadius: 10, border: `1px solid ${t.outlineVariant}`,
            background: t.surfaceContainer, cursor: 'pointer',
          }}>
            <Dot color={loadStatus === 'loaded' ? t.success : t.outline} size={7} />
            <span style={{ fontFamily: MONO, fontSize: 11.5, fontWeight: 600, color: t.onSurface, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{loadStatus === 'loaded' ? loadedLabel : (desc ? desc.id : model.family)}</span>
            <span style={{ color: t.onSurfaceVar, fontSize: 13 }}>⚙</span>
          </button>
          <div style={{ flex: 1 }} />
          {streaming ? (
            <button onClick={onStop} style={{ width: 40, height: 40, borderRadius: 12, border: 'none', cursor: 'pointer', background: t.error, color: '#fff', fontSize: 14, flexShrink: 0 }}>■</button>
          ) : (
            <button onClick={canSend ? onSend : undefined} style={{
              width: 40, height: 40, borderRadius: 12, border: 'none', flexShrink: 0,
              cursor: canSend ? 'pointer' : 'default',
              background: canSend ? t.primary : t.outlineVariant, color: canSend ? t.onPrimary : t.onSurfaceVar,
              fontSize: 16, boxShadow: canSend ? `0 3px 12px ${hexA(t.primary, 0.4)}` : 'none',
            }}>▶</button>
          )}
        </div>
      </div>
      <div style={{ marginTop: 7, fontFamily: MONO, fontSize: 9.5, color: imgWarn ? t.error : t.onSurfaceVar, display: 'flex', justifyContent: 'space-between' }}>
        <span>{imgWarn ? '⚠ vision model only — image will be ignored' : '⏎ send · ⇧⏎ newline'}</span>
        <span style={{ color: t.onSurfaceVar }}>{model.quant}</span>
      </div>
    </div>
  );
}

function JsonRunBar({ t, ctx }) {
  const { onSend, streaming, onStop, jsonErr } = ctx;
  return (
    <div style={{ padding: '10px 12px 12px', borderTop: `1px solid ${t.outlineVariant}`, background: t.bg, display: 'flex', gap: 8 }}>
      {streaming
        ? <Btn t={t} flex kind="filled" danger label="■  stop" onClick={onStop} />
        : <Btn t={t} flex label="▶  run (streaming)" onClick={onSend} disabled={!!jsonErr} />}
      <Btn t={t} kind="tonal" label="blocking" onClick={onSend} disabled={streaming || !!jsonErr} />
    </div>
  );
}

function ConversationView({ t, ctx }) {
  const { messages, streaming, msgFormat, setMsgFormat, jsonView, setJsonView,
    openaiJson, setOpenaiJson, onSyncJson, onClear, onSuggest,
    model, loadedLabel, loadStatus } = ctx;
  const desc = cat.resolve(model.family, model.runtime, model.backend);
  const modelId = loadStatus === 'loaded' ? loadedLabel : (desc ? desc.id : model.family);
  const showJson = msgFormat === 'openai' && jsonView;

  const threadRef = React.useRef(null);
  React.useEffect(() => { if (threadRef.current) threadRef.current.scrollTop = threadRef.current.scrollHeight; }, [messages, streaming, showJson]);

  // openai json parse → role preview
  let parsed = null, jsonErr = null;
  if (msgFormat === 'openai') {
    try {
      const arr = JSON.parse(openaiJson);
      if (!Array.isArray(arr)) throw new Error('expected a JSON array of messages');
      parsed = arr.map(m => ({ role: String(m.role || '').toUpperCase(), content: typeof m.content === 'string' ? m.content : JSON.stringify(m.content) }));
    } catch (e) { jsonErr = e.message; }
  }
  ctx.jsonErr = jsonErr;
  const roleColor = (r) => r === 'SYSTEM' ? [t.tertiaryContainer, t.tertiary] : r === 'USER' ? [t.primaryContainer, t.onPrimaryContainer] : [t.secondaryContainer, t.onSurface];

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      {/* format header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px 10px' }}>
        <span style={{ fontFamily: MONO, fontSize: 9.5, fontWeight: 700, letterSpacing: '0.12em', color: t.onSurfaceVar, textTransform: 'uppercase' }}>format</span>
        <Segmented t={t} small value={msgFormat} onPick={setMsgFormat}
          options={[{ key: 'chat', label: '⌬ Chat' }, { key: 'openai', label: '{ } OpenAI' }]} />
        <div style={{ flex: 1 }} />
        {msgFormat === 'openai' && (
          <button onClick={() => setJsonView(v => !v)} title="toggle JSON view" style={{
            fontFamily: MONO, fontSize: 11, fontWeight: 700, cursor: 'pointer', padding: '6px 10px', borderRadius: 9,
            border: `1px solid ${jsonView ? 'transparent' : t.outlineVariant}`,
            background: jsonView ? t.primary : t.surface, color: jsonView ? t.onPrimary : t.onSurfaceVar,
          }}>{'{ }'} json</button>
        )}
        {messages.length > 0 && !showJson && (
          <button onClick={onClear} title="clear conversation" style={{ fontFamily: MONO, fontSize: 12, cursor: 'pointer', padding: '6px 9px', borderRadius: 9, border: `1px solid ${t.outlineVariant}`, background: t.surface, color: t.onSurfaceVar }}>⌫</button>
        )}
      </div>

      {/* body */}
      <div ref={threadRef} style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '2px 12px 8px' }}>
        {showJson ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <Card t={t} color={t.surface}>
              <div style={{ display: 'flex', alignItems: 'center', marginBottom: 10 }}>
                <Lbl t={t}>messages preview</Lbl>
                <div style={{ flex: 1 }} />
                <button onClick={onSyncJson} style={{ fontFamily: MONO, fontSize: 10, cursor: 'pointer', padding: '4px 8px', borderRadius: 8, border: `1px solid ${t.outlineVariant}`, background: t.surfaceContainer, color: t.onSurfaceVar }}>↻ from chat</button>
              </div>
              {parsed && (
                <div style={{ background: t.surfaceContainerHigh, borderRadius: 12, padding: 9, display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {parsed.map((m, i) => {
                    const [bg, fg] = roleColor(m.role);
                    return (
                      <div key={i} style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                        <span style={{ fontFamily: MONO, fontSize: 9, fontWeight: 700, color: fg, background: bg, padding: '3px 7px', borderRadius: 999, minWidth: 62, textAlign: 'center', flexShrink: 0 }}>{m.role || '?'}</span>
                        <span style={{ fontFamily: MONO, fontSize: 11.5, color: t.onSurface, lineHeight: 1.5, wordBreak: 'break-word' }}>{m.content}</span>
                      </div>
                    );
                  })}
                </div>
              )}
              {jsonErr && <div style={{ fontFamily: MONO, fontSize: 11, color: t.error, background: t.errorContainer, padding: '8px 11px', borderRadius: 9 }}>ⓘ {jsonErr}</div>}
            </Card>
            <div>
              <Lbl t={t}>messages json</Lbl>
              <Field t={t} rows={10} value={openaiJson} onChange={setOpenaiJson} />
              <div style={{ marginTop: 6, fontFamily: MONO, fontSize: 9.5, color: t.onSurfaceVar }}>› edits here are sent on the next run</div>
            </div>
          </div>
        ) : messages.length === 0 ? (
          <EmptyState t={t} modelId={modelId} backend={model.backend} onSuggest={onSuggest} />
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14, paddingTop: 6 }}>
            {messages.map((m, i) => <Bubble key={i} t={t} msg={m} streaming={streaming && i === messages.length - 1 && m.role === 'assistant'} />)}
          </div>
        )}
      </div>

      {/* composer or run bar */}
      {showJson ? <JsonRunBar t={t} ctx={ctx} /> : <Composer t={t} ctx={ctx} />}
    </div>
  );
}

/* ───────────────── Metrics ───────────────── */

function MetricsTab({ t, ctx }) {
  const { metrics, onFetchMetrics } = ctx;
  if (!metrics) {
    return (
      <div style={{ padding: 12 }}>
        <Card t={t} style={{ textAlign: 'center', padding: '32px 20px' }}>
          <div style={{ width: 56, height: 56, borderRadius: 16, margin: '0 auto 14px', background: t.surfaceContainerHigh, color: t.onSurfaceVar, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: MONO, fontSize: 26 }}>▤</div>
          <div style={{ fontFamily: DISP, fontSize: 16, fontWeight: 600, color: t.onSurface }}>No metrics yet</div>
          <div style={{ fontFamily: MONO, fontSize: 11.5, color: t.onSurfaceVar, marginTop: 6, lineHeight: 1.5 }}>Run a message, then fetch counters from the most recent generation.</div>
          <div style={{ height: 16 }} />
          <Btn t={t} flex label="fetch metrics" onClick={onFetchMetrics} />
        </Card>
      </div>
    );
  }
  const m = metrics;
  const tps = m.genDur > 0 ? (m.genTok / (m.genDur / 1000)).toFixed(1) : '—';
  const prefillFrac = Math.max(0, Math.min(1, m.prefillDur / Math.max(1, m.totalDur)));

  return (
    <div style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 10, overflowY: 'auto', height: '100%' }}>
      <Card t={t} color={t.primaryContainer} style={{ border: `1px solid ${hexA(t.primary, 0.3)}` }}>
        <Lbl t={t} color={t.onPrimaryContainer}>tokens per second</Lbl>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
          <span style={{ fontFamily: DISP, fontSize: 52, fontWeight: 700, color: t.onPrimaryContainer, letterSpacing: '-0.03em', lineHeight: 1 }}>{tps}</span>
          <span style={{ fontFamily: MONO, fontSize: 14, color: hexA(t.onPrimaryContainer, 0.7), fontWeight: 600 }}>tok/s</span>
        </div>
        <Sparkline t={t} />
      </Card>
      <div style={{ display: 'flex', gap: 10 }}>
        <Tile t={t} label="ttft" value={m.prefillDur.toFixed(0)} unit="ms" />
        <Tile t={t} label="total" value={(m.totalDur / 1000).toFixed(2)} unit="s" />
      </div>
      <div style={{ display: 'flex', gap: 10 }}>
        <Tile t={t} label="prefill tok" value={String(m.prefillTok)} unit="tok" />
        <Tile t={t} label="gen tok" value={String(m.genTok)} unit="tok" />
      </div>
      <div style={{ display: 'flex', gap: 10 }}>
        <Tile t={t} label="init" value={m.initDur.toFixed(0)} unit="ms" />
        <Tile t={t} label="peak mem" value={(m.peakKb / 1024).toFixed(1)} unit="MB" />
      </div>
      <Card t={t}>
        <Lbl t={t}>prefill ▸ generation</Lbl>
        <div style={{ display: 'flex', height: 10, borderRadius: 999, overflow: 'hidden', background: t.surfaceContainerHigh }}>
          <div style={{ width: `${prefillFrac * 100}%`, background: t.tertiary }} />
          <div style={{ flex: 1, background: t.primary }} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8, fontFamily: MONO, fontSize: 10.5 }}>
          <span style={{ color: t.tertiary }}>● prefill {m.prefillDur.toFixed(0)}ms</span>
          <span style={{ color: t.primary }}>● gen {m.genDur.toFixed(0)}ms</span>
        </div>
      </Card>
      <Btn t={t} kind="tonal" flex label="↻  refresh metrics" onClick={onFetchMetrics} />
    </div>
  );
}

function Tile({ t, label, value, unit }) {
  return (
    <div style={{ flex: 1, background: t.surfaceContainer, border: `1px solid ${t.outlineVariant}`, borderRadius: 16, padding: 14, boxShadow: t.shadow }}>
      <Lbl t={t} mark={null}>{label}</Lbl>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 5 }}>
        <span style={{ fontFamily: DISP, fontSize: 26, fontWeight: 700, color: t.onSurface, letterSpacing: '-0.02em' }}>{value}</span>
        <span style={{ fontFamily: MONO, fontSize: 11, color: t.onSurfaceVar }}>{unit}</span>
      </div>
    </div>
  );
}

function Sparkline({ t }) {
  const pts = [6, 9, 7, 12, 10, 14, 11, 15, 13, 16, 14, 17];
  const max = Math.max(...pts);
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 3, height: 28, marginTop: 12 }}>
      {pts.map((p, i) => <div key={i} style={{ flex: 1, height: `${(p / max) * 100}%`, background: hexA(t.onPrimaryContainer, 0.35), borderRadius: 2 }} />)}
    </div>
  );
}

Object.assign(window, { Segmented, ConversationView, Composer, JsonRunBar, Bubble, EmptyState, AttChip, MetricsTab, Tile, Sparkline, buildJson });
