// quick-chrome.jsx — app chrome: TopBar, HeroStatus, TabBar, ModelSection, OutputPanel
/* global React, MONO, DISP, hexA, cat, QUANTS, Mark, Lbl, Dot, Chip, ChipRow, Dropdown, Btn, Card, Collapsible, Field */

function TopBar({ t, dark, onToggleTheme, loadStatus, loadedLabel }) {
  const loaded = loadStatus === 'loaded';
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 12,
      padding: '10px 16px 12px', background: t.bg,
    }}>
      <Mark t={t} size={40} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontFamily: DISP, fontSize: 19, fontWeight: 700, color: t.onSurface, letterSpacing: '-0.02em', lineHeight: 1.1 }}>
          Quick<span style={{ color: t.primary }}>.AI</span>
        </div>
        <div style={{ fontFamily: MONO, fontSize: 10.5, letterSpacing: '0.02em', marginTop: 2 }}>
          <span style={{ color: t.onSurfaceVar }}>in-process aar</span>
          <span style={{ color: t.outline, margin: '0 5px' }}>·</span>
          <span style={{ color: loaded ? t.success : t.onSurfaceVar, fontWeight: 600 }}>{loaded ? loadedLabel : 'no model'}</span>
        </div>
      </div>
      <div style={{
        fontFamily: MONO, fontSize: 9, fontWeight: 700, letterSpacing: '0.08em',
        color: t.primary, background: t.primaryContainer, padding: '4px 8px', borderRadius: 7,
      }}>v0.4.0</div>
      <button onClick={onToggleTheme} title="theme" style={{
        width: 40, height: 40, borderRadius: 12, cursor: 'pointer',
        border: `1px solid ${t.outlineVariant}`, background: t.surfaceContainer,
        color: t.onSurfaceVar, fontSize: 16, display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>{dark ? '☀' : '☾'}</button>
    </div>
  );
}

// Hero terminal status line
function HeroStatus({ t, tone, text, streaming, onStop }) {
  const map = {
    error:    [t.errorContainer, t.error, t.error],
    success:  [t.successContainer, t.success, t.success],
    progress: [t.primaryContainer, t.primary, t.onPrimaryContainer],
    idle:     [t.surfaceContainer, t.outline, t.onSurfaceVar],
  };
  const [bg, dotC, fg] = map[tone] || map.idle;
  return (
    <div style={{ padding: '0 12px 12px', background: t.bg }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10,
        padding: '11px 14px', borderRadius: 14, background: bg,
        border: `1px solid ${hexA(dotC, 0.28)}`,
      }}>
        <Dot color={dotC} size={9} pulse={tone === 'progress' || streaming} />
        <span style={{
          flex: 1, minWidth: 0, fontFamily: MONO, fontSize: 12, color: fg,
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {text}
          {tone === 'idle' && <span className="qcaret" style={{ color: t.primary }}>▋</span>}
        </span>
        {streaming && (
          <button onClick={onStop} style={{
            fontFamily: MONO, fontSize: 11, fontWeight: 700, color: '#fff', cursor: 'pointer',
            background: t.error, border: 'none', padding: '5px 11px', borderRadius: 999,
            whiteSpace: 'nowrap', flexShrink: 0,
          }}>■ stop</button>
        )}
      </div>
    </div>
  );
}

function TabBar({ t, value, onPick }) {
  const tabs = [['chat', '⌬', 'Chat'], ['metrics', '▤', 'Metrics']];
  return (
    <div style={{ padding: '0 12px 10px', background: t.bg }}>
      <div style={{
        display: 'flex', gap: 4, padding: 4, borderRadius: 999,
        background: t.surfaceContainer, border: `1px solid ${t.outlineVariant}`,
      }}>
        {tabs.map(([key, glyph, label]) => {
          const on = key === value;
          return (
            <button key={key} onClick={() => onPick(key)} style={{
              flex: 1, cursor: 'pointer', border: 'none', borderRadius: 999,
              padding: '9px 6px', fontFamily: MONO, fontSize: 12.5,
              fontWeight: on ? 700 : 500, whiteSpace: 'nowrap',
              background: on ? t.primary : 'transparent',
              color: on ? t.onPrimary : t.onSurfaceVar,
              boxShadow: on ? `0 2px 10px ${hexA(t.primary, 0.34)}` : 'none',
              transition: 'all .15s ease',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
            }}>
              <span style={{ opacity: on ? 1 : 0.7 }}>{glyph}</span>{label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

// Collapsible model section (Family ▾ / Runtime · Backend · Quant chips / path / load)
function ModelSection({ t, expanded, onToggle, sel, setSel, loadStatus, loadedLabel, onLoad, onUnload }) {
  const { family, runtime, backend, quant } = sel;
  const desc = cat.resolve(family, runtime, backend);
  const subtitle = loadStatus === 'loaded'
    ? `${loadedLabel} · ${backend}`
    : `${desc ? desc.display : family} · ${runtime} · ${backend}`;
  const dotColor = loadStatus === 'loaded' ? t.success : loadStatus === 'loading' ? t.primary : t.outline;

  const pickFamily = (f) => {
    const rt = cat.runtimes(f)[0];
    const be = cat.backends(f, rt)[0];
    setSel({ family: f, runtime: rt, backend: be, quant });
  };
  const pickRuntime = (rt) => {
    const be = cat.backends(family, rt)[0];
    setSel({ ...sel, runtime: rt, backend: be });
  };

  return (
    <Collapsible t={t} glyph="▦" title="Model"
      subtitle={subtitle} expanded={expanded} onToggle={onToggle}
      right={<Dot color={dotColor} pulse={loadStatus === 'loading'} />}>
      <Lbl t={t}>family</Lbl>
      <Dropdown t={t} options={cat.families()} value={family} onPick={pickFamily} />
      <div style={{ height: 14 }} />
      <Lbl t={t}>runtime</Lbl>
      <ChipRow t={t} options={cat.runtimes(family)} value={runtime} onPick={pickRuntime} />
      <div style={{ height: 14 }} />
      <Lbl t={t}>backend</Lbl>
      <ChipRow t={t} options={cat.backends(family, runtime)} value={backend} onPick={(b) => setSel({ ...sel, backend: b })} />
      <div style={{ height: 14 }} />
      <Lbl t={t}>quantization</Lbl>
      <ChipRow t={t} options={QUANTS} value={quant} onPick={(q) => setSel({ ...sel, quant: q })} />
      <div style={{ height: 14 }} />
      <Lbl t={t}>model path</Lbl>
      <div style={{
        fontFamily: MONO, fontSize: 12, color: t.onSurfaceVar, padding: '11px 13px',
        borderRadius: 12, border: `1px dashed ${t.outline}`, background: t.surfaceDim,
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
      }}>/data/local/tmp/models/{desc ? desc.id : family}</div>
      <div style={{ height: 16 }} />
      <div style={{ display: 'flex', gap: 8 }}>
        <Btn t={t} flex label={loadStatus === 'loaded' ? '↻  reload model' : '↓  load model'}
          onClick={onLoad} disabled={loadStatus === 'loading'} />
        {loadStatus === 'loaded' && <Btn t={t} kind="tonal" danger label="✕ unload" onClick={onUnload} />}
      </div>
    </Collapsible>
  );
}

// Terminal output panel with traffic lights + line gutter + caret
function OutputPanel({ t, lines, streaming, onClear }) {
  const ref = React.useRef(null);
  React.useEffect(() => { if (ref.current) ref.current.scrollTop = ref.current.scrollHeight; }, [lines, streaming]);
  const dots = ['#FF5F57', '#FEBC2E', '#28C840'];
  return (
    <div style={{ borderRadius: 18, overflow: 'hidden', background: t.codeBg, boxShadow: t.shadow }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px', borderBottom: `1px solid ${hexA('#FFFFFF', 0.07)}` }}>
        <div style={{ display: 'flex', gap: 7 }}>
          {dots.map(c => <span key={c} style={{ width: 11, height: 11, borderRadius: 999, background: c }} />)}
        </div>
        <span style={{ flex: 1, fontFamily: MONO, fontSize: 11, color: hexA('#FFFFFF', 0.5) }}>output · stream.kt</span>
        <button onClick={onClear} title="clear" style={{
          fontFamily: MONO, fontSize: 11, color: hexA('#FFFFFF', 0.5), background: 'transparent',
          border: 'none', cursor: 'pointer', padding: '2px 6px',
        }}>clear ⌫</button>
      </div>
      <div ref={ref} style={{ maxHeight: 230, overflowY: 'auto', padding: '12px 14px' }}>
        {lines.length === 0 && (
          <div style={{ fontFamily: MONO, fontSize: 12.5, color: t.codeDim }}>// streaming output appears here…</div>
        )}
        {lines.map((ln, i) => (
          <div key={i} style={{ display: 'flex', gap: 12, alignItems: 'baseline' }}>
            <span style={{ fontFamily: MONO, fontSize: 11, color: hexA(t.codeFg, 0.28), userSelect: 'none', minWidth: 18, textAlign: 'right' }}>{i + 1}</span>
            <span style={{ fontFamily: MONO, fontSize: 12.5, lineHeight: 1.6, color: ln.startsWith('›') || ln.startsWith('$') ? t.codeFg : hexA(t.codeFg, 0.82), whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
              {ln}
              {streaming && i === lines.length - 1 && <span className="qcaret" style={{ color: t.codeFg }}>▋</span>}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

Object.assign(window, { TopBar, HeroStatus, TabBar, ModelSection, OutputPanel });
