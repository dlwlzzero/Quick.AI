// quick-core.jsx — Quick.AI v0.4.0 design tokens, model catalog, mono primitives
// Bolder developer/terminal refresh of the QuickDotAI M3 Expressive system.
// Palette lifted verbatim from SampleTestAPP MainActivity.kt (LIGHT / DARK M3Tokens).

const MONO = "'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace";
const DISP = "'Space Grotesk', system-ui, sans-serif";

const LIGHT = {
  name: 'light',
  bg: '#F6F7F8', surface: '#FFFFFF', surfaceDim: '#ECEEF0',
  surfaceContainer: '#EFF1F3', surfaceContainerHigh: '#E6E9EC',
  outline: '#C4C8CC', outlineVariant: '#E1E4E7',
  onSurface: '#1B2024', onSurfaceVar: '#545A60',
  primary: '#3E6076', onPrimary: '#FFFFFF',
  primaryContainer: '#D8E5EE', onPrimaryContainer: '#142E3D',
  secondary: '#5A6470', secondaryContainer: '#E2E6EA',
  tertiary: '#876155', tertiaryContainer: '#F0E2DB',
  error: '#9E4B44', errorContainer: '#F2DEDB',
  success: '#3F7350', successContainer: '#DBEBDF',
  codeBg: '#14181C', codeFg: '#E6EAED', codeDim: '#69727A',
  shadow: '0 1px 2px rgba(27,32,36,0.06), 0 8px 24px rgba(62,96,118,0.08)',
  grid: 'rgba(62,96,118,0.05)',
};

const DARK = {
  name: 'dark',
  bg: '#14171A', surface: '#1B1F23', surfaceDim: '#101316',
  surfaceContainer: '#21262B', surfaceContainerHigh: '#2A3036',
  outline: '#434B52', outlineVariant: '#2A3036',
  onSurface: '#E3E7EA', onSurfaceVar: '#B2BABF',
  primary: '#9CC0D6', onPrimary: '#10303F',
  primaryContainer: '#2D4C5D', onPrimaryContainer: '#CFE4F0',
  secondary: '#B6C0CA', secondaryContainer: '#3A434B',
  tertiary: '#D7AE9C', tertiaryContainer: '#4C3A30',
  error: '#E3A39B', errorContainer: '#4A2622',
  success: '#8EC79E', successContainer: '#1E3B29',
  codeBg: '#0C0F11', codeFg: '#9CC0D6', codeDim: '#6A737A',
  shadow: '0 1px 2px rgba(0,0,0,0.4), 0 10px 30px rgba(0,0,0,0.45)',
  grid: 'rgba(156,192,214,0.05)',
};

// hex + alpha(0..1) -> rgba
function hexA(hex, a) {
  const h = hex.replace('#', '');
  const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${a})`;
}

// ── Model catalog (selectable subset; embedding-only tiny-bert filtered out) ──
const CATALOG = [
  { id: 'qwen3-0.6b',     family: 'qwen3',          display: 'Qwen3 0.6B',        runtime: 'NATIVE', backends: ['CPU', 'GPU'], caps: ['STREAMING', 'TOOL_USE'] },
  { id: 'qwen3-1.7b-q40', family: 'qwen3',          display: 'Qwen3 1.7B · Q4_0',  runtime: 'NATIVE', backends: ['CPU', 'GPU'], caps: ['STREAMING', 'TOOL_USE'] },
  { id: 'function-gemma', family: 'function-gemma', display: 'Function Gemma',     runtime: 'NATIVE', backends: ['CPU', 'GPU'], caps: ['TOOL_USE'] },
  { id: 'gemma4-cpu',     family: 'gemma4',         display: 'Gemma4 · CPU',       runtime: 'NATIVE', backends: ['CPU'],        caps: ['STREAMING'] },
  { id: 'gemma4-e2b-qnn', family: 'gemma4',         display: 'Gemma4 E2B · QNN',   runtime: 'NATIVE', backends: ['NPU'],        caps: ['MESSAGES_API'] },
  { id: 'gemma4',         family: 'gemma4',         display: 'Gemma4 · LiteRT',    runtime: 'LITERT', backends: ['GPU'],        caps: ['MULTIMODAL', 'MESSAGES_API', 'STREAMING'] },
  { id: 'vjepa-qnn',      family: 'vjepa',          display: 'V-JEPA · QNN',       runtime: 'NATIVE', backends: ['NPU'],        caps: ['MULTIMODAL', 'MULTI_IMAGE', 'MESSAGES_API'] },
];
const QUANTS = ['W4A32', 'W16A16', 'W8A16', 'W32A32'];

const cat = {
  families: () => [...new Set(CATALOG.map(d => d.family))],
  runtimes: (fam) => [...new Set(CATALOG.filter(d => d.family === fam).map(d => d.runtime))],
  backends: (fam, rt) => [...new Set(CATALOG.filter(d => d.family === fam && d.runtime === rt).flatMap(d => d.backends))],
  resolve: (fam, rt, be) => CATALOG.find(d => d.family === fam && d.runtime === rt && d.backends.includes(be)),
  multimodal: (d) => !!d && (d.caps.includes('MULTIMODAL') || d.caps.includes('MULTI_IMAGE')),
  multiImage: (d) => !!d && d.caps.includes('MULTI_IMAGE'),
};

/* ───────────────────────── primitives ───────────────────────── */

// Comet brand mark — gradient tile with a ☄ glyph
function Mark({ t, size = 38 }) {
  return (
    <div style={{
      width: size, height: size, borderRadius: size * 0.32, flexShrink: 0,
      background: `linear-gradient(135deg, ${t.primary}, ${t.tertiary})`,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      boxShadow: `0 4px 14px ${hexA(t.primary, 0.4)}`,
      position: 'relative', overflow: 'hidden',
    }}>
      <span style={{ fontSize: size * 0.5, lineHeight: 1, filter: 'saturate(1.2)' }}>☄</span>
    </div>
  );
}

// Uppercase tracked mono micro-label, optional index marker
function Lbl({ t, children, mark = '▌', color }) {
  return (
    <div style={{
      fontFamily: MONO, fontSize: 10.5, fontWeight: 600, letterSpacing: '0.14em',
      textTransform: 'uppercase', color: color || t.onSurfaceVar,
      display: 'flex', alignItems: 'center', gap: 6, marginBottom: 7,
    }}>
      {mark && <span style={{ color: t.primary, opacity: 0.85 }}>{mark}</span>}
      {children}
    </div>
  );
}

function Dot({ color, size = 9, pulse = false }) {
  return (
    <span style={{
      width: size, height: size, borderRadius: 999, background: color,
      display: 'inline-block', flexShrink: 0,
      boxShadow: pulse ? `0 0 0 0 ${hexA(color, 0.55)}` : 'none',
      animation: pulse ? 'qpulse 1.4s ease-out infinite' : 'none',
    }} />
  );
}

// Bracketed terminal chip
function Chip({ t, label, active, onClick, disabled }) {
  return (
    <button onClick={disabled ? undefined : onClick} style={{
      fontFamily: MONO, fontSize: 12, fontWeight: active ? 700 : 500,
      letterSpacing: '0.02em', cursor: disabled ? 'default' : 'pointer',
      padding: '7px 12px', borderRadius: 9, whiteSpace: 'nowrap',
      border: `1px solid ${active ? 'transparent' : t.outlineVariant}`,
      background: active ? t.primary : t.surface,
      color: active ? t.onPrimary : (disabled ? t.outline : t.onSurfaceVar),
      opacity: disabled ? 0.5 : 1, transition: 'all .14s ease',
      boxShadow: active ? `0 2px 8px ${hexA(t.primary, 0.32)}` : 'none',
    }}>
      <span style={{ opacity: active ? 0.6 : 0.45, marginRight: 5 }}>{active ? '▶' : '·'}</span>
      {label}
    </button>
  );
}

function ChipRow({ t, options, value, onPick }) {
  return (
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
      {options.map(o => <Chip key={o} t={t} label={o} active={o === value} onClick={() => onPick(o)} />)}
    </div>
  );
}

// Dropdown (FAMILY) — opens an absolute menu
function Dropdown({ t, options, value, onPick }) {
  const [open, setOpen] = React.useState(false);
  return (
    <div style={{ position: 'relative' }}>
      <button onClick={() => setOpen(o => !o)} style={{
        width: '100%', textAlign: 'left', cursor: 'pointer',
        fontFamily: MONO, fontSize: 13.5, fontWeight: 600,
        padding: '12px 14px', borderRadius: 12,
        border: `1px solid ${open ? t.primary : t.outlineVariant}`,
        background: t.surface, color: t.onSurface,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        transition: 'border-color .14s',
      }}>
        <span>{value}</span>
        <span style={{ color: t.onSurfaceVar, fontSize: 11, transform: open ? 'rotate(180deg)' : 'none', transition: 'transform .16s' }}>▾</span>
      </button>
      {open && (
        <div style={{
          position: 'absolute', top: 'calc(100% + 6px)', left: 0, right: 0, zIndex: 40,
          background: t.surface, border: `1px solid ${t.outlineVariant}`, borderRadius: 12,
          boxShadow: t.shadow, overflow: 'hidden', padding: 4,
        }}>
          {options.map(o => {
            const on = o === value;
            return (
              <button key={o} onClick={() => { onPick(o); setOpen(false); }} style={{
                width: '100%', textAlign: 'left', cursor: 'pointer', border: 'none',
                fontFamily: MONO, fontSize: 13, fontWeight: on ? 700 : 500,
                padding: '10px 12px', borderRadius: 8,
                background: on ? t.primaryContainer : 'transparent',
                color: on ? t.onPrimaryContainer : t.onSurface,
                display: 'flex', alignItems: 'center', gap: 8,
              }}>
                <span style={{ color: t.primary, opacity: on ? 1 : 0 }}>›</span>{o}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

// Buttons
function Btn({ t, label, onClick, kind = 'filled', danger = false, disabled = false, flex = false }) {
  const base = {
    fontFamily: MONO, fontSize: 13, fontWeight: 700, letterSpacing: '0.01em',
    padding: '12px 16px', borderRadius: 12, cursor: disabled ? 'default' : 'pointer',
    flex: flex ? 1 : 'none', transition: 'all .14s ease', whiteSpace: 'nowrap',
    opacity: disabled ? 0.45 : 1, border: '1px solid transparent',
  };
  let st;
  if (kind === 'filled') st = { background: danger ? t.error : t.primary, color: danger ? '#fff' : t.onPrimary, boxShadow: disabled ? 'none' : `0 3px 12px ${hexA(danger ? t.error : t.primary, 0.34)}` };
  else if (kind === 'tonal') st = { background: danger ? t.errorContainer : t.secondaryContainer, color: danger ? t.error : t.onSurface };
  else st = { background: 'transparent', color: t.onSurfaceVar, border: `1px solid ${t.outline}` };
  return <button onClick={disabled ? undefined : onClick} style={{ ...base, ...st }}>{label}</button>;
}

// Card with hairline outline (technical look)
function Card({ t, children, color, pad = 14, style }) {
  return (
    <div style={{
      background: color || t.surfaceContainer, borderRadius: 18, padding: pad,
      border: `1px solid ${t.outlineVariant}`, boxShadow: t.shadow,
      ...style,
    }}>{children}</div>
  );
}

function SectionHeader({ t, glyph, title, subtitle, iconBg, iconFg, right }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 11 }}>
      <div style={{
        width: 34, height: 34, borderRadius: 10, flexShrink: 0,
        background: iconBg || t.secondaryContainer, color: iconFg || t.onSurface,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontFamily: MONO, fontSize: 15, fontWeight: 700,
      }}>{glyph}</div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontFamily: DISP, fontSize: 15, fontWeight: 600, color: t.onSurface, letterSpacing: '-0.01em' }}>{title}</div>
        {subtitle && <div style={{ fontFamily: MONO, fontSize: 10.5, color: t.onSurfaceVar, marginTop: 2, letterSpacing: '0.01em' }}>{subtitle}</div>}
      </div>
      {right}
    </div>
  );
}

function Collapsible({ t, glyph, iconBg, iconFg, title, subtitle, right, expanded, onToggle, children }) {
  return (
    <Card t={t} pad={0} style={{ overflow: 'visible' }}>
      <button onClick={onToggle} style={{
        width: '100%', background: 'transparent', border: 'none', cursor: 'pointer',
        padding: 14, display: 'flex', alignItems: 'center', gap: 11, textAlign: 'left',
      }}>
        <div style={{
          width: 34, height: 34, borderRadius: 10, flexShrink: 0,
          background: iconBg || t.primaryContainer, color: iconFg || t.onPrimaryContainer,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontFamily: MONO, fontSize: 15, fontWeight: 700,
        }}>{glyph}</div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontFamily: DISP, fontSize: 15, fontWeight: 600, color: t.onSurface }}>{title}</div>
          {subtitle && <div style={{ fontFamily: MONO, fontSize: 10.5, color: t.onSurfaceVar, marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{subtitle}</div>}
        </div>
        {right}
        <span style={{ color: t.onSurfaceVar, fontSize: 12, transform: expanded ? 'rotate(180deg)' : 'none', transition: 'transform .18s' }}>▾</span>
      </button>
      {expanded && <div style={{ padding: '2px 14px 16px' }}>{children}</div>}
    </Card>
  );
}

// mono input field
function Field({ t, value, onChange, placeholder, rows, mono = true, mini = false }) {
  const shared = {
    width: '100%', boxSizing: 'border-box', resize: 'none',
    fontFamily: mono ? MONO : DISP, fontSize: mini ? 13 : 13.5,
    padding: mini ? '10px 12px' : '12px 14px', borderRadius: 12,
    border: `1px solid ${t.outlineVariant}`, background: t.surface,
    color: t.onSurface, outline: 'none', lineHeight: 1.5,
  };
  if (rows) return <textarea rows={rows} value={value} placeholder={placeholder}
    onChange={e => onChange(e.target.value)} style={shared}
    onFocus={e => e.target.style.borderColor = t.primary}
    onBlur={e => e.target.style.borderColor = t.outlineVariant} />;
  return <input value={value} placeholder={placeholder}
    onChange={e => onChange(e.target.value)} style={shared}
    onFocus={e => e.target.style.borderColor = t.primary}
    onBlur={e => e.target.style.borderColor = t.outlineVariant} />;
}

Object.assign(window, {
  MONO, DISP, LIGHT, DARK, hexA, CATALOG, QUANTS, cat,
  Mark, Lbl, Dot, Chip, ChipRow, Dropdown, Btn, Card, SectionHeader, Collapsible, Field,
});
