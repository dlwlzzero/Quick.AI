// quick-settings.jsx — SettingsPanel (model / sampling / format) overlay + AttachMenu
/* global React, MONO, DISP, hexA, cat, QUANTS, Lbl, Dot, ChipRow, Dropdown, Btn, Field, Segmented */

function AttachMenu({ t, open, onClose, onAttach, multimodal }) {
  if (!open) return null;
  const item = (glyph, label, kind, disabled, note) => (
    <button onClick={disabled ? undefined : () => { onAttach(kind); onClose(); }} style={{
      width: '100%', display: 'flex', alignItems: 'center', gap: 10, textAlign: 'left',
      padding: '10px 11px', borderRadius: 10, border: 'none', cursor: disabled ? 'default' : 'pointer',
      background: 'transparent', opacity: disabled ? 0.45 : 1,
    }}>
      <span style={{ width: 26, height: 26, borderRadius: 7, background: t.surfaceContainerHigh, color: t.onSurface, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: MONO, fontSize: 13, flexShrink: 0 }}>{glyph}</span>
      <span style={{ flex: 1 }}>
        <span style={{ display: 'block', fontFamily: MONO, fontSize: 12.5, fontWeight: 600, color: t.onSurface }}>{label}</span>
        {note && <span style={{ display: 'block', fontFamily: MONO, fontSize: 9.5, color: t.onSurfaceVar }}>{note}</span>}
      </span>
    </button>
  );
  return (
    <React.Fragment>
      <div onClick={onClose} style={{ position: 'fixed', inset: 0, zIndex: 30 }} />
      <div style={{
        position: 'absolute', bottom: 'calc(100% + 8px)', left: 0, zIndex: 31, width: 210,
        background: t.surface, border: `1px solid ${t.outlineVariant}`, borderRadius: 14,
        boxShadow: t.shadow, padding: 5,
      }}>
        {item('▣', 'Image', 'image', !multimodal, multimodal ? 'jpeg · png' : 'vision model only')}
        {item('▤', 'File', 'file', false, 'txt · json · md')}
        {item('◎', 'Camera', 'camera', !multimodal, multimodal ? 'capture a photo' : 'vision model only')}
      </div>
    </React.Fragment>
  );
}

function SettingsPanel({ t, ctx }) {
  const { model, setModel, loadStatus, loadedLabel, onLoad, onUnload,
    sampling, setSampling, msgFormat, setMsgFormat, onCloseSettings } = ctx;
  const { family, runtime, backend, quant } = model;
  const desc = cat.resolve(family, runtime, backend);
  const dotColor = loadStatus === 'loaded' ? t.success : loadStatus === 'loading' ? t.primary : t.outline;

  const pickFamily = (f) => { const rt = cat.runtimes(f)[0]; const be = cat.backends(f, rt)[0]; setModel({ family: f, runtime: rt, backend: be, quant }); };
  const pickRuntime = (rt) => { const be = cat.backends(family, rt)[0]; setModel({ ...model, runtime: rt, backend: be }); };

  const Divider = () => <div style={{ height: 1, background: t.outlineVariant, margin: '18px 0' }} />;
  const Group = ({ children }) => <div style={{ marginBottom: 4 }}>{children}</div>;

  return (
    <div style={{ position: 'absolute', inset: 0, zIndex: 50, display: 'flex', flexDirection: 'column', justifyContent: 'flex-end' }}>
      <div onClick={onCloseSettings} style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.45)', backdropFilter: 'blur(2px)' }} />
      <div style={{
        position: 'relative', background: t.bg, borderTopLeftRadius: 24, borderTopRightRadius: 24,
        maxHeight: '90%', display: 'flex', flexDirection: 'column',
        border: `1px solid ${t.outlineVariant}`, borderBottom: 'none',
        boxShadow: '0 -12px 40px rgba(0,0,0,0.3)',
      }}>
        <div style={{ padding: '10px 0 4px', display: 'flex', justifyContent: 'center' }}>
          <div style={{ width: 38, height: 4, borderRadius: 999, background: t.outline }} />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '6px 18px 12px' }}>
          <span style={{ fontFamily: DISP, fontSize: 17, fontWeight: 700, color: t.onSurface, flex: 1, letterSpacing: '-0.01em' }}>Model & settings</span>
          <button onClick={onCloseSettings} style={{ width: 32, height: 32, borderRadius: 9, border: `1px solid ${t.outlineVariant}`, background: t.surfaceContainer, color: t.onSurfaceVar, cursor: 'pointer', fontSize: 14 }}>✕</button>
        </div>

        <div style={{ overflowY: 'auto', padding: '0 18px 18px' }}>
          {/* MODEL */}
          <Group>
            <Lbl t={t}>family</Lbl>
            <Dropdown t={t} options={cat.families()} value={family} onPick={pickFamily} />
            <div style={{ height: 14 }} />
            <Lbl t={t}>runtime</Lbl>
            <ChipRow t={t} options={cat.runtimes(family)} value={runtime} onPick={pickRuntime} />
            <div style={{ height: 14 }} />
            <Lbl t={t}>backend</Lbl>
            <ChipRow t={t} options={cat.backends(family, runtime)} value={backend} onPick={(b) => setModel({ ...model, backend: b })} />
            <div style={{ height: 14 }} />
            <Lbl t={t}>quantization</Lbl>
            <ChipRow t={t} options={QUANTS} value={quant} onPick={(q) => setModel({ ...model, quant: q })} />
            <div style={{ height: 16 }} />
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, fontFamily: MONO, fontSize: 11, color: t.onSurfaceVar }}>
              <Dot color={dotColor} pulse={loadStatus === 'loading'} />
              {loadStatus === 'loaded' ? `loaded · ${loadedLabel}` : loadStatus === 'loading' ? 'loading…' : `not loaded · ${desc ? desc.id : family}`}
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <Btn t={t} flex label={loadStatus === 'loaded' ? '↻  reload model' : '↓  load model'} onClick={onLoad} disabled={loadStatus === 'loading'} />
              {loadStatus === 'loaded' && <Btn t={t} kind="tonal" danger label="✕ unload" onClick={onUnload} />}
            </div>
          </Group>

          <Divider />

          {/* SAMPLING */}
          <Group>
            <Lbl t={t}>system prompt</Lbl>
            <Field t={t} mono={false} rows={2} value={sampling.system} placeholder="You are a helpful assistant."
              onChange={(v) => setSampling({ ...sampling, system: v })} />
            <div style={{ height: 12 }} />
            <div style={{ display: 'flex', gap: 10 }}>
              <div style={{ flex: 1 }}><Lbl t={t} mark={null}>temperature</Lbl><Field t={t} mini value={sampling.temp} onChange={(v) => setSampling({ ...sampling, temp: v })} /></div>
              <div style={{ flex: 1 }}><Lbl t={t} mark={null}>top_k</Lbl><Field t={t} mini value={sampling.topk} onChange={(v) => setSampling({ ...sampling, topk: v })} /></div>
            </div>
            <div style={{ height: 10 }} />
            <div style={{ display: 'flex', gap: 10 }}>
              <div style={{ flex: 1 }}><Lbl t={t} mark={null}>top_p</Lbl><Field t={t} mini value={sampling.topp} onChange={(v) => setSampling({ ...sampling, topp: v })} /></div>
              <div style={{ flex: 1 }}><Lbl t={t} mark={null}>seed</Lbl><Field t={t} mini value={sampling.seed} onChange={(v) => setSampling({ ...sampling, seed: v })} /></div>
            </div>
            <div style={{ height: 14 }} />
            <Lbl t={t}>enable_thinking</Lbl>
            <ChipRow t={t} options={['default', 'true', 'false']} value={sampling.thinking} onPick={(v) => setSampling({ ...sampling, thinking: v })} />
          </Group>

          <Divider />

          {/* MESSAGE FORMAT */}
          <Group>
            <Lbl t={t}>message format</Lbl>
            <Segmented t={t} value={msgFormat} onPick={setMsgFormat} full
              options={[{ key: 'chat', label: '⌬ Chat' }, { key: 'openai', label: '{ } OpenAI' }]} />
            <div style={{ marginTop: 10, fontFamily: MONO, fontSize: 10.5, color: t.onSurfaceVar, lineHeight: 1.6 }}>
              {msgFormat === 'chat'
                ? '› stateful session · chatRunStreaming() · turn-by-turn'
                : '› stateless messages[] · runModelHandleWithMessagesStreaming() · OpenAI-style array'}
            </div>
          </Group>

          <div style={{ height: 18 }} />
          <Btn t={t} flex label="done" onClick={onCloseSettings} />
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { AttachMenu, SettingsPanel });
