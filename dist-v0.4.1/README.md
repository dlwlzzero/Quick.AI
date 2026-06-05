# Quick.AI v0.4.1 — Android UI design bundle

Interactive hi-fi prototype of the QuickDotAI app (merged Chat + OpenAI,
calm steel-blue palette, no terminal panel). Open `Quick.AI v0.4.0.html`
in a browser; the `.jsx` files load alongside it.

Files:
- Quick.AI v0.4.0.html  — entry point (React + phone frame + app root)
- android-frame.jsx     — device bezel/status bar/nav (bg-aware)
- quick-core.jsx        — design tokens (light/dark), model catalog, primitives
- quick-chrome.jsx      — TopBar, HeroStatus, TabBar
- quick-screens.jsx     — ConversationView (Chat/OpenAI), Composer, Metrics
- quick-settings.jsx    — model/sampling/format settings sheet + attach menu

---

## Commit & push to a new v0.4.1 branch

I have read-only GitHub access and cannot push for you. From a local
clone of dlwlzzero/Quick.AI (on the v0.4.0 branch), run:

```bash
git checkout v0.4.0
git checkout -b v0.4.1

# copy this bundle into the repo's design-bundle location, e.g.:
#   Applications/QuickAI/QuickDotAI/
# then stage and commit (-s adds your DCO Signed-off-by trailer):

git add .
git commit -s -m "[Android] redesign QuickDotAI: merge chat/openai, calm palette" -m "
- Merge the Chat and OpenAI tabs into one conversational view with a
  Chat/OpenAI message-format toggle and an editable messages[] JSON view.
- Add a Claude-style unified composer: attach menu (image/file/camera),
  model chip, and a slide-up model/sampling/format settings sheet.
- Replace the purple M3 palette with a calmer steel-blue + warm-neutral
  scheme (light/dark); drop the bottom terminal output panel.
"

git push -u origin v0.4.1
```

Subject is imperative and <= 72 chars; body explains *why*, per CLAUDE.md.
Adjust the destination path to wherever the HTML design bundle lives in
the repo.
