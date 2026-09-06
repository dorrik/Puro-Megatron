#!/usr/bin/env python3
"""Generate the Puro-0.6B frontier-budget plan page. Every number on the page is
computed here from measured throughput and published model facts."""
import math, html

# ------------------------------------------------------------------ inputs --
A, B, AL, BE = 406.4, 410.7, 0.34, 0.28          # Chinchilla (Hoffmann 2022) form
PER_GPU = 441.4e12 * 0.95                        # measured 441.4 TFLOP/s/GPU BF16 MBS8, 95% scaling
N_DENSE, NE_DENSE = 0.596e9, 0.44e9              # Qwen3-0.6B shape, tied, vocab 151936
N_HYB,   NE_HYB   = 0.654e9, 0.50e9              # hybrid GDN shape C
FPT = 6 * N_DENSE                                # FLOP per training token
PURO_CORPUS = 1.377e12
FIR_TOKENS  = 19 * 1.0995e12 / 4                 # int32 resident on 19 TiB
YEAR8       = 8 * PER_GPU * 365 * 86400 / FPT

def excess(N, D): return A / N**AL + B / D**BE
def parity_tokens(Nt, Dt, No):
    need = excess(Nt, Dt) - A / No**AL
    return (B / need) ** (1 / BE) if need > 0 else math.inf
def days(D, g): return FPT * D / (g * PER_GPU) / 86400
def T(x):
    if x == math.inf: return "∞"
    return f"{x/1e12:.1f}T" if x < 10e12 else f"{x/1e12:.0f}T"
def sci(x):
    e = int(math.floor(math.log10(x))); m = x / 10**e
    return f"{m:.1f}×10<sup>{e}</sup>"

# name, total N, non-emb N, tokens, arch, license, evidence
TARGETS = [
 ("Qwen3-0.6B",        0.596e9, 0.44e9, 36e12, "dense, 28L",              "Apache-2.0", "base MMLU 52.8 · GSM8K 59.6 · MATH 32.4"),
 ("Qwen3.5-0.8B",      0.79e9,  0.54e9, 36e12, "hybrid GDN, 24L",         "Apache-2.0", "AA index 9 vs Qwen3-0.6B 6.5; tokens undisclosed"),
 ("LFM2-700M",         0.70e9,  0.60e9, 10e12, "hybrid conv+attn",        "LFM Open (≥$10M rev. restricted)", "instruct MMLU 49.9 · IFEval 72.2 · GSM8K 46.4"),
 ("Qwen2.5-0.5B",      0.49e9,  0.36e9, 18e12, "dense",                   "Apache-2.0", "base MMLU 47.5 · GSM8K 41.6"),
 ("Granite 4.0 H-350M",0.35e9,  0.27e9, 15e12, "hybrid Mamba2",           "Apache-2.0", "IFEval-strong, tool-calling"),
 ("SmolLM2-360M",      0.36e9,  0.31e9, 4e12,  "dense, 32L",              "Apache-2.0", "beats Qwen2.5-1.5B on MMLU-Pro"),
 ("Gemma 3 1B",        1.0e9,   0.70e9, 2e12,  "dense, sliding window",   "Gemma",      "base MMLU 26.3 · GSM8K 2.2 (Qwen3 report eval)"),
]
TIERS = [1.377e12, 4e12, 9e12, 18e12, 36e12]

# ------------------------------------------------------------------ chart ---
# log-x range plot: tokens to parity at our dense N for k = 4 (left) .. 1 (right)
CH_ROWS = ["Qwen3-0.6B", "Qwen3.5-0.8B", "Qwen2.5-0.5B", "Gemma 3 1B", "Granite 4.0 H-350M", "SmolLM2-360M"]
X0, X1 = 0.1e12, 100e12
W, PADL, PADR, PADT, ROWH = 1000, 190, 150, 64, 46
def X(D): return PADL + (math.log10(D) - math.log10(X0)) / (math.log10(X1) - math.log10(X0)) * (W - PADL - PADR)

svg = []
H = PADT + ROWH * len(CH_ROWS) + 52
svg.append(f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-labelledby="chart-title chart-desc">')
svg.append('<title id="chart-title">Tokens needed for a from-scratch 0.6B to reach loss parity with each frontier model</title>')
svg.append('<desc id="chart-desc">Horizontal range bars on a log token axis; each bar spans the optimistic (4x recipe efficiency) to conservative (1x) estimate. Dashed lines mark the Puro corpus, what fits on Fir scratch, and one year on eight H100s.</desc>')
# grid + ticks
for t in (0.1e12, 0.3e12, 1e12, 3e12, 10e12, 30e12, 100e12):
    x = X(t)
    svg.append(f'<line class="grid" x1="{x:.1f}" y1="{PADT-10}" x2="{x:.1f}" y2="{H-40}"/>')
    svg.append(f'<text class="tick" x="{x:.1f}" y="{H-18}" text-anchor="middle">{T(t)}</text>')
svg.append(f'<text class="axis" x="{(PADL+W-PADR)/2:.1f}" y="{H-2}" text-anchor="middle">training tokens (log scale)</text>')
# reference lines
for D, lab in ((PURO_CORPUS, "Puro corpus on hand"), (FIR_TOKENS, "fits Fir scratch"), (YEAR8, "one year, 8×H100")):
    x = X(D)
    svg.append(f'<line class="ref" x1="{x:.1f}" y1="{PADT-40}" x2="{x:.1f}" y2="{H-40}"/>')
    svg.append(f'<text class="reflab" x="{x+5:.1f}" y="{PADT-44}">{lab} · {T(D)}</text>')
# bars
for i, name in enumerate(CH_ROWS):
    Nt, Nn, Dt = next((t[1], t[2], t[3]) for t in TARGETS if t[0] == name)
    d1 = parity_tokens(Nn, Dt, NE_DENSE); d2, d4 = d1 / 2, d1 / 4
    y = PADT + i * ROWH + ROWH / 2
    svg.append(f'<text class="rowlab" x="{PADL-14}" y="{y+5:.1f}" text-anchor="end">{html.escape(name)}{"†" if name=="Gemma 3 1B" else ""}</text>')
    xa, xb = X(max(d4, X0)), X(min(d1, X1))
    off = d1 > X1
    svg.append(f'<g class="bar" data-name="{html.escape(name)}" data-k4="{T(d4)}" data-k2="{T(d2)}" data-k1="{T(d1)}" data-d8="{days(d2,8):.0f}" data-d32="{days(d2,32):.0f}" data-d64="{days(d2,64):.0f}">')
    svg.append(f'<rect class="hit" x="{PADL}" y="{y-ROWH/2:.1f}" width="{W-PADL-PADR}" height="{ROWH}"/>')
    svg.append(f'<rect class="range" x="{xa:.1f}" y="{y-5:.1f}" width="{max(xb-xa,4):.1f}" height="10" rx="4"/>')
    if not off:
        svg.append(f'<line class="mid" x1="{X(d2):.1f}" y1="{y-9:.1f}" x2="{X(d2):.1f}" y2="{y+9:.1f}"/>')
        svg.append(f'<text class="val" x="{xb+10:.1f}" y="{y+5:.1f}">{T(d4)} – {T(d1)}</text>')
    else:
        svg.append(f'<path class="chev" d="M{xb-2:.1f},{y-7:.1f} l7,7 l-7,7"/>')
        svg.append(f'<text class="val" x="{xa-8:.1f}" y="{y+5:.1f}" text-anchor="end">{T(d4)} – {T(d1)} (off scale)</text>')
    svg.append('</g>')
svg.append('</svg>')
SVG = "\n".join(svg)

# ------------------------------------------------------------------ tables --
def frontier_rows():
    out = []
    for name, Nt, Nn, Dt, arch, lic, ev in TARGETS:
        tok = "undisclosed (≥36T assumed)" if name == "Qwen3.5-0.8B" else T(Dt)
        out.append(f"<tr><th scope='row'>{html.escape(name)}</th><td>{arch}</td><td class='num'>{Nn/1e9:.2f}B</td><td class='num'>{tok}</td><td class='num'>{sci(6*Nt*Dt)}</td><td>{html.escape(ev)}</td><td>{html.escape(lic)}</td></tr>")
    return "\n".join(out)

def parity_rows():
    out = []
    for name, Nt, Nn, Dt, *_ in TARGETS:
        d = parity_tokens(Nn, Dt, NE_DENSE); dh = parity_tokens(Nn, Dt, NE_HYB)
        note = "†" if name == "Gemma 3 1B" else ""
        out.append(f"<tr><th scope='row'>{html.escape(name)}{note}</th><td class='num'>{T(d/4)}</td><td class='num'>{T(d/2)}</td><td class='num'>{T(d)}</td><td class='num'>{T(dh)}</td><td class='num'>{'—' if d==math.inf else f'{days(d/2,8):.0f} d'}</td></tr>")
    return "\n".join(out)

BEATS = {
 1.377e12: ("SmolLM2-360M and Granite H-350M outright; Qwen2.5-0.5B only if k≈4; below Qwen3-0.6B on knowledge", "Puro phase 1 + phase 2 (on hand)"),
 4e12:     ("Qwen2.5-0.5B-class overall; can pass Qwen3-0.6B on math and code with the Puro phase-2 mix", "Puro + ~2.6T from the Marin catalog"),
 9e12:     ("Qwen3-0.6B parity on general benchmarks if k≈4 holds", "Marin catalog (23.1T deduped available)"),
 18e12:    ("Qwen3-0.6B parity if k≈2 holds", "Marin catalog"),
 36e12:    ("Qwen3-0.6B compute parity, no efficiency assumed", "Marin catalog"),
}
def tier_rows():
    out = []
    for D in TIERS:
        beats, src = BEATS[D]
        out.append(f"<tr><th scope='row' class='num'>{T(D)}</th><td class='num'>{sci(FPT*D)}</td>"
                   f"<td class='num days' data-d8='{days(D,8):.0f}' data-d32='{days(D,32):.0f}' data-d64='{days(D,64):.0f}'>{days(D,8):.0f} d</td>"
                   f"<td class='num'>{D*4/1e12:.0f} TB{' · tranche' if D > FIR_TOKENS else ''}</td><td>{src}</td><td>{beats}</td></tr>")
    return "\n".join(out)

# ------------------------------------------------------------------ page ----
page = f"""<title>Brief Frontier Budget</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..125,500..800&family=Source+Serif+4:ital,opsz,wght@0,8..60,400..600;1,8..60,400&family=JetBrains+Mono:wght@400;500;600&display=swap">
<style>
:root{{
  --bg:#F3F5F8; --surface:#FFFFFF; --ink:#1A2530; --ink-2:#4C5B6A; --ink-3:#7C8A98;
  --rule:#D6DDE5; --rule-2:#E7ECF1; --acc:#2450C8; --acc-ink:#1B3E9E; --acc-soft:#E4EAFB;
  --acc-2:#C8641E; --acc-2-soft:#FBEBDD; --tile:#FFFFFF; --code:#EEF2F6;
  --f-display:"Archivo",system-ui,"Helvetica Neue",Arial,sans-serif;
  --f-body:"Source Serif 4",Georgia,"Times New Roman",serif;
  --f-mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}}
@media (prefers-color-scheme: dark){{ :root:not([data-theme="light"]){{
  --bg:#0F151B; --surface:#161E26; --ink:#E8EDF2; --ink-2:#AEBAC6; --ink-3:#7F8C99;
  --rule:#2A3541; --rule-2:#1E2731; --acc:#4F78EA; --acc-ink:#A9BEFB; --acc-soft:#1C2740;
  --acc-2:#D2773A; --acc-2-soft:#33231A; --tile:#161E26; --code:#1B242E;
}}}}
:root[data-theme="dark"]{{
  --bg:#0F151B; --surface:#161E26; --ink:#E8EDF2; --ink-2:#AEBAC6; --ink-3:#7F8C99;
  --rule:#2A3541; --rule-2:#1E2731; --acc:#4F78EA; --acc-ink:#A9BEFB; --acc-soft:#1C2740;
  --acc-2:#D2773A; --acc-2-soft:#33231A; --tile:#161E26; --code:#1B242E;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--f-body);font-size:17px;line-height:1.55;-webkit-font-smoothing:antialiased}}
.page{{max-width:1120px;margin:0 auto;padding:44px 32px 96px}}
@media (max-width:640px){{.page{{padding:28px 18px 72px}}}}
h1,h2,h3{{font-family:var(--f-display);font-stretch:95%;letter-spacing:-0.015em;text-wrap:balance;margin:0}}
h1{{font-size:clamp(34px,5vw,50px);font-weight:750;line-height:1.02}}
h2{{font-size:26px;font-weight:700;line-height:1.15;margin-top:64px}}
h3{{font-size:19px;font-weight:650;margin-top:28px}}
p{{margin:14px 0;max-width:68ch}}
.eyebrow{{font-family:var(--f-display);font-size:12px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:var(--acc-ink)}}
.dek{{font-size:20px;line-height:1.4;color:var(--ink-2);max-width:62ch;margin-top:16px}}
.meta{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px 28px;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);padding:14px 0;margin:28px 0 8px;font-family:var(--f-mono);font-size:13px}}
.meta dt{{color:var(--ink-3);font-size:11px;letter-spacing:0.08em;text-transform:uppercase}}
.meta dd{{margin:2px 0 0;color:var(--ink)}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin:26px 0 6px}}
.tile{{background:var(--tile);border:1px solid var(--rule);border-radius:6px;padding:18px 20px 16px}}
.tile .lab{{font-family:var(--f-display);font-size:12px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:var(--ink-3)}}
.tile .big{{font-family:var(--f-mono);font-size:30px;font-weight:600;letter-spacing:-0.02em;margin:8px 0 4px;font-variant-numeric:tabular-nums;color:var(--ink)}}
.tile .big sup{{font-size:0.55em;vertical-align:0.6em}}
.tile .sub{{font-size:14.5px;color:var(--ink-2);line-height:1.4}}
.tile.acc .big{{color:var(--acc-ink)}}
.wide{{overflow-x:auto;margin:18px 0 6px}}
table{{border-collapse:collapse;width:100%;font-size:14.5px;line-height:1.35}}
th,td{{text-align:left;vertical-align:top;padding:9px 12px;border-bottom:1px solid var(--rule-2)}}
thead th{{font-family:var(--f-display);font-size:12px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;color:var(--ink-3);border-bottom:1px solid var(--rule);white-space:nowrap}}
tbody th{{font-family:var(--f-display);font-weight:600;font-size:14.5px;white-space:nowrap}}
td.num,th.num{{font-family:var(--f-mono);font-size:13.5px;font-variant-numeric:tabular-nums;white-space:nowrap}}
td.num sup{{font-size:0.7em}}
tbody tr:hover td,tbody tr:hover th{{background:var(--acc-soft)}}
.callout{{border-left:3px solid var(--acc);background:var(--surface);padding:14px 18px;margin:22px 0;max-width:72ch;border-radius:0 6px 6px 0}}
.callout p{{margin:6px 0}}
.note{{color:var(--ink-2);font-size:14.5px}}
code,.mono{{font-family:var(--f-mono);font-size:0.9em;background:var(--code);padding:1px 5px;border-radius:4px}}
.chartwrap{{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:18px 18px 10px;margin:18px 0 6px;position:relative}}
.chart{{width:100%;height:auto;display:block}}
.chart .grid{{stroke:var(--rule-2);stroke-width:1}}
.chart .tick,.chart .axis,.chart .rowlab,.chart .val,.chart .reflab{{font-family:var(--f-mono);fill:var(--ink-2);font-size:13px}}
.chart .rowlab{{fill:var(--ink);font-family:var(--f-display);font-weight:600;font-size:14px}}
.chart .axis{{fill:var(--ink-3);font-size:12px}}
.chart .reflab{{fill:var(--acc-2);font-size:12px}}
.chart .ref{{stroke:var(--acc-2);stroke-width:1.5;stroke-dasharray:4 4}}
.chart .range{{fill:var(--acc)}}
.chart .mid{{stroke:var(--surface);stroke-width:2}}
.chart .chev{{fill:none;stroke:var(--acc);stroke-width:2}}
.chart .hit{{fill:transparent}}
.chart .bar:hover .range{{fill:var(--acc-ink)}}
.legend{{display:flex;flex-wrap:wrap;gap:8px 22px;font-family:var(--f-mono);font-size:12.5px;color:var(--ink-2);margin:6px 4px 4px}}
.legend span{{display:inline-flex;align-items:center;gap:8px}}
.sw{{width:22px;height:8px;border-radius:4px;background:var(--acc);display:inline-block}}
.sw.ref{{height:0;border-top:2px dashed var(--acc-2);border-radius:0}}
.tip{{position:absolute;pointer-events:none;background:var(--ink);color:var(--bg);font-family:var(--f-mono);font-size:12.5px;line-height:1.4;padding:8px 10px;border-radius:5px;box-shadow:0 4px 14px rgba(0,0,0,.18);display:none;max-width:260px;z-index:2}}
.seg{{display:inline-flex;border:1px solid var(--rule);border-radius:6px;overflow:hidden;font-family:var(--f-mono);font-size:13px;margin-left:10px;vertical-align:middle}}
.seg button{{background:var(--surface);color:var(--ink-2);border:0;padding:6px 12px;cursor:pointer;font:inherit}}
.seg button+button{{border-left:1px solid var(--rule)}}
.seg button[aria-pressed="true"]{{background:var(--acc);color:var(--bg);font-weight:600}}
.seg button:focus-visible{{outline:2px solid var(--acc);outline-offset:-2px}}
.phases{{display:grid;grid-template-columns:1fr;gap:0;margin-top:14px}}
.phase{{display:grid;grid-template-columns:150px 1fr 170px;gap:18px;padding:16px 0;border-top:1px solid var(--rule-2)}}
.phase:last-child{{border-bottom:1px solid var(--rule-2)}}
@media (max-width:760px){{.phase{{grid-template-columns:1fr}}}}
.phase .when{{font-family:var(--f-display);font-weight:650;font-size:15px}}
.phase .when small{{display:block;font-family:var(--f-mono);font-weight:400;color:var(--ink-3);font-size:12px;margin-top:4px}}
.phase .what p{{margin:4px 0;font-size:15.5px}}
.phase .cost{{font-family:var(--f-mono);font-size:13px;color:var(--ink-2);text-align:right}}
.phase .cost b{{display:block;color:var(--ink);font-weight:600;font-size:15px}}
@media (max-width:760px){{.phase .cost{{text-align:left}}}}
ul{{padding-left:1.2em;max-width:68ch}} li{{margin:6px 0}}
.srcs{{font-size:14px;color:var(--ink-2);columns:2;column-gap:32px;max-width:none}}
@media (max-width:760px){{.srcs{{columns:1}}}}
.srcs li{{break-inside:avoid;margin:4px 0}}
a{{color:var(--acc-ink)}}
sup{{line-height:0}}
.lineage{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:8px 40px;margin-top:6px}}
.lineage h3{{margin-top:10px}}
.lineage ul{{max-width:none}}
.decide{{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:6px 20px 8px;max-width:72ch}}
</style>

<main class="page">
<header>
  <div class="eyebrow">Brief · a 0.6B model from scratch · Alliance H100s</div>
  <h1 style="margin-top:10px">Brief Frontier Budget</h1>
  <p class="dek">Brief is a ~0.6B language model for real-time systems, trained from scratch with Puro's recipe and Marin's discipline. This page answers how many tokens and FLOPs it needs to overtake today's frontier small models, what that costs on the hardware we have, and the order of work that gets us a defensible answer in the first week.</p>
  <dl class="meta">
    <div><dt>Prepared</dt><dd>2026-09-05</dd></div>
    <div><dt>Cluster</dt><dd>Fir · 8×H100 SXM, 19 TiB scratch</dd></div>
    <div><dt>Measured throughput</dt><dd>441 TFLOP/s per GPU · BF16 · MBS 8 (44.6% MFU)</dd></div>
    <div><dt>Lineage</dt><dd>Puro recipe · Marin methodology · Qwen3.5-class shape</dd></div>
    <div><dt>Data on hand</dt><dd>Puro phase 1 439B built (verification pending) · phase 2 938B available</dd></div>
  </dl>
</header>


<section>
  <h2>Where Brief comes from</h2>
  <p>Brief borrows deliberately from two sources and adds a short list of its own calls. The split matters because the budget estimates above lean on Puro's measured efficiency, while the way we will find out whether they hold is Marin's.</p>
  <div class="lineage">
    <div>
      <h3>From Puro</h3>
      <ul>
        <li><strong>MuonHyperball</strong> with AdamW routing for embeddings, norms and the head; the <em>effective</em> LR (base LR × multiplier, m=3 at this size) as the primary hyperparameter.</li>
        <li><strong>Open-ended power schedule</strong> in phase 1, chosen precisely because the total budget is decided later; linear decay in phase 2; a constant-LR tail with checkpoint averaging as an option.</li>
        <li><strong>The two-phase data recipe and released corpus</strong> (439B + 938B tokens), its curriculum ordering, and proxy benchmarking for any data we add.</li>
        <li><strong>Qwen3 family architecture and tokenizer</strong> (vocab 151,936, ChatML end-of-turn as EOS), and its cost-accounting habit: every claim carries its GPU-hours.</li>
        <li>Blockwise FP8, <em>evaluated and not adopted</em>: +1.1% at this scale on H100.</li>
      </ul>
    </div>
    <div>
      <h3>From Marin's Hero Run</h3>
      <ul>
        <li><strong>A fitted hyperparameter law</strong> (<code>heuristic.py</code>, R²=0.978): MuonH LR = 13/3 × 0.0876 × tokens<sup>-0.346</sup> × hidden<sup>-0.345</sup> × √(tokens per batch). For Brief it gives an effective peak of 0.0164 at 12B tokens, where Puro measured 0.012, and 0.0032 at 1.38T. It sets every ladder rung's LR without a sweep, and it answers how the peak falls with horizon.</li>
        <li><strong>The scaling ladder</strong>: the same recipe at several widths and a fixed tokens-per-parameter, fitted to a power law, for about 1% of the main run's compute.</li>
        <li><strong>Preregistered predictions</strong>: the expected loss at every 5% of the run is written down before launch and checked as it trains.</li>
        <li><strong>Dynamics as a monitored signal</strong>: gradient norm and mixture-boundary loss bumps compared against the small rungs, so "is this normal?" has an answer.</li>
        <li><strong>Optimizer details worth ablating</strong>: MuonH momentum 0.95 with Nesterov, no gradient clipping, β₂ from the batch size (0.984 at Brief's batch), and the LM head on hyperball-Adam rather than AdamW.</li>
        <li><strong>z-loss</strong> at 1×10<sup>-4</sup> and an EMA of the weights for evaluation (neither exists in Puro-Megatron yet).</li>
        <li><strong>Architecture tricks for the dense shape</strong>: init 0.5/√hidden, query scaling 1.3, a kernel-4 short convolution at the K, attention and MLP sites, and sliding-window attention with a full-attention layer every fourth.</li>
        <li><strong>Data provenance and decontamination</strong>: a token-counts table with licenses, and n-gram decontamination against the evaluation sets, before any corpus extension from Marin's catalog.</li>
        <li><strong>Simulated epoching</strong>: a short ladder run is fed the long run's mixture schedule scaled down, so it crosses the phase-1 to phase-2 boundary at the same fraction and sees the same per-source exposure the long run will.</li>
        <li><strong>A delay policy</strong>: if the run falls behind, shrink the horizon and re-derive the decay to reach the same final LR, rather than improvise.</li>
      </ul>
    </div>
  </div>
  <p><strong>Brief's own calls.</strong> A ~0.6B budget chosen for real-time inference; the Qwen3.5-style hybrid layout for its 9× smaller KV cache, gated on a bake-off; tied embeddings; 4,096-token pretraining with a late extension to 32K; a base checkpoint saved before terminal decay, with optimizer state, for fine-tuners; 1–2% instruction-formatted data late in pretraining; quantization-aware training and a speculative-decoding draft head before release; and an export that loads in the standard Hugging Face classes so every fine-tuning tool works on day one.</p>
</section>

<section>
  <h2>What the frontier cost to make</h2>
  <p>The benchmark to beat at this size is Qwen3-0.6B. Its pretraining ran 36 trillion tokens. At six FLOPs per parameter per token on a 0.6B model, that is the compute below, and our measured rate turns it into calendar time.</p>
  <div class="tiles">
    <div class="tile"><div class="lab">Qwen3-0.6B pretraining compute</div><div class="big">{sci(6*0.596e9*36e12)}</div><div class="sub">FLOPs · 36T tokens × 3.6 GFLOP/token</div></div>
    <div class="tile acc"><div class="lab">Our measured rate</div><div class="big">{sci(8*PER_GPU)}</div><div class="sub">FLOP/s on 8×H100 · 441 TFLOP/s/GPU at 95% scaling · FP8 measured at +1%, not used</div></div>
    <div class="tile"><div class="lab">Days to compute parity</div><div class="big">{days(36e12,8):.0f} <span style="font-size:.5em;color:var(--ink-3)">@8</span> · {days(36e12,64):.0f} <span style="font-size:.5em;color:var(--ink-3)">@64</span></div><div class="sub">calendar days at 8 and 64 GPUs, no efficiency advantage assumed</div></div>
  </div>
  <p>So the direct answer is: matching Qwen3-0.6B's compute takes about {days(36e12,8)/365:.1f} years on eight GPUs, or two months on sixty-four. Nobody plans that. The question that matters is how much <em>less</em> our recipe needs, and that has a measurable answer.</p>
</section>

<section>
  <h2>The frontier at 0.6B, September 2026</h2>
  <p>Every credible model in the 0.3B to 1B band, with the compute that produced it. Note the two things the table makes obvious: the strong entries were all trained on 10T tokens or more, and the newest one is not a plain transformer.</p>
  <div class="wide"><table>
    <thead><tr><th>Model</th><th>Architecture</th><th>Non-embedding params</th><th>Tokens</th><th>Pretrain FLOPs</th><th>Evidence</th><th>License</th></tr></thead>
    <tbody>{frontier_rows()}</tbody>
  </table></div>
  <p class="note">Llama 3.2 1B is omitted: it was pruned and distilled from Llama 3.1 8B, so its token count does not describe a from-scratch run. Gemma 4 ships nothing under 2B; Qwen3.6 and 3.8 have no small dense models; SmolLM3 is 3B only.</p>
</section>

<section>
  <h2>How far our recipe has to reach</h2>
  <p>A scaling law of the Chinchilla form, <span class="mono">L = E + A/N<sup>0.34</sup> + B/D<sup>0.28</sup></span>, lets us ask a precise question: at our parameter count, how many tokens reach the same loss each frontier model reached with its own parameters and tokens? Those are the bars below. The law knows nothing about data quality or optimizer, so each bar is drawn as a range: the right end assumes our tokens are worth exactly theirs, the left end assumes each of ours is worth four of theirs.</p>
  <div class="callout">
    <p><strong>Why a range of 1× to 4×.</strong> Puro's own result is the only evidence we have for the recipe: Puro-2B reached Qwen2-1.5B quality at 3.8× less compute and approached Qwen2.5-1.5B at about 10× less. Those baselines are 2024 recipes; Qwen3's 36T corpus is better curated. Planning on 2× to 4× is the honest read, and the first week of work replaces this guess with a measurement.</p>
  </div>
  <div class="chartwrap">
    {SVG}
    <div class="legend"><span><i class="sw"></i>tokens to loss parity, 4× → 1× recipe efficiency; tick at 2×</span><span><i class="sw ref"></i>our budget lines</span></div>
    <div class="tip" id="tip"></div>
  </div>
  <div class="wide"><table>
    <thead><tr><th>Target</th><th>k = 4</th><th>k = 2</th><th>k = 1</th><th>k = 1, hybrid shape (0.50B non-emb)</th><th>Days @ 8 GPU, k = 2</th></tr></thead>
    <tbody>{parity_rows()}</tbody>
  </table></div>
  <p class="note">† The law overstates Gemma 3 1B: it has more non-embedding parameters than we do, yet its base checkpoint scores MMLU 26.3 and GSM8K 2.2 in Qwen's evaluation. Compute parity is not benchmark parity, which is exactly why the plan measures rather than extrapolates. The hybrid column shows a second lever: carrying 0.50B rather than 0.44B non-embedding parameters cuts the Qwen3-0.6B parity budget from 36T to 14T before any efficiency is assumed.</p>
</section>

<section>
  <h2>Budget tiers</h2>
  <p>What each token budget costs and what it credibly buys. Days assume the measured rate; disk assumes int32 shards, which the 151,936-token vocabulary requires. Beyond about 5T tokens the shards no longer fit Fir's scratch at once and the batched pipeline runs in tranches.</p>
  <p class="note">Show days for <span class="seg" role="group" aria-label="GPU count"><button aria-pressed="true" data-g="8">8 GPUs</button><button aria-pressed="false" data-g="32">32 GPUs</button><button aria-pressed="false" data-g="64">64 GPUs</button></span></p>
  <div class="wide"><table id="tiers">
    <thead><tr><th>Tokens</th><th>FLOPs</th><th>Wall-clock</th><th>Shards on disk</th><th>Data source</th><th>Credibly beats</th></tr></thead>
    <tbody>{tier_rows()}</tbody>
  </table></div>
  <h3>Beat it on which axis</h3>
  <p>"Frontier" at this size is a bundle, and the parts scale differently. Broad knowledge (MMLU, MMLU-Pro) tracks total tokens and is the hardest to win; that is where the 9T to 18T lives. Math and code track the data mix far more than the token count, and the Puro phase-2 mixture is 18% mathematics, so a 1.4T to 4T run can plausibly pass Qwen3-0.6B on GSM8K and MATH while trailing it on MMLU. Instruction following is decided in post-training. Latency is decided by architecture. Choosing the axis is choosing the budget.</p>
</section>

<section>
  <h2>Measure it in week one</h2>
  <p>The efficiency factor k is the whole uncertainty in this plan, and it can be measured for about three days of GPU time. This is the Marin scaling-ladder method pointed at a specific question. Each rung takes its learning rate from Marin's fitted law (<code>MARIN_LR=1</code> in the recipe), so the rungs are comparable by construction rather than individually tuned.</p>
  <ul>
    <li><strong>Build a neutral held-out set.</strong> Marin's two are reusable: <em>Uncheatable Eval</em> (dated dumps of Wikipedia, BBC, arXiv, GitHub, AO3; public) and <em>Paloma</em>. The latest Uncheatable slices are September 2025, which is after Qwen3-0.6B's training but not after Qwen3.5-0.8B's, so a fresh September-2026 crawl of the same sources is needed for that comparison. Score in bits per byte so tokenizers compare fairly, and add our per-domain Puro validation sets alongside.</li>
    <li><strong>Score the frontier on it.</strong> Qwen3-0.6B-Base, Qwen3.5-0.8B-Base, Qwen2.5-0.5B and SmolLM2-360M, a few GPU-hours in total. Their bits-per-byte become horizontal lines on our loss chart.</li>
    <li><strong>Run our ladder.</strong> Three widths at the same recipe and 100 tokens per parameter: 188M, 365M and 596M, costing about {6*0.188e9*18.8e9/(8*PER_GPU)/3600:.0f}, {6*0.365e9*36.5e9/(8*PER_GPU)/3600:.0f} and {6*0.596e9*59.6e9/(8*PER_GPU)/3600:.0f} hours on 8 GPUs. Fit the law to our own points.</li>
    <li><strong>Read off the crossing.</strong> Where our fitted 0.6B curve crosses each frontier line is the token count that beats it, with error bars, on the same text. That number replaces every k in this document.</li>
  </ul>
  <p>The ladder also delivers a preregistered loss curve for the long run: at every 5% of training we know what loss to expect, so a run that drifts off it is caught in hours rather than discovered at the end.</p>
</section>

<section>
  <h2>Architecture</h2>
  <p>Two shapes are in contention for Brief. The dense one is the Qwen3-0.6B layout that the recipe and every tool already handle. The hybrid one is the Qwen3.5 layout, three Gated DeltaNet layers to every full-attention layer, and it is what the newest frontier model at this size actually uses.</p>
  <div class="wide"><table>
    <thead><tr><th>Shape</th><th>Layers</th><th>Params (tied, vocab 151,936)</th><th>KV cache per token</th><th>Ecosystem</th><th>Risk in our stack</th></tr></thead>
    <tbody>
      <tr><th scope="row">Dense (Qwen3-0.6B)</th><td>28 · h1024 · FFN 3072 · 16/8 heads × 128</td><td class="num">596M</td><td class="num">112 KB</td><td>Universal: HF, vLLM, llama.cpp, TensorRT-LLM, PEFT, Unsloth</td><td>None; validated end-to-end this week at 441 TFLOP/s/GPU</td></tr>
      <tr><th scope="row">Hybrid (Qwen3.5 layout, trimmed)</th><td>24 · h1024 · FFN 3072 · 6 attn (8/2 × 256) + 18 GDN (16 × 128)</td><td class="num">654M</td><td class="num">12 KB</td><td>vLLM 0.17 native, llama.cpp GDN kernels, Unsloth, MNN and LiteRT ports</td><td>Megatron support is marked experimental; needs flash-linear-attention and causal-conv1d installed; MuonHyperball treats the fused GDN projection as one matrix rather than per-logical-matrix</td></tr>
    </tbody>
  </table></div>
  <p>For a real-time system the hybrid's 9× smaller cache is a structural advantage that no amount of quantization recovers for the dense model. It also carries more non-embedding parameters, which the parity table shows is worth a factor of 2.5 in tokens. The recommendation is the hybrid, gated on a bake-off: both shapes at 188M and 365M, same recipe, same budget. If MuonHyperball misbehaves on the GDN layers, the dense shape is the fallback and nothing is lost.</p>
</section>

<section>
  <h2>The plan</h2>
  <div class="phases">
    <div class="phase"><div class="when">Week 1<small>gate: measured k, LR, shape</small></div><div class="what"><p>Finalize phase-1 data and per-domain validation sets. Build the neutral held-out set; score the frontier models on it. Run the dense-vs-hybrid bake-off at 188M and 365M, the peak-LR sweep at 20 tokens per parameter, and the three-rung ladder on the winning shape. Fit the law, write down the predicted loss at every 5% of the long run.</p></div><div class="cost"><b>~3 days</b>8×H100, detached runs</div></div>
    <div class="phase"><div class="when">Phase 1<small>gate: on the preregistered curve</small></div><div class="what"><p>439B tokens, the Puro phase-1 mixture, on the open-ended power schedule: the schedule the paper chose precisely because the total budget is decided later. Checkpoint every 500 steps with bounded retention; a pre-decay base checkpoint and optimizer state are kept for fine-tuners.</p></div><div class="cost"><b>{days(439e9,8):.1f} days</b>8×H100</div></div>
    <div class="phase"><div class="when">Decision<small>choose the horizon</small></div><div class="what"><p>With the measured crossing point and a real loss curve, pick the terminal budget: 1.4T (all of Puro), 4T, or the 9T-plus that contests Qwen3-0.6B on knowledge. Anything past 1.4T starts the Marin-catalog extension on the CPU allocation, in tranches, while training continues.</p></div><div class="cost"><b>0 GPU-days</b>a meeting</div></div>
    <div class="phase"><div class="when">Phase 2<small>gate: eval every 5%</small></div><div class="what"><p>The Puro phase-2 mixture (heavier on math and code, 1.3% instruction-formatted) on linear decay to the chosen horizon. Optional: logit distillation from a stronger teacher during the final anneal, which does not change what "from scratch" means for the weights but is the largest single quality lever available at this size.</p></div><div class="cost"><b>{days(938e9,8):.0f} days</b>to 1.4T at 8 GPUs; scale GPUs for more</div></div>
    <div class="phase"><div class="when">Hardening<small>real-time and fine-tunability</small></div><div class="what"><p>Context extension to 32K, quantization-aware training for FP8 and INT4, an EAGLE3 draft head for speculative decoding, exports to HF, vLLM and GGUF, latency at p50/p99 on the target device, and LoRA and full fine-tune smoke tests against the base checkpoint.</p></div><div class="cost"><b>~2 days</b>8×H100</div></div>
  </div>
</section>

<section>
  <h2>Decisions Brief needs from you</h2>
  <div class="decide">
    <p><strong>Which axis is "frontier" for Brief.</strong> Knowledge benchmarks set the budget at 9T to 18T and 32 or more GPUs. Math, code and latency are reachable within the Puro corpus on 8.</p>
    <p><strong>GPU count for the long run.</strong> Eight schedules in seconds on Fir and matches the original brief; 32 starts in about 27 minutes and divides every wall-clock figure by four. Checkpoints reshard, so this can change between phases.</p>
    <p><strong>Deployment target.</strong> GPU server, edge GPU, or CPU/mobile decides whether the hybrid's cache advantage is decisive and which quantization stage matters.</p>
  </div>
</section>

<section>
  <h2>Assumptions and sources</h2>
  <ul class="note">
    <li>Throughput: 441.4 TFLOP/s/GPU measured on one H100 at MBS 8, BF16; 95% multi-GPU scaling assumed; FP8 measured at +1.1% and not used. H100 BF16 dense peak taken as 989.4 TFLOP/s.</li>
    <li>Scaling law constants A=406.4, B=410.7, α=0.34, β=0.28 from Hoffmann et al. 2022; used for relative comparisons at similar N only.</li>
    <li>Qwen3.5-0.8B's token count is not published; 36T is assumed as a floor. Its 0.8B headline includes a ~116M vision tower; text-only is ~0.79B with 254M in embeddings.</li>
    <li>Puro corpus: 439B phase 1 + 938B phase 2 = 1.377T. Marin's deduplicated pool is 23.1T, largely ODC-BY and NVIDIA Data Agreement licensed.</li>
  </ul>
  <ul class="srcs">
    <li><a href="https://arxiv.org/abs/2608.27370">Puro-2B paper (arXiv 2608.27370)</a></li>
    <li><a href="https://ar5iv.labs.arxiv.org/html/2505.09388">Qwen3 Technical Report, Table 8</a></li>
    <li><a href="https://huggingface.co/Qwen/Qwen3.5-0.8B-Base">Qwen3.5-0.8B-Base model card</a></li>
    <li><a href="https://artificialanalysis.ai/articles/qwen3-5-small-models">Artificial Analysis on Qwen3.5 small models</a></li>
    <li><a href="https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models">Liquid AI LFM2 benchmarks</a></li>
    <li><a href="https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai">Liquid AI LFM2.5</a></li>
    <li><a href="https://huggingface.co/blog/ibm-granite/granite-4-nano">Granite 4.0 Nano</a></li>
    <li><a href="https://github.com/marin-community/marin/issues/8435">Marin Hero Run scaling ladder</a></li>
    <li><a href="https://huggingface.co/datasets/marin-community/token-counts">Marin data provenance table</a></li>
    <li><a href="https://arxiv.org/abs/2203.15556">Hoffmann et al., Training Compute-Optimal LLMs</a></li>
  </ul>
</section>
</main>

<script>
(function(){{
  var tip=document.getElementById('tip'), wrap=document.querySelector('.chartwrap');
  document.querySelectorAll('.chart .bar').forEach(function(g){{
    g.addEventListener('mousemove',function(e){{
      var d=g.dataset, r=wrap.getBoundingClientRect();
      tip.innerHTML='<b>'+d.name+'</b><br>4×: '+d.k4+' &nbsp;·&nbsp; 2×: '+d.k2+' &nbsp;·&nbsp; 1×: '+d.k1+'<br>at 2×: '+d.d8+' d @8 · '+d.d32+' d @32 · '+d.d64+' d @64';
      tip.style.display='block';
      var x=e.clientX-r.left+14, y=e.clientY-r.top+14;
      if(x+270>r.width) x=e.clientX-r.left-270;
      tip.style.left=x+'px'; tip.style.top=y+'px';
    }});
    g.addEventListener('mouseleave',function(){{tip.style.display='none';}});
  }});
  document.querySelectorAll('.seg button').forEach(function(b){{
    b.addEventListener('click',function(){{
      document.querySelectorAll('.seg button').forEach(function(x){{x.setAttribute('aria-pressed','false');}});
      b.setAttribute('aria-pressed','true');
      var g=b.dataset.g;
      document.querySelectorAll('#tiers td.days').forEach(function(td){{td.textContent=td.dataset['d'+g]+' d';}});
    }});
  }});
}})();
</script>
"""
out = "/private/tmp/claude-501/-Users-md-projects-brevity-brief/6975d33a-1322-4893-bfb6-a11e0daf0905/scratchpad/puro-0p6b-frontier-budget.html"
open(out, "w").write(page)
print("wrote", out, len(page), "bytes")
print(f"checks: parity Qwen3 dense={T(parity_tokens(0.44e9,36e12,NE_DENSE))} hybrid={T(parity_tokens(0.44e9,36e12,NE_HYB))}; year@8={T(YEAR8)}; fir={T(FIR_TOKENS)}")
