from nbtools import md, code, loader_cell

TITLE = "Part B — Auditing the UN SDG palette"

cells = [
md("""# Colour Brings People Together
### Part B · Auditing the UN SDG palette across languages, colour vision and age

The United Nations identifies its 17 Sustainable Development Goals with 17 colours, used unchanged in every country and every language. In the next 45 minutes you and your partner will test how well that palette communicates along two axes:

| Axis | Question | Assessed |
|---|---|---|
| **Perceptual** | Can people still tell the colours apart with a colour vision deficiency, or with an ageing eye? | within a community |
| **Cognitive** | Do people reliably agree on what each colour is called, and do the names in different languages point to the same colour? | across languages |

<img src="https://raw.githubusercontent.com/akvilesinkeviciute/colour-brings-people-together/v1.0/images/sdg_poster.png" alt="The 17 Sustainable Development Goals" width="620">

*The 17 Sustainable Development Goals and their colours. Source: United Nations, [Sustainable Development Goals communications materials](https://www.un.org/sustainabledevelopment/news/communications-material/). Visit the UN SDG website: [un.org/sustainabledevelopment](https://www.un.org/sustainabledevelopment). The content of this publication has not been approved by the United Nations and does not reflect the views of the United Nations or its officials or Member States.*

A palette can succeed on one axis and fail on the other. By the end you will have a two-axis picture of a subset of SDG colours, your **test colours**, and a comparison with what an AI system says the same colours are called.

**How to use this notebook.** Run each cell in order with **Shift + Enter**. The only things you need to change are the grey form fields on the right of some cells. Keep the [Colour Spinner](https://colornaming.net/spinner) open in another tab."""),

loader_cell(),

md("""## 1 · Meet the palette

Every swatch in this notebook sits on a neutral mid-grey surround. A white or black page changes how a colour looks, so colour scientists judge colours against grey."""),

code("""show_palette(SDG.hex.tolist(), labels=SDG.index.tolist(), captions=SDG.goal.tolist(),
             note="The 17 UN Sustainable Development Goal colours (United Nations 2015).")"""),

md("""## 2 · Choose your subset and languages

Pick **three to five** SDG colours. Good audits mix an easy colour with a hard one: try one of the reds (1, 4, 5, 8), one of the blues (6, 14, 16, 17) and something in between such as 12.

Tick the **languages** to audit, two or more. The Colour Spinner currently covers six. You'll choose which language to translate *from* later, in the translatability step. If either of you speaks one of them natively, include it, and include **OtjiHimba**, spoken by the Himba of northern Namibia, which has just seven indispensable colour terms: *serandu*, for example, covers what English splits into red, pink and orange.

**Backend.** *Colour Spinner API* fetches everything for you. If the network is slow, choose *Copied from the browser* and fill in the sheet in section 4."""),

code("""#@title Your pair's choices { run: "auto" }
Subset = "1, 6, 12, 14"  #@param {type:"string"}
SourceLanguage = "British English"  #@param ["American English", "British English", "French", "Greek", "OtjiHimba", "Spanish"]
#@markdown **Target languages** (tick any):
AmericanEnglish = False  #@param {type:"boolean"}
BritishEnglish = True  #@param {type:"boolean"}
French = False  #@param {type:"boolean"}
Greek = True  #@param {type:"boolean"}
OtjiHimba = True  #@param {type:"boolean"}
Spanish = True  #@param {type:"boolean"}
Backend = "Colour Spinner API"  #@param ["Colour Spinner API", "Copied from the browser"]

ticked = {"American English": AmericanEnglish, "British English": BritishEnglish,
          "French": French, "Greek": Greek, "OtjiHimba": OtjiHimba, "Spanish": Spanish}
LANGUAGES = [SourceLanguage] + [l for l, on in ticked.items() if on and l != SourceLanguage]
PICK = [int(s) for s in re.findall(r"\\d+", Subset) if 1 <= int(s) <= 17]
sub = SDG.loc[PICK]

show_palette(sub.hex.tolist(), labels=[f"SDG {i}" for i in sub.index], captions=sub.goal.tolist(),
             note=f"Auditing {len(PICK)} colours in {len(LANGUAGES)} languages: {', '.join(LANGUAGES)}.")"""),

md("""## 3 · The perceptual axis: can everyone tell them apart?

About 1 in 12 men and 1 in 200 women of Northern European descent have a colour vision deficiency, most often a reduced ability to separate reds from greens. Independently, the lens of the eye yellows with age and absorbs more short-wavelength light, so blues and greens drift towards one another.

The rows below show the whole palette under each condition, simulated by the Colour Spinner's own models, the same ones you see with its "Simulate age/CVD" menu: protanopia, deuteranopia and tritanopia, and lens yellowing at ages 40, 60 and 80. Look down each column for colours that change, and across each row for colours that merge.

These simulations show how a colour might *appear*, not how people with a colour vision deficiency or an older eye would *name* it: the Spinner's naming models were trained on people with typical colour vision."""),

code("""prefetch_simulations(SDG.hex.tolist())      # ask the Colour Spinner for all 17 colours at once
show_conditions(SDG.hex.tolist(), labels=[str(i) for i in SDG.index])"""),

md("""The chart below makes "merge" measurable. For each colour in your subset it finds the **nearest other SDG colour** under each condition and reports the CIEDE2000 colour difference (ΔE00) between them. A difference of about 1 is just noticeable side by side; for colours that must be told apart at a glance in a chart or a sign, aim for 10 or more. The small number under each value names the nearest neighbour.

**A note on the distances.** All colour differences in this notebook are CIEDE2000 (ΔE00). CIEDE2000 was derived from colour-difference judgements by observers with typical colour vision. Applied to simulated colours, it estimates how different those colours would look to a typical observer, which only approximates what people with a colour vision deficiency or an older eye can tell apart."""),

code("""#@title Nearest-neighbour distinctiveness { run: "auto" }
ConfusableBelow = 10  #@param {type:"slider", min:3, max:25, step:1}

def nearest(sdg_id, condition):
    me = simulate(SDG.hex[sdg_id], condition)
    best = min(((de00(me, simulate(h, condition)), j) for j, h in SDG.hex.items() if j != sdg_id))
    return best

nn = pd.DataFrame({c: {i: nearest(i, c) for i in PICK} for c in CONDITIONS})
nn_de = nn.map(lambda t: t[0]); nn_id = nn.map(lambda t: t[1])

fig, ax = plt.subplots(figsize=(1.1 * len(CONDITIONS) + 1.6, 0.62 * len(PICK) + 1.4))
ax.imshow(nn_de.values, cmap=RISK_CMAP, vmin=0, vmax=2 * ConfusableBelow, aspect="auto")
for (r, c), v in np.ndenumerate(nn_de.values):
    ink = "white" if v < ConfusableBelow * 0.6 else INK
    ax.text(c, r - 0.08, f"{v:.1f}", ha="center", va="center", fontsize=11, fontweight="bold", color=ink)
    ax.text(c, r + 0.27, f"vs {nn_id.values[r, c]}", ha="center", va="center", fontsize=8, color=ink)
ax.set_xticks(range(len(CONDITIONS)), [condition_label(c) for c in CONDITIONS], rotation=30, ha="right")
row_chips(ax, [SDG.hex[i] for i in PICK], [f"SDG {i}" for i in PICK])
ax.tick_params(axis="x", length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title(f"Distance to the nearest other SDG colour (ΔE00); red cells fall below {ConfusableBelow}")
plt.tight_layout(); plt.show()

at_risk = {}
for i in PICK:
    for c in CONDITIONS:
        j, d = nn_id.loc[i, c], nn_de.loc[i, c]
        if d < ConfusableBelow:
            at_risk.setdefault(tuple(sorted((i, j))), []).append((c, d))
if at_risk:
    rows = ""
    for (i, j), hits in sorted(at_risk.items(), key=lambda kv: min(d for _, d in kv[1])):
        worst_c = min(hits, key=lambda t: t[1])[0]
        chips = "".join(f'<div class="mini" style="background:{simulate(SDG.hex[k], worst_c)}"></div>' for k in (i, j))
        rows += (f'<tr><td><div class="row">{chips}</div></td><td><b>SDG {i} and {j}</b></td><td>' +
                 " ".join(f'<span style="white-space:nowrap">{_h(c)} {de_badge(d, ConfusableBelow, small_is_good=False)}</span>'
                          for c, d in hits) + "</td></tr>")
    show('<b>Pairs at risk</b>, shown as they look under the condition where they are closest'
         f'<table class="babel">{rows}</table>')
else:
    show(f"Every colour in your subset stays at least ΔE00 {ConfusableBelow} from its nearest neighbour under every condition.")"""),

md("""## 4 · The cognitive axis: what are these colours called?

Here you ask the Colour Spinner what speakers of each language actually call your colours. Its naming model, Spin Colour Forest, is trained on about 81,000 free colour names from 2,450 speakers, collected online at [colornaming.net](https://colornaming.net) and, for OtjiHimba, in fieldwork in Namibia. Treat it as a reference baseline rather than ground truth: it covers six languages, and in each only the **indispensable vocabulary**, the minimal set of terms needed to name every colour. That ranges from 7 terms in OtjiHimba to 47 in American English (32 in British English, Greek and Spanish; 27 in French).

Two measures come out of this step:

- **Nameability**: the Namer's *confidence*, the probability of its top-ranked name in the model's naming distribution for that colour. High confidence marks the centre of a category, where speakers agree; low confidence marks a contested boundary between categories.
- **Translatability**: whether each language's name points to the same colour. The Mapper's *best example* of each name, its **focal colour** (the top-left tile), shows where that name is centred; if the focal colours drift apart across languages, a "translation" of the colour lands somewhere else.

**Only if you chose *Copied from the browser*:** the next cell shows a row of boxes for each test colour and language. In the Spinner, type the colour's hex code into the Picker and set the language. Then copy the Namer's name and confidence, and the RGB of the Mapper's best example (its top-left tile), into the boxes. Otherwise skip it."""),

code("""SHEET = \"\"\"
hex, language, name, confidence, best_example
# C34F4F, American English, Red, 20%, 232 21 56      ← example row, as the Spinner shows #C34F4F
\"\"\"
"""),

code("""#@title Ask the Colour Spinner
spinner = ApiSpinner() if Backend == "Colour Spinner API" else SheetSpinner(SHEET)

records, problems = [], []
for i in PICK:
    for lang in LANGUAGES:
        hx = SDG.hex[i]
        got = try_call(spinner.name, hx, lang)
        if isinstance(got, Exception):
            problems.append(f"SDG {i} in {lang}: {got}"); continue
        best = try_call(spinner.best_example, got["name"], lang)
        best = None if isinstance(best, Exception) else best
        records.append(dict(sdg=i, hex=hx, language=lang, name=got["name"],
                            confidence=got["confidence"], best_example=best,
                            drift=de00(hx, best) if best else np.nan))
naming = pd.DataFrame(records)

if problems:
    show("<b>Some lookups did not work</b><br>" + "<br>".join(map(_h, problems[:8])) +
         ('<br>…' if len(problems) > 8 else "") +
         '<div class="note">With the API: check the connection or SPINNER_API. '
         'With the sheet: add the missing rows and run this cell again.</div>')
if len(naming):
    wide = naming.pivot(index="sdg", columns="language", values="name")
    conf = naming.pivot(index="sdg", columns="language", values="confidence").add_suffix("|conf")
    babel = SDG.loc[wide.index, ["hex"]].join(wide).join(conf)
    show_babel(babel, [l for l in LANGUAGES if l in wide.columns])"""),

md("""### Nameability

Each marker is one language. Colours near the right have a confident top name; colours on the left are where the Spinner is less certain what speakers would call them, and where a label such as "the red goal" will mean different things to different readers."""),

code("""MARKERS = dict(zip(SPINNER_LANGUAGES, ["o", "s", "D", "^", "v", "P"]))
rows = [(i, l) for i in PICK for l in LANGUAGES
        if not naming[(naming.sdg == i) & (naming.language == l)].empty]
gap = {i: k for k, i in enumerate(PICK)}
ys = [n + 0.6 * gap[i] for n, (i, l) in enumerate(rows)]
fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(rows) + 0.5 * len(PICK) + 1))
blend = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
for y, (i, l) in zip(ys, rows):
    r = naming[(naming.sdg == i) & (naming.language == l)].iloc[0]
    ax.hlines(y, 0, r.confidence, color="#DADAD5", lw=6, capstyle="round", zorder=1)
    ax.scatter(r.confidence, y, s=120, color=SDG.hex[i], edgecolor=INK, lw=0.7, zorder=3)
    ax.text(r.confidence + 0.025, y, f"{r['name']}  {r.confidence:.0%}", va="center", fontsize=10)
for i in PICK:
    g = [y for y, (j, _) in zip(ys, rows) if j == i]
    if not g: continue
    ax.add_patch(mpl.patches.Rectangle((-0.3, min(g) - 0.4), 0.07, max(g) - min(g) + 0.8,
                 color=SDG.hex[i], transform=blend, clip_on=False))
    ax.text(-0.265, (min(g) + max(g)) / 2, str(i), transform=blend, ha="center", va="center",
            rotation=90, color=_ink_on(SDG.hex[i]), fontweight="bold", clip_on=False)
ax.set_yticks(ys, [l for _, l in rows]); ax.tick_params(axis="y", length=0)
ax.invert_yaxis(); ax.set_xlim(0, 1.18); ax.spines["left"].set_visible(False)
ax.set_xticks([0, .25, .5, .75, 1]); ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
ax.set_xlabel("Spinner confidence in its top name")
ax.set_title("Nameability: how confident is the Spinner's top name?")
plt.tight_layout(); plt.show()"""),

md("""### Translatability

Translation is where a source and a target language come in. Choose the **source language** in the form below; every other language you ticked becomes a **target language**.

For each test colour, the first column shows the source language's name for it and that name's **best example** (focal colour), with its ΔE00 from the test colour. Each target column shows the target language's name for the same colour and its focal colour, with its ΔE00 from the **source's** focal colour. A small distance means the names translate: they point to the same colour. A large one means a translation would land on a different colour."""),

code("""rows = ""
for i in PICK:
    sub_n = naming[naming.sdg == i]
    cells = ""
    for lang in LANGUAGES:
        r = sub_n[sub_n.language == lang]
        if r.empty or not r.best_example.iloc[0]:
            cells += f'<td style="color:{MUTED}">—</td>'; continue
        b = r.best_example.iloc[0]
        cells += (f'<td><div class="pair" style="width:110px"><div style="background:{b};color:{_ink_on(b)}">'
                  f'{_h(r.name.iloc[0])}<br>{b}</div></div><div style="margin-top:3px">'
                  f'{de_badge(r.drift.iloc[0], ConfusableBelow)}</div></td>')
    sp = spread(sub_n.best_example.tolist())
    rows += (f'<tr><td><div class="pair" style="width:96px"><div style="background:{SDG.hex[i]};color:{_ink_on(SDG.hex[i])};'
             f'font-weight:700;font-size:13px">SDG {i}</div></div></td>{cells}'
             f'<td style="color:{MUTED}">{"" if pd.isna(sp) else "spread " + format(sp, ".1f")}</td></tr>')
show('<table class="babel"><tr><th></th>' + "".join(f"<th>{_h(l)}</th>" for l in LANGUAGES) +
     f'<th></th></tr>{rows}</table><div class="note">Badges: ΔE00 from the SDG colour. '
     'Spread: the largest ΔE00 between two languages\\' best examples.</div>')"""),

md("""The same information as a map. This is a top-down view of CAM16-UCS colour space, the perceptually uniform space the Colour Spinner works in: hue runs around the centre and colourfulness increases outwards. Each large disc is a test colour; the lines lead to where each language's name for it is centred."""),

code("""fig, ax = plt.subplots(figsize=(7, 7))
ax.axhline(0, color="#E4E4E0", lw=1, zorder=0); ax.axvline(0, color="#E4E4E0", lw=1, zorder=0)
for i in PICK:
    o = rgb_to_cam16ucs(SDG.hex[i])
    for _, r in naming[(naming.sdg == i) & naming.best_example.notna()].iterrows():
        b = rgb_to_cam16ucs(r.best_example)
        ax.plot([o[1], b[1]], [o[2], b[2]], color=MUTED, lw=1, zorder=1)
        ax.scatter(b[1], b[2], s=90, marker=MARKERS[r.language], color=r.best_example,
                   edgecolor=INK, lw=0.7, zorder=3)
    ax.scatter(o[1], o[2], s=520, color=SDG.hex[i], edgecolor="white", lw=2.5, zorder=2)
    ax.text(o[1], o[2], str(i), ha="center", va="center", color=_ink_on(SDG.hex[i]),
            fontweight="bold", fontsize=10, zorder=4)
ax.set_aspect("equal"); ax.set_xlabel("a′  (green ← → red)"); ax.set_ylabel("b′  (blue ← → yellow)")
ax.legend(handles=[mpl.lines.Line2D([], [], marker=MARKERS[l], ls="", color="white",
          markeredgecolor=INK, markersize=8, label=l) for l in LANGUAGES], frameon=False, loc="best")
ax.set_title("Where each language's name pulls the colour")
plt.tight_layout(); plt.show()"""),

md("""## 5 · Two axes, one picture

Each disc is a test colour. Across: its **perceptual distance**, the ΔE00 to the nearest other SDG colour, for typical colour vision unless you choose another condition in the form. Up: its **naming confidence**, averaged over your languages unless you choose one. Both axes run from 0 to 1, so the two are on the same scale. The grey circle marks where the two thresholds meet. The table underneath gives the numbers, with each language's name and confidence."""),

code("""#@title Scorecard { run: "auto" }
HardToNameBelow = 0.5  #@param {type:"slider", min:0.2, max:0.9, step:0.05}
#@markdown A colour whose top name has a Spinner confidence below this counts as hard to name.

score = pd.DataFrame({
    "worst_case_dE00": nn_de.min(axis=1),
    "worst_condition": nn_de.idxmin(axis=1),
    "mean_confidence": naming.groupby("sdg").confidence.mean(),
    "language_spread": naming.groupby("sdg").best_example.apply(lambda s: spread(s.tolist())),
}).loc[PICK]

fig, ax = plt.subplots(figsize=(7.5, 6))
xmax = max(2.2 * ConfusableBelow, score.worst_case_dE00.max() * 1.15)
ax.axvline(ConfusableBelow, color="#BDBDB8", ls="--", lw=1); ax.axhline(HardToNameBelow, color="#BDBDB8", ls="--", lw=1)
for x, y, t in [(0.02, 0.98, "named, but easily confused"), (0.98, 0.98, "robust on both axes"),
                (0.02, 0.02, "fragile on both axes"), (0.98, 0.02, "distinct, but hard to name")]:
    ax.text(x, y, t, transform=ax.transAxes, ha="left" if x < .5 else "right",
            va="top" if y > .5 else "bottom", color=MUTED, fontsize=10, style="italic")
for i, r in score.iterrows():
    ax.scatter(r.worst_case_dE00, r.mean_confidence, s=900, color=SDG.hex[i], edgecolor="white", lw=2.5, zorder=3)
    ax.text(r.worst_case_dE00, r.mean_confidence, str(i), ha="center", va="center",
            color=_ink_on(SDG.hex[i]), fontweight="bold", zorder=4)
ax.set_xlim(0, xmax); ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
ax.set_xlabel("Perceptual: worst-case ΔE00 to nearest neighbour"); ax.set_ylabel("Cognitive: mean naming confidence")
ax.set_title("Where your colours sit on the two axes")
plt.tight_layout(); plt.show()
display(score.round(2))"""),

md("""## 6 · What does an AI system say?

Now ask a current AI system to name the same colours in the same languages, and compare its answers with how speakers actually name them.

**Which AI?** Use the AI you use day to day: the comparison is most meaningful that way. The next cell shows the prompt with buttons for ChatGPT, Claude and Gemini. Each button copies the prompt and opens the tool; if the prompt isn't filled in there, paste it. Using another AI? Click **Copy prompt** and paste it there. Below the prompt is one numbered row per test colour: paste the names your AI gives into the boxes."""),

code("""lang_note = "; ".join(f"{l}" + (" (spoken by the Himba of northern Namibia)" if l == "OtjiHimba" else "")
                      for l in LANGUAGES)
PROMPT_B = f\"\"\"You are helping with a cross-cultural colour naming study.
For each colour below, give the single colour name that most native speakers of each language would use for it.
Write each name in that language and in its usual script; for a non-Latin script, add the Latin spelling in brackets.

Languages: {lang_note}
Colours (sRGB hex): {", ".join(SDG.hex[i] for i in PICK)}

Reply with one line for every colour and language, and nothing else, in exactly this format:
#E5243B | {LANGUAGES[-1]} | name\"\"\"
print(PROMPT_B)"""),

code("""#@title AI system
AI = "Gemini built into Colab (no key needed)"  #@param ["Gemini built into Colab (no key needed)", "Paste the reply myself", "Anthropic", "OpenAI-compatible"]
AI_MODEL = ""  #@param {type:"string"}
AI_BASE_URL = ""  #@param {type:"string"}
#@markdown Colab's Gemini is for demonstration; any AI you regularly use works via *Paste the reply myself*.
#@markdown Leave **AI_MODEL** empty to use Colab's default Gemini model. Run `from google.colab import ai; ai.list_models()` to see the others.
#@markdown *Anthropic* and *OpenAI-compatible* need your own key in Colab ▸ Secrets (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) and a model name."""),

code("""AI_REPLY = \"\"\"
paste the AI's reply here, only if you chose "Paste the reply myself"
\"\"\"
"""),

code("""reply_b = AI_REPLY if AI.startswith("Paste") else ask_ai(PROMPT_B, AI, AI_MODEL, AI_BASE_URL)
ai_names = pd.DataFrame(parse_json_reply(reply_b))
ai_names["hex"] = ai_names["hex"].map(parse_colour)
ai_names = ai_names.rename(columns={"name": "ai_name"})[["hex", "language", "ai_name"]]

cmp_b = naming.merge(ai_names, on=["hex", "language"], how="left")
cmp_b["agrees"] = [isinstance(a, str) and norm(a) == norm(s) for a, s in zip(cmp_b.ai_name, cmp_b.name)]

rows = ""
for _, r in cmp_b.iterrows():
    mark = ("<span class='badge' style='background:#DDEFE3;color:#14532D'>same</span>" if r.agrees else
            "<span class='badge' style='background:#F9DADA;color:#7F1D1D'>differs</span>")
    rows += (f'<tr><td><div class="mini" style="background:{r.hex}"></div></td><td>SDG {r.sdg}</td>'
             f'<td>{_h(r.language)}</td><td><b>{_h(r["name"])}</b> '
             f'<span style="color:{MUTED};font-size:11px">{r.confidence:.0%}</span></td>'
             f'<td><b>{_h(r.ai_name if isinstance(r.ai_name, str) else "—")}</b></td><td>{mark}</td></tr>')
show('<table class="babel"><tr><th></th><th></th><th>Language</th><th>Speakers (Spinner)</th>'
     f'<th>AI system</th><th></th></tr>{rows}</table>')

by_lang = cmp_b.groupby("language").agrees.mean().reindex(LANGUAGES)
fig, ax = plt.subplots(figsize=(8, 0.45 * len(LANGUAGES) + 1.2))
ax.barh(by_lang.index, by_lang.values, color=INK, height=0.5)
for y, v in enumerate(by_lang.values):
    ax.text(v + 0.02, y, f"{v:.0%}", va="center", fontsize=10)
ax.set_xlim(0, 1.1); ax.invert_yaxis(); ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
ax.set_title("How often the AI agrees with speakers")
plt.tight_layout(); plt.show()"""),

md("""A match on a colour with low confidence says little: when the Spinner's confidence in its top name is only around 30%, other names are also defensible. So for every name that differs, the last column asks the Spinner's Mapper where the AI's name is centred for speakers of that language. If that colour is close to the test colour, the AI chose a plausible neighbouring name. If it is far away, or the name isn't one speakers use at all, the AI's name would mislead them. Large distances on high-agreement colours are the ones to bring to the group.

## 7 · Save your results for Part C

Part C picks up the names you found here. Run the next cell to download `partB_results.csv`. If you are working in Colab, keep the tab open: Part C can also reload it from your Downloads."""),

code("""out = naming.merge(ai_names, on=["hex", "language"], how="left") if "ai_names" in globals() else naming
out.to_csv("partB_results.csv", index=False)
download("partB_results.csv")"""),

md("""## Before you share with the room

Agree with your partner on one sentence for each:

1. Which colour performs best on both axes, and which worst?
2. Is your worst colour failing **perceptually**, **cross-linguistically** or **both**?
3. Where did the AI system's names diverge from how speakers name the colour, and does the divergence look like a default towards English or "universal" colour categories?"""),
]


# ── short cells: charts and tables now live in colour_masterclass.py ──
def _set(i, src, hidden=None):
    assert cells[i]["cell_type"] == "code", i
    cells[i] = code(src, hidden=hidden)

_form = "".join(cells[5]["source"]).split("ticked = ")[0]
_set(5, _form + """ticked = {"American English": AmericanEnglish, "British English": BritishEnglish,
          "French": French, "Greek": Greek, "OtjiHimba": OtjiHimba, "Spanish": Spanish}
LANGUAGES = pick_languages(SourceLanguage, ticked)
PICK = parse_subset(Subset)
show_subset(PICK, LANGUAGES)""")

_set(9, """#@title Nearest-neighbour distinctiveness { run: "auto" }
ConfusableBelow = 10  #@param {type:"slider", min:3, max:25, step:1}
#@markdown Two colours closer than this CIEDE2000 colour difference count as easily confused. About 1 is just noticeable side by side; 10 is a common rule of thumb for telling colours apart at a glance.

nn_de, nn_id = nearest_neighbours(PICK)      # ΔE00 to the nearest other SDG colour, per condition
plot_nearest_neighbours(nn_de, nn_id, ConfusableBelow)
show_at_risk(nn_de, nn_id, ConfusableBelow)""")

_set(12, """spinner = connect(Backend, SHEET)

rows, problems = [], []
for sdg in PICK:
    for language in LANGUAGES:
        colour = SDG.hex[sdg]
        try:
            namer = spinner.name(colour, language)                  # Namer: top name and its confidence
        except Exception as e:
            problems.append(f"SDG {sdg} in {language}: {e}"); continue
        best = safe(spinner.best_example, namer["name"], language)  # Mapper: where that name is centred
        rows.append({"sdg": sdg, "hex": colour, "language": language,
                     "name": namer["name"], "romanized": namer.get("romanized"),   # Latin spelling, e.g. kokkino
                     "confidence": namer["confidence"], "best_example": best,
                     "drift": de00(colour, best) if best else np.nan})  # how far the name pulls the colour
naming = pd.DataFrame(rows)

report_problems(problems)
show_names_table(naming, LANGUAGES)""")

_set(14, "plot_nameability(naming, PICK, LANGUAGES)")
_set(16, "show_translatability(naming, PICK, LANGUAGES, ConfusableBelow)")
_set(18, "plot_name_map(naming, PICK, LANGUAGES)")
_set(20, """#@title Scorecard { run: "auto" }
HardToNameBelow = 0.5  #@param {type:"slider", min:0.2, max:0.9, step:0.05}
#@markdown A colour whose top name has a Spinner confidence below this counts as hard to name.

score = scorecard(nn_de, naming)
plot_two_axes(score, ConfusableBelow, HardToNameBelow)
display(score.round(2))""")
_set(25, """reply_b = ai_reply(PROMPT_B, AI, AI_MODEL, AI_BASE_URL, pasted=AI_REPLY)
ai_names = read_ai_names(reply_b)

comparison = naming.merge(ai_names, on=["hex", "language"], how="left")
comparison["agrees"] = [same_name(ai, name, rom)    # any script; ignores case and accents
                        for ai, name, rom in zip(comparison.ai_name, comparison.name, comparison.romanized)]
comparison = where_ai_names_point(comparison, spinner)   # Mapper: where each differing AI name is centred
show_name_comparison(comparison, ConfusableBelow)
plot_agreement_by_language(comparison, LANGUAGES)""")


# ── AI step: choose first, then prompt (+ paste box), then read the reply ──
_p = next(k for k, c in enumerate(cells) if c["cell_type"] == "code" and "PROMPT_B = f" in "".join(c["source"]))
_prompt_src = "".join(cells[_p]["source"]).replace("print(PROMPT_B)", "show_prompt(PROMPT_B, AI)")
_form = next(k for k, c in enumerate(cells) if c["cell_type"] == "code" and "".join(c["source"]).startswith("#@title AI system"))
_reply = next(k for k, c in enumerate(cells) if c["cell_type"] == "code" and "".join(c["source"]).startswith("AI_REPLY ="))
assert _p < _form < _reply
cells[_p] = code('#@title Which AI?\nAI = "Gemini built into Colab (no key needed)"  #@param ["Gemini built into Colab (no key needed)", "Paste the reply from my own AI", "Anthropic", "OpenAI-compatible"]\nAI_MODEL = ""  #@param {type:"string"}\nAI_BASE_URL = ""  #@param {type:"string"}\n#@markdown **Gemini built into Colab** needs nothing more: just run the cells below.\n#@markdown **Your own AI** (ChatGPT, Claude, Copilot, …): choose *Paste the reply from my own AI*; the next cell gives you the prompt and a box to paste the reply into.\n#@markdown *Anthropic* and *OpenAI-compatible* call an API with your own key from Colab ▸ Secrets (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) and a model name. Leave AI_MODEL empty for Colab\'s default Gemini.')
cells[_form] = code(_prompt_src)
del cells[_reply]
_r = next(k for k, c in enumerate(cells) if c["cell_type"] == "code" and "reply_b = ai_reply" in "".join(c["source"]))
cells[_r] = code("".join(cells[_r]["source"]).replace(", pasted=AI_REPLY", ""))


# ── round of notes, October 2026 ──
def _find(pred):
    return next(k for k, c in enumerate(cells) if pred("".join(c["source"]), c["cell_type"]))
cells[_find(lambda s, t: t == "code" and s.startswith("#@title Which AI?"))] = code('#@title Which AI?\nAI = "Paste answers from my own AI"  #@param ["Paste answers from my own AI", "Gemini built into Colab"]\n#@markdown **Your own AI** (ChatGPT, Claude, Gemini, Copilot, …): the next cell shows a prompt to copy into it and one box per colour and language. The colour codes are filled in for you; just copy in the names your AI gives.\n#@markdown **Gemini built into Colab** is for demonstration. It needs no set-up but stops working once Colab\'s free usage limit is reached.')
cells[_find(lambda s, t: t == "code" and "PROMPT_B = f" in s)] = code('lines = "\\n".join(f"{r.hex} | {r.language}" for _, r in naming.iterrows())\nhimba = "\\n(OtjiHimba is spoken by the Himba of northern Namibia.)" if "OtjiHimba" in LANGUAGES else ""\nPROMPT_B = f"""You are helping with a cross-cultural colour naming study.\nFor each colour code and language below, give the single colour name that most native speakers of that\nlanguage would use for it. Write each name in that language and in its usual script; for a non-Latin\nscript, add the Latin spelling in brackets.{himba}\n\n{lines}\n\nReply with exactly one line for each, in this format and nothing else:\n#E5243B | {LANGUAGES[-1]} | name"""\nshow_prompt(PROMPT_B, AI, rows=naming[["hex", "language"]], kind="name")')
cells[_find(lambda s, t: t == "code" and "PROMPT_B, AI" in s and "comparison" in s)] = code('if AI == PASTE:\n    ai_names = collected_answers()                        # the names typed into the boxes\nelse:\n    ai_names = read_ai_names(ask_ai(PROMPT_B, AI))\nai_names["language"] = ai_names["language"].map(canonical_language)\n\ncomparison = naming.merge(ai_names, on=["hex", "language"], how="left")\ncomparison["agrees"] = [same_name(ai, name, rom)    # any script; ignores case and accents\n                        for ai, name, rom in zip(comparison.ai_name, comparison.name, comparison.romanized)]\ncomparison = where_ai_names_point(comparison, spinner)   # Mapper: where each differing AI name is centred\nshow_name_comparison(comparison, ConfusableBelow)\nplot_agreement_by_language(comparison, LANGUAGES)')
cells[_find(lambda s, t: t == "code" and "card = scorecard" in s or "score = scorecard" in s)] = code('#@title Scorecard { run: "auto" }\nLanguage = "Average of my languages"  #@param ["Average of my languages", "American English", "British English", "French", "Greek", "OtjiHimba", "Spanish"]\nVision = "Typical"  #@param ["Typical", "Protanopia", "Deuteranopia", "Tritanopia", "Age 40", "Age 60", "Age 80", "Worst case of all conditions"]\nNormalise = "Largest value in my subset"  #@param ["Largest value in my subset", "Range across all 17 SDG colours"]\nHardToNameBelow = 0.5  #@param {type:"slider", min:0.2, max:0.9, step:0.05}\n#@markdown **Language**: whose naming confidence to plot; choose one of the languages you ticked in section 2, or the average. **Vision**: whose vision the perceptual distance is for (*Typical* = typical colour vision).\n#@markdown A colour whose top name has a Spinner confidence below **HardToNameBelow** counts as hard to name.\n\ncard = scorecard(PICK, naming, Language, Vision, Normalise)\nplot_two_axes(card, ConfusableBelow, HardToNameBelow)\nshow_scorecard(card)', hidden=False)


# ── languages chosen once; source language only where translating ──
cells[_find(lambda s, t: t == "code" and s.startswith("#@title Your pair's choices"))] = code('#@title Your pair\'s choices { run: "auto" }\nSubset = "1, 6, 12, 14"  #@param {type:"string"}\n#@markdown **Languages to audit** (tick two or more):\nAmericanEnglish = False  #@param {type:"boolean"}\nBritishEnglish = True  #@param {type:"boolean"}\nFrench = False  #@param {type:"boolean"}\nGreek = True  #@param {type:"boolean"}\nOtjiHimba = True  #@param {type:"boolean"}\nSpanish = True  #@param {type:"boolean"}\nBackend = "Colour Spinner API"  #@param ["Colour Spinner API", "Copied from the browser"]\n\nticked = {"American English": AmericanEnglish, "British English": BritishEnglish,\n          "French": French, "Greek": Greek, "OtjiHimba": OtjiHimba, "Spanish": Spanish}\nLANGUAGES = [language for language, on in ticked.items() if on]\nPICK = parse_subset(Subset)\nshow_subset(PICK, LANGUAGES)')
cells[_find(lambda s, t: t == "code" and s.strip().startswith("show_translatability("))] = code('#@title Translatability { run: "auto" }\nSourceLanguage = "British English"  #@param ["American English", "British English", "French", "Greek", "OtjiHimba", "Spanish"]\n#@markdown The language to translate **from**. Every other language you ticked is a **target language**.\n\nshow_translatability(naming, PICK, LANGUAGES, ConfusableBelow, source=SourceLanguage)', hidden=False)


# ── boxes instead of code for values copied from the browser ──
cells[_find(lambda s, t: t == "code" and s.startswith('SHEET = """'))] = code("""#@title Only if Backend is "Copied from the browser"
if Backend == "Copied from the browser":
    sheet_boxes(PICK, LANGUAGES)      # one row of boxes per test colour and language
else:
    show("You're using the Colour Spinner directly, so there's nothing to fill in here.")""", hidden=False)
_k = _find(lambda s, t: t == "code" and s.startswith("spinner = connect(Backend, SHEET)"))
cells[_k] = code("".join(cells[_k]["source"]).replace("connect(Backend, SHEET)", "connect(Backend)"))


# ── numbered prompt and numbered answer boxes ──
cells[_find(lambda s, t: t == "code" and "PROMPT_B = f" in s)] = code('lines = "\\n".join(numbered_items(naming[["hex", "language"]], "name"))\nhimba = "\\n(OtjiHimba is spoken by the Himba of northern Namibia.)" if "OtjiHimba" in LANGUAGES else ""\nPROMPT_B = f"""You are helping with a cross-cultural colour naming study.\nFor each numbered colour code below, give the single colour name that most native speakers of that\nlanguage would use for it. Write each name in that language and in its usual script; for a non-Latin\nscript, add the Latin spelling in brackets.{himba}\n\n{lines}\n\nReply with one line per number, in this format and nothing else:\n1. name\n2. …"""\nshow_prompt(PROMPT_B, AI, rows=naming[["hex", "language"]], kind="name")')
_k = _find(lambda s, t: t == "code" and 'ai_names = read_ai_names(ask_ai(PROMPT_B, AI))' in s)
cells[_k] = code("".join(cells[_k]["source"]).replace('ai_names = read_ai_names(ask_ai(PROMPT_B, AI))', 'ai_names = answers_from_reply(ask_ai(PROMPT_B, AI), naming[["sdg", "hex", "language"]], "name")'))


# ── one numbered row per test colour, a box per language ──
cells[_find(lambda s, t: t == "code" and "PROMPT_B = f" in s)] = code('lines = "\\n".join(numbered_items(naming, "name"))          # one number per test colour\nhimba = "\\n(OtjiHimba is spoken by the Himba of northern Namibia.)" if "OtjiHimba" in LANGUAGES else ""\nreply_format = " | ".join(f"{language}: name" for language in LANGUAGES)\nPROMPT_B = f"""You are helping with a cross-cultural colour naming study.\nFor each numbered colour code below, give the single colour name that most native speakers would use for it in each of these languages: {", ".join(LANGUAGES)}. Write each name in that language and in its usual script; for a non-Latin script, add the Latin spelling in brackets.{himba}\n\n{lines}\n\nReply with one line per number, in this format and nothing else:\n{PICK[0]}. {reply_format}\n…"""\nshow_prompt(PROMPT_B, AI, rows=naming[["sdg", "hex", "language"]], kind="name")')


# ── one AI route for everyone: buttons to open any AI, numbered boxes for the answers ──
del cells[_find(lambda s, t: t == "code" and s.startswith("#@title Which AI?"))]
_k = _find(lambda s, t: t == "code" and "show_prompt(PROMPT_B, AI," in s)
cells[_k] = code("".join(cells[_k]["source"]).replace("show_prompt(PROMPT_B, AI,", "show_prompt(PROMPT_B, PASTE,"))
_k = _find(lambda s, t: t == "code" and s.startswith("if AI == PASTE:"))
_lines = "".join(cells[_k]["source"]).split("\n")
cells[_k] = code("\n".join(['ai_names = collected_answers()                         # the names pasted into the boxes'] + _lines[4:]))


# ── one story: your palette, your language, your audience ──
def _md(i, start, new):
    assert cells[i]["cell_type"] == "markdown" and "".join(cells[i]["source"]).startswith(start), (i, start)
    cells[i] = md(new)
def _code(i, contains, new, hidden=None):
    assert cells[i]["cell_type"] == "code" and contains in "".join(cells[i]["source"]), (i, contains)
    cells[i] = code(new, hidden=hidden)
def _sub(i, a, b):
    src = "".join(cells[i]["source"]); assert a in src, (i, a); cells[i] = code(src.replace(a, b), hidden=cells[i]["metadata"].get("cellView") == "form")

_m0 = "".join(cells[0]["source"])
cells[0] = md(_m0.replace("A palette can succeed on one axis and fail on the other.",
    "**The story.** You are a designer working in your own language, choosing a small palette of SDG colours for an "
    "international audience. You'll check whether everyone can tell your colours apart, whether your audience calls them "
    "what you call them, and whether an AI gets it right. In Part C you'll fix the palette for each audience.\n\n"
    "A palette can succeed on one axis and fail on the other.", 1))

_md(4, "## 2 ·", """## 2 · Your palette, your language and your audience

**Your test palette.** Pick **three to five** SDG colours, as if for a chart, a map or a set of signs. Good choices mix an easy colour with a hard one: one of the reds (1, 4, 5, 8), one of the blues (6, 14, 16, 17) and something in between such as 12. These are your **test colours** for the rest of Parts B and C.

**Your language.** The language you, the designer, name and choose colours in. Usually your own.

**Your audience.** Who will read your design. Tick their languages; include at least one very different from yours, such as **OtjiHimba**, spoken by the Himba of northern Namibia, which has just seven indispensable colour terms (*serandu*, for example, covers what English splits into red, pink and orange).

**Backend.** *Colour Spinner API* fetches everything for you. If the network is slow, choose *Copied from the browser* and fill in the boxes in section 4.""")

_code(5, "#@title Your pair's choices", """#@title Your choices { run: "auto" }
Subset = "1, 6, 12, 14"  #@param {type:"string"}
YourLanguage = "British English"  #@param ["American English", "British English", "French", "Greek", "OtjiHimba", "Spanish"]
#@markdown **Your audience's languages** (tick one or more; your own language is always included):
AmericanEnglish = False  #@param {type:"boolean"}
BritishEnglish = False  #@param {type:"boolean"}
French = False  #@param {type:"boolean"}
Greek = True  #@param {type:"boolean"}
OtjiHimba = True  #@param {type:"boolean"}
Spanish = True  #@param {type:"boolean"}
Backend = "Colour Spinner API"  #@param ["Colour Spinner API", "Copied from the browser"]

ticked = {"American English": AmericanEnglish, "British English": BritishEnglish,
          "French": French, "Greek": Greek, "OtjiHimba": OtjiHimba, "Spanish": Spanish}
AUDIENCE = [language for language, on in ticked.items() if on and language != YourLanguage]
LANGUAGES = [YourLanguage] + AUDIENCE                 # you first, then your audience
PICK = parse_subset(Subset)
show_subset(PICK, LANGUAGES)
show(f"<b>You:</b> {YourLanguage} &nbsp;·&nbsp; <b>Your audience:</b> {', '.join(AUDIENCE) or 'none ticked yet'}")""")

_md(6, "## 3 ·", """## 3 · Can everyone tell your colours apart?

About 1 in 12 men and 1 in 200 women of Northern European descent have a colour vision deficiency, most often a reduced ability to separate reds from greens. Independently, the lens of the eye yellows with age and absorbs more blue light.

**First, some context:** the whole UN palette under each condition, simulated by the Colour Spinner's own models (the ones behind its "Simulate age/CVD" menu): protanopia, deuteranopia, tritanopia, and lens yellowing at ages 40, 60 and 80. Look down each column for colours that change.

These simulations show how a colour might *appear*, not how people with a colour vision deficiency or an older eye would *name* it: the Spinner's naming models were trained on people with typical colour vision.""")

_md(8, "The chart below makes", """**Now your palette.** For each of your test colours, the chart finds the **nearest other test colour** under each condition and reports the CIEDE2000 colour difference (ΔE00) between them. About 1 is just noticeable side by side; for colours that must be told apart at a glance, as in a chart legend or on a sign, aim for 10 or more. The table underneath shows each pair that falls below your threshold, and why those two colours look alike.

**A note on the distances.** All colour differences in this notebook are CIEDE2000 (ΔE00). CIEDE2000 was derived from colour-difference judgements by observers with typical colour vision. Applied to simulated colours, it estimates how different those colours would look to a typical observer, which only approximates what people with a colour vision deficiency or an older eye can tell apart.""")
_sub(9, "nn_de, nn_id = nearest_neighbours(PICK)", "palette_strip(PICK)\nnn_de, nn_id = nearest_neighbours(PICK)")
_sub(9, "# ΔE00 to the nearest other SDG colour, per condition", "# ΔE00 to the nearest other test colour, per condition")

_m10 = "".join(cells[10]["source"])
cells[10] = md(_m10.replace("## 4 · The cognitive axis: what are these colours called?\n\nHere you ask the Colour Spinner what speakers of each language actually call your colours.",
    "## 4 · Do you and your audience call these colours the same?\n\nHere you ask the Colour Spinner what speakers of your language and of each audience language actually call your test colours.", 1))
_sub(12, '"name": namer["name"], "romanized": namer.get("romanized"),',
         '"role": "your language" if language == YourLanguage else "audience",\n                     "name": namer["name"], "romanized": namer.get("romanized"),')
_sub(12, "show_names_table(naming, LANGUAGES)", "palette_strip(PICK)\nshow_names_table(naming, LANGUAGES, you=YourLanguage)")

_md(15, "### Translatability", """### Translatability: do your names point where your audience's do?

The first column is **your language**: your name for each test colour and that name's **best example** (focal colour), with its ΔE00 from the test colour. Each audience column shows that language's name for the same colour and its focal colour, with its ΔE00 from **yours**. A small distance means the names translate: they point to the same colour. A large one means a translation would land on a different colour.""")
_code(16, "show_translatability(", "show_translatability(naming, PICK, LANGUAGES, ConfusableBelow, source=YourLanguage)")

cells[18:19] = [cells[18],
    md("""### Your names in all six languages

The Colour Spinner's **Translator** takes your name for each test colour and finds the best-matching name in every language it covers, including those outside your audience. *Translation confidence* is how closely the two names are used for the same colours: low confidence means there is no clean equivalent."""),
    code("show_translations(spinner, naming, PICK, YourLanguage)")]

_i = next(k for k, c in enumerate(cells) if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("## 5 ·"))
_md(_i, "## 5 ·", """## 5 · Two axes, one picture

Each disc is a test colour. Across: its **perceptual distance**, the ΔE00 to the nearest other test colour, for typical colour vision unless you choose another condition. Up: its **naming confidence**, for your language, one audience language, or averaged over your audience. Both axes run from 0 to 1. The grey circle marks where your two thresholds meet. The table underneath gives the numbers, with each language's name and confidence.""")
_k = next(k for k, c in enumerate(cells) if c["cell_type"] == "code" and "card = scorecard(" in "".join(c["source"]))
_src = "".join(cells[_k]["source"])
_src = _src.replace('Language = "Average of my languages"  #@param ["Average of my languages", "American English", "British English", "French", "Greek", "OtjiHimba", "Spanish"]',
                    'Language = "Average of my audience"  #@param ["Average of my audience", "My language", "Average of all my languages", "American English", "British English", "French", "Greek", "OtjiHimba", "Spanish"]')
_src = _src.replace("card = scorecard(PICK, naming, Language, Vision, Normalise)",
                    'chosen = {"My language": YourLanguage, "Average of all my languages": ALL_LANGUAGES}.get(Language, Language)\n'
                    'palette_strip(PICK)\ncard = scorecard(PICK, naming, chosen, Vision, Normalise, audience=AUDIENCE)')
_src = _src.replace("#@markdown **Language**: whose naming confidence to plot; choose one of the languages you ticked in section 2, or the average.",
                    "#@markdown **Language**: whose naming confidence to plot: your audience on average, your own language, or one language you chose in section 2.")
cells[_k] = code(_src, hidden=False)

_i = next(k for k, c in enumerate(cells) if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("## 6 ·"))
_m = "".join(cells[_i]["source"])
cells[_i] = md(_m.replace("## 6 · What does an AI system say?\n\nNow ask a current AI system to name the same colours in the same languages, and compare its answers with how speakers actually name them.",
    "## 6 · Does an AI name them the way your audience does?\n\nDesigners increasingly ask AI tools for colour names and translations. Ask a current AI system to name your test colours in your language and your audience's, and compare its answers with how speakers actually name them.", 1))

_i = next(k for k, c in enumerate(cells) if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("## Before you share"))
cells[_i] = md("""## Before you share with the room

Agree with your partner on one sentence for each:

1. Which of your test colours works best for **all** your audience, and which worst?
2. Is your worst colour failing **perceptually** (people can't tell it apart), **cross-linguistically** (your audience calls it something else), or **both**?
3. Where did the AI's names diverge from your audience's, and does that look like a default towards English or "universal" colour categories?

In Part C you'll take your palette's names and fix the palette for each audience.""")


# ── the Colour Spinner screenshot where participants first use the Spinner ──
_i = next(k for k, c in enumerate(cells) if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("## 4 ·"))
_src = "".join(cells[_i]["source"])
_head, _rest = _src.split("\n\n", 1)
cells[_i] = md(_head + "\n\n" + '<img src="https://raw.githubusercontent.com/akvilesinkeviciute/colour-brings-people-together/v1.0/images/colour_spinner.png" alt="The Colour Spinner" width="560">\n\n*The Colour Spinner\'s six linked views: Picker, Namer, Thesaurus, Translator, Mapper and Profiler, with controls for language and for colour vision deficiency and ageing simulations.*' + "\n\n" + _rest)
