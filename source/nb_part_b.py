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

A palette can succeed on one axis and fail on the other. By the end you will have a two-axis picture of a subset of SDG colours and a comparison with what an AI system says the same colours are called.

**How to use this notebook.** Run each cell in order with **Shift + Enter**. The only things you need to change are the grey form fields on the right of some cells. Keep the [Colour Spinner](https://colornaming.net/spinner) open in another tab."""),

loader_cell(),

md("""## 1 · Meet the palette

Every swatch in this notebook sits on a neutral mid-grey surround. A white or black page changes how a colour looks, so colour scientists judge colours against grey."""),

code("""show_palette(SDG.hex.tolist(), labels=SDG.index.tolist(), captions=SDG.goal.tolist(),
             note="The 17 UN Sustainable Development Goal colours (United Nations 2015).")"""),

md("""## 2 · Choose your subset and languages

Pick **three to five** SDG colours. Good audits mix an easy colour with a hard one: try one of the reds (1, 4, 5, 8), one of the blues (6, 14, 16, 17) and something in between such as 12.

Tick the languages you want to audit. The Colour Spinner currently covers six. If either of you speaks one of them natively, include it, and include **Himba**, whose speakers partition colour very differently from European languages.

**Backend.** *Colour Spinner API* fetches everything for you. If the network is slow, choose *Copied from the browser* and fill in the sheet in section 4."""),

code("""#@title Your pair's choices { run: "auto" }
SUBSET = "1, 6, 12, 14"  #@param {type:"string"}
SOURCE_LANGUAGE = "British English"  #@param ["American English", "British English", "French", "Greek", "Himba", "Spanish"]
American_English = False  #@param {type:"boolean"}
British_English = True  #@param {type:"boolean"}
French = False  #@param {type:"boolean"}
Greek = True  #@param {type:"boolean"}
Himba = True  #@param {type:"boolean"}
Spanish = True  #@param {type:"boolean"}
BACKEND = "Colour Spinner API"  #@param ["Colour Spinner API", "Copied from the browser"]

ticked = {"American English": American_English, "British English": British_English,
          "French": French, "Greek": Greek, "Himba": Himba, "Spanish": Spanish}
LANGUAGES = [SOURCE_LANGUAGE] + [l for l, on in ticked.items() if on and l != SOURCE_LANGUAGE]
PICK = [int(s) for s in re.findall(r"\\d+", SUBSET) if 1 <= int(s) <= 17]
sub = SDG.loc[PICK]

show_palette(sub.hex.tolist(), labels=[f"SDG {i}" for i in sub.index], captions=sub.goal.tolist(),
             note=f"Auditing {len(PICK)} colours in {len(LANGUAGES)} languages: {', '.join(LANGUAGES)}.")"""),

md("""## 3 · The perceptual axis: can everyone tell them apart?

About 1 in 12 men and 1 in 200 women of Northern European descent have a colour vision deficiency, most often a reduced ability to separate reds from greens. Independently, the lens of the eye yellows with age and absorbs more short-wavelength light, so blues and greens drift towards one another.

The rows below show the whole palette under each condition, simulated by the Colour Spinner's own models, the same ones you see with its "Simulate age/CVD" menu. The age rows are age bands (16–40, 40–60, 60–80). Look down each column for colours that change, and across each row for colours that merge."""),

code("""prefetch_simulations(SDG.hex.tolist())      # ask the Colour Spinner for all 17 colours at once
show_conditions(SDG.hex.tolist(), labels=[str(i) for i in SDG.index])"""),

md("""The chart below makes "merge" measurable. For each colour in your subset it finds the **nearest other SDG colour** under each condition and reports the CIEDE2000 colour difference (ΔE00) between them. A difference of about 1 is just noticeable side by side; for colours that must be told apart at a glance in a chart or a sign, aim for 10 or more. The small number under each value names the nearest neighbour."""),

code("""#@title Nearest-neighbour distinctiveness { run: "auto" }
DISTINCT_AT = 10  #@param {type:"slider", min:3, max:25, step:1}

def nearest(sdg_id, condition):
    me = simulate(SDG.hex[sdg_id], condition)
    best = min(((de00(me, simulate(h, condition)), j) for j, h in SDG.hex.items() if j != sdg_id))
    return best

nn = pd.DataFrame({c: {i: nearest(i, c) for i in PICK} for c in CONDITIONS})
nn_de = nn.map(lambda t: t[0]); nn_id = nn.map(lambda t: t[1])

fig, ax = plt.subplots(figsize=(1.1 * len(CONDITIONS) + 1.6, 0.62 * len(PICK) + 1.4))
ax.imshow(nn_de.values, cmap=RISK_CMAP, vmin=0, vmax=2 * DISTINCT_AT, aspect="auto")
for (r, c), v in np.ndenumerate(nn_de.values):
    ink = "white" if v < DISTINCT_AT * 0.6 else INK
    ax.text(c, r - 0.08, f"{v:.1f}", ha="center", va="center", fontsize=11, fontweight="bold", color=ink)
    ax.text(c, r + 0.27, f"vs {nn_id.values[r, c]}", ha="center", va="center", fontsize=8, color=ink)
ax.set_xticks(range(len(CONDITIONS)), [condition_label(c) for c in CONDITIONS], rotation=30, ha="right")
row_chips(ax, [SDG.hex[i] for i in PICK], [f"SDG {i}" for i in PICK])
ax.tick_params(axis="x", length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title(f"Distance to the nearest other SDG colour (ΔE00); red cells fall below {DISTINCT_AT}")
plt.tight_layout(); plt.show()

at_risk = {}
for i in PICK:
    for c in CONDITIONS:
        j, d = nn_id.loc[i, c], nn_de.loc[i, c]
        if d < DISTINCT_AT:
            at_risk.setdefault(tuple(sorted((i, j))), []).append((c, d))
if at_risk:
    rows = ""
    for (i, j), hits in sorted(at_risk.items(), key=lambda kv: min(d for _, d in kv[1])):
        worst_c = min(hits, key=lambda t: t[1])[0]
        chips = "".join(f'<div class="mini" style="background:{simulate(SDG.hex[k], worst_c)}"></div>' for k in (i, j))
        rows += (f'<tr><td><div class="row">{chips}</div></td><td><b>SDG {i} and {j}</b></td><td>' +
                 " ".join(f'<span style="white-space:nowrap">{_h(c)} {de_badge(d, DISTINCT_AT, small_is_good=False)}</span>'
                          for c, d in hits) + "</td></tr>")
    show('<b>Pairs at risk</b>, shown as they look under the condition where they are closest'
         f'<table class="babel">{rows}</table>')
else:
    show(f"Every colour in your subset stays at least ΔE00 {DISTINCT_AT} from its nearest neighbour under every condition.")"""),

md("""## 4 · The cognitive axis: what are these colours called?

Here you ask the Colour Spinner what speakers of each language actually call your colours. Its data come from thousands of people who named colours online at [colornaming.net](https://colornaming.net). Treat it as a reference baseline rather than ground truth: it currently covers six languages and only the essential colour terms in each, the minimal set that can name every colour.

Two measures come out of this step:

- **Nameability**: the Namer's *confidence*, the share of speakers who would use the top name. Low confidence means people disagree.
- **Translatability**: whether each language's name points to the same colour. The Mapper's *best example* of each name shows where that name is centred; if the best examples drift apart across languages, a "translation" of the colour lands somewhere else.

**Only if you chose *Copied from the browser*:** fill in the sheet below, one row per colour and language. In the Spinner, type the hex into the Picker, set the Namer to the language, and copy the name, its confidence and the Mapper's best-example RGB. Otherwise skip this cell."""),

code("""SHEET = \"\"\"
hex, language, name, confidence, best_example
# C34F4F, American English, Red, 20%, 232 21 56      ← example row, as the Spinner shows #C34F4F
\"\"\"
"""),

code("""#@title Ask the Colour Spinner
spinner = ApiSpinner() if BACKEND == "Colour Spinner API" else SheetSpinner(SHEET)

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

Each marker is one language. Colours near the right are named consistently; colours on the left are where speakers disagree, and where a label such as "the red goal" will mean different things to different readers."""),

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
ax.set_xlabel("Share of speakers who use the top name")
ax.set_title("Nameability: how much do speakers agree?")
plt.tight_layout(); plt.show()"""),

md("""### Translatability

The left swatch is the SDG colour. Each swatch to its right is the **best example** of the name that language uses for it, with its distance from the original. When those swatches differ from each other, the "same" colour word does not mean the same colour in each language."""),

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
                  f'{de_badge(r.drift.iloc[0], DISTINCT_AT)}</div></td>')
    sp = spread(sub_n.best_example.tolist())
    rows += (f'<tr><td><div class="pair" style="width:96px"><div style="background:{SDG.hex[i]};color:{_ink_on(SDG.hex[i])};'
             f'font-weight:700;font-size:13px">SDG {i}</div></div></td>{cells}'
             f'<td style="color:{MUTED}">{"" if pd.isna(sp) else "spread " + format(sp, ".1f")}</td></tr>')
show('<table class="babel"><tr><th></th>' + "".join(f"<th>{_h(l)}</th>" for l in LANGUAGES) +
     f'<th></th></tr>{rows}</table><div class="note">Badges: ΔE00 from the SDG colour. '
     'Spread: the largest ΔE00 between two languages\\' best examples.</div>')"""),

md("""The same information as a map. This is a top-down view of CAM16-UCS colour space, the perceptually uniform space the Colour Spinner works in: hue runs around the centre and colourfulness increases outwards. Each large disc is an SDG colour; the lines lead to where each language's name for it is centred."""),

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

Each disc is a colour from your subset. Its horizontal position is the **worst case** on the perceptual axis: the smallest nearest-neighbour ΔE00 across all seven viewing conditions. Its vertical position is the **average naming agreement** across your languages."""),

code("""#@title Scorecard { run: "auto" }
AGREE_AT = 0.5  #@param {type:"slider", min:0.2, max:0.9, step:0.05}

score = pd.DataFrame({
    "worst_case_dE00": nn_de.min(axis=1),
    "worst_condition": nn_de.idxmin(axis=1),
    "mean_agreement": naming.groupby("sdg").confidence.mean(),
    "language_spread": naming.groupby("sdg").best_example.apply(lambda s: spread(s.tolist())),
}).loc[PICK]

fig, ax = plt.subplots(figsize=(7.5, 6))
xmax = max(2.2 * DISTINCT_AT, score.worst_case_dE00.max() * 1.15)
ax.axvline(DISTINCT_AT, color="#BDBDB8", ls="--", lw=1); ax.axhline(AGREE_AT, color="#BDBDB8", ls="--", lw=1)
for x, y, t in [(0.02, 0.98, "named, but easily confused"), (0.98, 0.98, "robust on both axes"),
                (0.02, 0.02, "fragile on both axes"), (0.98, 0.02, "distinct, but hard to name")]:
    ax.text(x, y, t, transform=ax.transAxes, ha="left" if x < .5 else "right",
            va="top" if y > .5 else "bottom", color=MUTED, fontsize=10, style="italic")
for i, r in score.iterrows():
    ax.scatter(r.worst_case_dE00, r.mean_agreement, s=900, color=SDG.hex[i], edgecolor="white", lw=2.5, zorder=3)
    ax.text(r.worst_case_dE00, r.mean_agreement, str(i), ha="center", va="center",
            color=_ink_on(SDG.hex[i]), fontweight="bold", zorder=4)
ax.set_xlim(0, xmax); ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
ax.set_xlabel("Perceptual: worst-case ΔE00 to nearest neighbour"); ax.set_ylabel("Cognitive: mean naming agreement")
ax.set_title("Where your colours sit on the two axes")
plt.tight_layout(); plt.show()
display(score.round(2))"""),

md("""## 6 · What does an AI system say?

Now ask a current AI system to name the same colours in the same languages, and compare its answers with how speakers actually name them.

**Which AI?** We demonstrate with the Gemini model built into Colab because it needs no set-up. It stands in for AI systems in general and isn't a recommendation. You are welcome to use the AI you use day to day (ChatGPT, Claude, Copilot, …) instead, and the comparison is often more meaningful that way.

- **Colab's Gemini:** leave the form as it is and run the cells in order.
- **Your own AI:** choose *Paste the reply myself* in the form, copy the printed prompt into your AI tool, and paste its whole reply into `AI_REPLY`."""),

code("""lang_note = "; ".join(f"{l}" + (" (Otjiherero, spoken by the Himba of Namibia)" if l == "Himba" else "")
                      for l in LANGUAGES)
PROMPT_B = f\"\"\"You are helping with a cross-cultural colour naming study.
For each colour below, give the single colour name that most native speakers of each language would use for it.
Write each name in that language and in its usual script.

Languages: {lang_note}
Colours (sRGB hex): {", ".join(SDG.hex[i] for i in PICK)}

Reply with JSON only, as a list of objects like
{{"hex": "#E5243B", "language": "{LANGUAGES[-1]}", "name": "..."}}
with one object for every colour and language.\"\"\"
print(PROMPT_B)"""),

code("""#@title AI system
AI_PROVIDER = "Gemini built into Colab (no key needed)"  #@param ["Gemini built into Colab (no key needed)", "Paste the reply myself", "Anthropic", "OpenAI-compatible"]
AI_MODEL = ""  #@param {type:"string"}
AI_BASE_URL = ""  #@param {type:"string"}
#@markdown Colab's Gemini is for demonstration; any AI you regularly use works via *Paste the reply myself*.
#@markdown Leave **AI_MODEL** empty to use Colab's default Gemini model. Run `from google.colab import ai; ai.list_models()` to see the others.
#@markdown *Anthropic* and *OpenAI-compatible* need your own key in Colab ▸ Secrets (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) and a model name."""),

code("""AI_REPLY = \"\"\"
paste the AI's reply here, only if you chose "Paste the reply myself"
\"\"\"
"""),

code("""reply_b = AI_REPLY if AI_PROVIDER.startswith("Paste") else ask_ai(PROMPT_B, AI_PROVIDER, AI_MODEL, AI_BASE_URL)
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

md("""A match on a colour with low speaker agreement says little: when only a third of speakers use the top name, other answers are also defensible. Mismatches on high-agreement colours are the ones to bring to the group.

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
_set(5, _form + """ticked = {"American English": American_English, "British English": British_English,
          "French": French, "Greek": Greek, "Himba": Himba, "Spanish": Spanish}
LANGUAGES = pick_languages(SOURCE_LANGUAGE, ticked)
PICK = parse_subset(SUBSET)
show_subset(PICK, LANGUAGES)""")

_set(9, """#@title Nearest-neighbour distinctiveness { run: "auto" }
DISTINCT_AT = 10  #@param {type:"slider", min:3, max:25, step:1}

nn_de, nn_id = nearest_neighbours(PICK)      # ΔE00 to the nearest other SDG colour, per condition
plot_nearest_neighbours(nn_de, nn_id, DISTINCT_AT)
show_at_risk(nn_de, nn_id, DISTINCT_AT)""")

_set(12, """spinner = connect(BACKEND, SHEET)

rows, problems = [], []
for sdg in PICK:
    for language in LANGUAGES:
        colour = SDG.hex[sdg]
        try:
            namer = spinner.name(colour, language)                  # Namer: top name and agreement
        except Exception as e:
            problems.append(f"SDG {sdg} in {language}: {e}"); continue
        best = safe(spinner.best_example, namer["name"], language)  # Mapper: where that name is centred
        rows.append({"sdg": sdg, "hex": colour, "language": language,
                     "name": namer["name"], "confidence": namer["confidence"], "best_example": best,
                     "drift": de00(colour, best) if best else np.nan})  # how far the name pulls the colour
naming = pd.DataFrame(rows)

report_problems(problems)
show_names_table(naming, LANGUAGES)""")

_set(14, "plot_nameability(naming, PICK, LANGUAGES)")
_set(16, "show_translatability(naming, PICK, LANGUAGES, DISTINCT_AT)")
_set(18, "plot_name_map(naming, PICK, LANGUAGES)")
_set(20, """#@title Scorecard { run: "auto" }
AGREE_AT = 0.5  #@param {type:"slider", min:0.2, max:0.9, step:0.05}

score = scorecard(nn_de, naming)
plot_two_axes(score, DISTINCT_AT, AGREE_AT)
display(score.round(2))""")
_set(25, """reply_b = ai_reply(PROMPT_B, AI_PROVIDER, AI_MODEL, AI_BASE_URL, pasted=AI_REPLY)
ai_names = read_ai_names(reply_b)

comparison = naming.merge(ai_names, on=["hex", "language"], how="left")
comparison["agrees"] = [isinstance(a, str) and norm(a) == norm(s)     # ignores case and accents
                        for a, s in zip(comparison.ai_name, comparison.name)]
show_name_comparison(comparison)
plot_agreement_by_language(comparison, LANGUAGES)""")
