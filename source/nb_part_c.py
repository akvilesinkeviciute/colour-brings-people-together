import re
from nbtools import md, code, loader_cell

TITLE = "Part C — Creating an empirically grounded colour palette"

cells = [
md("""# Colour Brings People Together
### Part C · Creating an empirically grounded colour palette

In Part B you audited your test palette: what you call each colour in your language, and what your audience calls it in theirs. Now reverse the direction, and fix the palette for each audience. You will give those **names**, yours and your audience's, to an AI system, ask it for the colour each name stands for, and check its answers against where speakers actually centre each name. Where the AI's colour diverges, you replace it with the Colour Spinner's best example, and you finish with a small palette you can defend with evidence. This matters: in the *Colour in Translation* benchmark (Mylonas et al., CHI 2026), a current large language model's colours were on average more than 17 ΔE from speakers' focal colours, because it mapped colour names onto the same prototypes whatever the language.

| Step | You do | Time |
|---|---|---|
| 1 | Load your names from Part B | 5 min |
| 2 | Ask an AI system for the best example of each name | 10 min |
| 3 | Compare its colours with the empirical best examples | 10 min |
| 4 | Replace divergent colours and check the result still works for every viewer | 15 min |
| 5 | Export the palette and prepare your finding | 5 min |
| | Plenary discussion | 15 min |

Run each cell with **Shift + Enter** and change only the grey form fields."""),

loader_cell(),

md("""## 1 · Load your names from Part B

Upload the `partB_results.csv` you downloaded at the end of Part B. If you lost it, choose *Type them in* and list a few colour names per language instead; the notebook will look up their best examples."""),

code("""#@title Where your names come from
NamesFrom = "Upload partB_results.csv"  #@param ["Upload partB_results.csv", "Type them in"]
Backend = "Colour Spinner API"  #@param ["Colour Spinner API", "Copied from the browser"]
#@markdown *Copied from the browser* only matters if you type names in: then also type each name's best example (the Mapper's top-left tile) into the boxes below."""),

code("""TYPED_NAMES = \"\"\"
language, name
# American English, Red          ← one row per name, only if you chose "Type them in"
\"\"\"

SHEET = \"\"\"
hex, language, name, confidence, best_example
\"\"\"
"""),

code("""if NamesFrom.startswith("Upload"):
    if IN_COLAB:
        from google.colab import files
        up = files.upload()
        partb = pd.read_csv(io.BytesIO(next(iter(up.values()))))
    else:
        partb = pd.read_csv("partB_results.csv")
    names = (partb.dropna(subset=["name"])
                  .drop_duplicates(["language", "name"])
                  [["sdg", "hex", "language", "name", "best_example"]]
                  .rename(columns={"hex": "sdg_hex", "best_example": "spinner_hex"}))
else:
    typed = _read_sheet(TYPED_NAMES)
    names = typed.assign(sdg=np.nan, sdg_hex=None, spinner_hex=None)[
        ["sdg", "sdg_hex", "language", "name", "spinner_hex"]]

names = names.reset_index(drop=True)
LANGUAGES = list(dict.fromkeys(names.language))
show('<table class="babel"><tr><th>Language</th><th>Names to test</th></tr>' + "".join(
     f"<tr><td>{_h(l)}</td><td><b>{', '.join(_h(n) for n in names[names.language == l].name)}</b></td></tr>"
     for l in LANGUAGES) + "</table>")"""),

md("""## 2 · Ask the AI system for the best example of each name

The prompt asks for the colour that a native speaker would consider the **best example** of each name: the most typical instance, not the average and not a brand colour. As in Part B, use the AI you use day to day, and the same one as in Part B so the two parts can be compared. The next cell gives you the prompt with **Copy prompt** and buttons for ChatGPT, Claude and Gemini, and a box: paste the AI's whole reply into it in one go, and the notebook reads the colour codes itself."""),

code("""items = "\\n".join(f"- {r.language}: {plain_with_roman(r)}" for _, r in names.iterrows())
PROMPT_C = f\"\"\"You are helping with a cross-cultural colour naming study.
For each colour name below, give the sRGB colour that native speakers of that language would consider
the best example of the name: its most typical instance.

{items}

Reply with one line per name, and nothing else, in exactly this format (R, G, B from 0 to 255):
{names.language.iloc[0]} | {names['name'].iloc[0]} | R, G, B\"\"\"
print(PROMPT_C)"""),

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

code("""reply_c = AI_REPLY if AI.startswith("Paste") else ask_ai(PROMPT_C, AI, AI_MODEL, AI_BASE_URL)
ai = pd.DataFrame(parse_json_reply(reply_c))
ai["ai_hex"] = ai.apply(lambda r: parse_colour(" ".join(map(str, r["rgb"]))) if "rgb" in r and
                        isinstance(r["rgb"], list) else parse_colour(r.get("hex", "")), axis=1)
ai["key"] = ai.language.map(norm) + "|" + ai.name.map(norm)
names["key"] = names.language.map(norm) + "|" + names.name.map(norm)
pal = names.merge(ai[["key", "ai_hex"]], on="key", how="left")

missing = pal[pal.ai_hex.isna()]
if len(missing):
    show("<b>The AI reply has no colour for:</b> " + ", ".join(
         f"{_h(r['name'])} ({_h(r.language)})" for _, r in missing.iterrows()) +
         '<div class="note">Ask it again for those names, or carry on without them.</div>')
show_palette(pal.ai_hex.dropna().tolist(), labels=["" for _ in pal.ai_hex.dropna()],
             captions=[f"{n} · {l}" for n, l, h in zip(pal.name, pal.language, pal.ai_hex) if isinstance(h, str)],
             note="The AI system's best examples.")"""),

md("""## 3 · Compare with how speakers actually use each name

For every name, the left half of each swatch is the AI's colour and the right half is the Colour Spinner's best example: the name's focal colour, the sample where speakers agree most on that name (the Mapper's top-left tile). The badge gives the CIEDE2000 difference between them.

A reminder of scale: about 1 is just noticeable with the two side by side, about 5 is obvious, and above 10 the two are usually read as different colours."""),

code("""#@title Fill in the Spinner's best examples and compare { run: "auto" }
ReplaceAbove = 10  #@param {type:"slider", min:3, max:30, step:1}

need = pal.spinner_hex.isna()
if need.any():
    spinner = ApiSpinner() if Backend == "Colour Spinner API" else SheetSpinner(SHEET)
    for k in pal.index[need]:
        got = try_call(spinner.best_example, pal.at[k, "name"], pal.at[k, "language"])
        pal.at[k, "spinner_hex"] = None if isinstance(got, Exception) else got

pal["de00"] = [de00(a, b) if isinstance(a, str) and isinstance(b, str) else np.nan
               for a, b in zip(pal.ai_hex, pal.spinner_hex)]
pal["de_ucs"] = [de_ucs(a, b) if isinstance(a, str) and isinstance(b, str) else np.nan
                 for a, b in zip(pal.ai_hex, pal.spinner_hex)]
pal = pal.sort_values(["language", "de00"], ascending=[True, False]).reset_index(drop=True)
show_pairs(pal, threshold=ReplaceAbove)"""),

md("""The map below is the same top-down view of CAM16-UCS colour space used in Part B. Each arrow starts at the AI's colour and ends at the speakers' best example. Long arrows are large errors; arrows that all point the same way for one language suggest a systematic bias, for instance towards English focal colours."""),

code("""MARKERS = dict(zip(SPINNER_LANGUAGES, ["o", "s", "D", "^", "v", "P"]))
ok = pal.dropna(subset=["ai_hex", "spinner_hex"])
fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1, 1.05]})

ax = axes[0]
ax.axhline(0, color="#E4E4E0", lw=1, zorder=0); ax.axvline(0, color="#E4E4E0", lw=1, zorder=0)
for _, r in ok.iterrows():
    a, s = rgb_to_cam16ucs(r.ai_hex), rgb_to_cam16ucs(r.spinner_hex)
    ax.annotate("", xy=(s[1], s[2]), xytext=(a[1], a[2]),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1, shrinkA=7, shrinkB=9), zorder=1)
    ax.scatter(a[1], a[2], s=70, marker=MARKERS.get(r.language, "o"), color=r.ai_hex,
               edgecolor=INK, lw=0.6, zorder=2)
    ax.scatter(s[1], s[2], s=170, marker=MARKERS.get(r.language, "o"), color=r.spinner_hex,
               edgecolor="white", lw=2, zorder=3)
ax.set_aspect("equal"); ax.set_xlabel("a′  (green ← → red)"); ax.set_ylabel("b′  (blue ← → yellow)")
ax.legend(handles=[mpl.lines.Line2D([], [], marker=MARKERS.get(l, "o"), ls="", color="white",
          markeredgecolor=INK, markersize=8, label=l) for l in LANGUAGES], frameon=False, loc="best")
ax.set_title("From the AI's colour to the speakers' best example")

ax = axes[1]
order = ok.sort_values("de00")
y = np.arange(len(order))
ax.hlines(y, 0, order.de00, color="#CFCFCA", lw=2)
ax.scatter(order.de00, y, s=150, c=order.spinner_hex.tolist(), edgecolor="white", lw=1.5, zorder=3)
ax.scatter(np.zeros(len(order)), y, s=60, c=order.ai_hex.tolist(), edgecolor=INK, lw=0.5, zorder=3)
ax.axvline(ReplaceAbove, color=INK, ls="--", lw=1)
ax.text(ReplaceAbove, len(order) - 0.4, f"  replace above {ReplaceAbove}", fontsize=9, color=MUTED, va="bottom")
ax.set_yticks(y, [f"{n}  ({l})" for n, l in zip(order.name, order.language)])
ax.set_xlabel("ΔE₀₀ between the AI's colour and the best example"); ax.set_xlim(left=-1)
ax.legend(handles=[mpl.lines.Line2D([], [], marker="o", ls="", color="#BDBDB8", markeredgecolor=INK,
                   markersize=6, label="AI's colour"),
                   mpl.lines.Line2D([], [], marker="o", ls="", color="#BDBDB8", markeredgecolor="white",
                   markersize=11, label="speakers' best example")], frameon=False, loc="lower right")
ax.set_title("Size of each error")
plt.tight_layout(); plt.show()

summary = ok.groupby("language").agg(names=("name", "size"), mean_dE00=("de00", "mean"),
                                     worst_dE00=("de00", "max"),
                                     diverging=("de00", lambda d: int((d > ReplaceAbove).sum())))
display(summary.reindex([l for l in LANGUAGES if l in summary.index]).round(1))"""),

md("""## 4 · Ground the palette

An AI colour diverges from observed language use if it is too far from the name's best example, or if speakers would call it by a different name. Choose which rule to apply in the form; by default a colour is replaced if either is true. A replaced colour becomes the Colour Spinner's best example of the name. The distance threshold is the slider in section 3."""),

code("""pal["grounded_hex"] = [s if isinstance(s, str) and (pd.isna(d) or d > ReplaceAbove) else a
                       for a, s, d in zip(pal.ai_hex, pal.spinner_hex, pal.de00)]
pal["replaced"] = pal.grounded_hex != pal.ai_hex

for lang in LANGUAGES:
    p = pal[(pal.language == lang) & pal.grounded_hex.notna()]
    if p.empty: continue
    tiles = ""
    for _, r in p.iterrows():
        ai_h = r.ai_hex if isinstance(r.ai_hex, str) else "#E4E4E0"
        tag = "replaced" if r.replaced else "kept"
        tiles += (f'<div class="tile"><div class="pair" style="width:86px;height:38px;border:none;border-radius:0">'
                  f'<div style="background:{ai_h};color:{_ink_on(ai_h)}">AI</div></div>'
                  f'<div class="chip" style="background:{r.grounded_hex};color:{_ink_on(r.grounded_hex)};height:56px">'
                  f'{_h(r["name"])}</div><div class="cap">{tag}<div class="hex">{r.grounded_hex}</div></div></div>')
    show(f'<div style="font-weight:700;margin:12px 0 4px">{_h(lang)}</div><div class="surround">{tiles}</div>')
show(f'<div class="note">{int(pal.replaced.sum())} of {int(pal.grounded_hex.notna().sum())} colours replaced '
     f'with the Colour Spinner\\'s best example. Top strip: the AI\\'s original colour.</div>')"""),

md("""### Does the grounded palette still work for every viewer?

Grounding fixes the cognitive axis, but best examples of neighbouring names can sit close together. Check the perceptual axis again: for each language's palette, the smallest ΔE₀₀ between any two of its colours under each viewing condition.

**A note on the distances.** All colour differences here are CIEDE2000 (ΔE₀₀). CIEDE2000 was derived from colour-difference judgements by observers with typical colour vision. Applied to simulated colours, it estimates how different those colours would look to a typical observer, which only approximates what people with a colour vision deficiency or an older eye can tell apart."""),

code("""def min_pair(hexes, cond):
    from itertools import combinations
    hs = [simulate(h, cond) for h in hexes if isinstance(h, str)]
    return min((de00(a, b) for a, b in combinations(hs, 2)), default=np.nan)

prefetch_simulations([h for h in pd.concat([pal.ai_hex, pal.grounded_hex]) if isinstance(h, str)])
check = []
for lang in LANGUAGES:
    p = pal[pal.language == lang]
    for version, col in [("AI", "ai_hex"), ("Grounded", "grounded_hex")]:
        check.append({"language": lang, "palette": version,
                      **{c: min_pair(p[col].tolist(), c) for c in CONDITIONS}})
check = pd.DataFrame(check).set_index(["language", "palette"])

fig, ax = plt.subplots(figsize=(1.1 * len(CONDITIONS) + 2.4, 0.5 * len(check) + 1.4))
ax.imshow(check.values, cmap=RISK_CMAP, vmin=0, vmax=2 * ReplaceAbove, aspect="auto")
for (r, c), v in np.ndenumerate(check.values):
    if not np.isnan(v):
        ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=10,
                color="white" if v < 0.7 * ReplaceAbove else INK)
ax.set_xticks(range(len(CONDITIONS)), [condition_label(c) for c in CONDITIONS], rotation=30, ha="right")
ax.set_yticks(range(len(check)), [f"{l} · {v}" for l, v in check.index])
for r in range(2, len(check), 2): ax.axhline(r - 0.5, color="white", lw=3)
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("Closest pair in each palette under each condition (ΔE₀₀)")
plt.tight_layout(); plt.show()

langs_shown = [l for l in LANGUAGES if (pal[pal.language == l].grounded_hex.notna()).sum() > 1]
for k, lang in enumerate(langs_shown):
    p = pal[(pal.language == lang) & pal.grounded_hex.notna()]
    show(f'<div style="font-weight:700;margin:10px 0 4px">{_h(lang)}, grounded</div>')
    show_conditions(p.grounded_hex.tolist(), labels=p.name.tolist(), width=64,
                    note=(k == len(langs_shown) - 1))"""),

md("""If a pair falls below the threshold under some condition, you have found a real design tension: the colours speakers expect for two names may be too similar for some viewers. Options to discuss: keep one colour and pick the next-best example of the other from the Spinner's Mapper panel, add a second cue such as a pattern, icon or label, or drop one name from the palette.

## 5 · Export your palette

The download has a picture of every palette, swatch files for design tools (Adobe, GIMP, Figma), CSS for websites, and the data with the evidence for every change."""),

code("""export = pal[["language", "name", "grounded_hex", "ai_hex", "spinner_hex", "de00", "replaced", "sdg", "sdg_hex"]]
export.round(2).to_csv("grounded_palette.csv", index=False)

palette_json = {lang: [{"name": r["name"], "hex": r.grounded_hex, "ai_hex": r.ai_hex,
                        "spinner_hex": r.spinner_hex,
                        "dE00_ai_vs_spinner": None if pd.isna(r.de00) else round(float(r.de00), 2),
                        "replaced": bool(r.replaced)}
                       for _, r in export[export.language == lang].iterrows()]
                for lang in LANGUAGES}
with open("grounded_palette.json", "w", encoding="utf-8") as f:
    json.dump(palette_json, f, ensure_ascii=False, indent=2)

def slug(s): return re.sub(r"[^\\w]+", "-", norm(s)).strip("-") or "colour"
css = ":root {\\n" + "".join(
    f"  --{slug(r.language)}-{slug(r['name'])}: {r.grounded_hex};"
    f"{'  /* replaced AI ' + r.ai_hex + ' */' if r.replaced and isinstance(r.ai_hex, str) else ''}\\n"
    for _, r in export.dropna(subset=["grounded_hex"]).iterrows()) + "}\\n"
with open("grounded_palette.css", "w", encoding="utf-8") as f:
    f.write(css)
print(css)

import zipfile
with zipfile.ZipFile("grounded_palette.zip", "w") as z:
    for fn in ("grounded_palette.csv", "grounded_palette.json", "grounded_palette.css"):
        z.write(fn)
download("grounded_palette.zip")"""),

md("""## 6 · Your most striking finding

The cell below picks the name where the AI system's colour was furthest from what speakers mean. You can choose a different one by typing its name in the form. Bring this card to the plenary."""),

code("""#@title Finding to share { run: "auto" }
FindingName = ""  #@param {type:"string"}
cand = pal.dropna(subset=["de00"])
row = (cand[cand.name.map(norm) == norm(FindingName)].iloc[0] if FindingName and
       (cand.name.map(norm) == norm(FindingName)).any() else cand.loc[cand.de00.idxmax()])

show(f\"\"\"<div class="card">
<h3>“{_h(row['name'])}” in {_h(row.language)}</h3>
<div style="display:flex;gap:10px;margin:12px 0;background:{SURROUND};padding:12px;border-radius:6px">
  <div class="tile"><div class="chip" style="background:{row.ai_hex};color:{_ink_on(row.ai_hex)};height:80px">AI</div>
       <div class="cap">AI system<div class="hex">{row.ai_hex}</div></div></div>
  <div class="tile"><div class="chip" style="background:{row.spinner_hex};color:{_ink_on(row.spinner_hex)};height:80px">Speakers</div>
       <div class="cap">Best example<div class="hex">{row.spinner_hex}</div></div></div>
  <div style="color:#fff;align-self:center;padding-left:8px;font-size:15px">ΔE₀₀<br><b style="font-size:30px">{row.de00:.1f}</b></div>
</div>
<p style="margin:0 0 8px">Talk it through with your partner:</p>
<p style="margin:0 0 4px"><b>What pattern of failure does this reveal?</b> Did the AI default to a Western-centric focal colour,
collapse a distinction the language maintains, or produce a colour that fails under CVD or ageing?</p>
<p style="margin:0"><b>What would a designer need to do differently</b> to avoid it in their own practice?</p>
</div>\"\"\")"""),

md("""---
*Colour Spinner: Sinkeviciute, Ahmed, Stutters, Koliousis & Mylonas (2026), under review. Spin Colour Forest and the AI colour-naming benchmark: Mylonas, Ahmed, Sinkeviciute & Koliousis, CHI 2026. Colour vision deficiency and age simulations from the Colour Spinner. Colour science with the colour-science library: CIELAB and CIEDE2000.*"""),
]


# ── short cells: charts and tables now live in colour_masterclass.py ──
def _set(i, src, hidden=None):
    assert cells[i]["cell_type"] == "code", i
    cells[i] = code(src, hidden=hidden)

_set(5, """names = load_names(NamesFrom, TYPED_NAMES)
LANGUAGES = list(dict.fromkeys(names.language))
show_names(names)""")
_set(10, """reply_c = ai_reply(PROMPT_C, AI, AI_MODEL, AI_BASE_URL, pasted=AI_REPLY)
pal = read_ai_colours(reply_c, names)
show_ai_palette(pal)""")
_set(12, """#@title Fill in the Spinner's best examples and compare { run: "auto" }
ReplaceAbove = 10  #@param {type:"slider", min:3, max:30, step:1}
#@markdown An AI colour further than this CIEDE2000 colour difference from the speakers' best example counts as divergent.

pal = add_best_examples(pal, connect(Backend, SHEET))     # looks up only the names that need it
pal["de00"] = [de00(ai, best) if isinstance(ai, str) and isinstance(best, str) else np.nan
               for ai, best in zip(pal.ai_hex, pal.spinner_hex)]
pal = pal.sort_values(["language", "de00"], ascending=[True, False]).reset_index(drop=True)
show_pairs(pal, threshold=ReplaceAbove)""")
_set(14, """plot_ai_vs_speakers(pal, LANGUAGES, ReplaceAbove)
display(error_summary(pal, LANGUAGES, ReplaceAbove))""")
_set(16, """# Keep the AI's colour unless it is more than ReplaceAbove from the speakers' best example.
pal["grounded_hex"] = [best if isinstance(best, str) and (pd.isna(d) or d > ReplaceAbove) else ai
                       for ai, best, d in zip(pal.ai_hex, pal.spinner_hex, pal.de00)]
pal["replaced"] = pal.grounded_hex != pal.ai_hex
show_grounded(pal, LANGUAGES)""")
_set(18, """#@title Can every viewer tell the colours apart? { run: "auto" }
ConfusableBelow = 10  #@param {type:"slider", min:3, max:25, step:1}
#@markdown Two colours closer than this CIEDE2000 colour difference count as easily confused, as in Part B.

check = closest_pairs(pal, LANGUAGES)      # smallest ΔE₀₀ within each palette, per condition
plot_closest_pairs(check, ConfusableBelow)
show_grounded_conditions(pal, LANGUAGES)""", hidden=False)
_set(20, "export_palette(pal, LANGUAGES)")
_set(22, """#@title Finding to share { run: "auto" }
FindingName = ""  #@param {type:"string"}
show_finding(pal, FindingName)""")


# ── round trip: send the AI's colours back to the Spinner's Namer ──
_i = next(k for k, c in enumerate(cells) if c["cell_type"] == "code" and "plot_ai_vs_speakers" in "".join(c["source"]))
cells[_i + 1:_i + 1] = [
md("""### Would speakers call the AI's colour by that name?

The distance above compares the AI's colour with a single point, the name's best example. But a name covers a range of colours, and the Spinner knows how speakers use it across that range. So send each AI colour back to the Spinner's Namer and ask what speakers would call it. If the answer is the name you gave the AI, the colour communicates as intended, even if it isn't the most typical example. If it's a different name, speakers would read the AI's colour as something else."""),
code("""spinner = connect(Backend, SHEET)
pal["rt_name"], pal["rt_conf"] = None, np.nan
for k, row in pal.iterrows():
    if isinstance(row.ai_hex, str):
        namer = safe(spinner.name, row.ai_hex, row.language)   # what speakers call the AI's colour
        if namer:
            pal.at[k, "rt_name"], pal.at[k, "rt_conf"] = namer["name"], namer["confidence"]
show_round_trip(pal)"""),
]
_g = next(k for k, c in enumerate(cells) if c["cell_type"] == "code" and 'pal["grounded_hex"]' in "".join(c["source"]))
cells[_g] = code("""#@title Ground the palette { run: "auto" }
ReplaceWhen = "Either"  #@param ["Either", "Too far from the best example", "Speakers would use a different name"]

too_far = [pd.isna(d) or d > ReplaceAbove for d in pal.de00]
renamed = [isinstance(n, str) and not same_name(n, m, r)
           for n, m, r in zip(pal.rt_name, pal.name, pal.get("romanized", pd.Series(None, index=pal.index)))]
rule = {"Either": [a or b for a, b in zip(too_far, renamed)],
        "Too far from the best example": too_far,
        "Speakers would use a different name": renamed}[ReplaceWhen]

# Replace the AI's colour with the speakers' best example wherever the rule says it diverges.
pal["grounded_hex"] = [best if isinstance(best, str) and replace else ai
                       for ai, best, replace in zip(pal.ai_hex, pal.spinner_hex, rule)]
pal["replaced"] = pal.grounded_hex != pal.ai_hex
show_grounded(pal, LANGUAGES)""", hidden=False)


# ── AI step: choose first, then prompt (+ paste box), then read the reply ──
_p = next(k for k, c in enumerate(cells) if c["cell_type"] == "code" and "PROMPT_C = f" in "".join(c["source"]))
_prompt_src = "".join(cells[_p]["source"]).replace("print(PROMPT_C)", "show_prompt(PROMPT_C, AI)")
_form = next(k for k, c in enumerate(cells) if c["cell_type"] == "code" and "".join(c["source"]).startswith("#@title AI system"))
_reply = next(k for k, c in enumerate(cells) if c["cell_type"] == "code" and "".join(c["source"]).startswith("AI_REPLY ="))
assert _p < _form < _reply
cells[_p] = code('#@title Which AI?\nAI = "Gemini built into Colab (no key needed)"  #@param ["Gemini built into Colab (no key needed)", "Paste the reply from my own AI", "Anthropic", "OpenAI-compatible"]\nAI_MODEL = ""  #@param {type:"string"}\nAI_BASE_URL = ""  #@param {type:"string"}\n#@markdown **Gemini built into Colab** needs nothing more: just run the cells below.\n#@markdown **Your own AI** (ChatGPT, Claude, Copilot, …): choose *Paste the reply from my own AI*; the next cell gives you the prompt and a box to paste the reply into.\n#@markdown *Anthropic* and *OpenAI-compatible* call an API with your own key from Colab ▸ Secrets (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) and a model name. Leave AI_MODEL empty for Colab\'s default Gemini.')
cells[_form] = code(_prompt_src)
del cells[_reply]
_r = next(k for k, c in enumerate(cells) if c["cell_type"] == "code" and "reply_c = ai_reply" in "".join(c["source"]))
cells[_r] = code("".join(cells[_r]["source"]).replace(", pasted=AI_REPLY", ""))


# ── round of notes, October 2026 ──
def _find(pred):
    return next(k for k, c in enumerate(cells) if pred("".join(c["source"]), c["cell_type"]))
cells[_find(lambda s, t: t == "code" and s.startswith("#@title Which AI?"))] = code('#@title Which AI?\nAI = "Paste answers from my own AI"  #@param ["Paste answers from my own AI", "Gemini built into Colab"]\n#@markdown **Your own AI** (ChatGPT, Claude, Gemini, Copilot, …): the next cell shows a prompt to copy into it and one box per name. The names are filled in for you; just copy in the colour code (#RRGGBB) your AI gives for each.\n#@markdown **Gemini built into Colab** is for demonstration. It needs no set-up but stops working once Colab\'s free usage limit is reached.')
cells[_find(lambda s, t: t == "code" and "PROMPT_C = f" in s)] = code('lines = "\\n".join(f"{r.language} | {plain_with_roman(r)}" for _, r in names.iterrows())\nfirst = names.iloc[0]\nPROMPT_C = f"""You are helping with a cross-cultural colour naming study.\nFor each colour name below, give the colour, as a hex code, that native speakers of that language\nwould consider the best example of the name: its most typical instance.\n\n{lines}\n\nReply with exactly one line for each, in this format and nothing else:\n{first.language} | {first[\'name\']} | #RRGGBB"""\nshow_prompt(PROMPT_C, AI, rows=names, kind="colour")')
cells[_find(lambda s, t: t == "code" and "PROMPT_C, AI" in s and "show_ai_palette" in s)] = code('if AI == PASTE:\n    pal = read_ai_colours(collected_answers(), names)    # the colour codes typed into the boxes\nelse:\n    pal = read_ai_colours(ask_ai(PROMPT_C, AI), names)\nshow_ai_palette(pal)')


# ── boxes instead of code for typed names ──
cells[_find(lambda s, t: t == "code" and s.startswith('TYPED_NAMES = """'))] = code("""#@title Only if you chose "Type them in"
if NamesFrom == "Type them in":
    name_boxes()      # choose a language, type a name; add its best example only for "Copied from the browser"
else:
    show("You're loading names from Part B, so there's nothing to fill in here.")""", hidden=False)
for _k, _c in enumerate(cells):
    if _c["cell_type"] == "code":
        _src = "".join(_c["source"])
        if "connect(Backend, SHEET)" in _src or "load_names(NamesFrom, TYPED_NAMES)" in _src:
            cells[_k] = code(_src.replace("connect(Backend, SHEET)", "connect(Backend)")
                                 .replace("load_names(NamesFrom, TYPED_NAMES)", "load_names(NamesFrom)"),
                             hidden=_c["metadata"].get("cellView") == "form")


# ── numbered prompt and numbered answer boxes ──
cells[_find(lambda s, t: t == "code" and "PROMPT_C = f" in s)] = code('lines = "\\n".join(numbered_items(names, "colour"))\nPROMPT_C = f"""You are helping with a cross-cultural colour naming study.\nFor each numbered colour name below, give the colour, as a hex code, that native speakers of that language would consider the best example of the name: its most typical instance.\n\n{lines}\n\nReply with one line per number, in this format and nothing else:\n1. #RRGGBB\n2. …"""\nshow_prompt(PROMPT_C, AI, rows=names, kind="colour")')
_k = _find(lambda s, t: t == "code" and 'pal = read_ai_colours(ask_ai(PROMPT_C, AI), names)' in s)
cells[_k] = code("".join(cells[_k]["source"]).replace('pal = read_ai_colours(ask_ai(PROMPT_C, AI), names)', 'pal = read_ai_colours(answers_from_reply(ask_ai(PROMPT_C, AI), names, "colour"), names)'))


# ── one AI route for everyone: buttons to open any AI, numbered boxes for the answers ──
del cells[_find(lambda s, t: t == "code" and s.startswith("#@title Which AI?"))]
_k = _find(lambda s, t: t == "code" and "show_prompt(PROMPT_C, AI," in s)
cells[_k] = code("".join(cells[_k]["source"]).replace("show_prompt(PROMPT_C, AI,", "show_prompt(PROMPT_C, PASTE,"))
_k = _find(lambda s, t: t == "code" and s.startswith("if AI == PASTE:"))
_lines = "".join(cells[_k]["source"]).split("\n")
cells[_k] = code("\n".join(['pal = read_ai_colours(collected_answers(), names)       # the colour codes read from the pasted reply'] + _lines[4:]))


# ── polished outputs: the error summary is its own table ──
_k = _find(lambda s, t: t == "code" and "display(error_summary(" in s)
cells[_k] = code("".join(cells[_k]["source"]).replace("display(error_summary(pal, LANGUAGES, ReplaceAbove))", "error_summary(pal, LANGUAGES, ReplaceAbove)"))


# ── repair perceptual problems the grounding created ──
_g = _find(lambda s, t: t == "code" and 'pal["replaced"] = pal.grounded_hex != pal.ai_hex' in s and "show_grounded" in s)
cells[_g + 1:_g + 1] = [
md("""### Fix colours that grounding made too similar

Grounding can put two colours of one palette close together: the focal colours of neighbouring names, such as two blues, may be hard to tell apart for some viewers. The next cell checks every palette under all seven viewing conditions. Where two colours are closer than the threshold, it swaps one of them for **another colour speakers still call by the same name**: the next tiles in the Spinner's Mapper, which run from the most to the least typical. It picks the swap that separates the palette most. If no colour of that name helps, the table says so; then consider a second cue such as a pattern, label or icon."""),
code("""#@title Fix perceptual problems { run: "auto" }
ConfusableBelow = 10  #@param {type:"slider", min:3, max:25, step:1}
#@markdown Two colours closer than this CIEDE2000 colour difference, under any viewing condition, count as too similar.

if "repaired_from" in pal:                 # rerunning: start again from the grounded colours
    pal["grounded_hex"] = pal.repaired_from.fillna(pal.grounded_hex)
fixed = repair_palettes(pal, connect(Backend), LANGUAGES, ConfusableBelow)
if fixed is not None:
    pal = fixed
    show_repairs(pal, ConfusableBelow)""", hidden=False),
]
# the later re-check uses the same slider value: drop its own slider
_k = _find(lambda s, t: t == "code" and s.startswith("#@title Can every viewer tell the colours apart?"))
cells[_k] = code("".join(cells[_k]["source"]).replace(
    '#@title Can every viewer tell the colours apart? { run: "auto" }\nConfusableBelow = 10  #@param {type:"slider", min:3, max:25, step:1}\n#@markdown Two colours closer than this CIEDE2000 colour difference count as easily confused, as in Part B.\n\n', ""))


# ── the palette in use: previews before export ──
_e = _find(lambda s, t: t == "code" and s.strip().startswith("export_palette("))
cells[_e:_e] = [
md("""### See the palette in use

Choose a language to see its grounded palette applied to three kinds of design: a **chart**, an **interface** (buttons and status dots) and **wayfinding** (colour-coded routes with labels). The second row shows the same designs for another kind of vision, so you can check the palette still works there. Each preview is also saved as a picture you can download."""),
code("""#@title Preview { run: "auto" }
PreviewLanguage = "British English"  #@param ["American English", "British English", "French", "Greek", "OtjiHimba", "Spanish"]
PreviewVision = "Deuteranopia"  #@param ["Typical", "Protanopia", "Deuteranopia", "Tritanopia", "Age 40", "Age 60", "Age 80"]

preview_applications(pal, PreviewLanguage, PreviewVision, path=f"preview_{PreviewLanguage.replace(' ', '_')}.png")""", hidden=False),
]


# ── save the finding (card + data), with the pair's own interpretation ──
_k = next(k for k, c in enumerate(cells) if c["cell_type"] == "code" and "show_finding(pal, FindingName)" in "".join(c["source"]))
cells[_k] = code("""#@title Finding to share { run: "auto" }
FindingName = ""  #@param {type:"string"}
#@markdown Leave empty for the name where the AI was furthest from speakers, or type another name.
show_finding(pal, FindingName)""")
cells.insert(_k + 1, code("""#@title Save your finding
Pattern = ""  #@param {type:"string"}
DesignerFix = ""  #@param {type:"string"}
#@markdown **Pattern**: what pattern of failure does it reveal? **DesignerFix**: what would a designer need to do differently? Then run this cell to download your finding as a picture card and as data.
save_finding(pal, FindingName, Pattern, DesignerFix)"""))


# ── Part C without repetition: each table appears once, each idea is explained once ──
def _c(pred):
    return next(c for c in cells if pred("".join(c["source"]), c["cell_type"]))
_loader  = cells[1]
_form    = _c(lambda s, t: t == "code" and s.startswith("#@title Where your names come from"))
_typed   = _c(lambda s, t: t == "code" and s.startswith('#@title Only if you chose "Type them in"'))
_load    = _c(lambda s, t: t == "code" and "names = load_names(NamesFrom)" in s)
_prompt  = _c(lambda s, t: t == "code" and "PROMPT_C = f" in s)
_preview = _c(lambda s, t: t == "code" and s.startswith("#@title Preview"))
_export  = _c(lambda s, t: t == "code" and s.strip().startswith("export_palette("))
_finding = _c(lambda s, t: t == "code" and s.startswith("#@title Finding to share"))
_save    = _c(lambda s, t: t == "code" and s.startswith("#@title Save your finding"))
_footer  = _c(lambda s, t: t == "markdown" and s.startswith("---"))

cells[:] = [
md("""# Colour Brings People Together
### Part C · Creating an empirically grounded colour palette

In Part B you audited your test palette: what you call each colour, and what your audience calls it. Now go the other way. You'll give those **names** to an AI system, ask it for the colour each name stands for, and check its answers against how speakers actually use each name. Where the AI's colour diverges, you replace it with the speakers' best example, and you finish with a palette for each language that you can defend with evidence.

This matters: in the *Colour in Translation* benchmark (Mylonas et al., CHI 2026), a current large language model's colours were on average more than 17 ΔE from speakers' focal colours, because it mapped colour names onto the same prototypes whatever the language.

| Step | You do | Time |
|---|---|---|
| 1–2 | Load your names and ask your AI for their colours | 15 min |
| 3 | See whether the AI's colours mean the same to speakers | 10 min |
| 4 | Ground the palette, and fix colours that became too similar | 10 min |
| 5–7 | See it in use, export it, and save your finding | 10 min |
| | Plenary discussion | 15 min |

Run each cell with **Shift + Enter** and change only the grey form fields."""),
_loader,
md("""## 1 · Your names from Part B

Upload the `partB_results.csv` you downloaded at the end of Part B. If you lost it, choose *Type them in* and enter a few colour names per language."""),
_form, _typed, _load,
md("""## 2 · Ask your AI for the colour of each name

The prompt asks for each name's **best example**: the colour native speakers would consider its most typical instance. Use the same AI as in Part B. Copy the prompt or open it with a button, send it, then paste the AI's whole reply into the box in one go."""),
_prompt,
code("pal = read_ai_colours(collected_answers(), names)       # the colour codes read from the pasted reply"),
md("""## 3 · Does the AI's colour mean the same to speakers?

Two checks for every name. **Distance:** how far the AI's colour is from the speakers' best example, the name's focal colour (the Mapper's top-left tile). **Round trip:** send the AI's colour back to the Spinner's Namer and see what speakers would call it. A colour can be some distance from the focal colour and still be read correctly, or close to it and still be called something else; together the two checks show whether the AI's colour communicates."""),
code("""#@title Compare { run: "auto" }
ReplaceAbove = 10  #@param {type:"slider", min:3, max:30, step:1}
#@markdown An AI colour further than this CIEDE2000 colour difference from the speakers' best example counts as divergent.

spinner = connect(Backend)
pal = add_best_examples(pal, spinner)                   # the speakers' best example of each name
pal["de00"] = [de00(ai, best) if isinstance(ai, str) and isinstance(best, str) else np.nan
               for ai, best in zip(pal.ai_hex, pal.spinner_hex)]
pal["rt_name"], pal["rt_conf"] = None, np.nan
for k, row in pal.iterrows():                           # round trip: what speakers call the AI's colour
    if isinstance(row.ai_hex, str):
        namer = safe(spinner.name, row.ai_hex, row.language)
        if namer:
            pal.at[k, "rt_name"], pal.at[k, "rt_conf"] = namer["name"], namer["confidence"]
pal = pal.sort_values(["language", "de00"], ascending=[True, False]).reset_index(drop=True)
show_comparison(pal, ReplaceAbove)
error_summary(pal, LANGUAGES, ReplaceAbove)""", hidden=False),
md("""The same comparison as a map, on the CIELAB a*–b* plane used in Part B. Arrows that all point the same way for one language suggest a systematic bias, for instance towards English focal colours."""),
code("plot_ai_vs_speakers(pal, LANGUAGES, ReplaceAbove)"),
md("""## 4 · Ground the palette

One cell, two steps. **Ground:** each AI colour that diverges from how speakers use the name (too far from its best example, or called something else; choose the rule in the form) is replaced by the speakers' best example. **Repair:** grounding can put two colours of one palette close together, such as the focal colours of two blues. Where two colours are closer than the threshold under any viewing condition, one is swapped for **another colour speakers still call by the same name** (the next tiles in the Spinner's Mapper), choosing the swap that separates the palette most.

The chart underneath compares each language's AI palette with its grounded one under all seven conditions. If a pair still falls below the threshold, you've found a real design tension: consider a second cue such as a pattern, label or icon, or drop one name.

All colour differences are CIEDE2000 (ΔE₀₀), derived from judgements by observers with typical colour vision; for simulated colours they only approximate what people with a colour vision deficiency or an older eye can tell apart."""),
code("""#@title Ground and repair { run: "auto" }
ReplaceWhen = "Either"  #@param ["Either", "Too far from the best example", "Speakers would use a different name"]
ConfusableBelow = 10  #@param {type:"slider", min:3, max:25, step:1}
#@markdown **ReplaceWhen**: which check decides that the AI's colour diverges. **ConfusableBelow**: two colours closer than this, under any viewing condition, count as too similar.

too_far = [pd.isna(d) or d > ReplaceAbove for d in pal.de00]
renamed = [isinstance(n, str) and not same_name(n, m, r)
           for n, m, r in zip(pal.rt_name, pal.name, pal.get("romanized", pd.Series(None, index=pal.index)))]
rule = {"Either": [a or b for a, b in zip(too_far, renamed)],
        "Too far from the best example": too_far,
        "Speakers would use a different name": renamed}[ReplaceWhen]

# Ground: the speakers' best example wherever the AI's colour diverges.
pal["grounded_hex"] = [best if isinstance(best, str) and replace else ai
                       for ai, best, replace in zip(pal.ai_hex, pal.spinner_hex, rule)]
pal["replaced"] = pal.grounded_hex != pal.ai_hex
pal = pal.drop(columns=["repaired_from"], errors="ignore")
# Repair: swap colours that became too similar for another colour of the same name.
fixed = repair_palettes(pal, spinner, LANGUAGES, ConfusableBelow)
if fixed is not None:
    pal = fixed
show_grounded(pal, LANGUAGES)
if fixed is not None:
    show_repairs(pal, ConfusableBelow)
plot_closest_pairs(closest_pairs(pal, LANGUAGES), ConfusableBelow)""", hidden=False),
md("""## 5 · See the palette in use

Choose a language to see its grounded palette in a **chart**, an **interface** and **wayfinding**, with typical vision and with another kind of vision. Each preview is saved for the download."""),
_preview,
md("""## 6 · Export your palette

The download has a picture of every palette, the previews you made, swatch files for design tools (Adobe, GIMP, Figma), CSS for websites, and the data with the evidence for every change."""),
_export,
md("""## 7 · Your most striking finding

The card shows the name where the AI's colour was furthest from what speakers mean; type another name to choose a different one. Then add your interpretation and save the card for the plenary."""),
_finding, _save, _footer,
]


# ── collaborator feedback, October 2026 ──
def _src(c): return "".join(c["source"])
for _k, _c in enumerate(cells):
    if _c["cell_type"] != "code": continue
    _s = _src(_c)
    if _s.startswith("#@title Compare"):
        _s = _s.split("\n", 1)[1]
        _s = re.sub(r"ReplaceAbove = 10  #@param[^\n]*\n#@markdown[^\n]*\n", "ReplaceAbove = DELTA_E_THRESHOLD              # ΔE₀₀ 10, the threshold used throughout\n", _s)
        cells[_k] = code(_s, hidden=False)
    elif _s.startswith("#@title Ground and repair"):
        _s = re.sub(r"ConfusableBelow = 10  #@param[^\n]*\n", "", _s)
        _s = _s.replace("#@markdown **ReplaceWhen**: which check decides that the AI's colour diverges. **ConfusableBelow**: two colours closer than this, under any viewing condition, count as too similar.",
                        "#@markdown **ReplaceWhen**: which check decides that the AI's colour diverges. Two colours closer than ΔE₀₀ 10, under any viewing condition, count as too similar.\nConfusableBelow = DELTA_E_THRESHOLD")
        _s = _s.replace("plot_closest_pairs(closest_pairs(pal, LANGUAGES), ConfusableBelow)", "show_closest_pairs(pal, LANGUAGES, ConfusableBelow)")
        cells[_k] = code(_s, hidden=False)
for _k, _c in enumerate(cells):
    if _c["cell_type"] == "markdown":
        _s = _src(_c)
        _s = _s.replace("The chart underneath compares each language's AI palette with its grounded one under all seven conditions.",
                        "The table underneath shows, for each language's AI and grounded palette and each viewing condition, the two closest colours as they look there and the ΔE₀₀ between the simulated colours.")
        _s = _s.replace("(too far from its best example, or called something else; choose the rule in the form)",
                        "(more than ΔE₀₀ 10 from its best example, or called something else; choose the rule in the form)")
        _s = _s.replace("Where two colours are closer than the threshold under any viewing condition,",
                        "Where two colours are closer than ΔE₀₀ 10 under any viewing condition,")
        cells[_k] = md(_s)
