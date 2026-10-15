from nbtools import md, code, loader_cell

TITLE = "Part C — Creating an empirically grounded colour palette"

cells = [
md("""# Colour Brings People Together
### Part C · Creating an empirically grounded colour palette

In Part B you found what speakers of several languages call a handful of SDG colours. Now reverse the direction. You will give those **names** to an AI system, ask it for the colour each name stands for, and check its answers against where speakers actually centre each name. Where the AI's colour diverges, you replace it with the Colour Spinner's best example, and you finish with a small palette you can defend with evidence.

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
NAMES_FROM = "Upload partB_results.csv"  #@param ["Upload partB_results.csv", "Type them in"]
BACKEND = "Colour Spinner API"  #@param ["Colour Spinner API", "Copied from the browser"]
#@markdown *Copied from the browser* only matters if you type names in: then add each name's best example to `SHEET` below."""),

code("""TYPED_NAMES = \"\"\"
language, name
# British English, Teal          ← one row per name, only if you chose "Type them in"
\"\"\"

SHEET = \"\"\"
hex, language, name, confidence, best_example
\"\"\"
"""),

code("""if NAMES_FROM.startswith("Upload"):
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

The prompt asks for the colour that a native speaker would consider the **best example** of each name: the most typical instance, not the average and not a brand colour. As in Part B, Colab's built-in Gemini is used for demonstration, and you can use any AI you use regularly instead: choose *Paste the reply myself*, copy the prompt into your tool and paste its reply into `AI_REPLY`. Use the same AI system as in Part B so the two parts can be compared."""),

code("""items = "\\n".join(f"- {r.language}: {r['name']}" for _, r in names.iterrows())
PROMPT_C = f\"\"\"You are helping with a cross-cultural colour naming study.
For each colour name below, give the sRGB colour that native speakers of that language would consider
the best example of the name: its most typical instance.

{items}

Reply with JSON only, as a list of objects like
{{"language": "{names.language.iloc[0]}", "name": "{names['name'].iloc[0]}", "rgb": [r, g, b]}}
with one object per name, in the same order.\"\"\"
print(PROMPT_C)"""),

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

code("""reply_c = AI_REPLY if AI_PROVIDER.startswith("Paste") else ask_ai(PROMPT_C, AI_PROVIDER, AI_MODEL, AI_BASE_URL)
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

For every name, the left half of each swatch is the AI's colour and the right half is the Colour Spinner's best example, the colour speakers most often call by that name. The badge gives the CIEDE2000 difference between them.

A reminder of scale: about 1 is just noticeable with the two side by side, about 5 is obvious, and above 10 the two are usually read as different colours."""),

code("""#@title Fill in the Spinner's best examples and compare { run: "auto" }
DIVERGES_AT = 10  #@param {type:"slider", min:3, max:30, step:1}

need = pal.spinner_hex.isna()
if need.any():
    spinner = ApiSpinner() if BACKEND == "Colour Spinner API" else SheetSpinner(SHEET)
    for k in pal.index[need]:
        got = try_call(spinner.best_example, pal.at[k, "name"], pal.at[k, "language"])
        pal.at[k, "spinner_hex"] = None if isinstance(got, Exception) else got

pal["de00"] = [de00(a, b) if isinstance(a, str) and isinstance(b, str) else np.nan
               for a, b in zip(pal.ai_hex, pal.spinner_hex)]
pal["de_ucs"] = [de_ucs(a, b) if isinstance(a, str) and isinstance(b, str) else np.nan
                 for a, b in zip(pal.ai_hex, pal.spinner_hex)]
pal = pal.sort_values(["language", "de00"], ascending=[True, False]).reset_index(drop=True)
show_pairs(pal, threshold=DIVERGES_AT)"""),

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
ax.axvline(DIVERGES_AT, color=INK, ls="--", lw=1)
ax.text(DIVERGES_AT, len(order) - 0.4, f"  replace above {DIVERGES_AT}", fontsize=9, color=MUTED, va="bottom")
ax.set_yticks(y, [f"{n}  ({l})" for n, l in zip(order.name, order.language)])
ax.set_xlabel("ΔE00 between the AI's colour and the best example"); ax.set_xlim(left=-1)
ax.legend(handles=[mpl.lines.Line2D([], [], marker="o", ls="", color="#BDBDB8", markeredgecolor=INK,
                   markersize=6, label="AI's colour"),
                   mpl.lines.Line2D([], [], marker="o", ls="", color="#BDBDB8", markeredgecolor="white",
                   markersize=11, label="speakers' best example")], frameon=False, loc="lower right")
ax.set_title("Size of each error")
plt.tight_layout(); plt.show()

summary = ok.groupby("language").agg(names=("name", "size"), mean_dE00=("de00", "mean"),
                                     worst_dE00=("de00", "max"),
                                     diverging=("de00", lambda d: int((d > DIVERGES_AT).sum())))
display(summary.reindex([l for l in LANGUAGES if l in summary.index]).round(1))"""),

md("""## 4 · Ground the palette

Every name whose AI colour is further than the threshold from the speakers' best example is replaced by that best example. Move the slider in the previous section to change the threshold; this section updates with it."""),

code("""pal["grounded_hex"] = [s if isinstance(s, str) and (pd.isna(d) or d > DIVERGES_AT) else a
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

Grounding fixes the cognitive axis, but best examples of neighbouring names can sit close together. Check the perceptual axis again: for each language's palette, the smallest ΔE00 between any two of its colours under each viewing condition."""),

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
ax.imshow(check.values, cmap=RISK_CMAP, vmin=0, vmax=2 * DIVERGES_AT, aspect="auto")
for (r, c), v in np.ndenumerate(check.values):
    if not np.isnan(v):
        ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=10,
                color="white" if v < 0.7 * DIVERGES_AT else INK)
ax.set_xticks(range(len(CONDITIONS)), [condition_label(c) for c in CONDITIONS], rotation=30, ha="right")
ax.set_yticks(range(len(check)), [f"{l} · {v}" for l, v in check.index])
for r in range(2, len(check), 2): ax.axhline(r - 0.5, color="white", lw=3)
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("Closest pair in each palette under each condition (ΔE00)")
plt.tight_layout(); plt.show()

langs_shown = [l for l in LANGUAGES if (pal[pal.language == l].grounded_hex.notna()).sum() > 1]
for k, lang in enumerate(langs_shown):
    p = pal[(pal.language == lang) & pal.grounded_hex.notna()]
    show(f'<div style="font-weight:700;margin:10px 0 4px">{_h(lang)}, grounded</div>')
    show_conditions(p.grounded_hex.tolist(), labels=p.name.tolist(), width=64,
                    note=(k == len(langs_shown) - 1))"""),

md("""If a pair falls below the threshold under some condition, you have found a real design tension: the colours speakers expect for two names may be too similar for some viewers. Options to discuss: keep one colour and pick the next-best example of the other from the Spinner's Mapper panel, add a second cue such as a pattern, icon or label, or drop one name from the palette.

## 5 · Export your palette

The files below contain every colour with its name, language, the AI's original and the evidence for any change."""),

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
PICK_NAME = ""  #@param {type:"string"}
cand = pal.dropna(subset=["de00"])
row = (cand[cand.name.map(norm) == norm(PICK_NAME)].iloc[0] if PICK_NAME and
       (cand.name.map(norm) == norm(PICK_NAME)).any() else cand.loc[cand.de00.idxmax()])

show(f\"\"\"<div class="card">
<h3>“{_h(row['name'])}” in {_h(row.language)}</h3>
<div style="display:flex;gap:10px;margin:12px 0;background:{SURROUND};padding:12px;border-radius:6px">
  <div class="tile"><div class="chip" style="background:{row.ai_hex};color:{_ink_on(row.ai_hex)};height:80px">AI</div>
       <div class="cap">AI system<div class="hex">{row.ai_hex}</div></div></div>
  <div class="tile"><div class="chip" style="background:{row.spinner_hex};color:{_ink_on(row.spinner_hex)};height:80px">Speakers</div>
       <div class="cap">Best example<div class="hex">{row.spinner_hex}</div></div></div>
  <div style="color:#fff;align-self:center;padding-left:8px;font-size:15px">ΔE00<br><b style="font-size:30px">{row.de00:.1f}</b></div>
</div>
<p style="margin:0 0 8px">Talk it through with your partner:</p>
<p style="margin:0 0 4px"><b>What pattern of failure does this reveal?</b> Did the AI default to a Western-centric focal colour,
collapse a distinction the language maintains, or produce a colour that fails under CVD or ageing?</p>
<p style="margin:0"><b>What would a designer need to do differently</b> to avoid it in their own practice?</p>
</div>\"\"\")"""),

md("""---
*Colour Spinner: Sinkeviciute, Stutters, Ahmed, Koliousis & Mylonas (2026). Benchmark of AI colour naming: Mylonas, Ahmed, Sinkeviciute & Koliousis, CHI 2026. CVD and age simulations from the Colour Spinner; local fallbacks after Machado, Oliveira & Fernandes (2009) and van de Kraats & van Norren (2007). CAM16-UCS after Li et al. (2017).*"""),
]


# ── short cells: charts and tables now live in colour_masterclass.py ──
def _set(i, src, hidden=None):
    assert cells[i]["cell_type"] == "code", i
    cells[i] = code(src, hidden=hidden)

_set(5, """names = load_names(NAMES_FROM, TYPED_NAMES)
LANGUAGES = list(dict.fromkeys(names.language))
show_names(names)""")
_set(10, """reply_c = ai_reply(PROMPT_C, AI_PROVIDER, AI_MODEL, AI_BASE_URL, pasted=AI_REPLY)
pal = read_ai_colours(reply_c, names)
show_ai_palette(pal)""")
_set(12, """#@title Fill in the Spinner's best examples and compare { run: "auto" }
DIVERGES_AT = 10  #@param {type:"slider", min:3, max:30, step:1}

pal = add_best_examples(pal, connect(BACKEND, SHEET))     # looks up only the names that need it
pal["de00"] = [de00(ai, best) if isinstance(ai, str) and isinstance(best, str) else np.nan
               for ai, best in zip(pal.ai_hex, pal.spinner_hex)]
pal = pal.sort_values(["language", "de00"], ascending=[True, False]).reset_index(drop=True)
show_pairs(pal, threshold=DIVERGES_AT)""")
_set(14, """plot_ai_vs_speakers(pal, LANGUAGES, DIVERGES_AT)
display(error_summary(pal, LANGUAGES, DIVERGES_AT))""")
_set(16, """# Keep the AI's colour unless it is more than DIVERGES_AT from the speakers' best example.
pal["grounded_hex"] = [best if isinstance(best, str) and (pd.isna(d) or d > DIVERGES_AT) else ai
                       for ai, best, d in zip(pal.ai_hex, pal.spinner_hex, pal.de00)]
pal["replaced"] = pal.grounded_hex != pal.ai_hex
show_grounded(pal, LANGUAGES)""")
_set(18, """check = closest_pairs(pal, LANGUAGES)      # smallest ΔE00 within each palette, per condition
plot_closest_pairs(check, DIVERGES_AT)
show_grounded_conditions(pal, LANGUAGES)""")
_set(20, "export_palette(pal, LANGUAGES)")
_set(22, """#@title Finding to share { run: "auto" }
PICK_NAME = ""  #@param {type:"string"}
show_finding(pal, PICK_NAME)""")
