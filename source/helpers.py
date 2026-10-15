

# ════════════════════════════════════════════════════════════════════════
#  6. Activity helpers: the charts and tables the notebooks show
# ════════════════════════════════════════════════════════════════════════
esc, ink_on, read_sheet = _h, _ink_on, _read_sheet      # public names for notebook cells
MARKERS = dict(zip(SPINNER_LANGUAGES, ["o", "s", "D", "^", "v", "P"]))

def _language_legend(ax, languages, loc="best", size=8):
    ax.legend(handles=[mpl.lines.Line2D([], [], marker=MARKERS.get(l, "o"), ls="", color="white",
              markeredgecolor=INK, markersize=size, label=l) for l in languages],
              frameon=False, loc=loc)

def _ucs_axes(ax):
    ax.axhline(0, color="#E4E4E0", lw=1, zorder=0); ax.axvline(0, color="#E4E4E0", lw=1, zorder=0)
    ax.set_aspect("equal"); ax.set_xlabel("a′  (green ← → red)"); ax.set_ylabel("b′  (blue ← → yellow)")

def _hide_spines(ax):
    for s in ax.spines.values(): s.set_visible(False)


# ── Part B: choosing ──────────────────────────────────────────────────────
def parse_subset(text):
    """'1, 6, 12' -> [1, 6, 12] (ignores anything outside 1–17)."""
    return list(dict.fromkeys(int(s) for s in re.findall(r"\d+", text) if 1 <= int(s) <= 17))

def pick_languages(source, ticked):
    """The source language first, then every other ticked language."""
    return [source] + [l for l, on in ticked.items() if on and l != source]

def show_subset(pick, languages):
    sub = SDG.loc[pick]
    show_palette(sub.hex.tolist(), labels=[f"SDG {i}" for i in sub.index], captions=sub.goal.tolist(),
                 note=f"Auditing {len(pick)} colours in {len(languages)} languages: {', '.join(languages)}.")


# ── Part B: perceptual axis ───────────────────────────────────────────────
def nearest_neighbours(pick, conditions=CONDITIONS):
    """For each colour in `pick` and each condition: the nearest other SDG colour and its ΔE00.
    Returns two tables with the same shape: distances and neighbour ids."""
    def nearest(i, c):
        me = simulate(SDG.hex[i], c)
        return min((de00(me, simulate(h, c)), j) for j, h in SDG.hex.items() if j != i)
    nn = pd.DataFrame({c: {i: nearest(i, c) for i in pick} for c in conditions})
    return nn.map(lambda t: t[0]), nn.map(lambda t: t[1])

def plot_nearest_neighbours(nn_de, nn_id, threshold):
    pick, conds = list(nn_de.index), list(nn_de.columns)
    fig, ax = plt.subplots(figsize=(1.1 * len(conds) + 1.6, 0.62 * len(pick) + 1.4))
    ax.imshow(nn_de.values, cmap=RISK_CMAP, vmin=0, vmax=2 * threshold, aspect="auto")
    for (r, c), v in np.ndenumerate(nn_de.values):
        ink = "white" if v < threshold * 0.6 else INK
        ax.text(c, r - 0.08, f"{v:.1f}", ha="center", va="center", fontsize=11, fontweight="bold", color=ink)
        ax.text(c, r + 0.27, f"vs {nn_id.values[r, c]}", ha="center", va="center", fontsize=8, color=ink)
    ax.set_xticks(range(len(conds)), [condition_label(c) for c in conds], rotation=30, ha="right")
    row_chips(ax, [SDG.hex[i] for i in pick], [f"SDG {i}" for i in pick])
    ax.tick_params(axis="x", length=0); _hide_spines(ax)
    ax.set_title(f"Distance to the nearest other SDG colour (ΔE00); red cells fall below {threshold}")
    plt.tight_layout(); plt.show()

def show_at_risk(nn_de, nn_id, threshold):
    at_risk = {}
    for i in nn_de.index:
        for c in nn_de.columns:
            if nn_de.loc[i, c] < threshold:
                at_risk.setdefault(tuple(sorted((i, nn_id.loc[i, c]))), []).append((c, nn_de.loc[i, c]))
    if not at_risk:
        show(f"Every colour in your subset stays at least ΔE00 {threshold} from its nearest "
             "neighbour under every condition.")
        return
    rows = ""
    for (i, j), hits in sorted(at_risk.items(), key=lambda kv: min(d for _, d in kv[1])):
        worst = min(hits, key=lambda t: t[1])[0]
        chips = "".join(f'<div class="mini" style="background:{simulate(SDG.hex[k], worst)}"></div>' for k in (i, j))
        rows += (f'<tr><td><div class="row">{chips}</div></td><td><b>SDG {i} and {j}</b></td><td>' +
                 " ".join(f'<span style="white-space:nowrap">{esc(c)} {de_badge(d, threshold, small_is_good=False)}</span>'
                          for c, d in hits) + "</td></tr>")
    show('<b>Pairs at risk</b>, shown as they look under the condition where they are closest'
         f'<table class="babel">{rows}</table>')


# ── Part B: cognitive axis ────────────────────────────────────────────────
def connect(backend, sheet=""):
    """The Colour Spinner itself, or the values you copied from it into SHEET."""
    return ApiSpinner() if backend == "Colour Spinner API" else SheetSpinner(sheet)

def report_problems(problems):
    if problems:
        show("<b>Some lookups did not work</b><br>" + "<br>".join(map(esc, problems[:8])) +
             ("<br>…" if len(problems) > 8 else "") +
             '<div class="note">Using the Colour Spinner: check your internet connection, or set BACKEND to '
             '<i>Copied from the browser</i>. Using the sheet: add the missing rows and run the cell again.</div>')

def show_names_table(naming, languages):
    if naming.empty: return
    wide = naming.pivot(index="sdg", columns="language", values="name")
    conf = naming.pivot(index="sdg", columns="language", values="confidence").add_suffix("|conf")
    show_babel(SDG.loc[wide.index, ["hex"]].join(wide).join(conf), [l for l in languages if l in wide.columns])

def plot_nameability(naming, pick, languages):
    rows = [(i, l) for i in pick for l in languages
            if not naming[(naming.sdg == i) & (naming.language == l)].empty]
    if not rows: return
    gap = {i: k for k, i in enumerate(pick)}
    ys = [n + 0.6 * gap[i] for n, (i, l) in enumerate(rows)]
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(rows) + 0.5 * len(pick) + 1))
    blend = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for y, (i, l) in zip(ys, rows):
        r = naming[(naming.sdg == i) & (naming.language == l)].iloc[0]
        ax.hlines(y, 0, r.confidence, color="#DADAD5", lw=6, capstyle="round", zorder=1)
        ax.scatter(r.confidence, y, s=120, color=SDG.hex[i], edgecolor=INK, lw=0.7, zorder=3)
        ax.text(r.confidence + 0.025, y, f"{r['name']}  {r.confidence:.0%}", va="center", fontsize=10)
    for i in pick:
        g = [y for y, (j, _) in zip(ys, rows) if j == i]
        if not g: continue
        ax.add_patch(mpl.patches.Rectangle((-0.3, min(g) - 0.4), 0.07, max(g) - min(g) + 0.8,
                     color=SDG.hex[i], transform=blend, clip_on=False))
        ax.text(-0.265, (min(g) + max(g)) / 2, str(i), transform=blend, ha="center", va="center",
                rotation=90, color=ink_on(SDG.hex[i]), fontweight="bold", clip_on=False)
    ax.set_yticks(ys, [l for _, l in rows]); ax.tick_params(axis="y", length=0)
    ax.invert_yaxis(); ax.set_xlim(0, 1.18); ax.spines["left"].set_visible(False)
    ax.set_xticks([0, .25, .5, .75, 1]); ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    ax.set_xlabel("Share of speakers who use the top name")
    ax.set_title("Nameability: how much do speakers agree?")
    plt.tight_layout(); plt.show()

def show_translatability(naming, pick, languages, threshold):
    rows = ""
    for i in pick:
        sub = naming[naming.sdg == i]
        cells = ""
        for lang in languages:
            r = sub[sub.language == lang]
            if r.empty or not isinstance(r.best_example.iloc[0], str):
                cells += f'<td style="color:{MUTED}">—</td>'; continue
            b = r.best_example.iloc[0]
            cells += (f'<td><div class="pair" style="width:110px"><div style="background:{b};color:{ink_on(b)}">'
                      f'{esc(r.name.iloc[0])}<br>{b}</div></div><div style="margin-top:3px">'
                      f'{de_badge(r.drift.iloc[0], threshold)}</div></td>')
        sp = spread(sub.best_example.tolist())
        rows += (f'<tr><td><div class="pair" style="width:96px"><div style="background:{SDG.hex[i]};'
                 f'color:{ink_on(SDG.hex[i])};font-weight:700;font-size:13px">SDG {i}</div></div></td>{cells}'
                 f'<td style="color:{MUTED}">{"" if pd.isna(sp) else "spread " + format(sp, ".1f")}</td></tr>')
    show('<table class="babel"><tr><th></th>' + "".join(f"<th>{esc(l)}</th>" for l in languages) +
         f'<th></th></tr>{rows}</table><div class="note">Badges: ΔE00 from the SDG colour. '
         "Spread: the largest ΔE00 between two languages' best examples.</div>")

def plot_name_map(naming, pick, languages):
    fig, ax = plt.subplots(figsize=(7, 7)); _ucs_axes(ax)
    for i in pick:
        o = rgb_to_cam16ucs(SDG.hex[i])
        for _, r in naming[(naming.sdg == i) & naming.best_example.notna()].iterrows():
            b = rgb_to_cam16ucs(r.best_example)
            ax.plot([o[1], b[1]], [o[2], b[2]], color=MUTED, lw=1, zorder=1)
            ax.scatter(b[1], b[2], s=90, marker=MARKERS.get(r.language, "o"), color=r.best_example,
                       edgecolor=INK, lw=0.7, zorder=3)
        ax.scatter(o[1], o[2], s=520, color=SDG.hex[i], edgecolor="white", lw=2.5, zorder=2)
        ax.text(o[1], o[2], str(i), ha="center", va="center", color=ink_on(SDG.hex[i]),
                fontweight="bold", fontsize=10, zorder=4)
    _language_legend(ax, languages)
    ax.set_title("Where each language's name pulls the colour")
    plt.tight_layout(); plt.show()


# ── Part B: two axes ──────────────────────────────────────────────────────
def scorecard(nn_de, naming):
    """One row per colour: worst-case perceptual distance and mean naming agreement."""
    return pd.DataFrame({
        "worst_case_dE00": nn_de.min(axis=1),
        "worst_condition": nn_de.idxmin(axis=1),
        "mean_agreement": naming.groupby("sdg").confidence.mean(),
        "language_spread": naming.groupby("sdg").best_example.apply(lambda s: spread(s.tolist())),
    }).loc[nn_de.index]

def plot_two_axes(score, distinct_at, agree_at):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    xmax = max(2.2 * distinct_at, score.worst_case_dE00.max() * 1.15)
    ax.axvline(distinct_at, color="#BDBDB8", ls="--", lw=1); ax.axhline(agree_at, color="#BDBDB8", ls="--", lw=1)
    for x, y, t in [(0.02, 0.98, "named, but easily confused"), (0.98, 0.98, "robust on both axes"),
                    (0.02, 0.02, "fragile on both axes"), (0.98, 0.02, "distinct, but hard to name")]:
        ax.text(x, y, t, transform=ax.transAxes, ha="left" if x < .5 else "right",
                va="top" if y > .5 else "bottom", color=MUTED, fontsize=10, style="italic")
    for i, r in score.iterrows():
        ax.scatter(r.worst_case_dE00, r.mean_agreement, s=900, color=SDG.hex[i], edgecolor="white", lw=2.5, zorder=3)
        ax.text(r.worst_case_dE00, r.mean_agreement, str(i), ha="center", va="center",
                color=ink_on(SDG.hex[i]), fontweight="bold", zorder=4)
    ax.set_xlim(0, xmax); ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    ax.set_xlabel("Perceptual: worst-case ΔE00 to nearest neighbour"); ax.set_ylabel("Cognitive: mean naming agreement")
    ax.set_title("Where your colours sit on the two axes")
    plt.tight_layout(); plt.show()


# ── Part B: the AI's names ────────────────────────────────────────────────
def ai_reply(prompt, provider, model="", base_url="", pasted=""):
    """The pasted reply, or the provider's answer to `prompt`."""
    return pasted if provider.startswith("Paste") else ask_ai(prompt, provider, model, base_url)

def read_ai_names(reply):
    t = pd.DataFrame(parse_json_reply(reply))
    t["hex"] = t["hex"].map(parse_colour)
    return t.rename(columns={"name": "ai_name"})[["hex", "language", "ai_name"]]

def show_name_comparison(cmp):
    same = "<span class='badge' style='background:#DDEFE3;color:#14532D'>same</span>"
    diff = "<span class='badge' style='background:#F9DADA;color:#7F1D1D'>differs</span>"
    rows = "".join(
        f'<tr><td><div class="mini" style="background:{r.hex}"></div></td><td>SDG {r.sdg}</td>'
        f'<td>{esc(r.language)}</td><td><b>{esc(r["name"])}</b> '
        f'<span style="color:{MUTED};font-size:11px">{r.confidence:.0%}</span></td>'
        f'<td><b>{esc(r.ai_name if isinstance(r.ai_name, str) else "—")}</b></td>'
        f'<td>{same if r.agrees else diff}</td></tr>' for _, r in cmp.iterrows())
    show('<table class="babel"><tr><th></th><th></th><th>Language</th><th>Speakers (Spinner)</th>'
         f'<th>AI system</th><th></th></tr>{rows}</table>')

def plot_agreement_by_language(cmp, languages):
    by_lang = cmp.groupby("language").agrees.mean().reindex(languages)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(languages) + 1.2))
    ax.barh(by_lang.index, by_lang.values, color=INK, height=0.5)
    for y, v in enumerate(by_lang.values):
        if not pd.isna(v): ax.text(v + 0.02, y, f"{v:.0%}", va="center", fontsize=10)
    ax.set_xlim(0, 1.1); ax.invert_yaxis(); ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    ax.set_title("How often the AI agrees with speakers")
    plt.tight_layout(); plt.show()


# ── Part C: names and the AI's colours ────────────────────────────────────
def load_names(source, typed=""):
    """Names to test: from partB_results.csv (uploaded in Colab) or typed in."""
    if source.startswith("Upload"):
        if IN_COLAB:
            from google.colab import files
            up = files.upload()
            partb = pd.read_csv(io.BytesIO(next(iter(up.values()))))
        else:
            partb = pd.read_csv("partB_results.csv")
        names = (partb.dropna(subset=["name"]).drop_duplicates(["language", "name"])
                 [["sdg", "hex", "language", "name", "best_example"]]
                 .rename(columns={"hex": "sdg_hex", "best_example": "spinner_hex"}))
    else:
        names = read_sheet(typed).assign(sdg=np.nan, sdg_hex=None, spinner_hex=None)[
            ["sdg", "sdg_hex", "language", "name", "spinner_hex"]]
    return names.reset_index(drop=True)

def show_names(names):
    langs = list(dict.fromkeys(names.language))
    show('<table class="babel"><tr><th>Language</th><th>Names to test</th></tr>' + "".join(
         f"<tr><td>{esc(l)}</td><td><b>{', '.join(esc(n) for n in names[names.language == l].name)}</b></td></tr>"
         for l in langs) + "</table>")

def read_ai_colours(reply, names):
    """Attach the AI's colour for each name (matched on language + name)."""
    ai = pd.DataFrame(parse_json_reply(reply))
    ai["ai_hex"] = [parse_colour(" ".join(map(str, r["rgb"]))) if isinstance(r.get("rgb"), list)
                    else parse_colour(r.get("hex", "")) for _, r in ai.iterrows()]
    key = lambda t: t.language.map(norm) + "|" + t.name.map(norm)
    pal = names.assign(key=key(names)).merge(ai.assign(key=key(ai))[["key", "ai_hex"]], on="key", how="left")
    missing = pal[pal.ai_hex.isna()]
    if len(missing):
        show("<b>The AI reply has no colour for:</b> " + ", ".join(
             f"{esc(r['name'])} ({esc(r.language)})" for _, r in missing.iterrows()) +
             '<div class="note">Ask it again for those names, or carry on without them.</div>')
    return pal.drop(columns="key")

def show_ai_palette(pal):
    have = pal[pal.ai_hex.notna()]
    show_palette(have.ai_hex.tolist(), labels=[""] * len(have),
                 captions=[f"{n} · {l}" for n, l in zip(have.name, have.language)],
                 note="The AI system's best examples.")

def add_best_examples(pal, spinner):
    """Fill in the Spinner's best example for any name that does not have one yet."""
    pal = pal.copy()
    for k in pal.index[pal.spinner_hex.isna()]:
        got = try_call(spinner.best_example, pal.at[k, "name"], pal.at[k, "language"])
        pal.at[k, "spinner_hex"] = None if isinstance(got, Exception) else got
    return pal


# ── Part C: comparing ─────────────────────────────────────────────────────
def plot_ai_vs_speakers(pal, languages, threshold):
    ok = pal.dropna(subset=["ai_hex", "spinner_hex"])
    if ok.empty: return
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1, 1.05]})
    ax = axes[0]; _ucs_axes(ax)
    for _, r in ok.iterrows():
        a, s = rgb_to_cam16ucs(r.ai_hex), rgb_to_cam16ucs(r.spinner_hex)
        ax.annotate("", xy=(s[1], s[2]), xytext=(a[1], a[2]),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1, shrinkA=7, shrinkB=9), zorder=1)
        ax.scatter(a[1], a[2], s=70, marker=MARKERS.get(r.language, "o"), color=r.ai_hex,
                   edgecolor=INK, lw=0.6, zorder=2)
        ax.scatter(s[1], s[2], s=170, marker=MARKERS.get(r.language, "o"), color=r.spinner_hex,
                   edgecolor="white", lw=2, zorder=3)
    _language_legend(ax, languages)
    ax.set_title("From the AI's colour to the speakers' best example")

    ax = axes[1]
    order = ok.sort_values("de00"); y = np.arange(len(order))
    ax.hlines(y, 0, order.de00, color="#CFCFCA", lw=2)
    ax.scatter(order.de00, y, s=150, c=order.spinner_hex.tolist(), edgecolor="white", lw=1.5, zorder=3)
    ax.scatter(np.zeros(len(order)), y, s=60, c=order.ai_hex.tolist(), edgecolor=INK, lw=0.5, zorder=3)
    ax.axvline(threshold, color=INK, ls="--", lw=1)
    ax.text(threshold, len(order) - 0.4, f"  replace above {threshold}", fontsize=9, color=MUTED, va="bottom")
    ax.set_yticks(y, [f"{n}  ({l})" for n, l in zip(order.name, order.language)])
    ax.set_xlabel("ΔE00 between the AI's colour and the best example"); ax.set_xlim(left=-1)
    ax.legend(handles=[
        mpl.lines.Line2D([], [], marker="o", ls="", color="#BDBDB8", markeredgecolor=INK, markersize=6,
                         label="AI's colour"),
        mpl.lines.Line2D([], [], marker="o", ls="", color="#BDBDB8", markeredgecolor="white", markersize=11,
                         label="speakers' best example")], frameon=False, loc="lower right")
    ax.set_title("Size of each error")
    plt.tight_layout(); plt.show()

def error_summary(pal, languages, threshold):
    ok = pal.dropna(subset=["ai_hex", "spinner_hex"])
    t = ok.groupby("language").agg(names=("name", "size"), mean_dE00=("de00", "mean"),
                                   worst_dE00=("de00", "max"),
                                   diverging=("de00", lambda d: int((d > threshold).sum())))
    return t.reindex([l for l in languages if l in t.index]).round(1)


# ── Part C: grounding ─────────────────────────────────────────────────────
def show_grounded(pal, languages):
    for lang in languages:
        p = pal[(pal.language == lang) & pal.grounded_hex.notna()]
        if p.empty: continue
        tiles = ""
        for _, r in p.iterrows():
            ai_h = r.ai_hex if isinstance(r.ai_hex, str) else "#E4E4E0"
            tiles += (f'<div class="tile"><div class="pair" style="width:86px;height:38px;border:none;border-radius:0">'
                      f'<div style="background:{ai_h};color:{ink_on(ai_h)}">AI</div></div>'
                      f'<div class="chip" style="background:{r.grounded_hex};color:{ink_on(r.grounded_hex)};height:56px">'
                      f'{esc(r["name"])}</div><div class="cap">{"replaced" if r.replaced else "kept"}'
                      f'<div class="hex">{r.grounded_hex}</div></div></div>')
        show(f'<div style="font-weight:700;margin:12px 0 4px">{esc(lang)}</div><div class="surround">{tiles}</div>')
    show(f'<div class="note">{int(pal.replaced.sum())} of {int(pal.grounded_hex.notna().sum())} colours '
         "replaced with the Colour Spinner's best example. Top strip: the AI's original colour.</div>")

def closest_pairs(pal, languages, conditions=CONDITIONS):
    """For each language: the smallest ΔE00 between any two of its colours, AI vs grounded."""
    from itertools import combinations
    def min_pair(hexes, c):
        hs = [simulate(h, c) for h in hexes if isinstance(h, str)]
        return min((de00(a, b) for a, b in combinations(hs, 2)), default=np.nan)
    prefetch_simulations([h for h in pd.concat([pal.ai_hex, pal.grounded_hex]) if isinstance(h, str)])
    rows = [{"language": l, "palette": v, **{c: min_pair(pal[pal.language == l][col].tolist(), c)
                                              for c in conditions}}
            for l in languages for v, col in [("AI", "ai_hex"), ("Grounded", "grounded_hex")]]
    return pd.DataFrame(rows).set_index(["language", "palette"])

def plot_closest_pairs(check, threshold):
    conds = list(check.columns)
    fig, ax = plt.subplots(figsize=(1.1 * len(conds) + 2.4, 0.5 * len(check) + 1.4))
    ax.imshow(check.values, cmap=RISK_CMAP, vmin=0, vmax=2 * threshold, aspect="auto")
    for (r, c), v in np.ndenumerate(check.values):
        if not np.isnan(v):
            ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=10,
                    color="white" if v < 0.6 * threshold else INK)
    ax.set_xticks(range(len(conds)), [condition_label(c) for c in conds], rotation=30, ha="right")
    ax.set_yticks(range(len(check)), [f"{l} · {v}" for l, v in check.index])
    for r in range(2, len(check), 2): ax.axhline(r - 0.5, color="white", lw=3)
    ax.tick_params(length=0); _hide_spines(ax)
    ax.set_title("Closest pair in each palette under each condition (ΔE00)")
    plt.tight_layout(); plt.show()

def show_grounded_conditions(pal, languages):
    langs = [l for l in languages if pal[pal.language == l].grounded_hex.notna().sum() > 1]
    for k, lang in enumerate(langs):
        p = pal[(pal.language == lang) & pal.grounded_hex.notna()]
        show(f'<div style="font-weight:700;margin:10px 0 4px">{esc(lang)}, grounded</div>')
        show_conditions(p.grounded_hex.tolist(), labels=p.name.tolist(), width=64, note=(k == len(langs) - 1))


# ── Part C: export and finding ────────────────────────────────────────────
def export_palette(pal, languages, stem="grounded_palette"):
    """Write CSV, JSON and CSS versions of the grounded palette, zip them and offer the download."""
    import zipfile
    export = pal[["language", "name", "grounded_hex", "ai_hex", "spinner_hex", "de00", "replaced", "sdg", "sdg_hex"]]
    export.round(2).to_csv(f"{stem}.csv", index=False)
    data = {lang: [{"name": r["name"], "hex": r.grounded_hex, "ai_hex": r.ai_hex, "spinner_hex": r.spinner_hex,
                    "dE00_ai_vs_spinner": None if pd.isna(r.de00) else round(float(r.de00), 2),
                    "replaced": bool(r.replaced)}
                   for _, r in export[export.language == lang].iterrows()] for lang in languages}
    with open(f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    slug = lambda s: re.sub(r"[^\w]+", "-", norm(s)).strip("-") or "colour"
    css = ":root {\n" + "".join(
        f"  --{slug(r.language)}-{slug(r['name'])}: {r.grounded_hex};"
        f"{'  /* replaced AI ' + r.ai_hex + ' */' if r.replaced and isinstance(r.ai_hex, str) else ''}\n"
        for _, r in export.dropna(subset=["grounded_hex"]).iterrows()) + "}\n"
    with open(f"{stem}.css", "w", encoding="utf-8") as f:
        f.write(css)
    with zipfile.ZipFile(f"{stem}.zip", "w") as z:
        for ext in ("csv", "json", "css"):
            z.write(f"{stem}.{ext}")
    print(css)
    download(f"{stem}.zip")

def show_finding(pal, name=""):
    cand = pal.dropna(subset=["de00"])
    if cand.empty: return
    match = cand[cand.name.map(norm) == norm(name)] if name else cand.iloc[:0]
    row = match.iloc[0] if len(match) else cand.loc[cand.de00.idxmax()]
    show(f"""<div class="card">
<h3>“{esc(row['name'])}” in {esc(row.language)}</h3>
<div style="display:flex;gap:10px;margin:12px 0;background:{SURROUND};padding:12px;border-radius:6px">
  <div class="tile"><div class="chip" style="background:{row.ai_hex};color:{ink_on(row.ai_hex)};height:80px">AI</div>
       <div class="cap">AI system<div class="hex">{row.ai_hex}</div></div></div>
  <div class="tile"><div class="chip" style="background:{row.spinner_hex};color:{ink_on(row.spinner_hex)};height:80px">Speakers</div>
       <div class="cap">Best example<div class="hex">{row.spinner_hex}</div></div></div>
  <div style="color:#fff;align-self:center;padding-left:8px;font-size:15px">ΔE00<br><b style="font-size:30px">{row.de00:.1f}</b></div>
</div>
<p style="margin:0 0 8px">Talk it through with your partner:</p>
<p style="margin:0 0 4px"><b>What pattern of failure does this reveal?</b> Did the AI default to a Western-centric focal colour,
collapse a distinction the language maintains, or produce a colour that fails under CVD or ageing?</p>
<p style="margin:0"><b>What would a designer need to do differently</b> to avoid it in their own practice?</p>
</div>""")

def safe(fn, *args, **kwargs):
    """fn(*args), or None if it fails (e.g. a name with no best example)."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None
