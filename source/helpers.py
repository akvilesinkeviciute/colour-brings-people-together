

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
    ax.set_xlabel("a′  (green ← → red)", color=MUTED); ax.set_ylabel("b′  (blue ← → yellow)", color=MUTED)

def _titles(ax, title, subtitle=""):
    """Bold title with a muted subtitle underneath, left-aligned and wrapped to the plot's width."""
    import textwrap
    fig = ax.figure
    width_in = ax.get_position().width * fig.get_figwidth()
    lines = textwrap.wrap(subtitle, width=max(40, int(width_in * 14))) if subtitle else []
    ax.set_title(title, loc="left", pad=10 + 13 * len(lines))
    if lines:
        ax.annotate("\n".join(lines), xy=(0, 1), xycoords="axes fraction", xytext=(0, 6),
                    textcoords="offset points", ha="left", va="bottom", fontsize=9.5, color=MUTED,
                    linespacing=1.3)

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
    show_palette(sub.hex.tolist(), labels=[str(i) for i in sub.index], captions=sub.goal.tolist(),
                 note=f"Auditing {len(pick)} colours in {len(languages)} languages: {', '.join(languages)}.")


# ── Part B: perceptual axis ───────────────────────────────────────────────
@needs_simulations(default=(None, None))
def nearest_neighbours(pick, conditions=CONDITIONS, among=None):
    """For each colour in `pick` and each condition: the nearest other colour among `among`
    (by default the other test colours) and its ΔE00. Returns distances and neighbour ids."""
    among = list(pick if among is None else among)
    def nearest(i, c):
        me = simulate(SDG.hex[i], c)
        return min(((de00(me, simulate(SDG.hex[j], c)), j) for j in among if j != i), default=(np.nan, i))
    nn = pd.DataFrame({c: {i: nearest(i, c) for i in pick} for c in conditions})
    return nn.map(lambda t: t[0]), nn.map(lambda t: t[1])

def _risk_colour(v, threshold):
    """Below the threshold: from deep red (0) to light red (at the threshold). Above: light grey."""
    if np.isnan(v):
        return "#FFFFFF"
    if v >= threshold:
        return "#F2F2F0"
    t = max(v, 0) / threshold
    lo, hi = np.array(mpl.colors.to_rgb("#8E1B1B")), np.array(mpl.colors.to_rgb("#EDB9AC"))
    return mpl.colors.to_hex(lo + (hi - lo) * t)

def _cell_grid(values, threshold, row_labels, title, subtitle, row_chip_colours=None, neighbours=None):
    """A grid of rounded cells: one row per test colour (or palette), one column per viewing condition."""
    nr, nc = values.shape
    cols = list(values.columns)
    W, H = 1.3 * nc + (1.6 if row_chip_colours else 2.4), 0.8 * nr + 1.6
    fig, ax = plt.subplots(figsize=(W, H))
    for r in range(nr):
        for c in range(nc):
            v = values.iat[r, c]
            ax.add_patch(mpl.patches.FancyBboxPatch((c - 0.45, r - 0.40), 0.90, 0.80,
                         boxstyle="round,pad=0,rounding_size=0.07", fc=_risk_colour(v, threshold), ec="none"))
            if np.isnan(v):
                continue
            ink = "white" if v < 0.6 * threshold else INK
            ax.text(c, r - (0.09 if neighbours is not None else 0), f"{v:.1f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=ink)
            if neighbours is not None:
                n = neighbours.iat[r, c]
                ax.add_patch(mpl.patches.Rectangle((c - 0.30, r + 0.13), 0.13, 0.15, fc=SDG.hex[n], ec="white", lw=0.8))
                ax.text(c - 0.12, r + 0.205, f"vs {n}", ha="left", va="center", fontsize=8.5, color=ink)
    for c, name in enumerate(cols):
        ax.text(c, -0.68, condition_label(name), ha="center", va="center", fontsize=10, color=INK)
    ax.set_xlim(-0.5, nc - 0.5); ax.set_ylim(nr - 0.5, -0.95)
    ax.set_xticks([])
    if row_chip_colours:
        ax.set_yticks([])
    else:
        ax.set_yticks(range(nr), row_labels, fontsize=10); ax.tick_params(axis="y", length=0, colors=INK)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(title, pad=24)
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points",
                ha="left", va="bottom", fontsize=9.5, color=MUTED)
    fig.tight_layout()
    if row_chip_colours:
        fig.subplots_adjust(left=max(fig.subplotpars.left, 1.3 / W))
        for r, (h, l) in enumerate(zip(row_chip_colours, row_labels)):
            chip(ax, r, h, l, height_in=0.62)
    plt.show()

def plot_nearest_neighbours(nn_de, nn_id, threshold):
    if nn_de is None: return
    pick = list(nn_de.index)
    below = int((nn_de.values < threshold).sum())
    _cell_grid(nn_de, threshold, [str(i) for i in pick],
               "How close is each test colour to its nearest neighbour in your palette?",
               f"ΔE00 to the nearest other test colour under each condition. Red: closer than {threshold}, "
               f"so easily confused ({below} of {nn_de.size} cells).",
               row_chip_colours=[SDG.hex[i] for i in pick], neighbours=nn_id)

CONDITION_WHY = {
    "Typical": "They are close even with typical colour vision.",
    "Protanopia": "Without long-wavelength (red) cones, the red–green difference disappears and reds look darker. "
                  "What's left is mainly lightness and the blue–yellow difference.",
    "Deuteranopia": "Without medium-wavelength (green) cones, the red–green difference disappears. "
                    "What's left is mainly lightness and the blue–yellow difference.",
    "Tritanopia": "Without short-wavelength (blue) cones, the blue–yellow difference disappears, "
                  "so blues and greens, or pinks and reds, can merge.",
    "Age 40": "The eye's lens yellows and absorbs blue light: blues darken and drift towards green, pale colours lose contrast.",
    "Age 60": "The eye's lens yellows and absorbs blue light: blues darken and drift towards green, pale colours lose contrast.",
    "Age 80": "The eye's lens yellows strongly and absorbs much of the blue light: blues darken and drift towards green.",
}

def palette_strip(pick, note="Your test palette"):
    """A one-line reminder of the colours this section is about."""
    chips = "".join(f'<div style="display:inline-block;margin-right:8px">{framed(SDG.hex[i], str(i), 56, 34, 12)}</div>'
                    for i in pick)
    show(f'<div style="display:flex;align-items:center;gap:12px">{small(note)}<div>{chips}</div></div>')

@needs_simulations()
def show_at_risk(nn_de, nn_id, threshold):
    if nn_de is None: return
    pairs = {}
    for i in nn_de.index:
        for c in nn_de.columns:
            if nn_de.loc[i, c] < threshold:
                pairs.setdefault(tuple(sorted((i, nn_id.loc[i, c]))), []).append((c, nn_de.loc[i, c]))
    if not pairs:
        show(f"<b>No pairs at risk.</b> {small(f'Your test colours stay at least ΔE00 {threshold} apart under every condition.')}")
        return
    rows = []
    for (i, j), hits in sorted(pairs.items(), key=lambda kv: min(d for _, d in kv[1])):
        c, d = min(hits, key=lambda t: t[1])
        typical = de00(SDG.hex[i], SDG.hex[j])
        two = lambda cond: (f'<div style="display:flex;gap:8px">{framed(simulate(SDG.hex[i], cond), str(i), 60, 38, 12)}'
                            f'{framed(simulate(SDG.hex[j], cond), str(j), 60, 38, 12)}</div>')
        also = [x for x in dict.fromkeys(x for x, _ in hits) if x != c]
        why = CONDITION_WHY.get(c, "") + (f" Also below the threshold: {', '.join(also)}." if also else "")
        rows.append([two("Typical"), two(c),
                     f'<span style="white-space:nowrap"><b>{typical:.1f} → {d:.1f}</b></span><br>{small("typical → " + _h(c))}',
                     f'<span style="font-size:12.5px">{_h(why)}</span>'])
    show("<b>Pairs at risk, and why</b>")
    table(["Typical vision", "Where they're closest", "ΔE00", "Why they look alike"], rows,
          "Each pair is shown with typical vision and under the condition where the two are closest.")

def connect(backend, sheet=None):
    """The Colour Spinner itself, or the values you copied from it into the boxes."""
    if backend == "Colour Spinner API":
        return ApiSpinner()
    return SheetSpinner(sheet if sheet is not None else boxes_as_sheet())

# ── Boxes for values copied from the Colour Spinner in the browser ────────
_SHEET = {"rows": [], "names": []}

def _w():
    import ipywidgets as w
    return w

def _swatch_html(hex_, text):
    return (f'<span style="display:inline-block;width:34px;height:22px;border-radius:3px;vertical-align:middle;'
            f'background:{hex_}"></span>&nbsp; {text}')

def sheet_boxes(pick, languages):
    """One row of boxes per test colour and language, for what the Spinner shows in the browser."""
    w = _w()
    lay = lambda px: w.Layout(width=f"{px}px")
    head = w.HBox([w.HTML("<b>Test colour · language</b>", layout=lay(300)), w.HTML("<b>Namer: name</b>", layout=lay(190)),
                   w.HTML("<b>Confidence %</b>", layout=lay(110)), w.HTML("<b>Mapper: best example RGB</b>", layout=lay(220))])
    items, rows = [head], []
    for i in pick:
        for lang in languages:
            name = w.Text(placeholder="e.g. Red", layout=lay(180))
            conf = w.Text(placeholder="e.g. 36", layout=lay(100))
            best = w.Text(placeholder="e.g. 232 21 56", layout=lay(200))
            rows.append((SDG.hex[i], lang, name, conf, best))
            items.append(w.HBox([w.HTML(_swatch_html(SDG.hex[i], f"<b>{i}</b> {SDG.hex[i]} · {_h(lang)}"), layout=lay(300)),
                                 name, conf, best]))
    _SHEET["rows"] = rows
    display(w.VBox(items))

def name_boxes(n=8):
    """Boxes to type colour names in, with an optional best example for each."""
    w = _w()
    lay = lambda px: w.Layout(width=f"{px}px")
    head = w.HBox([w.HTML("<b>Language</b>", layout=lay(190)), w.HTML("<b>Colour name</b>", layout=lay(200)),
                   w.HTML("<b>Best example RGB</b> (only for <i>Copied from the browser</i>)", layout=lay(330))])
    items, rows = [head], []
    for _ in range(n):
        lang = w.Dropdown(options=SPINNER_LANGUAGES, layout=lay(180))
        name = w.Text(placeholder="e.g. Red", layout=lay(190))
        best = w.Text(placeholder="e.g. 232 21 56", layout=lay(200))
        rows.append((lang, name, best))
        items.append(w.HBox([lang, name, best]))
    _SHEET["names"] = rows
    display(w.VBox(items))

def boxes_as_sheet():
    """Everything typed into the boxes, in the table form SheetSpinner reads."""
    lines = ["hex, language, name, confidence, best_example"]
    for hex_, lang, name, conf, best in _SHEET["rows"]:
        if name.value.strip():
            lines.append(f"{hex_.lstrip('#')}, {lang}, {name.value.strip()}, "     # no '#': that marks a comment
                         f"{conf.value.strip().rstrip('%') or 0}, {best.value.strip()}")
    for lang, name, best in _SHEET["names"]:
        if name.value.strip() and best.value.strip():
            lines.append(f", {lang.value}, {name.value.strip()}, 0, {best.value.strip()}")
    return "\n".join(lines) if len(lines) > 1 else ""

def typed_names():
    t = [(lang.value, name.value.strip()) for lang, name, _ in _SHEET["names"] if name.value.strip()]
    if not t:
        raise ValueError("Type at least one colour name into the boxes above, then run this cell again.")
    return pd.DataFrame(t, columns=["language", "name"])

def report_problems(problems):
    if problems:
        show("<b>Some lookups did not work</b><br>" + "<br>".join(map(esc, problems[:8])) +
             ("<br>…" if len(problems) > 8 else "") +
             '<div class="note">Using the Colour Spinner: check your internet connection, or set Backend to '
             '<i>Copied from the browser</i>. Using the sheet: add the missing rows and run the cell again.</div>')

def show_names_table(naming, languages, you=None):
    if naming.empty: return
    wide = naming.pivot(index="sdg", columns="language", values="name")
    conf = naming.pivot(index="sdg", columns="language", values="confidence").add_suffix("|conf")
    table = SDG.loc[wide.index, ["hex"]].join(wide).join(conf)
    if "romanized" in naming:
        table = table.join(naming.pivot(index="sdg", columns="language", values="romanized").add_suffix("|rom"))
    labels = {you: f"<b style='color:{INK}'>{_h(you)}</b> (you)"} if you else None
    show_babel(table, [l for l in languages if l in wide.columns], labels=labels)

def plot_nameability(naming, pick, languages):
    rows = [(i, l) for i in pick for l in languages
            if not naming[(naming.sdg == i) & (naming.language == l)].empty]
    if not rows: return
    gap = {i: k for k, i in enumerate(pick)}
    ys = [n + 0.6 * gap[i] for n, (i, l) in enumerate(rows)]
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(rows) + 0.5 * len(pick) + 1))
    for y, (i, l) in zip(ys, rows):
        r = naming[(naming.sdg == i) & (naming.language == l)].iloc[0]
        ax.hlines(y, 0, r.confidence, color="#DADAD5", lw=6, capstyle="round", zorder=1)
        ax.scatter(r.confidence, y, s=120, color=SDG.hex[i], edgecolor=INK, lw=0.7, zorder=3)
        ax.text(r.confidence + 0.025, y, f"{r['name']}  {r.confidence:.0%}", va="center", fontsize=10)
    for i in pick:
        g = [y for y, (j, _) in zip(ys, rows) if j == i]
        if g:
            chip(ax, (min(g) + max(g)) / 2, SDG.hex[i], str(i), height_in=0.62, gap_in=1.45)
    ax.set_yticks(ys, [l for _, l in rows]); ax.tick_params(axis="y", length=0)
    ax.invert_yaxis(); ax.set_xlim(0, 1.18); ax.spines["left"].set_visible(False)
    ax.set_xticks([0, .25, .5, .75, 1]); ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    ax.set_xlabel("Spinner confidence in its top name")
    plt.tight_layout(); fig.subplots_adjust(left=0.33, top=0.90)
    _titles(ax, "Nameability: how confident is the Spinner's top name?",
            "One row per language. Each bar runs to the confidence of the Spinner's top name.")
    plt.show()

def show_translatability(naming, pick, languages, threshold, source=None):
    """Source language first: its focal colour vs the test colour. Then each target language:
    its focal colour vs the source's focal colour."""
    if source is not None and source not in languages:
        show(f"<b>{esc(source)}</b> isn't one of the languages you ticked in section 2. "
             "Tick it there, or choose another source language.")
        return
    order = ([source] + [l for l in languages if l != source]) if source else list(languages)
    W = 160                                         # every language column has the same width
    cell = f'style="width:{W}px;vertical-align:top;padding:0 0 18px 0"'
    rows = ""
    for i in pick:
        sub = naming[naming.sdg == i]
        best = {l: sub[sub.language == l].best_example.iloc[0] for l in order
                if not sub[sub.language == l].empty and isinstance(sub[sub.language == l].best_example.iloc[0], str)}
        cells = ""
        for k, lang in enumerate(order):
            r = sub[sub.language == lang]
            if lang not in best:
                cells += f'<td {cell}><span style="color:{MUTED}">—</span></td>'
                continue
            b = best[lang]
            ref = SDG.hex[i] if (k == 0 or not source or source not in best) else best[source]
            cells += (f'<td {cell}><div class="pair" style="width:{W - 22}px"><div style="background:{b};color:{ink_on(b)}">'
                      f'{esc(r.name.iloc[0])}<br>{b}</div></div><div style="margin-top:4px">'
                      f'{de_badge(de00(ref, b), threshold)}</div></td>')
        sp = spread(list(best.values()))
        rows += (f'<tr><td style="width:120px;vertical-align:top;padding:0 0 18px 0">'
                 f'<div class="pair" style="width:96px"><div style="background:{SDG.hex[i]};'
                 f'color:{ink_on(SDG.hex[i])};font-weight:700;font-size:15px">{i}</div></div></td>'
                 f'<td style="width:28px"></td>{cells}'
                 f'<td style="vertical-align:top;padding-top:20px;color:{MUTED};white-space:nowrap">'
                 f'{"" if pd.isna(sp) else "spread " + format(sp, ".1f")}</td></tr>')
    heads = "".join(f'<th style="width:{W}px;text-align:left;padding:0 0 8px 0;font-weight:400;font-size:12px;white-space:nowrap;color:{MUTED}">'
                    f'{esc(l)}{(" (you)" if k == 0 else " · audience") if source else ""}</th>'
                    for k, l in enumerate(order))
    note = ("Your column: ΔE00 between the test colour and your name's best example (focal colour). "
            "Audience columns: ΔE00 between each audience name's focal colour and yours. " if source else
            "Badges: ΔE00 between the test colour and each language's best example (focal colour). ")
    show('<table style="table-layout:fixed;border-collapse:collapse">'
         f'<tr><th style="width:120px;text-align:left;padding:0 0 8px 0;font-weight:400;font-size:12px;color:{MUTED}">Test colour</th>'
         f'<th style="width:28px"></th>{heads}<th></th></tr>{rows}</table>'
         f'<div class="note">{note}Spread: the largest ΔE00 between any two languages\' focal colours.</div>')

def show_translations(spinner, naming, pick, you):
    """The Spinner's Translator: your language's name for each test colour, translated into every
    language it covers, with the translation confidence."""
    if not hasattr(spinner, "lookup"):
        show("The translations come from the Colour Spinner directly, so they need Backend set to <i>Colour Spinner API</i>.")
        return
    others = [l for l in SPINNER_LANGUAGES if l != you]
    audience = set(naming.language) - {you}
    rows = []
    for i in pick:
        mine = naming[(naming.sdg == i) & (naming.language == you)]
        if mine.empty: continue
        cells = []
        for l in others:
            res = safe(spinner.lookup, SDG.hex[i], you, l)
            if not res or not res.get("translation"):
                cells.append(small("—")); continue
            conf = res.get("translation_confidence")
            rom = res.get("translation_romanized")
            extra = f" {small(_h(rom))}" if rom and norm(rom) != norm(res["translation"]) else ""
            cells.append(f'<b>{_h(res["translation"])}</b>{extra}' +
                         (f'{bar(conf)}{small(f"{conf:.0%} translation confidence")}' if conf is not None else ""))
        rows.append([framed(SDG.hex[i], str(i)), f'<b>{_h(mine.name.iloc[0])}</b>'] + cells)
    head = ["Test colour", f"{_h(you)} (you)"] + [f"{_h(l)}" + (" · audience" if l in audience else "") for l in others]
    table(head, rows, valign="top", note=
          "From the Spinner's Translator: the name in each language whose naming distribution best matches yours. "
          "Translation confidence is how closely the two names are used for the same colours.")

def plot_name_map(naming, pick, languages):
    fig, ax = plt.subplots(figsize=(8, 6.2)); _ucs_axes(ax)
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
    ax.set_aspect("equal", adjustable="datalim")
    _titles(ax, "Where each language's name pulls the test colour",
            "Top-down view of CAM16-UCS colour space. Large disc: the test colour. Lines lead to each "
            "language's best example (focal colour) for its name.")
    plt.tight_layout(); plt.show()


# ── Part B: two axes ──────────────────────────────────────────────────────
ALL_LANGUAGES = "Average of my languages"
AUDIENCE_AVG = "Average of my audience"
WORST_CASE = "Worst case of all conditions"
SUBSET_MAX = "Largest value in my subset"
SDG_RANGE = "Range across all 17 SDG colours"

def _distances(ids, condition):
    conds = CONDITIONS if condition == WORST_CASE else [condition]
    de, _ = nearest_neighbours(ids, conds)
    if de is None:
        return None
    return de.min(axis=1) if condition == WORST_CASE else de[condition]

def scorecard(pick, naming, language=ALL_LANGUAGES, condition="Typical", normalise=SUBSET_MAX, audience=None):
    """One row per test colour: its perceptual distance (ΔE00 to the nearest other SDG colour, under
    `condition`) and naming confidence (in `language`, or averaged), both on a 0–1 scale."""
    de = _distances(pick, condition)
    if de is None:
        return None
    if normalise == SDG_RANGE:
        all_de = _distances(list(SDG.index), condition)          # the whole palette, as a reference scale
        lo, hi = all_de.min(), all_de.max()
    else:
        lo, hi = 0.0, de.max()
    conf = naming.pivot(index="sdg", columns="language", values="confidence")
    names = naming.pivot(index="sdg", columns="language", values="name")
    if language not in (ALL_LANGUAGES, AUDIENCE_AVG) and language not in conf.columns:
        show(f'<div class="card"><b>{esc(language)} isn\'t one of the languages you audited</b>'
             f'<p style="margin:6px 0 0">Showing the average of your languages instead. To see {esc(language)}, '
             f'tick it in section 2 and run sections 2 and 4 again. Your languages: '
             f'{esc(", ".join(conf.columns))}.</p></div>')
        language = ALL_LANGUAGES
    if language == AUDIENCE_AVG:
        aud = [l for l in (audience or []) if l in conf.columns] or list(conf.columns)
        y = conf[aud].mean(axis=1)
    else:
        y = conf.mean(axis=1) if language == ALL_LANGUAGES else conf[language]
    card = pd.DataFrame({"dE00": de, "perceptual_0to1": (de - lo) / (hi - lo) if hi > lo else 1.0,
                         "confidence_0to1": y.reindex(pick)}, index=pick)
    for l in conf.columns:
        card[l] = [f"{names.at[i, l]} ({conf.at[i, l]:.0%})" if i in conf.index and not pd.isna(conf.at[i, l]) else ""
                   for i in pick]
    card.attrs.update(lo=lo, hi=hi, language=language, condition=condition, normalise=normalise)
    card.index.name = "SDG"
    return card

def plot_two_axes(card, confusable_below, hard_below):
    if card is None:
        return
    lo, hi = card.attrs["lo"], card.attrs["hi"]
    x_cut = (confusable_below - lo) / (hi - lo) if hi > lo else 0.5
    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    ax.axvline(x_cut, color="#BDBDB8", ls="--", lw=1); ax.axhline(hard_below, color="#BDBDB8", ls="--", lw=1)
    if 0 <= x_cut <= 1:
        ax.scatter(x_cut, hard_below, s=900, facecolor="none", edgecolor="#BDBDB8", lw=2, zorder=2)
    for x, y, t in [(0.02, 0.98, "named, but easily confused"), (0.98, 0.98, "robust on both axes"),
                    (0.02, 0.02, "fragile on both axes"), (0.98, 0.02, "distinct, but hard to name")]:
        ax.text(x, y, t, transform=ax.transAxes, ha="left" if x < .5 else "right",
                va="top" if y > .5 else "bottom", color=MUTED, fontsize=10, style="italic")
    for i, r in card.iterrows():
        if pd.isna(r.confidence_0to1): continue
        ax.scatter(r.perceptual_0to1, r.confidence_0to1, s=900, color=SDG.hex[i], edgecolor="white", lw=2.5, zorder=3)
        ax.text(r.perceptual_0to1, r.confidence_0to1, str(i), ha="center", va="center",
                color=ink_on(SDG.hex[i]), fontweight="bold", zorder=4)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    cond = card.attrs["condition"]
    vision = "typical colour vision" if cond == "Typical" else ("worst case of all conditions" if cond == WORST_CASE else cond)
    scale = "÷ largest in subset" if card.attrs["normalise"] == SUBSET_MAX else "scaled to all 17 SDG colours"
    ax.set_xlabel(f"Perceptual distance (ΔE00 to nearest test colour, {vision}; {scale})")
    lab = card.attrs["language"]
    ax.set_ylabel("Naming confidence (" + ("average of your audience" if lab == AUDIENCE_AVG else
                  "average of all your languages" if lab == ALL_LANGUAGES else lab) + ")")
    _titles(ax, "Naming confidence × perceptual distance",
            "Each disc is a test colour. The dashed lines are your two thresholds; the grey circle marks where they meet.")
    if not 0 <= x_cut <= 1:
        ax.text(0.5, -0.16, f"The ΔE00 {confusable_below} line falls outside this scale.", transform=ax.transAxes,
                ha="center", color=MUTED, fontsize=9)
    plt.tight_layout(); plt.show()

def show_scorecard(card):
    if card is None:
        return
    langs = [c for c in card.columns if c not in ("dE00", "perceptual_0to1", "confidence_0to1")]
    rows = []
    for i, r in card.iterrows():
        perc = f'<b>{r.perceptual_0to1:.2f}</b>{bar(r.perceptual_0to1)}{small(f"ΔE00 {r.dE00:.1f}")}'
        conf = ("—" if pd.isna(r.confidence_0to1) else
                f'<b>{r.confidence_0to1:.2f}</b>{bar(r.confidence_0to1)}{small("naming confidence")}')
        names = []
        for l in langs:
            m = re.match(r"(.*) \((\d+)%\)$", r[l] or "")
            names.append(f'<b>{_h(m.group(1))}</b>{bar(int(m.group(2)) / 100)}{small(m.group(2) + "% confidence")}'
                         if m else small("—"))
        rows.append([framed(SDG.hex[i], str(i)), perc, conf] + names)
    table(["Test colour", "Perceptual (0–1)", "Confidence (0–1)"] + [_h(l) for l in langs], rows, valign="top", note=
          "Perceptual: distance to the nearest SDG colour, scaled 0–1. Confidence: the Spinner's confidence in its top name.")

# ── Part B: the AI's names ────────────────────────────────────────────────
# ── Asking the AI, and reading its reply ─────────────────────────────────
PASTE = "Paste answers from my own AI"
_ANS = {"boxes": [], "rows": None, "kind": None}

def numbered_items(rows, kind):
    """The numbered list shared by the prompt and the boxes.
    kind='name': one number per test colour ('1. #E5243B'); kind='colour': one per name."""
    rows = rows.reset_index(drop=True)
    if kind == "name":
        colours = rows.drop_duplicates("hex")
        if "sdg" not in colours:
            return [f"{n}. {r.hex}" for n, (_, r) in enumerate(colours.iterrows(), 1)]
        return [f"{int(r.sdg)}. {r.hex}" for _, r in colours.iterrows()]   # numbered by SDG goal
    return [f"{n}. {r.language}: {plain_with_roman(r)}" for n, (_, r) in enumerate(rows.iterrows(), 1)]

def answer_boxes(rows, kind):
    """kind='name': one numbered row per test colour, with its swatch and hex code filled in and one box
    per language. kind='colour': one numbered box per name."""
    try:
        import ipywidgets as w
    except ImportError:
        print("Answer boxes need ipywidgets, which Colab includes.")
        return
    rows = rows.reset_index(drop=True)
    lay = lambda px, **k: w.Layout(width=f"{px}px", **k)
    items, boxes = [], {}
    if kind == "name":
        colours = rows.drop_duplicates("hex")
        langs = list(dict.fromkeys(rows.language))
        items.append(w.HBox([w.HTML("", layout=lay(34)), w.HTML("<b>Test colour</b>", layout=lay(190))] +
                            [w.HTML(f"<b>{_h(l)}</b>", layout=lay(170)) for l in langs]))
        for k, (_, r) in enumerate(colours.iterrows(), 1):
            n = int(r.sdg) if "sdg" in r and not pd.isna(r.sdg) else k
            swatch = (f'<span style="display:inline-block;width:34px;height:22px;border-radius:3px;vertical-align:middle;'
                      f'background:{r.hex};box-shadow:0 0 0 3px {SURROUND}"></span>&nbsp;&nbsp; {r.hex}')
            row = [w.HTML(f'<b style="font-size:15px">{n}</b>', layout=lay(34)), w.HTML(swatch, layout=lay(190))]
            for l in langs:
                box = w.Text(placeholder=f"name in {l}", layout=lay(160))
                boxes[(r.hex, l)] = box
                row.append(box)
            items.append(w.HBox(row))
    else:
        for n, (_, r) in enumerate(rows.iterrows(), 1):
            box = w.Text(placeholder="paste the colour code, e.g. #E5243B", layout=lay(300))
            boxes[n] = box
            items.append(w.HBox([w.HTML(f'<b style="font-size:15px">{n}</b>', layout=lay(34)), box,
                                 w.HTML(f'<span style="color:{MUTED}">{_h(r.language)}: {_h(r["name"])}</span>',
                                        layout=lay(260, margin="0 0 0 10px"))]))
    _ANS.update(boxes=boxes, rows=rows, kind=kind)
    display(w.VBox(items))

def _answers_table(values, rows, kind):
    """values: {(hex, language): name} for kind 'name'; [colour code, …] in order for kind 'colour'."""
    t = rows.reset_index(drop=True).copy()
    if kind == "name":
        t["ai_name"] = [values.get((h, l)) or None for h, l in zip(t.hex, t.language)]
        return t[["hex", "language", "ai_name"]]
    t["ai_hex"] = [parse_colour(v) if v and (_HEX.search(v) or len(re.findall(r"\d+", v)) == 3) else None
                   for v in values]
    return t[["language", "name", "ai_hex"]]

def collected_answers():
    """What was typed into the boxes, as a table."""
    rows, kind, boxes = _ANS["rows"], _ANS["kind"], _ANS["boxes"]
    if rows is None:
        raise ValueError("Run the cell above first: it shows the boxes to fill in.")
    if kind == "name":
        values = {k: b.value.strip() for k, b in boxes.items()}
        filled = any(values.values())
    else:
        values = [boxes[n].value.strip() for n in sorted(boxes)]
        filled = any(values)
    if not filled:
        raise ValueError("The boxes are empty: paste your AI's answers into them, then run this cell again.")
    return _answers_table(values, rows, kind)

def answers_from_reply(reply, rows, kind):
    """Read a numbered reply into the same table as the boxes.
    kind='name': '1. British English: Red | Greek: Κόκκινο | …'; kind='colour': '1. #E5243B'."""
    found = {}
    for line in re.sub(r"```\w*", "", reply).splitlines():
        m = re.match(r"\s*\**\s*(\d+)\s*[.):\-]\s*(.+?)\s*$", line)
        if m:
            found.setdefault(int(m.group(1)), m.group(2).strip())
    if not found:
        raise ValueError("Couldn't find numbered answers ('1. …') in the AI's reply.")
    rows = rows.reset_index(drop=True)
    clean = lambda v: re.sub(r"[*_`]", "", v).strip().strip('"').strip()
    if kind == "colour":
        return _answers_table([clean(found.get(n, "")) for n in range(1, len(rows) + 1)], rows, kind)
    colours = rows.drop_duplicates("hex")
    langs = list(dict.fromkeys(rows.language))
    values = {}
    numbers = colours.sdg.astype(int) if "sdg" in colours else range(1, len(colours) + 1)
    for n, hex_ in zip(numbers, colours.hex):
        parts = [p for p in re.split(r"\s*[|;]\s*", found.get(n, "")) if p.strip()]
        for k, part in enumerate(parts):
            if ":" in part:
                lang, name = part.split(":", 1)
                lang = next((l for l in langs if norm(l) == norm(canonical_language(clean(lang)))), None)
            else:
                lang, name = (langs[k] if k < len(langs) else None), part
            if lang:
                values[(hex_, lang)] = clean(name)
    return _answers_table(values, rows, kind)

# Where each button goes. Sites that accept a prompt in the link get it filled in; for the others
# the click copies the prompt, so participants paste it (Cmd/Ctrl + V).
AI_SITES = [
    ("ChatGPT",    "https://chatgpt.com/?q={q}"),
    ("Claude",     "https://claude.ai/new?q={q}"),
    ("Gemini",     "https://gemini.google.com/app?q={q}"),
]

def prompt_box(prompt):
    """The prompt in a read-only box, a Copy button, and one button per AI tool. Every button copies
    the prompt as it opens the tool, so it can be pasted if the tool doesn't fill it in."""
    from urllib.parse import quote
    import uuid
    uid = "p" + uuid.uuid4().hex[:8]
    q = quote(prompt)
    btn = ("display:inline-block;padding:7px 13px;margin:8px 6px 0 0;border-radius:6px;font-weight:700;"
           "font-size:13px;text-decoration:none;cursor:pointer;border:1px solid #1F1F1F;")
    links = "".join(f'<a class="{uid}a" href="{url.format(q=q)}" target="_blank" rel="noopener" '
                    f'style="{btn}background:#FFFFFF;color:{INK}">{name} ↗</a>' for name, url in AI_SITES)
    show(f"""
<textarea id="{uid}" readonly rows="{min(22, prompt.count(chr(10)) + 2)}"
  style="width:100%;max-width:820px;font-family:ui-monospace,Menlo,monospace;font-size:12.5px;line-height:1.45;
         padding:10px;border:1px solid #CFCFCA;border-radius:6px;background:#FFFFFF;color:{INK}">{_h(prompt)}</textarea>
<div><button id="{uid}b" style="{btn}background:{INK};color:#FFFFFF">Copy prompt</button>{links}</div>
<div id="{uid}m" style="color:{MUTED};font-size:12.5px;margin-top:6px">
  Open your AI with one of the buttons. The prompt is copied as you click: if it isn't already in the
  AI's message box, paste it (Cmd/Ctrl + V). Another AI? Copy the prompt and paste it there.</div>
<script>
(function() {{
  const ta = document.getElementById("{uid}"), msg = document.getElementById("{uid}m");
  async function copy() {{
    let ok = false;
    try {{ await navigator.clipboard.writeText(ta.value); ok = true; }} catch (e) {{}}
    if (!ok) {{ ta.focus(); ta.select(); try {{ ok = document.execCommand("copy"); }} catch (e) {{}} }}
    return ok;
  }}
  document.getElementById("{uid}b").onclick = async () => {{
    msg.textContent = (await copy()) ? "Copied ✓  Now paste it into your AI tool."
                                     : "Select the text above (Cmd/Ctrl + A inside the box), then copy it.";
  }};
  document.querySelectorAll("a.{uid}a").forEach(a => a.addEventListener("click", async () => {{
    if (await copy()) msg.textContent = "Prompt copied ✓  If it isn't already in the AI's message box, paste it (Cmd/Ctrl + V).";
  }}));
}})();
</script>""")

def show_prompt(prompt, provider, rows=None, kind="name"):
    """Show the prompt with copy and open-in buttons; for your own AI, also the answer boxes."""
    if provider == PASTE:
        prompt_box(prompt)
        step2 = ("For each test colour, paste the names your AI gives for that number into its row of boxes."
                 if kind == "name" else
                 "Paste the colour code your AI gives for number 1 into box 1, number 2 into box 2, and so on.")
        show(f"<b>1.</b> Open your AI with a button above (or copy the prompt) and send it.<br>"
             f"<b>2.</b> {step2} Then run the next cell.")
        answer_boxes(rows, kind)
    else:
        print(prompt)
        show(f'<div class="note">The next cell sends this prompt to {esc(provider)}.</div>')

_HEX = re.compile(r"#?[0-9A-Fa-f]{6}")

def parse_ai_table(text, fields):
    """Read an AI reply given as JSON, a table, or lines like 'a | b | c'."""
    try:
        obj = parse_json_reply(text)
        if isinstance(obj, dict): obj = next((v for v in obj.values() if isinstance(v, list)), [obj])
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return pd.DataFrame(obj)
    except ValueError:
        pass
    rows = []
    for line in text.splitlines():
        line = line.strip().strip("`").strip()
        if not line or set(line) <= set("|-:+= "):
            continue
        line = re.sub(r"^(\d+[.)]|[-*•])\s+", "", line)            # list numbering or bullets
        parts = [p.strip().strip("*") for p in re.split(r"\s*[|\t]\s*", line.strip("|"))]
        if len(parts) >= len(fields):
            rows.append(dict(zip(fields, parts[:len(fields)])))
    return pd.DataFrame(rows, columns=fields)

def _clean_name(n):
    """'κόκκινο (kokkino)' -> 'κόκκινο'"""
    return re.sub(r"\s*\(.*?\)\s*$", "", str(n)).strip().strip('"')

def read_ai_names(reply):
    t = parse_ai_table(reply, ["hex", "language", "name"])
    t = t[t["hex"].astype(str).str.fullmatch(r"\s*#?[0-9A-Fa-f]{6}\s*")].copy()  # skips header rows
    if t.empty:
        raise ValueError("Couldn't find any 'colour | language | name' lines in the reply.")
    t["hex"] = t["hex"].map(parse_colour)
    t["language"] = t["language"].map(canonical_language)
    t["name"] = t["name"].astype(str).str.strip().str.strip('"')
    return t.rename(columns={"name": "ai_name"})[["hex", "language", "ai_name"]]

def where_ai_names_point(cmp, spinner):
    """For each AI name that differs from the speakers' top name, ask the Spinner's Mapper
    where that name is centred in the language, and how far that is from the test colour."""
    cmp = cmp.copy()
    cmp["ai_best"], cmp["ai_drift"], cmp["ai_status"] = None, np.nan, ""
    for k, r in cmp.iterrows():
        if r.agrees or not isinstance(r.ai_name, str):
            continue
        try:
            best = None
            for v in name_variants(r.ai_name):        # either script, if the AI gave both
                try:
                    best = spinner.best_example(v, r.language); break
                except KeyError:
                    continue
            if best is None:
                raise KeyError(r.ai_name)
        except KeyError:                          # not one of the Spinner's names in this language
            cmp.at[k, "ai_status"] = ("not in your sheet" if isinstance(spinner, SheetSpinner)
                                      else "not a Spinner name")
            continue
        except Exception:
            cmp.at[k, "ai_status"] = "lookup failed"
            continue
        cmp.at[k, "ai_best"], cmp.at[k, "ai_drift"] = best, de00(r.hex, best)
    return cmp

def show_name_comparison(cmp, threshold=10):
    """One row per test colour, one column per language: the speakers' name, the AI's name, and
    for names that differ, where the AI's name points."""
    langs = list(dict.fromkeys(cmp.language))
    rows = []
    for sdg, g in cmp.groupby("sdg", sort=False):
        hex_ = g.hex.iloc[0]
        cells = []
        for l in langs:
            r = g[g.language == l]
            if r.empty:
                cells.append(small("—")); continue
            r = r.iloc[0]
            spk = f'<b>{_h(r["name"])}</b>{bar(r.confidence)}{small(f"{r.confidence:.0%} confidence")}'
            if not isinstance(r.ai_name, str):
                ai = small("no AI answer")
            elif r.agrees:
                ai = f'{pill("AI: same", "good")}'
            else:
                where = ""
                if isinstance(r.get("ai_best"), str):
                    where = f' {swatch(r.ai_best, 22, 14)} {small(f"ΔE00 {r.ai_drift:.1f}")}'
                elif r.get("ai_status"):
                    where = " " + small(_h(r.ai_status))
                ai = f'{pill("AI: " + _h(r.ai_name), "bad")}<div style="margin-top:4px">{where}</div>'
            cells.append(f'{spk}<div style="margin-top:6px">{ai}</div>')
        rows.append([framed(hex_, str(sdg))] + cells)
    table(["Test colour"] + [_h(l) for l in langs], rows, valign="top", note=
          "Each cell: the speakers' top name (Spinner) and its confidence, then the AI's answer. For a name that differs: "
          "the Spinner's best example of the AI's name and its ΔE00 from the test colour. Small: a plausible neighbouring "
          "name. Large: the name means a different colour to speakers.")

def plot_agreement_by_language(cmp, languages):
    """How often the AI's name matches the speakers' top name, per language."""
    rows = []
    for l in languages:
        g = cmp[cmp.language == l]
        if g.empty: continue
        share, n = g.agrees.mean(), len(g)
        rows.append([f"<b>{_h(l)}</b>", f'<b>{share:.0%}</b>{bar(share, None, 220)}{small(f"{int(g.agrees.sum())} of {n} names")}'])
    show("<b>How often the AI agrees with speakers</b>")
    table(["Language", "Same top name"], rows)

# ── Part C: names and the AI's colours ────────────────────────────────────
def load_names(source):
    """Names to test: from partB_results.csv (uploaded in Colab) or typed into the boxes."""
    if source.startswith("Upload"):
        if IN_COLAB:
            from google.colab import files
            up = files.upload()
            partb = pd.read_csv(io.BytesIO(next(iter(up.values()))))
        else:
            partb = pd.read_csv("partB_results.csv")
        names = (partb.dropna(subset=["name"]).drop_duplicates(["language", "name"])
                 [[c for c in ["sdg", "hex", "language", "role", "name", "romanized", "best_example"] if c in partb]]
                 .rename(columns={"hex": "sdg_hex", "best_example": "spinner_hex"}))
    else:
        names = typed_names().assign(sdg=np.nan, sdg_hex=None, spinner_hex=None)[
            ["sdg", "sdg_hex", "language", "name", "spinner_hex"]]
    names["language"] = names["language"].map(canonical_language)
    return names.reset_index(drop=True)

def plain_with_roman(r):
    """'κόκκινο (kokkino)' for prompts; just the name when it is already in Latin script."""
    rom = r.get("romanized")
    return r["name"] + (f" ({rom})" if isinstance(rom, str) and rom and norm(rom) != norm(r["name"]) else "")

def _with_roman(r):
    rom = r.get("romanized")
    return esc(r["name"]) + (f' <span style="color:{MUTED};font-weight:400">({esc(rom)})</span>'
                             if isinstance(rom, str) and rom and norm(rom) != norm(r["name"]) else "")

def show_names(names):
    if "sdg" in names and names.sdg.notna().any():
        palette_strip([int(i) for i in dict.fromkeys(names.sdg.dropna())], "Your test palette from Part B")
    role = dict(zip(names.language, names.get("role", pd.Series("", index=names.index))))
    rows = [[f"<b>{_h(l)}</b>" + (f" {small('(you)')}" if role.get(l) == "your language" else ""),
             ", ".join(f"<b>{_with_roman(r)}</b>" for _, r in g.iterrows())]
            for l, g in names.groupby("language", sort=False)]
    table(["Language", "Names to test"], rows)

def read_ai_colours(reply, names):
    """Attach the AI's colour for each name (matched on language + name). `reply` is the AI's text,
    or the table from the answer boxes."""
    if isinstance(reply, pd.DataFrame):
        ai = reply.rename(columns={"ai_hex": "colour"}).copy()
    else:
        ai = parse_ai_table(reply, ["language", "name", "colour"])
    def colour_of(r):
        for key in ("rgb", "colour", "color", "hex"):
            v = r.get(key)
            if isinstance(v, list) and len(v) == 3:
                return rgb_to_hex(v)
            if isinstance(v, str):
                m = _HEX.search(v)
                if m: return parse_colour(m.group(0))
                if len(re.findall(r"\d+", v)) == 3: return parse_colour(v)
        return None
    ai["ai_hex"] = [colour_of(r) for _, r in ai.iterrows()]
    ai = ai[ai.ai_hex.notna()].copy()
    ai["language"] = ai["language"].map(canonical_language)
    if ai.empty:
        raise ValueError("Couldn't find any 'language | name | colour' lines in the reply.")
    pal = names.copy()
    rom = pal["romanized"] if "romanized" in pal else pd.Series(None, index=pal.index)
    def find(r, romanized):                   # the AI's line for this name, in either script
        hits = ai[(ai.language.astype(str).map(norm) == norm(r.language)) &
                  ai.name.map(lambda n: same_name(n, r["name"], romanized))]
        return hits.ai_hex.iloc[0] if len(hits) else None
    pal["ai_hex"] = [find(r, ro) for (_, r), ro in zip(pal.iterrows(), rom)]
    missing = pal[pal.ai_hex.isna()]
    if len(missing):
        show("<b>The AI reply has no colour for:</b> " + ", ".join(
             f"{esc(r['name'])} ({esc(r.language)})" for _, r in missing.iterrows()) +
             '<div class="note">Ask it again for those names, or carry on without them.</div>')
    return pal

def show_ai_palette(pal):
    rows = []
    for l, g in pal.groupby("language", sort=False):
        missing = (f'<span style="display:inline-block;width:60px;height:34px;vertical-align:middle;border-radius:3px;'
                   f'border:2px dashed #BDBDB8"></span>')
        cells = "".join(
            f'<div style="display:inline-block;width:104px;margin:0 8px 8px 0;vertical-align:top">'
            f'{swatch(r.ai_hex, 60, 34) if isinstance(r.ai_hex, str) else missing}'
            f'<div style="margin-top:7px"><b>{_h(r["name"])}</b><br>'
            f'{small(r.ai_hex if isinstance(r.ai_hex, str) else "no answer")}</div></div>'
            for _, r in g.iterrows())
        rows.append([f"<b>{_h(l)}</b>", cells])
    show("<b>The AI system's best examples</b>")
    table(["Language", "Name and the AI's colour"], rows)

def add_best_examples(pal, spinner):
    """Fill in the Spinner's best example for any name that does not have one yet."""
    pal = pal.copy()
    for k in pal.index[pal.spinner_hex.isna()]:
        got = try_call(spinner.best_example, pal.at[k, "name"], pal.at[k, "language"])
        pal.at[k, "spinner_hex"] = None if isinstance(got, Exception) else got
    return pal


# ── Part C: round trip ────────────────────────────────────────────────────
def show_round_trip(pal):
    rows, last = [], None
    for _, r in pal.iterrows():
        if not isinstance(r.ai_hex, str): continue
        lang = f"<b>{_h(r.language)}</b>" if r.language != last else ""
        last = r.language
        if isinstance(r.get("rt_name"), str):
            ok = same_name(r.rt_name, r["name"], r.get("romanized"))
            said = f'<b>{_h(r.rt_name)}</b>{bar(r.rt_conf)}{small(f"{r.rt_conf:.0%} confidence")}'
            verdict = pill("same name", "good") if ok else pill("different name", "bad")
        else:
            said, verdict = small("not looked up"), ""
        rows.append([lang, f"<b>{_h(r['name'])}</b>", swatch(r.ai_hex), said, verdict])
    table(["Language", "Name given to the AI", "AI's colour", "What speakers call that colour", ""], rows,
          "Confidence: the Spinner's confidence in that name for the AI's colour.")

# ── Part C: comparing ─────────────────────────────────────────────────────
def plot_ai_vs_speakers(pal, languages, threshold):
    ok = pal.dropna(subset=["ai_hex", "spinner_hex"])
    if ok.empty: return
    fig, ax = plt.subplots(figsize=(7.6, 6.6)); _ucs_axes(ax)
    for _, r in ok.iterrows():
        a, s = rgb_to_cam16ucs(r.ai_hex), rgb_to_cam16ucs(r.spinner_hex)
        far = r.de00 > threshold
        ax.annotate("", xy=(s[1], s[2]), xytext=(a[1], a[2]),
                    arrowprops=dict(arrowstyle="-|>", color="#B4342D" if far else "#9A9A95", lw=1.2 if far else 0.9,
                                    shrinkA=7, shrinkB=9), zorder=1)
        ax.scatter(a[1], a[2], s=60, marker=MARKERS.get(r.language, "o"), color=r.ai_hex, edgecolor=INK, lw=0.6, zorder=2)
        ax.scatter(s[1], s[2], s=180, marker=MARKERS.get(r.language, "o"), color=r.spinner_hex,
                   edgecolor="white", lw=2, zorder=3)
    _language_legend(ax, languages)
    ax.set_aspect("equal", adjustable="datalim")
    _titles(ax, "From the AI's colour to the speakers' best example",
            f"Top-down view of CAM16-UCS colour space. Small marker: the AI's colour. Large marker: the speakers' "
            f"best example. Red arrows: more than ΔE00 {threshold:g} apart.")
    plt.tight_layout(); plt.show()

def error_summary(pal, languages, threshold):
    """Per language: how far the AI's colours were from speakers' best examples."""
    ok = pal.dropna(subset=["ai_hex", "spinner_hex"])
    top = max(2 * threshold, float(ok.de00.max()) if len(ok) else 1)
    rows = []
    for l in languages:
        g = ok[ok.language == l]
        if g.empty: continue
        n_bad = int((g.de00 > threshold).sum())
        rows.append([f"<b>{_h(l)}</b>", f'<b>{g.de00.mean():.1f}</b>{bar(g.de00.mean() / top, None, 160)}',
                     f"<b>{g.de00.max():.1f}</b>", pill(f"{n_bad} of {len(g)}", "bad" if n_bad else "good")])
    show("<b>Errors by language</b>")
    table(["Language", "Mean ΔE00", "Worst ΔE00", f"Above {threshold:g}"], rows)

# ── Part C: grounding ─────────────────────────────────────────────────────
def show_grounded(pal, languages):
    rows = []
    for lang in languages:
        p = pal[(pal.language == lang) & pal.grounded_hex.notna()]
        if p.empty: continue
        cells = ""
        for _, r in p.iterrows():
            ai = swatch(r.ai_hex, 26, 16) if isinstance(r.ai_hex, str) else small("—")
            cells += (f'<div style="display:inline-block;margin:0 16px 10px 0;vertical-align:top">'
                      f'{framed(r.grounded_hex, r["name"], 112, 54, 13)}'
                      f'<div style="margin-top:6px">{pill("replaced", "mid") if r.replaced else pill("kept", "good")} '
                      f'{small("AI:")} {ai}</div></div>')
        rows.append([f"<b>{_h(lang)}</b>", cells])
    table(["Language", "Grounded palette"], rows,
          f"{int(pal.replaced.sum())} of {int(pal.grounded_hex.notna().sum())} colours replaced with the Colour Spinner's "
          "best example. The small swatch is the AI's original colour.")

@needs_simulations(default=None)
def repair_palettes(pal, spinner, languages, threshold, candidates=8):
    """Where a grounded palette has two colours closer than `threshold` (ΔE00) under any viewing
    condition, swap one of them for another colour speakers still call by the same name (the next
    tiles in the Spinner's Mapper), choosing the swap that best separates the palette."""
    from itertools import combinations
    pal = pal.copy()
    pal["repaired_from"] = None
    log = []
    def worst(hexes):
        best = (np.inf, None, None, None)
        for (a, ha), (b, hb) in combinations(list(enumerate(hexes)), 2):
            for c in CONDITIONS:
                d = de00(simulate(ha, c), simulate(hb, c))
                if d < best[0]:
                    best = (d, a, b, c)
        return best
    for lang in languages:
        idx = list(pal.index[(pal.language == lang) & pal.grounded_hex.notna()])
        if len(idx) < 2:
            continue
        for _ in range(len(idx)):                       # at most one swap per colour
            hexes = [pal.at[k, "grounded_hex"] for k in idx]
            d, a, b, cond = worst(hexes)
            if d >= threshold:
                break
            options = []
            for pos in (a, b):
                k = idx[pos]
                if pal.at[k, "repaired_from"]:
                    continue
                members = safe(spinner.category_members, pal.at[k, "name"], lang) or []
                tried = [m for m in members[:candidates + 1] if m != pal.at[k, "grounded_hex"]][:candidates]
                prefetch_simulations(tried)
                for m in tried:
                    trial = hexes.copy(); trial[pos] = m
                    options.append((worst(trial)[0], -members.index(m), k, m))
            if not options:
                log.append(dict(language=lang, pair=(pal.at[idx[a], "name"], pal.at[idx[b], "name"]),
                                before=d, condition=cond, fixed=False))
                break
            new_d, _, k, m = max(options)
            if new_d <= d:
                log.append(dict(language=lang, pair=(pal.at[idx[a], "name"], pal.at[idx[b], "name"]),
                                before=d, condition=cond, fixed=False))
                break
            log.append(dict(language=lang, name=pal.at[k, "name"], old=pal.at[k, "grounded_hex"], new=m,
                            pair=(pal.at[idx[a], "name"], pal.at[idx[b], "name"]), before=d, after=new_d,
                            condition=cond, fixed=new_d >= threshold))
            pal.at[k, "repaired_from"] = pal.at[k, "grounded_hex"]
            pal.at[k, "grounded_hex"] = m
    pal.attrs["repairs"] = log
    return pal

def show_repairs(pal, threshold):
    log = pal.attrs.get("repairs", [])
    if not log:
        show(f"<b>No perceptual problems.</b> {small(f'Every grounded palette keeps its colours at least ΔE00 {threshold:g} apart under every condition.')}")
        return
    rows = []
    for e in log:
        if "new" in e:
            change = (f'{swatch(e["old"])} <span style="color:{MUTED}">→</span> {swatch(e["new"])}')
            result = (pill(f'{e["before"]:.1f} → {e["after"]:.1f}', "good" if e["fixed"] else "mid") + " " +
                      small("fixed" if e["fixed"] else "improved, still below the threshold"))
            what = f'<b>{_h(e["name"])}</b>'
        else:
            change, what = small("no alternative found"), "—"
            result = pill(f'{e["before"]:.1f}', "bad") + " " + small("still too close: consider a pattern, label or icon")
        rows.append([f"<b>{_h(e['language'])}</b>", f'{_h(e["pair"][0])} & {_h(e["pair"][1])}<br>{small("closest under " + _h(e["condition"]))}',
                     what, change, result])
    show("<b>Perceptual repairs</b>")
    table(["Language", "Pair too close", "Colour changed", "Before → after", "Closest ΔE00"], rows,
          "Each replacement is another colour speakers call by the same name, taken from the Colour Spinner's Mapper "
          "(its tiles run from the most to the least typical), chosen to separate the palette most.")

@needs_simulations()
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
    if check is None: return
    below = int((check.values < threshold).sum())
    _cell_grid(check, threshold, [f"{l} · {v}" for l, v in check.index],
               "Can every viewer still tell each palette's colours apart?",
               f"Smallest ΔE00 between any two colours in the palette. Red: closer than {threshold} "
               f"({below} of {int(np.isfinite(check.values).sum())} cells).")

@needs_simulations()
def show_grounded_conditions(pal, languages):
    if not _SIM["available"]: return          # the problem was already explained above
    langs = [l for l in languages if pal[pal.language == l].grounded_hex.notna().sum() > 1]
    for k, lang in enumerate(langs):
        p = pal[(pal.language == lang) & pal.grounded_hex.notna()]
        show(f'<div style="font-weight:700;margin:10px 0 4px">{esc(lang)}, grounded</div>')
        show_conditions(p.grounded_hex.tolist(), labels=p.name.tolist(), width=64, note=(k == len(langs) - 1))


# ── Part C: export and finding ────────────────────────────────────────────
# ── Part C: the grounded palette as a picture, in use, and in design-tool formats ──
def _status(r):
    if isinstance(r.get("repaired_from"), str): return "repaired"
    return "replaced" if r.get("replaced") else "kept"

def palette_card(pal, languages, path=None, title="Grounded palette"):
    """A clean card of every language's grounded palette: swatch, name, hex code and status."""
    langs = [l for l in languages if pal[(pal.language == l) & pal.grounded_hex.notna()].shape[0]]
    n = max([pal[(pal.language == l) & pal.grounded_hex.notna()].shape[0] for l in langs] or [1])
    W, Hd = n * 1.55 + 2.0, len(langs) * 1.45 + 0.1             # data units; one unit is drawn as 0.95 inch
    fig, ax = plt.subplots(figsize=(W * 0.95, Hd * 0.95 + 1.0))
    ax.set_xlim(0, W); ax.set_ylim(Hd, -0.2); ax.set_aspect("equal"); ax.axis("off")
    for row, lang in enumerate(langs):
        y = row * 1.45
        ax.text(0, y + 0.42, lang, fontsize=11.5, fontweight="bold", va="center", color=INK)
        for k, (_, r) in enumerate(pal[(pal.language == lang) & pal.grounded_hex.notna()].iterrows()):
            x = 2.0 + k * 1.55
            ax.add_patch(mpl.patches.FancyBboxPatch((x, y), 1.35, 0.78, boxstyle="round,pad=0,rounding_size=0.06",
                         fc=SURROUND, ec="none"))
            ax.add_patch(mpl.patches.FancyBboxPatch((x + 0.06, y + 0.06), 1.23, 0.66,
                         boxstyle="round,pad=0,rounding_size=0.03", fc=r.grounded_hex, ec="none"))
            ax.text(x + 0.14, y + 0.62, r["name"], fontsize=9.5, fontweight="bold", color=_ink_on(r.grounded_hex))
            ax.text(x, y + 0.98, r.grounded_hex, fontsize=8.5, color=INK, family="monospace")
            ax.text(x, y + 1.18, _status(r), fontsize=8, color=MUTED)
    import textwrap
    note = textwrap.fill("Each colour shows what speakers of that language mean by the name, from the Colour Spinner. "
                         "kept: the AI's colour matched · replaced: the speakers' best example · "
                         "repaired: changed so every viewer can tell the colours apart.", width=int(W * 0.95 * 15))
    ax.set_title(title, loc="left", fontsize=14, pad=12 + 12 * note.count("\n") + 12)
    ax.annotate(note, xy=(0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                fontsize=8.5, color=MUTED, va="bottom", linespacing=1.3)
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()

def preview_applications(pal, language, vision="Typical", path=None):
    """The grounded palette for one language, applied to a chart, an interface and wayfinding labels,
    as seen with typical vision and under `vision`."""
    p = pal[(pal.language == language) & pal.grounded_hex.notna()]
    if p.empty:
        show(f"<b>No grounded palette for {_h(language)}.</b> {small('Choose one of: ' + _h(', '.join(dict.fromkeys(pal.language))))}")
        return
    names, hexes = list(p["name"]), list(p.grounded_hex)
    rows = ["Typical"] + ([vision] if vision != "Typical" else [])
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 3.6 * len(rows)), squeeze=False,
                             gridspec_kw={"width_ratios": [1.2, 1, 1]})
    vals = [5, 8, 6, 9, 4, 7, 3, 6][:len(names)] + [5] * max(0, len(names) - 8)
    for r, cond in enumerate(rows):
        cols = [simulate(h, cond) for h in hexes]
        ax = axes[r][0]                                     # a chart with a legend
        ax.bar(range(len(names)), vals, color=cols, width=0.65)
        ax.set_xticks(range(len(names)), names, fontsize=9, rotation=20 if len(names) > 4 else 0)
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_title(f"Chart · {condition_label(cond, ' ')}", fontsize=11)
        n = len(names)
        ax = axes[r][1]                                     # interface: buttons and status dots
        ax.set_xlim(0, 6); ax.set_ylim(n * 1.1 + 0.2, -0.4); ax.set_aspect("equal"); ax.axis("off")
        for k, (nm, c) in enumerate(zip(names, cols)):
            y = k * 1.1 + 0.3
            ax.add_patch(mpl.patches.FancyBboxPatch((0.2, y - 0.36), 2.9, 0.72,
                         boxstyle="round,pad=0,rounding_size=0.36", fc=c, ec="none"))
            ax.text(1.65, y, nm, ha="center", va="center", fontsize=9.5, fontweight="bold", color=_ink_on(c))
            ax.add_patch(mpl.patches.Circle((3.65, y), 0.22, fc=c, ec="none"))
            ax.text(4.05, y, "status", va="center", fontsize=9, color=MUTED)
        ax.set_title(f"Interface · {condition_label(cond, ' ')}", fontsize=11)
        ax = axes[r][2]                                     # wayfinding: colour-coded routes with labels
        ax.set_xlim(0, 10); ax.set_ylim(n * 1.6 + 0.6, -0.6); ax.set_aspect("equal"); ax.axis("off")
        for k, (nm, c) in enumerate(zip(names, cols)):
            y = k * 1.6 + 0.4
            ax.plot([0.4, 2.6, 3.6, 6.4], [y, y, y + 0.6, y + 0.6], color=c, lw=6, solid_capstyle="round",
                    solid_joinstyle="round")
            ax.add_patch(mpl.patches.Circle((7.0, y + 0.6), 0.42, fc=c, ec="white", lw=2))
            ax.text(7.7, y + 0.6, nm, va="center", fontsize=9.5, color=INK)
        ax.set_title(f"Wayfinding · {condition_label(cond, ' ')}", fontsize=11)
    fig.suptitle(f"The {language} grounded palette in use", x=0.01, ha="left", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()

def _ase(entries):
    """Adobe Swatch Exchange (.ase) bytes for (name, hex) pairs: Illustrator, Photoshop, InDesign."""
    import struct
    blocks = b""
    for name, hex_ in entries:
        nm = (name + "\0").encode("utf-16-be")
        r, g, b = (hex_to_rgb(hex_) / 255).tolist()
        body = struct.pack(">H", len(name) + 1) + nm + b"RGB " + struct.pack(">fffH", r, g, b, 2)
        blocks += struct.pack(">HI", 0x0001, len(body)) + body
    return b"ASEF" + struct.pack(">HHI", 1, 0, len(entries)) + blocks

def export_palette(pal, languages, stem="grounded_palette"):
    """Write the grounded palette as data, design-tool swatches and pictures, zip them and offer the download."""
    import zipfile
    cols = ["language", "name", "grounded_hex", "ai_hex", "spinner_hex", "de00", "rt_name", "rt_conf",
            "replaced", "repaired_from", "sdg", "sdg_hex"]
    export = pal[[c for c in cols if c in pal.columns]]
    final = export.dropna(subset=["grounded_hex"])
    files = []
    export.round(2).to_csv(f"{stem}.csv", index=False); files.append(f"{stem}.csv")
    data = {lang: [{"name": r["name"], "hex": r.grounded_hex, "ai_hex": r.ai_hex, "spinner_hex": r.spinner_hex,
                    "dE00_ai_vs_spinner": None if pd.isna(r.de00) else round(float(r.de00), 2),
                    "speakers_call_ai_colour": r.get("rt_name") if isinstance(r.get("rt_name"), str) else None,
                    "repaired_from": r.get("repaired_from") if isinstance(r.get("repaired_from"), str) else None,
                    "replaced": bool(r.replaced)}
                   for _, r in export[export.language == lang].iterrows()] for lang in languages}
    with open(f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    files.append(f"{stem}.json")
    slug = lambda s: re.sub(r"[^\w]+", "-", norm(s)).strip("-") or "colour"
    css = ":root {\n" + "".join(f"  --{slug(r.language)}-{slug(r['name'])}: {r.grounded_hex};\n"
                                for _, r in final.iterrows()) + "}\n"
    with open(f"{stem}.css", "w", encoding="utf-8") as f: f.write(css)
    files.append(f"{stem}.css")
    tokens = {slug(l): {slug(r["name"]): {"$type": "color", "$value": r.grounded_hex,
                                          "$description": f"{r['name']} ({l}), {_status(r)}"}
                        for _, r in final[final.language == l].iterrows()} for l in languages}
    with open(f"{stem}.tokens.json", "w", encoding="utf-8") as f:        # Figma Tokens / Style Dictionary
        json.dump(tokens, f, ensure_ascii=False, indent=2)
    files.append(f"{stem}.tokens.json")
    entries = [(f"{r.language} · {r['name']}", r.grounded_hex) for _, r in final.iterrows()]
    with open(f"{stem}.gpl", "w", encoding="utf-8") as f:                 # GIMP, Inkscape, Krita
        f.write(f"GIMP Palette\nName: {stem}\nColumns: 4\n#\n" + "".join(
            "{:3d} {:3d} {:3d}\t{}\n".format(*[int(v) for v in hex_to_rgb(h)], n) for n, h in entries))
    files.append(f"{stem}.gpl")
    with open(f"{stem}.ase", "wb") as f: f.write(_ase(entries))          # Adobe apps
    files.append(f"{stem}.ase")
    palette_card(pal, languages, path=f"{stem}_card.png"); files.append(f"{stem}_card.png")
    import glob
    files += sorted(glob.glob("preview_*.png"))                         # any previews made above
    with zipfile.ZipFile(f"{stem}.zip", "w") as z:
        for fn in files:
            z.write(fn)
    show("<b>In the download</b>")
    table(["File", "Use it in"], [
        [f"<code>{stem}_card.png</code>", "Slides, reports, sharing: a picture of every palette"],
        ["<code>preview_….png</code>", "The palette in a chart, an interface and wayfinding, for each language you previewed"],
        [f"<code>{stem}.ase</code>", "Adobe Illustrator, Photoshop, InDesign (Swatches ▸ Import)"],
        [f"<code>{stem}.gpl</code>", "GIMP, Inkscape, Krita"],
        [f"<code>{stem}.tokens.json</code>", "Figma (Tokens Studio), Style Dictionary, design systems"],
        [f"<code>{stem}.css</code>", "Websites and apps: CSS custom properties"],
        [f"<code>{stem}.csv</code> · <code>{stem}.json</code>", "Analysis: every colour with the evidence for any change"]])
    download(f"{stem}.zip")

def show_finding(pal, name=""):
    cand = pal.dropna(subset=["de00"])
    if cand.empty: return
    match = cand[cand.name.map(norm) == norm(name)] if name else cand.iloc[:0]
    row = match.iloc[0] if len(match) else cand.loc[cand.de00.idxmax()]
    rt_line = ""
    if isinstance(row.get("rt_name"), str):
        rt_line = (f'<p style="margin:0 0 10px">Speakers would call the AI\'s colour '
                   f'<b>{esc(row.rt_name)}</b> ({row.rt_conf:.0%} confidence).</p>')
    show(f"""<div class="card">
<h3>“{esc(row['name'])}” in {esc(row.language)}</h3>
<div style="display:flex;gap:10px;margin:12px 0;background:{SURROUND};padding:12px;border-radius:6px">
  <div class="tile"><div class="chip" style="background:{row.ai_hex};color:{ink_on(row.ai_hex)};height:80px">AI</div>
       <div class="cap">AI system<div class="hex">{row.ai_hex}</div></div></div>
  <div class="tile"><div class="chip" style="background:{row.spinner_hex};color:{ink_on(row.spinner_hex)};height:80px">Speakers</div>
       <div class="cap">Best example<div class="hex">{row.spinner_hex}</div></div></div>
  <div style="color:#fff;align-self:center;padding-left:8px;font-size:15px">ΔE00<br><b style="font-size:30px">{row.de00:.1f}</b></div>
</div>
{rt_line}
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
