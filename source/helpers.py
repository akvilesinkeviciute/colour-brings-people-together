

# ════════════════════════════════════════════════════════════════════════
#  6. Activity helpers: the charts and tables the notebooks show
# ════════════════════════════════════════════════════════════════════════
esc, ink_on, read_sheet = _h, _ink_on, _read_sheet      # public names for notebook cells
MARKERS = dict(zip(SPINNER_LANGUAGES, ["p", "s", "D", "*", "^", "v", "h"]))   # Spanish keeps its hexagon   # the test colour itself is a circle

def _language_legend(ax, languages, loc="best", size=8):
    ax.legend(handles=[mpl.lines.Line2D([], [], marker=MARKERS.get(l, "o"), ls="", color="white",
              markeredgecolor=INK, markersize=size, label=l) for l in languages],
              frameon=False, loc=loc)

def _lab_axes(ax):
    """The CIELAB a*–b* plane, seen from above (lightness not shown)."""
    ax.axhline(0, color="#E4E4E0", lw=1, zorder=0); ax.axvline(0, color="#E4E4E0", lw=1, zorder=0)
    ax.set_xlabel("a*", color=MUTED); ax.set_ylabel("b*", color=MUTED, rotation=0, labelpad=10)

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

def _n(k, word):
    return f"{k} {word}{'' if k == 1 else 's'}"

def show_subset(pick, languages):
    sub = SDG.loc[pick]
    show_palette(sub.hex.tolist(), labels=[str(i) for i in sub.index], captions=sub.goal.tolist(),
                 note=f"Auditing {_n(len(pick), 'colour')} in {_n(len(languages), 'language')}: {', '.join(languages)}.")

def check_choices(text, pick, audience):
    """Say plainly what was understood, and what would make the audit work better."""
    tips = []
    entries = [t for t in re.split(r"[,\s;]+", text.strip()) if t]
    bad = [t for t in entries if not (t.isdigit() and 1 <= int(t) <= 17)]
    nums = [int(t) for t in entries if t.isdigit() and 1 <= int(t) <= 17]
    if bad:
        tips.append(f"Ignored {', '.join(_h(t) for t in bad)}: test colours are the SDG numbers 1 to 17.")
    if len(nums) != len(set(nums)):
        tips.append("Each colour counts once, so the repeated numbers were dropped.")
    if len(pick) < 2:
        tips.append("Choose at least two test colours: the checks compare your colours with each other.")
    elif len(pick) > 6:
        tips.append(f"{len(pick)} colours make long tables and a long AI prompt; three to five work best in the time.")
    if not audience:
        tips.append("Tick at least one audience language: the checks compare your names with your audience's.")
    if tips:
        show('<div class="note" style="color:#8A5A00">' + "<br>".join(tips) + "</div>")

def _available_language(pal, wanted, what):
    """`wanted` if the palette has it; otherwise the first audience language, saying so."""
    have = list(dict.fromkeys(pal.dropna(subset=["grounded_hex"]).language)) if "grounded_hex" in pal else list(dict.fromkeys(pal.language))
    if wanted in have or not have:
        return wanted
    role = dict(zip(pal.language, pal.get("role", pd.Series("", index=pal.index))))
    pick = next((l for l in have if role.get(l) == "audience"), have[0])
    show(small(f"{_h(wanted)} isn't among your languages, so this shows {_h(pick)}."))
    return pick


# ── Part B: perceptual axis ───────────────────────────────────────────────
@needs_simulations(default=(None, None))
def nearest_neighbours(pick, conditions=CONDITIONS, among=None):
    """For each colour in `pick` and each condition: the nearest other colour among `among`
    (by default the other test colours) and its ΔE₀₀. Returns distances and neighbour ids."""
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
    W, H = 1.5 * nc + (1.6 if row_chip_colours else 2.4), 0.8 * nr + 1.8
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
                seen = simulate(SDG.hex[n], cols[c])            # the neighbour as it looks under this condition
                ax.add_patch(mpl.patches.Rectangle((c - 0.30, r + 0.13), 0.13, 0.15, fc=seen, ec="white", lw=0.8))
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
    import textwrap
    sub = textwrap.fill(subtitle, width=int(1.5 * nc * 13))
    ax.set_title(title, pad=12 + 13 * (sub.count("\n") + 1))
    ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points",
                ha="left", va="bottom", fontsize=9.5, color=MUTED, linespacing=1.3)
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
               f"Each cell: ΔE₀₀ between the simulated colours, i.e. between the test colour and its nearest other test colour, "
               f"both simulated for that column (for example, ΔE₀₀ between deuteranopia-simulated colours). The small patch "
               f"is that neighbour as it looks there. Red: closer than {threshold}, "
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
        show(f"<b>No pairs at risk.</b> {small(f'Your test colours stay at least ΔE₀₀ {threshold} apart under every condition.')}")
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
                     f'<span style="white-space:nowrap"><b>{typical:.1f} → {d:.1f}</b></span><br>'
                     f'{small("between typical colours → between " + _h(sim_phrase(c)))}',
                     f'<span style="font-size:12.5px">{_h(why)}</span>'])
    show("<b>Pairs at risk, and why</b>")
    table(["Typical vision", "Where they're closest", "ΔE₀₀", "Why they look alike"], rows,
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
    _SHEET["sheet_box"] = w.VBox(items)
    display(_SHEET["sheet_box"])

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
    _SHEET["names_box"] = w.VBox(items)
    display(_SHEET["names_box"])

class _FillInFirst(Exception):
    """Stops a cell quietly: the boxes it showed need filling in before the cell is run again."""
    def _render_traceback_(self):
        return []

def boxes_first(backend, pick, languages):
    """Only for Backend 'Copied from the browser': the first run shows the boxes and stops; the next run
    shows them again, with what was typed, and carries on. With the Colour Spinner API it does nothing."""
    if backend != "Copied from the browser":
        return
    key = (tuple(pick), tuple(languages))
    if _SHEET.get("sheet_key") != key:
        show("<b>Copy the Colour Spinner's answers into the boxes.</b> " + small(
            "For each row, look the colour up in the Spinner and copy the Namer's name and confidence, and the RGB "
            "of the Mapper's best example. Then run this cell again."))
        sheet_boxes(pick, languages)
        _SHEET["sheet_key"] = key
        raise _FillInFirst()
    display(_SHEET["sheet_box"])

def names_first(source):
    """Only for 'Type them in': the first run shows the boxes and stops; the next run reads them."""
    if source != "Type them in":
        return
    if "names_box" not in _SHEET:
        show("<b>Type the colour names to test.</b> " + small("Choose a language and type a name in each row you need. "
             "Add a best example only if you're using <i>Copied from the browser</i>. Then run this cell again."))
        name_boxes()
        raise _FillInFirst()
    display(_SHEET["names_box"])
    if not any(name.value.strip() for _, name, _ in _SHEET["names"]):
        show("<b>The boxes are still empty.</b> " + small("Type at least one colour name, then run this cell again."))
        raise _FillInFirst()

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
             '<div class="note">Using the Colour Spinner: check your internet connection, or ask the facilitators to switch to '
             '<i>Copied from the browser</i>. Using the sheet: add the missing rows and run the cell again.</div>')

def show_names_table(naming, languages, you=None):
    if naming.empty: return
    wide = naming.pivot(index="sdg", columns="language", values="name")
    conf = naming.pivot(index="sdg", columns="language", values="confidence").add_suffix("|conf")
    table = SDG.loc[wide.index, ["hex"]].join(wide).join(conf)
    if "romanized" in naming:
        table = table.join(naming.pivot(index="sdg", columns="language", values="romanized").add_suffix("|rom"))
    labels = {you: f"<b style='color:{INK}'>{_h(you)}</b> (you)"} if you else None
    show_babel(table, [l for l in languages if l in wide.columns], labels=labels, first="Your test palette")

def plot_nameability(naming, pick, languages, threshold=0.5, you=None):
    """One small panel per test colour: for each language, a bar in the test colour running to the
    Spinner's confidence in its top name, with the name beside it."""
    import math
    pick = [i for i in pick if not naming[naming.sdg == i].empty]
    if not pick: return
    langs = [l for l in languages if l in set(naming.language)]
    ncols = 2 if len(pick) > 1 else 1
    nrows = math.ceil(len(pick) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 0.48 * len(langs) * nrows + 0.9 * nrows + 0.8),
                             squeeze=False)
    for ax in axes.flat[len(pick):]:
        ax.axis("off")
    for ax, i in zip(axes.flat, pick):
        hex_ = SDG.hex[i]
        for k, l in enumerate(langs):
            r = naming[(naming.sdg == i) & (naming.language == l)]
            ax.barh(k, 1.0, height=0.56, color="#EEEEEA", zorder=1)
            if r.empty: continue
            r = r.iloc[0]
            ax.barh(k, r.confidence, height=0.56, color=hex_, zorder=2)
            inside = r.confidence > 0.62
            ax.text(r.confidence - 0.02 if inside else r.confidence + 0.02, k, r["name"],
                    ha="right" if inside else "left", va="center", fontsize=10.5, fontweight="bold",
                    color=_ink_on(hex_) if inside else INK, zorder=3)
            ax.text(1.03, k, f"{r.confidence:.0%}", ha="left", va="center", fontsize=10, color=MUTED)
        ax.axvline(threshold, color=INK, lw=0.8, ls=(0, (3, 3)), alpha=0.35, zorder=1.5)
        ax.set_yticks(range(len(langs)), [f"{l} (you)" if l == you else l for l in langs], fontsize=10)
        for t in ax.get_yticklabels():
            t.set_color(INK if t.get_text().endswith("(you)") else MUTED)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.set_xlim(0, 1.12); ax.set_ylim(len(langs) - 0.4, -1.25); ax.set_xticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.text(0, -0.95, f" {i} ", fontsize=11, fontweight="bold", color=_ink_on(hex_), va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.25", fc=hex_, ec=SURROUND, lw=2.5))
        ax.text(0.075 + 0.028 * len(str(i)), -0.95, SDG.goal[i], fontsize=10, color=MUTED, va="center", ha="left")
    fig.suptitle("Nameability: how confident is the Spinner's top name?", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.text(0.01, 1 - 0.55 / fig.get_figheight(),
             f"Each bar runs to the Spinner's confidence in its top name for the test colour, in each language. "
             f"Dashed line: {threshold:.0%}; shorter bars mean speakers' names are spread across several terms.",
             fontsize=9.5, color=MUTED, ha="left", va="top", wrap=True)
    plt.tight_layout(rect=(0, 0, 1, 1 - 0.7 / fig.get_figheight()), h_pad=0.6)
    plt.show()

def show_translatability(naming, pick, languages, threshold, source=None):
    """For each test colour: the best example (focal colour) of each language's name for it. Your language is the
    reference; under each audience language, the ΔE₀₀ between its best example and yours."""
    if source is not None and source not in languages:
        show(f"<b>{esc(source)}</b> isn't one of the languages you ticked in section 3. "
             "Tick it there, or choose another source language.")
        return
    order = ([source] + [l for l in languages if l != source]) if source else list(languages)
    SW = 132                                          # every swatch, test colours included, has the same size
    td = 'style="vertical-align:top;padding:0 14px 16px 0"'
    def chip(hex_, text, size=10.5, bold=False):
        return (f'<div class="pair" style="width:{SW}px;height:58px"><div style="background:{hex_};color:{ink_on(hex_)};'
                f'font-size:{size}px;{"font-weight:700;" if bold else ""}">{text}</div></div>')
    rows = ""
    for i in pick:
        sub = naming[naming.sdg == i]
        best = {l: sub[sub.language == l].best_example.iloc[0] for l in order
                if not sub[sub.language == l].empty and isinstance(sub[sub.language == l].best_example.iloc[0], str)}
        cells = ""
        for k, lang in enumerate(order):
            if lang not in best:
                cells += f'<td {td}><span style="color:{MUTED}">—</span></td>'
                continue
            r = sub[sub.language == lang]; b = best[lang]
            badge = ""
            if k > 0 and source and source in best:
                badge = f'<div style="margin-top:5px">{de_badge(de00(best[source], b), threshold, prefix="ΔE₀₀ ")}</div>'
            cells += f'<td {td}>{chip(b, f"{esc(r.name.iloc[0])}<br>{b}")}{badge}</td>'
        rows += f'<tr><td {td}>{chip(SDG.hex[i], str(i), 15, True)}</td>{cells}</tr>'
    th = f'style="text-align:left;padding:0 14px 8px 0;font-weight:400;font-size:12px;white-space:nowrap;color:{MUTED}"'
    heads = "".join(f'<th {th}>{esc(l)}{(" (you)" if k == 0 else " · audience") if source else ""}</th>'
                    for k, l in enumerate(order))
    show(f'<table style="border-collapse:collapse"><tr><th {th}>Your test palette</th>{heads}</tr>{rows}</table>'
         f'<div class="note">Under each audience language: the ΔE₀₀ between your best example and that language\'s. '
         f'Green: below {threshold / 2:g}; amber: up to {threshold:g}; red: more than {threshold:g}, where a translation '
         f'would land on a different colour.</div>')

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

def _lab_panels(n, title, subtitle, points, pad=8, height=3.4):
    """One CIELAB a*–b* panel per language, two across, all on the same scale and sized to the data."""
    import math, textwrap
    pts = np.asarray(points, float)
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    ratio = (hi[0] - lo[0]) / (hi[1] - lo[1])
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)
    w = max(3.2, height * ratio) + 0.9
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, height * nrows + 1.2), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 1 - 0.5 / fig.get_figheight(), textwrap.fill(subtitle, int(w * ncols * 13)),
             fontsize=9.5, color=MUTED, ha="left", va="top", linespacing=1.3)
    for ax in axes.flat[:n]:
        _lab_axes(ax)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=8.5, colors=MUTED)
    return fig, list(axes.flat[:n])

def _point_label(ax, x, y, text, dx=5, dy=5):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75), zorder=2.5)

def _clean_lab_axes(ax, points=None, pad=8, ratio=1.55):
    """The CIELAB a*–b* plane in the style of our colour-naming figures: no grid, light frame, fitted to the data.
    a* and b* keep the same scale; the shorter range is widened around the data so the plot fills a wide frame."""
    if points is not None and len(points):
        pts = np.asarray(points, float); lo, hi = pts.min(0) - pad, pts.max(0) + pad
        w, h = hi - lo
        if w < h * ratio:
            cx = (lo[0] + hi[0]) / 2; lo[0], hi[0] = cx - h * ratio / 2, cx + h * ratio / 2
        else:
            cy = (lo[1] + hi[1]) / 2; lo[1], hi[1] = cy - w / ratio / 2, cy + w / ratio / 2
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#8A8A85")
    ax.tick_params(colors="#5E5E5E", labelsize=9)
    ax.set_xlabel("a*", color=INK); ax.set_ylabel("b*", color=INK, rotation=0, labelpad=10)
    ax.set_aspect("equal", adjustable="box")

def _marker_size(m, base):
    return 0.8 * base * {"*": 2.6, "p": 1.15, "h": 1.1}.get(m, 1.0)   # stars look small at the same area

def _shape_legend(ax, entries):
    ax.legend(handles=[mpl.lines.Line2D([], [], marker=m, ls="", markerfacecolor=fc, markeredgecolor=INK,
                                        markeredgewidth=0.8, markersize=8 * (1.55 if m == "*" else 1), label=l)
                       for l, m, fc in entries],
              frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9.5, handletextpad=0.4)

MAP_HALF_WINDOW = 20        # ΔE-sized half-width of each map panel; points further out are drawn at the edge

def _to_window(ax, p, centre):
    """A point inside the panel, or the place where its line leaves the panel (and whether it was moved)."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    p, c = np.asarray(p, float), np.asarray(centre, float)
    if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
        return p, False
    d = p - c
    t = min([((x1 if d[0] > 0 else x0) - c[0]) / d[0] if d[0] else np.inf,
             ((y1 if d[1] > 0 else y0) - c[1]) / d[1] if d[1] else np.inf])
    return c + d * t * 0.93, True

def _edge_arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color="#1C1C1C", lw=0.7,
                shrinkA=0, shrinkB=0, mutation_scale=9), zorder=1)

def _panel_grid(groups, title, subtitle, legend_entries):
    """A 2 × 2 (or smaller) grid of CIELAB a*–b* panels, one per group, every window the same size so distances
    compare across panels. groups: list of (panel title, centre (a*, b*), [all (a*, b*) points in the panel])."""
    import math, textwrap
    half = min(MAP_HALF_WINDOW, max(np.abs(np.asarray(pts, float) - np.asarray(c, float)).max() for _, c, pts in groups) + 7)
    n = len(groups); ncols = 2 if n > 1 else 1; nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows + 1.6 + 0.3 * (len(legend_entries) > 5)), squeeze=False)
    for ax in axes.flat[n:]: ax.axis("off")
    for ax, (t, c, _) in zip(axes.flat, groups):
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color("#8A8A85")
        ax.tick_params(colors="#5E5E5E", labelsize=8.5)
        ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half); ax.set_aspect("equal")
    for k, (ax, (t, c, _)) in enumerate(zip(axes.flat, groups)):
        if k // ncols == nrows - 1 or k + ncols >= n:          # axis labels once: bottom row and left column
            ax.set_xlabel("a*", color=INK)
        if k % ncols == 0:
            ax.set_ylabel("b*", color=INK, rotation=0, labelpad=10)
        ax.set_title(t, loc="left", fontsize=11, fontweight="bold", pad=6)
    fig.suptitle(title, x=0.01, y=1 - 0.12 / fig.get_figheight(), ha="left", va="top", fontsize=13, fontweight="bold")
    fig.text(0.01, 1 - 0.5 / fig.get_figheight(), textwrap.fill(subtitle, int(5.2 * ncols * 14)),
             fontsize=9.5, color=MUTED, ha="left", va="top", linespacing=1.3)
    fig.legend(handles=[mpl.lines.Line2D([], [], marker=m, ls="", markerfacecolor=fc, markeredgecolor=INK,
                                         markeredgewidth=0.8, markersize=8 * (1.55 if m == "*" else 1), label=l)
                        for l, m, fc in legend_entries],
               loc="upper left", ncol=len(legend_entries) if len(legend_entries) <= 5 else -(-len(legend_entries) // 2), frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.005, 1 - 0.95 / fig.get_figheight()))
    plt.tight_layout(rect=(0, 0, 1, 1 - (1.35 + 0.3 * (len(legend_entries) > 5)) / fig.get_figheight()), h_pad=2, w_pad=2)
    return fig, list(axes.flat[:n])

def plot_name_map(naming, pick, languages):
    """One panel per test colour: the test colour (numbered circle) joined by thin lines to the best example of each
    language's name for it, one shape per language, on the CIELAB a*–b* plane."""
    pick = [i for i in pick if not naming[(naming.sdg == i) & naming.best_example.notna()].empty]
    if not pick: return
    groups = []
    for i in pick:
        o = rgb_to_lab(SDG.hex[i])[1:]
        pts = [o] + [rgb_to_lab(b)[1:] for b in naming[(naming.sdg == i) & naming.best_example.notna()].best_example]
        groups.append((f"Test colour {i}", o, pts))
    fig, axes = _panel_grid(groups, "Where each language's name pulls the test colour",
                            "CIELAB a*–b* plane, seen from above (lightness not shown). An arrow at the edge: that best "
                            "example lies further out.",
                            [("Test colour", "o", "white")] + [(l, MARKERS.get(l, "o"), "white") for l in languages])
    for ax, i in zip(axes, pick):
        o = rgb_to_lab(SDG.hex[i])
        for _, r in naming[(naming.sdg == i) & naming.best_example.notna()].iterrows():
            b = rgb_to_lab(r.best_example); m = MARKERS.get(r.language, "o")
            pb, moved = _to_window(ax, b[1:], o[1:])
            if moved:                                   # far away: an arrow to the panel's edge
                _edge_arrow(ax, o[1:], pb)
            else:
                ax.plot([o[1], pb[0]], [o[2], pb[1]], color="#1C1C1C", lw=0.7, zorder=1)
            ax.scatter(*pb, s=_marker_size(m, 150), marker=m, color=r.best_example, edgecolor="white", lw=0.9,
                       zorder=3.5 if m == "*" else 3,          # stars on top: the shapes underneath show between their points
                       alpha=0.85 if moved else 1)   # a thin white outline keeps overlapping shapes apart
        ax.scatter(o[1], o[2], s=440, color=SDG.hex[i], edgecolor="white", lw=2, zorder=4)
        ax.text(o[1], o[2], str(i), ha="center", va="center", fontsize=10, fontweight="bold", color=ink_on(SDG.hex[i]), zorder=5)
    plt.show()

def _plot_name_map_single(naming, pick, languages):
    fig, ax = plt.subplots(figsize=(8, 6.2)); _lab_axes(ax)
    for i in pick:
        o = rgb_to_lab(SDG.hex[i])
        for _, r in naming[(naming.sdg == i) & naming.best_example.notna()].iterrows():
            b = rgb_to_lab(r.best_example)
            ax.plot([o[1], b[1]], [o[2], b[2]], color=MUTED, lw=1, zorder=1)
            ax.scatter(b[1], b[2], s=90, marker=MARKERS.get(r.language, "o"), color=r.best_example,
                       edgecolor=INK, lw=0.7, zorder=3)
        ax.scatter(o[1], o[2], s=520, color=SDG.hex[i], edgecolor="white", lw=2.5, zorder=2)
        ax.text(o[1], o[2], str(i), ha="center", va="center", color=ink_on(SDG.hex[i]),
                fontweight="bold", fontsize=10, zorder=4)
    _language_legend(ax, languages)
    ax.set_aspect("equal", adjustable="datalim")
    _titles(ax, "Where each language's name pulls the test colour",
            "CIELAB a*–b* plane, seen from above. Large disc: the test colour. Lines lead to each "
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
    """One row per test colour: its perceptual distance (ΔE₀₀ to the nearest other SDG colour, under
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
             f'tick it in section 3 and run sections 3 to 5 again. Your languages: '
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

def plot_two_axes(card, confusable_below=DELTA_E_THRESHOLD, hard_below=HARD_TO_NAME, top=40):
    """Colour identity scorecard: naming confidence (across) against nearest-colour ΔE₀₀ (up),
    on fixed axes so every palette is drawn on the same scale."""
    if card is None:
        return
    if card["dE00"].isna().all():
        show("<b>The scorecard needs at least two test colours</b> " + small("to measure how far each is from the nearest other one. Add more in section 3."))
        return
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    zones = [((0, confusable_below), hard_below, top - confusable_below, "#F4F4F2", "Distinct, hard to name", "left", "top"),
             ((hard_below, confusable_below), 1 - hard_below, top - confusable_below, "#E2EEE7", "Distinct + nameable", "right", "top"),
             ((0, 0), hard_below, confusable_below, "#F6E9E2", "Confusable + hard to name", "left", "bottom"),
             ((hard_below, 0), 1 - hard_below, confusable_below, "#F4F4F2", "Nameable, confusable", "right", "bottom")]
    for (x, y), w, hgt, col, label, ha, va in zones:
        ax.add_patch(mpl.patches.Rectangle((x, y), w, hgt, fc=col, ec="none", zorder=0))
        ax.text(x + (0.015 if ha == "left" else w - 0.015), y + (hgt - top * 0.012 if va == "top" else top * 0.012), label,
                ha=ha, va=va, fontsize=9, color="#7A7A75", zorder=1)
    ax.axvline(hard_below, color="#9A9A95", ls=(0, (4, 3)), lw=1.2, zorder=1)
    ax.axhline(confusable_below, color="#9A9A95", ls=(0, (4, 3)), lw=1.2, zorder=1)
    pts = [(i, r.confidence_0to1, min(r.dE00, top * 0.97), r.dE00) for i, r in card.iterrows()
           if not (pd.isna(r.confidence_0to1) or pd.isna(r.dE00))]
    # Two test colours that are each other's nearest neighbour share the same ΔE₀₀, so their discs can land on
    # top of each other: nudge any overlapping discs apart sideways so every colour stays visible.
    groups = []
    for p_ in pts:
        g = next((g for g in groups if abs(g[0][1] - p_[1]) < 0.06 and abs(g[0][2] - p_[2]) < top * 0.05), None)
        (g.append(p_) if g else groups.append([p_]))
    nudged = False
    for g in groups:
        for k, (i, x, y, raw) in enumerate(g):
            dx = (k - (len(g) - 1) / 2) * 0.075
            nudged |= len(g) > 1
            ax.scatter(x + dx, y, s=900, color=SDG.hex[i], edgecolor=SURROUND, lw=1.5, zorder=3)
            ax.text(x + dx, y, str(i), ha="center", va="center", color=ink_on(SDG.hex[i]), fontweight="bold",
                    fontsize=12, zorder=4)
            if raw > top:
                ax.text(x + dx, y + top * 0.035, f"↑ {raw:.0f}", ha="center", va="bottom", fontsize=8.5, color=MUTED, zorder=4)
    ax.set_xlim(0, 1); ax.set_ylim(0, top)
    ax.set_xticks([0, hard_below, 1], ["Low", f"{hard_below:.0%}", "High"])
    ax.set_yticks([0, confusable_below, 20, 30, top])
    ax.spines["left"].set_color(INK); ax.spines["bottom"].set_color(INK)
    lab = card.attrs["language"]
    who = ("your audience on average" if lab == AUDIENCE_AVG else "all your languages on average" if lab == ALL_LANGUAGES else lab)
    ax.set_xlabel("Naming confidence")
    ax.set_ylabel("Nearest-colour ΔE₀₀")
    between = sim_phrase(card.attrs["condition"]) if card.attrs["condition"] != WORST_CASE else "simulated colours, worst case"
    _titles(ax, "Colour identity scorecard",
            f"Confidence: {who}. ΔE₀₀: between {between}. Dashed lines: ΔE₀₀ {confusable_below:g} and "
            f"{hard_below:.0%} confidence." + (" Discs at the same spot are set side by side." if nudged else ""))
    plt.tight_layout(); plt.show()

def show_scorecard(card, top=40):
    if card is None:
        return
    langs = [c for c in card.columns if c not in ("dE00", "perceptual_0to1", "confidence_0to1")]
    rows = []
    for i, r in card.iterrows():
        perc = f'<b>{r.dE00:.1f}</b>{bar(r.dE00 / top)}{small("to the nearest test colour")}'
        conf = ("—" if pd.isna(r.confidence_0to1) else
                f'<b>{r.confidence_0to1:.0%}</b>{bar(r.confidence_0to1)}{small("naming confidence")}')
        names = []
        for l in langs:
            m = re.match(r"(.*) \((\d+)%\)$", r[l] or "")
            names.append(f'<b>{_h(m.group(1))}</b>{bar(int(m.group(2)) / 100)}{small(m.group(2) + "% confidence")}'
                         if m else small("—"))
        rows.append([framed(SDG.hex[i], str(i)), perc, conf] + names)
    table(["Test colour", "ΔE₀₀", "Confidence"] + [_h(l) for l in langs], rows, valign="top")

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
    """One box to paste the AI's whole reply into; the notebook reads the numbered lines itself."""
    try:
        import ipywidgets as w
    except ImportError:
        print("The paste box needs ipywidgets, which Colab includes.")
        return
    rows = rows.reset_index(drop=True)
    n = rows.drop_duplicates("hex").shape[0] if kind == "name" else len(rows)
    box = w.Textarea(placeholder="Paste your AI's whole reply here (all lines at once), then run the next cell.",
                     layout=w.Layout(width="100%", max_width="820px", height=f"{min(320, 60 + 24 * n)}px"))
    _ANS.update(boxes={"reply": box}, rows=rows, kind=kind)
    display(box)

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
    """Read the pasted reply into a table, and say if any answers are missing."""
    rows, kind, boxes = _ANS["rows"], _ANS["kind"], _ANS["boxes"]
    if rows is None:
        raise ValueError("Run the cell above first: it shows the box to paste into.")
    text = boxes["reply"].value.strip()
    if not text:
        raise ValueError("The box is empty: paste your AI's whole reply into it, then run this cell again.")
    t = answers_from_reply(text, rows, kind)
    col = "ai_name" if kind == "name" else "ai_hex"
    missing = t[t[col].isna()]
    if len(missing):
        what = (", ".join(f"{r.hex} in {r.language}" for _, r in missing.iterrows()) if kind == "name"
                else ", ".join(f"{r['name']} ({r.language})" for _, r in missing.iterrows()))
        show(f"<b>Read {len(t) - len(missing)} of {len(t)} answers.</b> {small('Missing: ' + _h(what) + '. Ask your AI again for those, or carry on without them.')}")
    else:
        show(f"<b>Read all {len(t)} answers ✓</b>")
    return t

def answers_from_reply(reply, rows, kind):
    """Read the AI's reply into the same table as the boxes. Numbers are optional and blank lines are fine:
    kind='name': '1. British English: Red | Greek: Κόκκινο | …' (or the same without '1.');
    kind='colour': '1. #E5243B' (or just '#E5243B'), one line per name, in order."""
    text = re.sub(r"```\w*", "", reply)
    text = re.sub(r"\\([.\-*_()#\[\]|!])", r"\1", text)      # markdown escapes copied from a chat: "1\." → "1."
    found = {}
    for line in text.splitlines():
        m = re.match(r"\s*\**\s*(\d+)\s*[.):\-]\s+(.+?)\s*$", line) or re.match(r"\s*\**\s*(\d+)\s*[.):]\s*(.+?)\s*$", line)
        if m:
            found.setdefault(int(m.group(1)), m.group(2).strip())
    def looks_like_answer(line):
        line = line.strip()
        if kind == "colour":
            return bool(_HEX.search(line)) or len(re.findall(r"\b\d{1,3}\b", line)) == 3
        return "|" in line or bool(re.search(r"\b(" + "|".join(map(re.escape, SPINNER_LANGUAGES + ["Himba"])) + r")\s*:", line, re.I))
    expected = rows.drop_duplicates("hex").shape[0] if kind == "name" else len(rows)
    if not found or len(found) < expected and len(found) < 2:
        plain = [re.sub(r"^\s*([-*•]|\d+\s*[.):\-])\s*", "", l).strip() for l in text.splitlines() if looks_like_answer(l)]
        if plain:
            found = dict(enumerate(plain, 1))
    if not found:
        raise ValueError("Couldn't find the answers in the AI's reply.")
    rows = rows.reset_index(drop=True)
    clean = lambda v: re.sub(r"[*_`]", "", v).strip().strip('"').strip()
    if kind == "colour":
        return _answers_table([clean(found.get(n, "")) for n in range(1, len(rows) + 1)], rows, kind)
    colours = rows.drop_duplicates("hex")
    langs = list(dict.fromkeys(rows.language))
    values = {}
    numbers = list(colours.sdg.astype(int)) if "sdg" in colours else list(range(1, len(colours) + 1))
    if not set(numbers) <= set(found) and len(found) == len(numbers):
        # The AI renumbered the lines (1, 2, 3, … instead of 1, 6, 12, …): match them in order.
        found = dict(zip(numbers, [found[k] for k in sorted(found)]))
    for n, hex_ in zip(numbers, colours.hex):
        parts = [p for p in re.split(r"\s*[|;]\s*", found.get(n, "")) if p.strip()]
        for k, part in enumerate(parts):
            m = re.match(r"\s*([^:–—=]+?)\s*(?::|\s[–—-]\s|=)\s*(.+)$", part)
            lang = _match_language(clean(m.group(1)), langs) if m else None
            name = m.group(2) if lang else part
            if not lang:                              # no label we recognise: take the answers in the prompt's order
                lang = langs[k] if k < len(langs) else None
            if lang and (hex_, lang) not in values:
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
<div id="{uid}m" style="color:{MUTED};font-size:12.5px;margin-top:6px"></div>
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
        show("<b>1.</b> Open your AI with a button (or <b>Copy prompt</b> for another AI) and send the prompt.<br>"
             "<b>2.</b> Paste the AI's <b>whole reply</b> into the box below, in one go, then run the next cell.")
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

def spelling_variant(ai, name, romanized=None):
    """True when the AI's name differs from the speakers' by a single letter (e.g. Buru / Burou): possibly the same
    word spelt differently. Short words are left out, where one letter often makes a different word."""
    if not isinstance(ai, str) or not isinstance(name, str):
        return False
    a = norm(ai)
    for b in {norm(name), norm(romanized) if isinstance(romanized, str) else ""} - {""}:
        if a == b or min(len(a), len(b)) < 4 or abs(len(a) - len(b)) > 1:
            continue
        prev = list(range(len(b) + 1))                       # edit distance, stopping once it passes 1
        for i, ca in enumerate(a, 1):
            cur = [i] + [0] * len(b)
            for j, cb in enumerate(b, 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            prev = cur
        if prev[-1] == 1:
            return True
    return False

def _match_language(label, langs):
    """The language among `langs` that the AI meant by `label`: 'British English', 'English (UK)', 'UK English',
    'English', 'Himba', 'Ελληνικά', 'Srpski', … Returns None if it can't tell."""
    exact = next((l for l in langs if norm(l) == norm(canonical_language(label))), None)
    if exact:
        return exact
    t = " ".join(re.sub(r"[^\w]+", " ", str(label).casefold()).split())
    words = set(t.split())
    keys = {"OtjiHimba": ("himba", "herero"), "Greek": ("greek", "ελληνικ"), "Spanish": ("spanish", "español", "espanol", "castellano"),
            "French": ("french", "français", "francais"), "Serbian": ("serbian", "srpski", "српск")}
    for lang, ks in keys.items():
        if lang in langs and any(k in t for k in ks):
            return lang
    if "english" in t or "en" in words:
        if words & {"us", "american", "usa"} and "American English" in langs:
            return "American English"
        if words & {"uk", "british", "gb", "britain"} and "British English" in langs:
            return "British English"
        english = [l for l in langs if "English" in l]
        if len(english) == 1:
            return english[0]
    return None

def _variants(cmp):
    return [(not ag) and spelling_variant(ai, n, rom) for ai, n, rom, ag in
            zip(cmp.ai_name, cmp["name"], cmp.get("romanized", pd.Series(None, index=cmp.index)), cmp.agrees)]

def show_name_comparison(cmp, threshold=10, you=None):
    """One row per test colour, one column per language: the speakers' name, the AI's name, and
    for names that differ, where the AI's name points."""
    langs = list(dict.fromkeys(cmp.language))
    cmp = cmp.assign(variant=_variants(cmp))
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
            elif r.variant:
                ai = pill(f"AI: {_h(r.ai_name)} (spelling variant?)", "mid")
            else:
                where = ""
                if isinstance(r.get("ai_best"), str):
                    where = f' {swatch(r.ai_best, 22, 14)} {small(f"ΔE₀₀ {r.ai_drift:.1f}")}'
                elif r.get("ai_status"):
                    where = " " + small(_h(r.ai_status))
                ai = f'{pill("AI: " + _h(r.ai_name), "bad")}<div style="margin-top:4px">{where}</div>'
            cells.append(f'{spk}<div style="margin-top:6px">{ai}</div>')
        rows.append([framed(hex_, str(sdg))] + cells)
    head = ["Your test palette"] + [(f"<b style='color:{INK}'>{_h(l)}</b> (you)" if l == you else _h(l)) for l in langs]
    table(head, rows, valign="top", note=
          "Each cell: the speakers' top name and its confidence, then the AI's answer. For an AI name that differs: where "
          "speakers centre it (the Spinner's Mapper) and its ΔE₀₀ from the test colour. Small: a plausible neighbour; "
          "large: a different colour to speakers. Amber: the AI's name differs by one letter, possibly the same word "
          "spelt differently.")

def plot_agreement_by_language(cmp, languages):
    """How often the AI's name matches the speakers' top name, per language."""
    rows = []
    for l in languages:
        g = cmp[cmp.language == l]
        if g.empty: continue
        var = sum(_variants(g))
        n = len(g) - var                                   # possible spelling variants are left out of the count
        share = g.agrees.sum() / n if n else np.nan
        extra = f" · {var} possible spelling variant{'s' if var > 1 else ''}, not counted" if var else ""
        rows.append([f"<b>{_h(l)}</b>", (f'<b>{share:.0%}</b>' if n else "—") + bar(0 if not n else share, None, 220)
                     + small(f"{int(g.agrees.sum())} of {n} names{extra}")])
    show("<b>How often the AI agrees with speakers</b>")
    table(["Language", "Same top name"], rows)

# ── Part C: names and the AI's colours ────────────────────────────────────
_BY_SDG = None

def load_names(source):
    """Names to test: from partB_results.csv (uploaded in Colab) or typed into the boxes."""
    if source.startswith("Upload"):
        if IN_COLAB:
            from google.colab import files
            up = files.upload()
            partb = pd.read_csv(io.BytesIO(next(iter(up.values()))))
        else:
            partb = pd.read_csv("partB_results.csv")
        global _BY_SDG                                 # every (test colour, language, name), shared names included
        _BY_SDG = partb.dropna(subset=["name"])[["sdg", "language", "name"]].assign(
            language=lambda t: t.language.map(canonical_language)).reset_index(drop=True)
        names = (partb.dropna(subset=["name"]).drop_duplicates(["language", "name"])
                 [[c for c in ["sdg", "hex", "language", "role", "name", "romanized", "best_example"] if c in partb]]
                 .rename(columns={"hex": "sdg_hex", "best_example": "spinner_hex"}))
    else:
        _BY_SDG = None
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
    """The names to test. From Part B: the same grid as Part B's names table, one row per test colour."""
    role = dict(zip(names.language, names.get("role", pd.Series("", index=names.index))))
    langs = list(dict.fromkeys(names.language))
    if _BY_SDG is not None and len(_BY_SDG):
        rom = {(r.language, r["name"]): r.get("romanized") for _, r in names.iterrows()}
        seen, shared, rows = {}, False, []
        for i in dict.fromkeys(_BY_SDG.sdg.astype(int)):
            cells = []
            for l in langs:
                g = _BY_SDG[(_BY_SDG.sdg == i) & (_BY_SDG.language == l)]
                if g.empty:
                    cells.append(small("—")); continue
                n = g["name"].iloc[0]
                r = rom.get((l, n))
                extra = f" {small(_h(r))}" if isinstance(r, str) and r and norm(r) != norm(n) else ""
                again = ""
                if (l, norm(n)) in seen:
                    again = "<br>" + small(f"same name as {seen[(l, norm(n))]}"); shared = True
                else:
                    seen[(l, norm(n))] = i
                cells.append(f"<b>{_h(n)}</b>{extra}{again}")
            rows.append([framed(SDG.hex[i], str(i))] + cells)
        head = ["Your test palette"] + [(f"<b style='color:{INK}'>{_h(l)}</b> (you)" if role.get(l) == "your language" else _h(l))
                                        for l in langs]
        table(head, rows, note="A name used for two test colours is asked about once." if shared else None)
        return
    rows = [[f"<b>{_h(l)}</b>", ", ".join(f"<b>{_with_roman(r)}</b>" for _, r in g.iterrows())]
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
    if len(missing) and not isinstance(reply, pd.DataFrame):     # pasted answers already report what's missing
        show("<b>The AI reply has no colour for:</b> " + ", ".join(f"{_h(r['name'])} ({_h(r.language)})" for _, r in missing.iterrows()))
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
def show_comparison(pal, threshold):
    """For each name: the AI's colour next to the speakers' best example, the ΔE₀₀ between them, and what speakers
    would call the AI's colour (the round trip). From Part B: one row per test colour, like Part B's grids."""
    by = {(r.language, norm(r["name"])): r for _, r in pal.iterrows()}
    def cell(r):
        rom = r.get("romanized")
        head = f'<b>{_h(r["name"])}</b>' + (f" {small(_h(rom))}" if isinstance(rom, str) and rom and norm(rom) != norm(r["name"]) else "")
        ai = swatch(r.ai_hex, 34, 24, "AI's colour") if isinstance(r.ai_hex, str) else small("no answer")
        best = swatch(r.spinner_hex, 34, 24, "Speakers' best example") if isinstance(r.spinner_hex, str) else small("missing")
        pair = (f'<div style="display:flex;gap:8px;align-items:center;margin:5px 0 2px">{ai}{best}</div>'
                f'<div style="display:flex;gap:8px;font-size:10.5px;color:{MUTED}"><span style="width:34px">AI</span><span>Speakers</span></div>')
        d = de_badge(r.de00, threshold, prefix="ΔE₀₀ ") if not pd.isna(r.get("de00", np.nan)) else small("n/a")
        if isinstance(r.get("rt_name"), str):
            ok = same_name(r.rt_name, r["name"], r.get("romanized"))
            rt = (f'<div style="font-size:11.5px;color:{MUTED};white-space:nowrap">The AI\'s colour is called</div>'
                  f'<div style="white-space:nowrap"><b>{_h(r.rt_name)}</b> {small(f"{r.rt_conf:.0%}")}</div>'
                  + (pill("same name", "good") if ok else pill("different name", "bad")))
        else:
            rt = small("not looked up")
        return f'{head}{pair}<div style="margin:5px 0">{d}</div>{rt}'
    langs = list(dict.fromkeys(pal.language))
    if _BY_SDG is not None and len(_BY_SDG):
        role = dict(zip(pal.language, pal.get("role", pd.Series("", index=pal.index))))
        rows = []
        for i in dict.fromkeys(_BY_SDG.sdg.astype(int)):
            cells = []
            for l in langs:
                g = _BY_SDG[(_BY_SDG.sdg == i) & (_BY_SDG.language == l)]
                r = by.get((l, norm(g["name"].iloc[0]))) if not g.empty else None
                cells.append(cell(r) if r is not None else small("—"))
            rows.append([framed(SDG.hex[i], str(i))] + cells)
        head = ["Your test palette"] + [(f"<b style='color:{INK}'>{_h(l)}</b> (you)" if role.get(l) == "your language" else _h(l))
                                        for l in langs]
        table(head, rows, valign="top")
    else:
        rows = [[f"<b>{_h(r.language)}</b>", cell(r)] for _, r in pal.iterrows()]
        table(["Language", "Name"], rows, valign="top")
    show(f'<div class="note">Each cell: the AI\'s colour next to the speakers\' best example, and the ΔE₀₀ between them '
         f'(green below {threshold / 2:g}, amber up to {threshold:g}, red above). Then the round trip: what speakers would call '
         f'<b>the AI\'s colour</b> (the left swatch); a different name means the AI\'s colour says something else.</div>')

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
    """One panel per test colour (or per language, for names typed in by hand): for each name, the AI's colour
    (outlined shape) joined by a thin line to the speakers' best example (filled shape), one shape per language."""
    ok = pal.dropna(subset=["ai_hex", "spinner_hex"])
    if ok.empty: return
    by_sdg = "sdg" in ok and ok.sdg.notna().all()
    keys = list(dict.fromkeys(ok.sdg.astype(int))) if by_sdg else [l for l in languages if l in set(ok.language)]
    groups = []
    for k in keys:
        g = ok[ok.sdg == k] if by_sdg else ok[ok.language == k]
        pts = [rgb_to_lab(h)[1:] for h in pd.concat([g.ai_hex, g.spinner_hex])]
        centre = rgb_to_lab(SDG.hex[k])[1:] if by_sdg else np.mean(pts, axis=0)
        groups.append((f"Names for test colour {k}" if by_sdg else k, centre, pts + [centre]))
    langs = [l for l in languages if l in set(ok.language)]
    fig, axes = _panel_grid(groups, "From the AI's colour to the speakers' best example",
                            "CIELAB a*–b* plane, seen from above (lightness not shown). Outlined shape: the AI's colour. "
                            "Filled shape: the speakers' best example. The longer the line, the further the AI is from what "
                            "speakers mean. Every panel has the same scale; an arrow at the edge means the AI's colour lies further out.",
                            [("AI's colour (outlined)", "o", "white"), ("Speakers' best example (filled)", "o", "#5E5E5E")]
                            + [(l, MARKERS.get(l, "o"), "white") for l in langs])
    for ax, (k, (_, centre, _)) in zip(axes, zip(keys, groups)):
        g = ok[ok.sdg == k] if by_sdg else ok[ok.language == k]
        for _, r in g.iterrows():
            a, s_ = rgb_to_lab(r.ai_hex), rgb_to_lab(r.spinner_hex); m = MARKERS.get(r.language, "o")
            pa, a_out = _to_window(ax, a[1:], centre); ps, s_out = _to_window(ax, s_[1:], centre)
            if a_out and not s_out:                    # the AI's colour lies outside the panel
                _edge_arrow(ax, ps, pa)
            else:
                ax.plot([pa[0], ps[0]], [pa[1], ps[1]], color="#1C1C1C", lw=0.7, zorder=1)
            ax.scatter(*pa, s=_marker_size(m, 120), marker=m, facecolor="white", edgecolor=r.ai_hex, lw=2, zorder=3)
            ax.scatter(*ps, s=_marker_size(m, 160), marker=m, color=r.spinner_hex, edgecolor="white", lw=0.9,
                       zorder=4.5 if m == "*" else 4)
    plt.show()

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
    table(["Language", "Mean ΔE₀₀", "Worst ΔE₀₀", f"Above {threshold:g}"], rows)

# ── Part C: grounding ─────────────────────────────────────────────────────
def show_grounded(pal, languages):
    """The grounded palette. From Part B: one row per test colour, like the other grids; otherwise per language."""
    def status(r):
        return (pill("repaired", "mid") if isinstance(r.get("repaired_from"), str) else
                pill("replaced", "mid") if r.replaced else pill("kept", "good"))
    def cell(r):
        ai = swatch(r.ai_hex, 26, 16, "AI's colour") if isinstance(r.ai_hex, str) else small("—")
        return (f'{framed(r.grounded_hex, r["name"], 112, 54, 13)}'
                f'<div style="margin-top:6px">{status(r)} {small("AI:")} {ai}</div>')
    ok = pal[pal.grounded_hex.notna()]
    note = (f"{int(ok.replaced.sum())} of {len(ok)} colours differ from the AI's. kept: the AI's colour matched · "
            "replaced: the speakers' best example · repaired: another colour of the same name, so every viewer can tell "
            "the colours apart. The small swatch is the AI's original colour.")
    if _BY_SDG is not None and len(_BY_SDG):
        by = {(r.language, norm(r["name"])): r for _, r in ok.iterrows()}
        role = dict(zip(pal.language, pal.get("role", pd.Series("", index=pal.index))))
        langs = [l for l in languages if l in set(ok.language)]
        rows = []
        for i in dict.fromkeys(_BY_SDG.sdg.astype(int)):
            cells = []
            for l in langs:
                g = _BY_SDG[(_BY_SDG.sdg == i) & (_BY_SDG.language == l)]
                r = by.get((l, norm(g["name"].iloc[0]))) if not g.empty else None
                cells.append(cell(r) if r is not None else small("—"))
            rows.append([framed(SDG.hex[i], str(i))] + cells)
        head = ["Your test palette"] + [(f"<b style='color:{INK}'>{_h(l)}</b> (you)" if role.get(l) == "your language" else _h(l))
                                        for l in langs]
        show("<b>Your grounded palette</b>")
        table(head, rows, valign="top", note=note)
        return
    rows = []
    for lang in languages:
        p = ok[ok.language == lang]
        if p.empty: continue
        cells = "".join(f'<div style="display:inline-block;margin:0 16px 10px 0;vertical-align:top">{cell(r)}</div>'
                        for _, r in p.iterrows())
        rows.append([f"<b>{_h(lang)}</b>", cells])
    show("<b>Your grounded palette</b>")
    table(["Language", "Grounded palette"], rows, note=note)

def repair_palettes(pal, spinner, languages, threshold, candidates=None):
    """Where a grounded palette has two colours closer than `threshold` (ΔE₀₀) under any viewing
    condition, swap one of them for another colour speakers still call by the same name (the next
    tiles in the Spinner's Mapper: all of them, unless `candidates` limits it), choosing the swap that best
    separates the palette."""
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
                tried = [m for m in dict.fromkeys(members) if m != pal.at[k, "grounded_hex"]]   # every tile the Mapper shows
                if candidates:
                    tried = tried[:candidates]
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
        return                      # nothing was repaired; the check underneath shows the palettes are fine
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
    table(["Language", "Pair too close", "Colour changed", "Before → after", "Closest ΔE₀₀"], rows,
          "Each replacement is another colour speakers call by the same name, taken from the Colour Spinner's Mapper "
          "(its tiles run from the most to the least typical), chosen to separate the palette most.")

@needs_simulations()
def closest_pairs(pal, languages, conditions=CONDITIONS):
    """For each language: the smallest ΔE₀₀ between any two of its colours, AI vs grounded."""
    from itertools import combinations
    def min_pair(hexes, c):
        hs = [simulate(h, c) for h in hexes if isinstance(h, str)]
        return min((de00(a, b) for a, b in combinations(hs, 2)), default=np.nan)
    prefetch_simulations([h for h in pd.concat([pal.ai_hex, pal.grounded_hex]) if isinstance(h, str)])
    rows = [{"language": l, "palette": v, **{c: min_pair(pal[pal.language == l][col].tolist(), c)
                                              for c in conditions}}
            for l in languages for v, col in [("AI", "ai_hex"), ("Grounded", "grounded_hex")]]
    return pd.DataFrame(rows).set_index(["language", "palette"])

@needs_simulations()
def show_closest_pairs(pal, languages, threshold=DELTA_E_THRESHOLD, conditions=CONDITIONS):
    """For each language's AI and grounded palette: its two closest colours, under the viewing condition where they
    are closest, as they look there, with the ΔE₀₀ between the simulated colours."""
    from itertools import combinations
    prefetch_simulations([h for h in pd.concat([pal.ai_hex, pal.grounded_hex]) if isinstance(h, str)])
    rows = []
    for lang in languages:
        p = pal[pal.language == lang]
        cells = []
        for label, col in [("AI palette", "ai_hex"), ("Grounded palette", "grounded_hex")]:
            hexes = [h for h in p[col] if isinstance(h, str)]
            if len(hexes) < 2:
                cells.append(small("fewer than two colours")); continue
            d, c, a, b = min((de00(simulate(x, cc), simulate(y, cc)), cc, x, y)
                             for cc in conditions for x, y in combinations(hexes, 2))
            sw = (f'<div style="display:flex;gap:4px;margin-bottom:4px">{swatch(simulate(a, c), 34, 24)}'
                  f'{swatch(simulate(b, c), 34, 24)}</div>')
            cells.append(f'{sw}{pill(f"{d:.1f}", "bad" if d < threshold else "good")} {small(_h(c))}')
        rows.append([f"<b>{_h(lang)}</b>"] + cells)
    show("<b>Can every viewer still tell each palette's colours apart?</b>")
    table(["Language", "AI palette", "Grounded palette"], rows, valign="top",
          note=f"For each palette: its two closest colours under the viewing condition where they are closest, as they "
               f"look there, and the ΔE₀₀ between the simulated colours. Red: below {threshold:g}.")

def plot_closest_pairs(check, threshold):
    if check is None: return
    below = int((check.values < threshold).sum())
    _cell_grid(check, threshold, [f"{l} · {v}" for l, v in check.index],
               "Can every viewer still tell each palette's colours apart?",
               f"Smallest ΔE₀₀ between any two colours in the palette. Red: closer than {threshold} "
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

def palette_card(pal, languages, path=None, title="Grounded palette", display_it=True):
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
    plt.show() if display_it else plt.close(fig)

def _explorer_data(n):
    """The same synthetic data as the palette explorer (seeded generator), so the previews match it."""
    state = [7]
    def R():
        state[0] = (state[0] * 1664525 + 1013904223) & 0xFFFFFFFF
        v = state[0] - (1 << 32) if state[0] >= (1 << 31) else state[0]
        return (v & 0xFFFFFFFF) / 4294967296
    MAXN = 40
    bar = [round(R() * 55 + 35) for _ in range(MAXN)]
    line = [[round(R() * 55 + 25) for _ in range(8)] for _ in range(MAXN)]
    def randn():
        u = v = 0.0
        while u == 0: u = R()
        while v == 0: v = R()
        return np.sqrt(-2.0 * np.log(u)) * np.cos(2.0 * np.pi * v)
    centres = []
    for _ in range(MAXN):
        tries = 0
        while True:
            cx, cy = R() * 70 + 15, R() * 70 + 15; tries += 1
            if tries >= 12 or not any(np.hypot(c[0] - cx, c[1] - cy) < 16 for c in centres): break
        centres.append((cx, cy))
    clamp = lambda v: min(98, max(2, round(v)))
    scat = [[(clamp(cx + randn() * 6), clamp(cy + randn() * 6)) for _ in range(8)] for cx, cy in centres]
    return bar[:n], line[:n], scat[:n]

APPLICATIONS = ["Colour swatches", "Bar chart", "Donut chart", "Line chart", "Scatter plot", "Map (world)", "Map (US states)"]

# The world map: Natural Earth countries at 1:110m (public domain), from vega-datasets, fetched when first needed.
WORLD_URLS = ["https://cdn.jsdelivr.net/npm/vega-datasets@2/data/world-110m.json",
              "https://cdn.jsdelivr.net/npm/vega-datasets@1.29.0/data/world-110m.json"]
_WORLD = None
MAP_CREDITS = {
    "Map (world)": "World map: Natural Earth 1:110m countries (public domain), from vega-datasets; values are example data.",
    "Map (US states)": "US states map and example values: vega-datasets (us-10m, US Census Bureau; unemployment, US Bureau of "
                       "Labor Statistics), as used in the Altair/Vega-Lite examples and our palette explorer.",
}

def _equal_earth(lon, lat):
    """The Equal Earth projection (Šavrič, Patterson & Jenny 2018), in projection units."""
    A1, A2, A3, A4, M = 1.340264, -0.081106, 0.000893, 0.003796, np.sqrt(3) / 2
    lam, phi = np.radians(lon), np.radians(lat)
    th = np.arcsin(M * np.sin(phi))
    x = 2 * np.sqrt(3) * lam * np.cos(th) / (3 * (9 * A4 * th ** 8 + 7 * A3 * th ** 6 + 3 * A2 * th ** 2 + A1))
    y = A4 * th ** 9 + A3 * th ** 7 + A2 * th ** 3 + A1 * th
    return x, y

def _decode_topojson(topo, name):
    """Outer rings (lon, lat) of every geometry in a TopoJSON object, with each geometry's id."""
    sc, tr = topo["transform"]["scale"], topo["transform"]["translate"]
    arcs = []
    for arc in topo["arcs"]:
        a = np.cumsum(np.asarray(arc, float), axis=0)
        arcs.append(np.column_stack([a[:, 0] * sc[0] + tr[0], a[:, 1] * sc[1] + tr[1]]))
    def ring(idx):
        parts = [arcs[i] if i >= 0 else arcs[~i][::-1] for i in idx]
        return np.concatenate([parts[0]] + [p[1:] for p in parts[1:]])
    out = []
    for g in topo["objects"][name]["geometries"]:
        if g.get("type") == "Polygon": polys = [g["arcs"]]
        elif g.get("type") == "MultiPolygon": polys = g["arcs"]
        else: continue
        out.append((g.get("id"), [ring(p[0]) for p in polys]))
    return out

def _world():
    """The world's countries, projected (Equal Earth), or None if the map can't be downloaded."""
    global _WORLD
    if _WORLD is None:
        import urllib.request
        for url in WORLD_URLS:
            try:
                with urllib.request.urlopen(url, timeout=15) as f:
                    topo = json.load(f)
                break
            except Exception:
                topo = None
        if topo is None:
            return None
        countries = []
        for cid, rings in _decode_topojson(topo, "countries"):
            if str(cid) in ("10", "010"):          # Antarctica: left out, as on most thematic world maps
                continue
            proj = []
            for r in rings:
                lon = np.degrees(np.unwrap(np.radians(r[:, 0])))   # keep rings that cross 180° in one piece
                for shift in (0, -360, 360):                        # …and draw their other side too (clipped to the frame)
                    if shift == 0 or (shift == -360 and lon.max() > 180) or (shift == 360 and lon.min() < -180):
                        proj.append(np.column_stack(_equal_earth(np.clip(lon + shift, -270, 270), r[:, 1])))
            countries.append((cid, proj))
        _WORLD = countries
    return _WORLD

# The US states map from the palette explorer (Albers USA projection, with its example values per state),
# compressed so the toolkit needs no extra files or downloads.
_US_STATES = None
_US_STATES_B64 = (
    "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"
    "S2ROkw=="
)

def _us_states():
    global _US_STATES
    if _US_STATES is None:
        import zlib, base64
        _US_STATES = json.loads(zlib.decompress(base64.b64decode(_US_STATES_B64)))
    return _US_STATES

def _draw_application(ax, kind, cols, names, data):
    bar, line, scat = data
    grid = "#EDEDEA"
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0, labelsize=8.5, colors=MUTED)
    n = len(cols)
    if kind == "Colour swatches":
        ax.set_xlim(0, n); ax.set_ylim(1.35, 0); ax.axis("off")
        for k, (c, nm) in enumerate(zip(cols, names)):
            ax.add_patch(mpl.patches.FancyBboxPatch((k + 0.06, 0.05), 0.88, 1.15, boxstyle="round,pad=0,rounding_size=0.06",
                         fc="white", ec=(0, 0, 0, 0.12), lw=0.6, mutation_aspect=1.2))
            ax.add_patch(mpl.patches.FancyBboxPatch((k + 0.06, 0.05), 0.88, 0.82, boxstyle="round,pad=0,rounding_size=0.06",
                         fc=c, ec="none", mutation_aspect=1.2))
            ax.text(k + 0.5, 1.04, nm, ha="center", va="center", fontsize=14 if n <= 4 else 12 if n <= 6 else 10,
                    color=INK, fontweight="bold")
    elif kind == "Bar chart":
        for k, (c, v) in enumerate(zip(cols, bar)):
            ax.add_patch(mpl.patches.FancyBboxPatch((k - 0.36, 0), 0.72, v, boxstyle="round,pad=0,rounding_size=0.08",
                         fc=c, ec="none", mutation_aspect=60))
        pad = (max(n, 4) - n) / 2                    # at least four slots, so few bars stay bar-shaped, centred
        ax.set_xlim(-0.7 - pad, n - 0.3 + pad); ax.set_ylim(0, 100)
        ax.set_xticks(range(n), [chr(65 + k) for k in range(n)]); ax.set_yticks(range(0, 101, 20))
        ax.yaxis.grid(True, color=grid, lw=0.8, alpha=1); ax.set_axisbelow(True)
    elif kind == "Donut chart":
        ax.pie(bar, colors=cols, startangle=90, counterclock=False,
               wedgeprops=dict(width=0.48, edgecolor="white", linewidth=2.5))
        ax.set_aspect("equal")
    elif kind == "Line chart":
        x = np.arange(8); xs = np.linspace(0, 7, 120)
        for c, ys in zip(cols, line):
            ys = np.asarray(ys, float)
            # Catmull-Rom smoothing, like the explorer's curved lines
            pts = np.concatenate([[ys[0]], ys, [ys[-1]]])
            sm = []
            for t in xs:
                i = min(int(t), 6); u = t - i
                p0, p1, p2, p3 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
                sm.append(0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
            ax.fill_between(xs, sm, 0, color=c, alpha=0.035, lw=0)
            ax.plot(xs, sm, color=c, lw=2.2); ax.scatter(x, ys, s=12, color=c, zorder=3)
        ax.set_xticks(x, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"])
        ax.grid(True, color=grid, lw=0.8, alpha=1); ax.set_axisbelow(True); ax.set_ylim(0, 100)
    elif kind == "Scatter plot":
        for c, pts in zip(cols, scat):
            xs_, ys_ = zip(*pts)
            ax.scatter(xs_, ys_, s=46, color=c, alpha=0.73, edgecolor=c, lw=1)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.grid(True, color=grid, lw=0.8, alpha=1); ax.set_axisbelow(True)
    elif kind == "Map (world)":
        world = _world()
        if world is None:
            ax.text(0.5, 0.5, "The world map couldn't be downloaded.\nChoose Map (US states) instead.", ha="center",
                    va="center", transform=ax.transAxes, fontsize=10, color=MUTED); ax.axis("off"); return
        lat = np.linspace(-90, 90, 181)
        bx, by = _equal_earth(np.r_[np.full(181, -180.0), np.full(181, 180.0)], np.r_[lat, lat[::-1]])
        frame = mpl.patches.Polygon(np.column_stack([bx, by]), closed=True, fc="#F4F4F2", ec="#D6D6D2", lw=0.6, zorder=0)
        ax.add_patch(frame)
        rng = np.random.default_rng(7)                  # example values, like the other charts
        vals = rng.random(len(world))
        cuts = np.quantile(vals, np.linspace(0, 1, n + 1)[1:-1])
        for (cid, rings), v in zip(world, vals):
            k = int(np.searchsorted(cuts, v))
            for r in rings:
                pch = mpl.patches.Polygon(r, closed=True, fc=cols[min(k, n - 1)], ec="white", lw=0.25)
                ax.add_patch(pch); pch.set_clip_path(frame)
        ax.set_xlim(bx.min() * 1.02, bx.max() * 1.02); ax.set_ylim(by.min() * 1.05, by.max() * 1.05)
        ax.set_aspect("equal"); ax.axis("off")
    elif kind == "Map (US states)":
        states = _us_states()
        rates = [st["rate"] for st in states if st["rate"] is not None]
        cuts = np.quantile(rates, np.linspace(0, 1, n + 1)[1:-1])      # one class of states per palette colour
        for st in states:
            k = int(np.searchsorted(cuts, st["rate"])) if st["rate"] is not None else 0
            for ring_ in st["rings"]:
                ax.add_patch(mpl.patches.Polygon(ring_, closed=True, fc=cols[min(k, n - 1)], ec="white", lw=0.4))
        ax.autoscale(); ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")

def preview_applications(pal, language, vision="Typical", path=None, kind="Bar chart"):
    """The grounded palette for one language applied to a chart, map or swatches, in the style of the
    palette explorer, with typical vision and under `vision` side by side."""
    language = _available_language(pal, language, "PreviewLanguage")
    p = pal[(pal.language == language) & pal.grounded_hex.notna()]
    if p.empty:
        show(f"<b>No grounded palette for {_h(language)}.</b> {small('Choose one of: ' + _h(', '.join(dict.fromkeys(pal.language))))}")
        return
    names, hexes = list(p["name"]), list(p.grounded_hex)
    data = _explorer_data(len(hexes))
    if kind == "Map (world)" and _world() is None:          # no download: the US states map, built in, instead
        show(small("The world map couldn't be downloaded, so here is the US states map instead."))
        kind = "Map (US states)"
    conds = ["Typical"] + ([vision] if vision != "Typical" else [])
    fig, axes = plt.subplots(1, len(conds), figsize=(6.2 * len(conds), 4.3), squeeze=False)
    fig.patch.set_facecolor("white")
    for ax, cond in zip(axes[0], conds):
        cols = [simulate(h, cond) for h in hexes]
        _draw_application(ax, kind, cols, names, data)
        ax.set_title(condition_label(cond, " ") if cond != "Typical" else "Typical colour vision", fontsize=11, loc="left", pad=10)
        if kind != "Colour swatches":          # a legend of small squares, as in the explorer
            handles = [mpl.patches.Patch(color=c, label=nm) for c, nm in zip(cols, names)]
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1 if kind not in ("Donut chart", "Map (US states)", "Map (world)") else -0.02),
                      ncol=min(len(names), 4), frameon=False, fontsize=9, handlelength=0.8, handleheight=0.8, labelcolor=MUTED)
    what = {"Colour swatches": "as swatches", "Map (US states)": "applied to a map of the US states",
            "Map (world)": "applied to a world map"}.get(kind, f"applied to a {kind.lower()}")
    fig.suptitle(f"The {language} grounded palette, {what}", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    # the maps' sources are credited in the notebook's credits cell, at the end
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

def corrected_sdg_card(pal, languages, path):
    """A picture of the UN's 17 colours, then one row per language with the test colours corrected (marked)."""
    if _BY_SDG is None or not len(_BY_SDG):
        return False
    grounded = {(r.language, norm(r["name"])): r.grounded_hex for _, r in pal.iterrows() if isinstance(r.grounded_hex, str)}
    rows = [("The UN's original palette", dict(SDG.hex), set())]
    for lang in languages:
        corr, marked = dict(SDG.hex), set()
        for _, r in _BY_SDG[_BY_SDG.language == lang].iterrows():
            h = grounded.get((lang, norm(r["name"])))
            if isinstance(h, str):
                corr[int(r.sdg)] = h
                if de00(h, SDG.hex[int(r.sdg)]) >= 0.5: marked.add(int(r.sdg))
        rows.append((f"Grounded for {lang}", corr, marked))
    W, Hrow = 17 * 0.62 + 0.3, 1.05
    fig, ax = plt.subplots(figsize=(W, Hrow * len(rows) + 0.7)); ax.axis("off")
    ax.set_xlim(0, W); ax.set_ylim(Hrow * len(rows) + 0.5, -0.5)
    ax.text(0.15, -0.25, "Your grounded colours in the full SDG palette", fontsize=13, fontweight="bold", color=INK)
    for k, (label, colours, marked) in enumerate(rows):
        y = k * Hrow + 0.25
        ax.text(0.15, y, label, fontsize=9.5, color=MUTED, va="center")
        for c, (i, h) in enumerate(colours.items()):
            x = 0.15 + c * 0.62
            ax.add_patch(mpl.patches.FancyBboxPatch((x, y + 0.15), 0.54, 0.5, boxstyle="round,pad=0,rounding_size=0.04",
                         fc=SURROUND, ec="none"))
            ax.add_patch(mpl.patches.Rectangle((x + 0.05, y + 0.2), 0.44, 0.4, fc=h, ec="none"))
            ax.text(x + 0.08, y + 0.56, str(i), fontsize=7.5, fontweight="bold", color=_ink_on(h), va="bottom")
            if i in marked:
                ax.add_patch(mpl.patches.Rectangle((x, y + 0.7), 0.54, 0.05, fc=INK, ec="none"))
    ax.text(0.15, Hrow * len(rows) + 0.35, "Marked: changed from the UN colour. Only the test colours were grounded; the others "
            "are the UN's. The SDG goals also have numbers, icons and names: an example, not a proposal to change the UN's colours.",
            fontsize=8, color=MUTED)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return True

def export_palette(pal, languages, stem="grounded_palette"):
    """Pictures of the palettes and the data behind them, zipped for one download."""
    import zipfile, glob
    cols = ["language", "name", "grounded_hex", "ai_hex", "spinner_hex", "de00", "rt_name", "rt_conf",
            "replaced", "repaired_from", "sdg", "sdg_hex"]
    export = pal[[c for c in cols if c in pal.columns]]
    files, rows = [], []
    palette_card(pal, languages, path=f"{stem}_card.png", display_it=False)
    files.append(f"{stem}_card.png"); rows.append([f"<code>{stem}_card.png</code>", "Your grounded palette for each language, as a picture"])
    if corrected_sdg_card(pal, languages, "grounded_sdg_palette.png"):
        files.append("grounded_sdg_palette.png")
        rows.append(["<code>grounded_sdg_palette.png</code>", "The full SDG palette with your grounded colours for each audience"])
    previews = sorted(glob.glob("preview_*.png"))
    if previews:
        files += previews; rows.append(["<code>preview_….png</code>", "The designs you previewed in section 5"])
    export.round(2).to_csv(f"{stem}.csv", index=False)
    files.append(f"{stem}.csv"); rows.append([f"<code>{stem}.csv</code>", "Every colour's hex code, with the evidence for any change"])
    with zipfile.ZipFile(f"{stem}.zip", "w") as z:
        for fn in files:
            z.write(fn)
    show("<b>In the download</b>")
    table(["File", "What it is"], rows)
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
    def sw(hex_, label, cap):
        return (f'<div style="flex:1"><div style="background:{hex_};color:{ink_on(hex_)};height:92px;border-radius:5px;'
                f'display:flex;align-items:flex-end;padding:8px 10px;font-weight:700;font-size:16px">{label}</div>'
                f'<div style="color:#fff;font-size:12px;margin-top:6px">{cap} · {hex_}</div></div>')
    show(f"""<div class="card" style="max-width:640px">
<h3>“{esc(row['name'])}” in {esc(row.language)}</h3>
<div style="display:flex;gap:18px;align-items:center;margin:12px 0">
  <div style="display:flex;gap:12px;flex:1;background:{SURROUND};padding:12px;border-radius:8px">
    {sw(row.ai_hex, "AI", "estimate")}{sw(row.spinner_hex, "Speakers", "best example")}
  </div>
  <div style="color:{MUTED};font-size:14px">ΔE₀₀<br><b style="font-size:34px;color:{INK}">{row.de00:.1f}</b></div>
</div>
{rt_line}
<p style="margin:0 0 8px">Talk it through with your partner, or think it through if you're working on your own:</p>
<p style="margin:0 0 4px"><b>What pattern of failure does this reveal?</b> Did the AI default to a Western-centric focal colour,
collapse a distinction the language maintains, or produce a colour that fails for colour vision deficiency or ageing?</p>
<p style="margin:0"><b>What would a designer need to do differently</b> to avoid it in their own practice?</p>
</div>""")


def safe(fn, *args, **kwargs):
    """fn(*args), or None if it fails (e.g. a name with no best example)."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None

def _finding_row(pal, name=""):
    cand = pal.dropna(subset=["de00"])
    if cand.empty:
        return None
    match = cand[cand.name.map(norm) == norm(name)] if name else cand.iloc[:0]
    return match.iloc[0] if len(match) else cand.loc[cand.de00.idxmax()]

def save_finding(pal, name="", pattern="", designer_fix="", stem="my_finding"):
    """Save the finding as a picture card (PNG) and as data (JSON), zipped for download, so it can be
    shown in the plenary or collected afterwards."""
    import zipfile, textwrap, datetime
    row = _finding_row(pal, name)
    if row is None:
        show("There's no finding to save yet: run the comparison in section 3 first."); return
    data = {"name": row["name"], "language": row.language, "ai_hex": row.ai_hex, "speakers_best_example": row.spinner_hex,
            "dE00": round(float(row.de00), 1),
            "speakers_call_ai_colour": row.get("rt_name") if isinstance(row.get("rt_name"), str) else None,
            "rt_confidence": None if pd.isna(row.get("rt_conf", np.nan)) else round(float(row.rt_conf), 2),
            "test_colour_sdg": None if pd.isna(row.get("sdg", np.nan)) else int(row.sdg),
            "pattern_of_failure": pattern.strip() or None, "what_a_designer_should_do": designer_fix.strip() or None,
            "saved": datetime.datetime.now().isoformat(timespec="minutes")}
    with open(f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    notes = [(lab, textwrap.wrap(t.strip(), 88)) for lab, t in
             [("Pattern of failure", pattern), ("What a designer should do", designer_fix)] if t.strip()]
    H = 3.4 + (0.4 if data["speakers_call_ai_colour"] else 0) + sum(0.38 + 0.3 * len(w) for _, w in notes) + 0.5
    fig, ax = plt.subplots(figsize=(8, H)); ax.set_xlim(0, 8); ax.set_ylim(H, 0); ax.axis("off")
    ax.text(0.3, 0.45, f"“{row['name']}” in {row.language}", fontsize=16, fontweight="bold", va="center")
    ax.add_patch(mpl.patches.FancyBboxPatch((0.3, 0.85), 5.6, 1.65, boxstyle="round,pad=0,rounding_size=0.08", fc=SURROUND, ec="none"))
    for k, (hex_, lab) in enumerate([(row.ai_hex, "AI system"), (row.spinner_hex, "Speakers' best example")]):
        x = 0.5 + k * 2.0
        ax.add_patch(mpl.patches.FancyBboxPatch((x, 1.0), 1.75, 1.0, boxstyle="round,pad=0,rounding_size=0.05", fc=hex_, ec="none"))
        ax.text(x + 0.1, 1.85, "AI" if k == 0 else "Speakers", fontsize=11, fontweight="bold", color=_ink_on(hex_))
        ax.text(x, 2.3, f"{lab} · {hex_}", fontsize=8.5, color="white")
    ax.text(4.65, 1.3, "ΔE₀₀", fontsize=10, color="white"); ax.text(4.65, 1.95, f"{row.de00:.1f}", fontsize=24, fontweight="bold", color="white")
    y = 2.95
    if data["speakers_call_ai_colour"]:
        ax.text(0.3, y, f"Speakers would call the AI's colour “{data['speakers_call_ai_colour']}”"
                        f" ({data['rt_confidence']:.0%} confidence).", fontsize=10, va="top"); y += 0.45
    for lab, lines in notes:
        ax.text(0.3, y, lab, fontsize=10, fontweight="bold", va="top"); y += 0.34
        ax.text(0.3, y, "\n".join(lines), fontsize=10, va="top", linespacing=1.35); y += 0.3 * len(lines) + 0.12
    ax.text(0.3, H - 0.2, "Colour Brings People Together · BCS HCI 2026 · data: Colour Spinner (colornaming.net)",
            fontsize=7.5, color=MUTED)
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    with zipfile.ZipFile(f"{stem}.zip", "w") as z:
        z.write(f"{stem}.png"); z.write(f"{stem}.json")
    show(f"<b>Saved your finding</b> {small(f'{stem}.png (a card for the plenary) and {stem}.json (the data), in {stem}.zip.')}")
    download(f"{stem}.zip")

# ── The optional questionnaire at the end of the masterclass ──────────────
QUESTIONNAIRE_URL = "https://forms.cloud.microsoft/r/bQ2e8faHsx"          # the Microsoft Forms link; the QR code for the slides comes from Forms itself

def show_questionnaire(url=None):
    """A button that opens the optional, anonymous questionnaire in a new tab."""
    url = url or QUESTIONNAIRE_URL
    if not url:
        show(f"<b>The questionnaire link will appear here.</b> {small('Ask the facilitators for the link or scan the QR code on the screen.')}")
        return
    show(f'''<div style="border:1px solid #E4E4E0;border-radius:10px;padding:16px 18px;max-width:620px">
<div style="font-weight:700;font-size:15px;margin-bottom:6px">Before you go: a short questionnaire (optional)</div>
<div style="color:{MUTED};font-size:13px;margin-bottom:12px">About 10 minutes and anonymous. Please read the participant information sheet at the start of the form.
Taking part is voluntary and separate from the masterclass.</div>
<a href="{url}" target="_blank" rel="noopener" style="display:inline-block;padding:8px 16px;border-radius:6px;background:{INK};
color:#FFFFFF;font-weight:700;font-size:13px;text-decoration:none">Open the questionnaire ↗</a>
<div style="margin-top:10px">{small(_h(url))}</div></div>''')

# ── Part C: the full SDG palette with the test colours corrected ──────────
@needs_simulations()
def show_corrected_sdg(pal, language, vision="Typical", threshold=DELTA_E_THRESHOLD):
    """The UN's 17 colours next to the same palette with the test colours replaced by their grounded colours
    for one language, under a chosen kind of vision. Arcs under the corrected row point to the pairs that the
    correction brought too close, so they can be checked again."""
    if "sdg" not in pal or pal.sdg.isna().all():
        show("This step needs the names from Part B (uploaded <code>partB_results.csv</code>), because it has to know "
             "which SDG colour each name belongs to.")
        return
    language = _available_language(pal, language, "CorrectedFor")
    p = pal[(pal.language == language) & pal.grounded_hex.notna() & pal.sdg.notna()]
    if p.empty:
        show(f"<b>No grounded colours for {_h(language)}.</b> " + small("Choose one of: " + _h(", ".join(dict.fromkeys(pal.language)))))
        return
    original = dict(SDG.hex)
    corrected, named = dict(original), {}
    pairs = (_BY_SDG[_BY_SDG.language == language] if _BY_SDG is not None else p[["sdg", "name"]])
    grounded = {norm(r["name"]): r.grounded_hex for _, r in p.iterrows()}
    for _, r in pairs.iterrows():
        h = grounded.get(norm(r["name"]))
        if isinstance(h, str):
            corrected[int(r.sdg)] = h; named[int(r.sdg)] = r["name"]
    changed = sorted(i for i in named if de00(original[i], corrected[i]) >= 0.5)
    prefetch_simulations(list(original.values()) + list(corrected.values()))
    sim = lambda h: simulate(h, vision)
    # what the correction changed for the whole palette: pairs to check again
    arcs, notes = [], []
    tests = sorted(named)
    for a_ in range(len(tests)):
        for b_ in range(a_ + 1, len(tests)):
            i, j = tests[a_], tests[b_]
            d = de00(sim(corrected[i]), sim(corrected[j]))
            if d < threshold and de00(sim(original[i]), sim(original[j])) >= threshold:
                why = (f"both are now <i>{_h(named[i])}</i>" if norm(named[i]) == norm(named[j]) else "they are now close")
                notes.append(f"<b>{i} and {j}</b>: {why}, ΔE₀₀ {d:.1f} apart, though the originals were distinct.")
                arcs.append((i, j, d, False))
    near = {}
    for i in tests:
        for j, h in original.items():
            if j in named: continue
            d = de00(sim(corrected[i]), sim(h))
            if d < threshold and de00(sim(original[i]), sim(h)) >= threshold:      # new, not inherited
                near.setdefault((corrected[i], j), []).append((i, d))
                arcs.append((i, j, d, True))
    for (hex_, j), hits in near.items():
        ids = " and ".join(str(i) for i, _ in hits)
        verb = "is" if len(hits) == 1 else "are"
        notes.append(f"<b>{ids}</b> (<i>{_h(named[hits[0][0]])}</i>) {verb} now ΔE₀₀ {hits[0][1]:.1f} from the unchanged SDG colour <b>{j}</b>.")
    old, old_notes, seen = [], [], set()          # already close in the UN palette, and still close now
    for i in tests:
        for j in original:
            if j == i or tuple(sorted((i, j))) in seen: continue
            d_now = de00(sim(corrected[i]), sim(corrected[j]))
            if d_now < threshold and de00(sim(original[i]), sim(original[j])) < threshold:
                seen.add(tuple(sorted((i, j))))
                old.append((i, j, d_now, False))
                old_notes.append(f"{min(i, j)} and {max(i, j)} (ΔE₀₀ {d_now:.1f})")
    TILE, GAP = 50, 3                                   # framed tile width (40 + frame) and the gap between tiles
    def row(colours, label, mark=()):
        tiles = ""
        for i, h in colours.items():
            dot = (f'<div style="height:4px;margin-top:3px;border-radius:2px;background:{INK}"></div>' if i in mark
                   else '<div style="height:4px;margin-top:3px"></div>')
            tiles += (f'<div style="display:inline-block;margin:0 {GAP}px 0 0;vertical-align:top;text-align:center">'
                      f'{framed(sim(h), str(i), 40, 36, 10.5)}{dot}</div>')
        return f'<div style="margin:6px 0 4px"><div style="font-size:12px;color:{MUTED};margin-bottom:4px">{label}</div><div style="white-space:nowrap">{tiles}</div></div>'
    def arcs_svg():
        allarcs = [(i, j, d, arrow, True) for i, j, d, arrow in arcs] + [(i, j, d, arrow, False) for i, j, d, arrow in old]
        if not allarcs: return ""
        x = lambda i: (i - 1) * (TILE + GAP) + TILE / 2
        parts = []
        for k, (i, j, d, arrow, new) in enumerate(sorted(allarcs, key=lambda t: (not t[4], abs(t[0] - t[1])))):
            x1, x2 = x(i), x(j)
            depth = 14 + 9 * k                          # each arc a little deeper, so none overlap
            mid = (x1 + x2) / 2
            style = ('stroke="#B4342D" stroke-width="1.7"' if new else
                     'stroke="#8A8A85" stroke-width="1.2" stroke-dasharray="4 3"')
            parts.append(f'<path d="M{x1:.0f},2 Q{mid:.0f},{depth * 2:.0f} {x2:.0f},2" fill="none" {style}'
                         f'{" marker-end=\"url(#ah)\"" if arrow and new else ""}/>')
        H = 14 + 9 * len(allarcs) + 8
        width = 17 * (TILE + GAP)
        return (f'<svg width="{width}" height="{H}" style="display:block;margin:0 0 6px">'
                f'<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">'
                f'<path d="M0,0 L8,4 L0,8 z" fill="#B4342D"/></marker></defs>{"".join(parts)}</svg>')
    view = "typical colour vision" if vision == "Typical" else condition_label(vision, " ").lower()
    html = (row(original, "The UN's original palette")
            + row(corrected, f"Your test colours grounded for {_h(language)} (marked: changed from the UN colour; the others are unchanged)", changed)
            + arcs_svg())
    show(f"<b>Your test colours, grounded for {_h(language)}, in the full SDG palette</b> {small('as seen with ' + _h(view))}" + html)
    body = ""
    if notes:
        body += (f'<div style="margin-top:4px"><b style="color:#B4342D">New, made by grounding</b> '
                 f'{small("(red arcs; an arrow points from a grounded colour to the unchanged colour it is now close to)")}</div>'
                 + "".join(f'<div style="font-size:13px;margin:3px 0">{n}</div>' for n in notes))
    if old_notes:
        body += (f'<div style="margin-top:8px"><b style="color:#5E5E5E">Already close in the UN palette</b> '
                 f'{small("(grey dashed arcs; still below ΔE₀₀ " + format(threshold, "g") + ", so worth checking too)")}</div>'
                 f'<div style="font-size:13px;margin:3px 0">{", ".join(old_notes)}</div>')
    if body:
        show(f"<b>Check these again</b>, as seen with {_h(view)}" + body +
             f'<div class="note">Grounding fixes what each name means; it can also merge colours that share a name, or '
             f'move a colour close to one you didn\'t test. Problems that were already in the UN palette stay unless '
             f'those colours are changed too.</div>')
    else:
        show(f"<b>No problems.</b> " + small(f"No colour pair involving your test colours is closer than ΔE₀₀ {threshold:g}, "
                                             f"as seen with {_h(view)}."))

def spinner_link():
    """The Colour Spinner logo as a link, in a cell's output: links there open directly, without the
    'redirect notice' page that links in Colab's text cells can show."""
    show(f'<a href="https://colornaming.net/spinner" target="_blank" rel="noopener" style="display:inline-block">'
         f'<img src="https://raw.githubusercontent.com/akvilesinkeviciute/colour-brings-people-together/v1.0/images/colour_spinner_logo.png" '
         f'alt="Open the Colour Spinner" width="240"></a><div style="font-size:12px;color:{MUTED};margin-top:2px">'
         f'Click to open the Colour Spinner in a new tab.</div>')


# ── Menus that list only what the pair chose, and redraw straight away ─────
VISIONS = ["Typical", "Protanopia", "Deuteranopia", "Tritanopia", "Age 40", "Age 60", "Age 80"]

def live_controls(controls, render):
    """Dropdown menus built when the cell runs (so they list only the pair's own languages), with the output
    redrawn whenever a choice changes. controls: {key: (label, options, default)}; render(**choices)."""
    try:
        import ipywidgets as w
    except ImportError:
        render(**{k: d for k, (_, _, d) in controls.items()}); return
    menus = {k: w.Dropdown(options=list(opts), value=d if d in opts else list(opts)[0], description=label,
                           style={"description_width": "initial"}, layout=w.Layout(width="auto", margin="0 18px 0 0"))
             for k, (label, opts, d) in controls.items()}
    out = w.Output()
    def draw(*_):
        out.clear_output(wait=True)
        with out:
            render(**{k: m.value for k, m in menus.items()})
    for m in menus.values():
        m.observe(draw, names="value")
    try:
        display(w.HBox(list(menus.values())), out)
    except Exception:
        pass
    draw()

PART_C_URL = ("https://colab.research.google.com/github/akvilesinkeviciute/colour-brings-people-together/"
              "blob/v1.0/Part_C_Ground_an_AI_palette.ipynb")

def open_part_c():
    """A button that opens Part C in a new tab (in a cell's output, so it opens directly)."""
    show(f'<a href="{PART_C_URL}" target="_blank" rel="noopener" style="display:inline-block;padding:8px 16px;border-radius:6px;'
         f'background:{INK};color:#FFFFFF;font-weight:700;font-size:13px;text-decoration:none;margin-top:6px">Open Part C ↗</a>'
         f'<div style="font-size:12px;color:{MUTED};margin-top:6px">Part C opens in a new tab; upload <code>partB_results.csv</code> there.</div>')
