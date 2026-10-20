# Facilitator guide

Notes for running the masterclass notebooks. Participants start from [README.md](README.md).

## Before the session

### 1. How the notebooks talk to the Colour Spinner

The Colour Spinner has no separate API. The notebooks make the same requests as its web page, read from the browser's Network panel in October 2026:

| Request | Used for | Status |
|---|---|---|
| `POST /spinner/color_picker` with `r`, `g`, `b`, `language-select`, `translation-select` | Namer name and confidence, Mapper best example of that name, Translator result | Reply format checked against a saved response |
| `POST /spinner/simulate` with JSON `{r, g, b, simulation_type}` | Colour vision deficiency and age simulations (`protanopia`, `deuteranopia`, `tritanopia`, `16-40`, `40-60`, `60-80`) | Request format seen; reply read flexibly (24-byte JSON, not yet seen) |
| `GET /spinner/mobile_panels?language_id=…&colour_id=…` | Best example of a name the Spinner hasn't already returned: the AI's names in Part B, section 7, and names typed in by hand in Part C | Reply format not yet seen; the notebook searches it for colours |

All of this lives in `SPINNER_API` and `ApiSpinner` in `colour_masterclass.py`. If the site changes, only those need updating.

**Two assumptions to check:**
- Languages are numbered in the order of the Spinner's language menu: 1 American English, 2 British English, 3 French, 4 Greek, 5 OtjiHimba ("Himba" in the menu), 6 Spanish. The saved reply confirms 4 is Greek.
- These are the website's internal requests, not a published API. Confirm with the Colour Spinner team that 10–25 people using them at once is fine, and run both notebooks on the venue network beforehand.

### 2. Hosting

Everything lives in the public repository [akvilesinkeviciute/colour-brings-people-together](https://github.com/akvilesinkeviciute/colour-brings-people-together).

- The notebooks load `colour_masterclass.py` from the **`v1.0` release**, so the links work only once that release exists (Releases ▸ Create a new release ▸ tag `v1.0`).
- The "Open in Colab" buttons in `README.md` also point at `v1.0`, so everyone gets the same version.
- To fix something later, edit the files, publish a `v1.1` release, and change `v1.0` to `v1.1` in `ToolkitURL` (first cell of each notebook; `TOOLKIT_URL` in `source/nbtools.py` before rebuilding) and in the README buttons. Copies people already saved keep working on `v1.0`.

If GitHub can't be reached, participants can upload `colour_masterclass.py` in Colab's Files panel (folder icon, left) and rerun the first cell, which skips the download when the file is already there.

Participants should use **File ▸ Save a copy in Drive** so their edits persist.

### 3. Run through once on the venue network

Run both notebooks on the venue network if you can. In Part B, the *Ask the Colour Spinner* cell should show the multilingual names table without the "Some lookups did not work" box.

## Fallbacks during the session

| Problem | What participants do |
|---|---|
| Spinner API unreachable | Set `Backend` to *Copied from the browser*. A row of boxes appears for each test colour and language; participants copy the Namer's name and confidence and the Mapper's best-example RGB from the Spinner in the browser. |
| Spinner simulations unreachable | Sections 3 and 5 of Part B and the re-check in Part C show a message. Once the connection is back, run `retry_simulations()` and rerun those cells. Meanwhile, use the Spinner's "Simulate age/CVD" menu in the browser. |
| Lost `partB_results.csv` | In Part C, choose *Type them in*: participants pick a language and type each colour name into the boxes. |
| No Google account | Do the whole audit in the Spinner's browser interface. The notebooks are optional enrichment, as stated in the proposal. |

### The AI system

Everyone uses the same route, with the AI they use day to day. The notebook shows the prompt with **Copy prompt** and buttons for ChatGPT, Claude and Gemini. Each button copies the prompt as it opens the tool. ChatGPT and Claude accept the prompt in the link and open with it typed in; for Gemini, participants paste it (Cmd/Ctrl + V) if it isn't filled in. Any other AI: copy the prompt and paste it there. The sites decide whether to accept a prompt in a link, so check the buttons shortly before the session. Participants then paste the AI's whole reply into one box; the notebook reads the numbered lines (numbered by SDG goal in Part B, by name in Part C) and says if any answer is missing.

Colab's built-in Gemini was removed from the notebooks: it depends on each person's small free quota, which runs out without warning.

## Methods and their limits

**Colour science.** All colour conversions and colour differences use the [colour-science](https://www.colour-science.org) library: hex colours are converted to CIELAB (D65), and every colour difference is `colour.delta_E(..., method="CIE 2000")`. The two maps show the CIELAB a*–b* plane. Colab doesn't include the library, so the toolkit installs it the first time it loads, which adds about 15 seconds to the first cell. On loading, the toolkit also checks two reference values (#FF0000 in CIELAB, and Sharma et al.'s first CIEDE2000 test pair). If a future version of the library changes its units, participants get a clear message instead of wrong results.

**Thresholds.** The masterclass uses one threshold throughout, with no sliders: ΔE₀₀ 10, a teaching heuristic for "reads as a different colour at a glance" (about 1 is just noticeable side by side). Naming confidence below 50% counts as hard to name. Both are set once in the toolkit (`DELTA_E_THRESHOLD`, `HARD_TO_NAME`).

**Colour vision deficiency and ageing.** Every simulation comes from the Colour Spinner's own models, the ones behind its "Simulate age/CVD" menu, so the notebooks match what participants see in the browser. The toolkit contains no simulation models of its own. Protanopia, deuteranopia and tritanopia use linear transformation matrices (Viénot, Brettel and Mollon 1999). Ageing is lens yellowing at ages 40, 60 and 80 (Pokorny, Smith and Lutze 1987): the colour is matched to the nearest of 606 measured spectral chips and integrated against age-adjusted cone fundamentals. The menu sends the ages as the codes `16-40`, `40-60` and `60-80`. As the Colour Spinner paper notes, these simulations show how colours appear; the naming models were trained on observers with typical colour vision, so they don't show how people with CVD or older eyes name colours.

Each notebook asks for the simulations of all its colours at once, six requests at a time, and caches them. Part B needs about 100 small requests per participant, and Part C a few dozen.

If the Spinner's simulations can't be reached, the perceptual cells show a message instead of results, and the rest of the notebook still runs. Once the connection is back, run `retry_simulations()` and rerun those cells. Meanwhile, participants can use the "Simulate age/CVD" menu in the browser.

**SDG colours.** The SDG hex values are the commonly published ones from the UN communications guidelines. Verify them against the current guidelines.

**Comparing the AI with speakers.** Part B checks each AI name against the speakers' top name, and for names that differ, asks the Mapper where the AI's name is centred and how far that is from the SDG colour. Part C sends each AI colour back to the Namer (a round trip) to see whether speakers would call it by the intended name. By default, a colour is replaced if it is too far from the name's best example or speakers would call it something else; the form in section 4 switches between the two rules.

**Naming confidence.** The Namer's confidence is the probability of the top-ranked term in the Spin Colour Forest naming distribution. It is not the Profiler's *Agreement* measure, and the notebooks call it confidence throughout. The best example of a name is its focal colour, the Mapper's top-left tile.

**Perceptual repair.** After grounding, Part C checks each palette under all seven viewing conditions. Where two colours are closer than the threshold, it swaps one for another colour speakers still call by the same name: one of the next eight tiles of that name in the Spinner's Mapper, which run from the most to the least typical. It chooses the swap that best separates the palette, at most once per colour. If none helps, it says so and suggests a second cue (pattern, label or icon). The export records any repaired colour's previous value.

**Colour identity scorecard.** Part B section 6 plots each test colour's naming confidence against its nearest-colour ΔE₀₀ on fixed axes (0–100% and ΔE₀₀ 0–40), with the thresholds ΔE₀₀ 10 and 50%. Values above 40 sit at the top edge, marked with their value; test colours that land on the same spot (for example, two colours that are each other's nearest neighbour) are set side by side.

**Palette previews and export.** Part C shows each language's grounded palette in a chart, an interface and wayfinding, with typical vision and one chosen condition, and saves each preview as a PNG. The download has a palette card (PNG), swatch files for Adobe apps (`.ase`), GIMP, Inkscape and Krita (`.gpl`) and Figma or design systems (`.tokens.json`), CSS custom properties, and the CSV and JSON data.

**The story.** Participants choose a test palette, *their* language (as the designer) and *their audience's* languages once, in Part B section 3, after seeing the whole palette under each condition in section 2. Everything follows from that: distinctness is measured within the test palette (the full 17-colour grid is only context), translatability compares the audience's focal colours with the designer's, and Part C builds one grounded palette per language.

**SDG poster.** Part B opens with the UN's official SDG poster (`images/sdg_poster.png`), from the UN's communications materials, with the link and the statement the UN's guidelines require. The UN also asks for permission when SDG branding appears on materials for events with registration fees (sdgpermissions@un.org); confirm this applies, or has been granted, before the session.

**Saving findings.** At the end of Part C, pairs type their interpretation into two short fields (the pattern of failure, and what a designer should do) and download `my_finding.zip`: a picture card for the plenary (`my_finding.png`) and the same finding as data (`my_finding.json`), so you can collect them afterwards.

**Name matching.** Names are compared ignoring case, accents, punctuation and Greek final sigma, so Κόκκινο matches κοκκινο. They are also compared across scripts. The Spinner gives a Latin spelling for each name (e.g. *kokkino*), and an AI answer counts as the same name if it gives the native word, that spelling, or both, as in "κόκκινο (kokkino)". Synonyms such as *Sky blue* and *Light blue* still count as different names, which is the point of the comparison.

## Reproducing or editing the notebooks

The notebooks are generated from the Python scripts in `source/`:

- `toolkit.py` and `helpers.py`: the two halves of `colour_masterclass.py` (colour maths and connections; charts and tables)
- `nb_part_b.py` and `nb_part_c.py`: the cells of each notebook
- `nbtools.py` and `build_all.py`: build helpers

To rebuild the notebooks and `colour_masterclass.py`, run `python build_all.py` inside `source/`, then move the files it writes up to the top level of the repository. You can also edit `colour_masterclass.py` and the notebooks directly.
