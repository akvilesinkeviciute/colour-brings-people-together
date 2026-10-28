# Facilitator guide

Notes for running the masterclass notebooks. Participants start from [README.md](README.md).

## Before the session

### 1. How the notebooks talk to the Colour Spinner

The Colour Spinner has no separate API. The notebooks make the same requests as its web page, read from the browser's Network panel in October 2026:

| Request | Used for | Status |
|---|---|---|
| `POST /spinner/color_picker` with `r`, `g`, `b`, `language-select`, `translation-select` | Namer name and confidence, Mapper best example of that name, Translator result | Reply format checked against a saved response |
| `POST /spinner/simulate` with JSON `{r, g, b, simulation_type}` | Colour vision deficiency and age simulations (`protanopia`, `deuteranopia`, `tritanopia`, `16-40`, `40-60`, `60-80`) | Request format seen; reply read flexibly (24-byte JSON, not yet seen) |
| `GET /spinner/mobile_panels?language_id=…&colour_id=…` (JSON: `swatches`, the Mapper's tiles with the best example first; `thesaurus`; `features`) | Best example of a name the Spinner hasn't already returned: the AI's names in Part B, section 7, and names typed in by hand in Part C | Reply format not yet seen; the notebook searches it for colours |

All of this lives in `SPINNER_API` and `ApiSpinner` in `colour_masterclass.py`. If the site changes, only those need updating.

**Two assumptions to check:**
- Languages are numbered in the order of the Spinner's language menu: 1 American English, 2 British English, 3 French, 4 Greek, 5 OtjiHimba ("Himba" in the menu), 6 Serbian, 7 Spanish (checked in the menu on 26 October 2026; the numbers follow the menu's alphabetical order, so they can shift when a language is added). The saved reply confirms 4 is Greek.
- These are the website's internal requests, not a published API. Confirm with the Colour Spinner team that 10–25 people using them at once is fine, and run both notebooks on the venue network beforehand.

### 2. Hosting

Everything lives in the public repository [akvilesinkeviciute/colour-brings-people-together](https://github.com/akvilesinkeviciute/colour-brings-people-together).

- The notebooks load `colour_masterclass.py` from the **`v1.0` release**, so the links work only once that release exists (Releases ▸ Create a new release ▸ tag `v1.0`).
- The "Open in Colab" buttons in `README.md` also point at `v1.0`, so everyone gets the same version.
- To fix something later, edit the files, publish a `v1.1` release, and change `v1.0` to `v1.1` in `TOOLKIT_URL` (first cell of each notebook) and in the README buttons. Copies people already saved keep working on `v1.0`.

If GitHub can't be reached, participants can upload `colour_masterclass.py` in Colab's Files panel (folder icon, left) and rerun the first cell, which skips the download when the file is already there.

Participants should use **File ▸ Save a copy in Drive** so their edits persist.

### 3. Run through once on the venue network

Run both notebooks on the venue network if you can. In Part B, the *Ask the Colour Spinner* cell should show the multilingual names table without the "Some lookups did not work" box.

## Fallbacks during the session

| Problem | What participants do |
|---|---|
| Spinner API unreachable | `Backend` is no longer a form field. Ask participants to click **Show code** in Part B's *Your choices* cell and in Part C's *Load your names* cell, and change `Backend = "Colour Spinner API"` to `Backend = "Copied from the browser"`. The cell in Part B section 5 then first shows a row of boxes per test colour and language to fill in from the Spinner in the browser, and runs when run again. |
| Best examples in the notebook differ from the Spinner's page | Check the Spinner's language menu: it opens in American English. Each language has its own best examples (British English *Blue* is #3431B8, American English *Blue* #4155CF). |
| Spinner simulations unreachable | Sections 3 and 5 of Part B and the re-check in Part C show a message. Once the connection is back, run `retry_simulations()` and rerun those cells. Meanwhile, use the Spinner's "Simulate age/CVD" menu in the browser. |
| Lost `partB_results.csv` | In Part C, choose *Type them in*: participants pick a language and type each colour name into the boxes. |
| No Google account | Do the whole audit in the Spinner's browser interface. The notebooks are optional enrichment, as stated in the proposal. |

### The AI system

Everyone uses the same route, with the AI they use day to day. The notebook shows the prompt with **Copy prompt** and buttons for ChatGPT, Claude and Gemini. Each button copies the prompt as it opens the tool. ChatGPT and Claude accept the prompt in the link and open with it typed in; for Gemini, participants paste it (Cmd/Ctrl + V) if it isn't filled in. Any other AI: copy the prompt and paste it there. The sites decide whether to accept a prompt in a link, so check the buttons shortly before the session. Participants then paste the AI's whole reply into one box; the notebook reads the numbered lines (numbered by SDG goal in Part B, by name in Part C) and says if any answer is missing.

Colab's built-in Gemini was removed from the notebooks: it depends on each person's small free quota, which runs out without warning.

## Methods and their limits

**Colour science.** All colour conversions and colour differences use the [colour-science](https://www.colour-science.org) library: hex colours are converted to CIELAB (D65), and every colour difference is `colour.delta_E(..., method="CIE 2000")`. The two maps show the CIELAB a*–b* plane. Colab doesn't include the library, so the toolkit installs it the first time it loads, which adds about 15 seconds to the first cell. On loading, the toolkit also checks two reference values (#FF0000 in CIELAB, and Sharma et al.'s first CIEDE2000 test pair). If a future version of the library changes its units, participants get a clear message instead of wrong results.

**Thresholds.** The masterclass uses one threshold throughout, with no sliders: ΔE₀₀ 5, a teaching heuristic for "reads as a different colour at a glance" (about 1 is just noticeable side by side). Naming confidence below 50% counts as hard to name. Both are set once in the toolkit (`DELTA_E_THRESHOLD`, `HARD_TO_NAME`).

**Colour vision deficiency and ageing.** Every simulation comes from the Colour Spinner's own models, the ones behind its "Simulate age/CVD" menu, so the notebooks match what participants see in the browser. The toolkit contains no simulation models of its own. Protanopia, deuteranopia and tritanopia use linear transformation matrices (Viénot, Brettel and Mollon 1999). Ageing is lens yellowing at ages 40, 60 and 80 (Pokorny, Smith and Lutze 1987): the colour is matched to the nearest of 606 measured spectral chips and integrated against age-adjusted cone fundamentals. The menu sends the ages as the codes `16-40`, `40-60` and `60-80`. As the Colour Spinner paper notes, these simulations show how colours appear; the naming models were trained on observers with typical colour vision, so they don't show how people with CVD or older eyes name colours.

Each notebook asks for the simulations of all its colours at once, six requests at a time, and caches them. Part B needs about 100 small requests per participant, and Part C a few dozen.

If the Spinner's simulations can't be reached, the perceptual cells show a message instead of results, and the rest of the notebook still runs. Once the connection is back, run `retry_simulations()` and rerun those cells. Meanwhile, participants can use the "Simulate age/CVD" menu in the browser.

**SDG colours.** The SDG hex values are the commonly published ones from the UN communications guidelines. Verify them against the current guidelines.

**Comparing the AI with speakers.** Part B checks each AI name against the speakers' top name, and for names that differ, asks the Mapper where the AI's name is centred and how far that is from the SDG colour. Part C sends each AI colour back to the Namer (a round trip) to see whether speakers would call it by the intended name. By default, a colour is replaced if it is too far from the name's best example or speakers would call it something else; the form in section 4 switches between the two rules.

**Naming confidence.** The Namer's confidence is the probability of the top-ranked term in the Spin Colour Forest naming distribution. It is not the Profiler's *Agreement* measure, and the notebooks call it confidence throughout. The best example of a name is its focal colour, the Mapper's top-left tile.

**Renaming and repair, chosen together.** After grounding, Part C gives each test colour its own entry and, for each language, tries every combination of allowed colours. Each test colour may: keep its grounded colour; move to another tile of its name in the Spinner's Mapper (any within ΔE₀₀ 5 of the name's best example, or further if the Spinner's Namer still gives that name for the tile, so boundary colours aren't used); or, if it shares a name with another test colour, take the best example of one of that name's top 6 synonyms from the Spinner's similarity data. The chosen palette has, in order: no shared names; every pair at least ΔE₀₀ 5 apart for the worst-off viewer (typical vision, three colour vision deficiencies, three ages) if any combination allows it, otherwise the largest such distance; then the fewest renamings and changes, and the most typical colours. Red pairs left in the closest-pairs check therefore mean these names can't separate the colours, so a second cue is needed. If there are too many combinations (over 200,000) or the Spinner doesn't answer, the cell falls back to the step-by-step renaming and repair and says so. In languages with few colour terms, such as OtjiHimba with seven, a renamed colour may get a name that means something rather different; that is a finding to discuss.

**Colour identity scorecard.** Part B section 6 plots one point per pair of test colours (the same pairs as section 4's "Pairs at risk" table), drawn as a disc split into the two colours and labelled e.g. 6–14. Up: the ΔE₀₀ between the two colours, with the chosen kind of vision (the same CIEDE2000 values as the rest of the notebook). Across: name distinctiveness between their two names in the chosen language, 1 minus the Colour Spinner's name similarity: its own JSD-based matrices behind the Thesaurus (checked: American English red's top six match the Thesaurus exactly), published as `data/spinner_similarity.json` (59 kB, all seven languages). 0 means the two share a name. Both axes therefore compare exactly the same two colours, so the pink corner means 'these two look alike and are named alike'. Dashed lines mark ΔE₀₀ 5 and name distinctiveness 0.7 (as alike as a #1 Thesaurus synonym). Each axis is stretched in two straight pieces so its threshold sits in the middle (four equal squares); tick labels show the real values. Points that would overlap are moved sideways only, never across a dashed line; their heights are exact. Language: your language or any single audience language; Vision: typical or one of the six simulated conditions.

**Spinner server errors.** The Colour Spinner's server occasionally returns an error (HTTP 500) for particular colour values, as happened for SDG 9 and 17 on 26 October 2026. The toolkit retries once, then looks up the nearest neighbouring colour (±1–2 in one channel, ΔE₀₀ well below 1) and says so in the output. If even the neighbours fail, the message says the problem is on the Spinner's side; worth reporting to the Spinner team with the hex codes.

**Grounded colours in the full SDG palette.** Part C section 6 shows the UN's 17 colours next to the same palette with the test colours replaced by their grounded colours for one audience language, under a chosen kind of vision. Red arcs mark close pairs (below ΔE₀₀ 5) that the correction created; grey dashed arcs mark close pairs involving a test colour that were already in the UN palette. Only the test colours are grounded. The notebooks also point out that the SDG goals are never identified by colour alone (each has a number, an icon and a name), so the palette is used as an example, not as a case for changing the UN's colours. It needs names from Part B, so it says so when names were typed in by hand.

**Menus that follow the pair's choices.** The scorecard (Part B, section 6), the preview (Part C, section 5) and the grounded colours in the full SDG palette (Part C, section 6) use drop-down menus created when the cell runs, so they list only the languages the pair chose, and the output redraws as soon as a choice changes. They need no rerun.

**Palette previews and export.** Part C shows each language's grounded palette in the style of the palette explorer (swatches, bar, donut, line and scatter charts, or a world map), with typical vision and one chosen condition side by side, for the test colours or all 17 SDG colours. Sections 5 and 6 both start on the pair's own language, as Part B does. The download, `grounded_palette.zip`, follows the screen: `grounded_palette_card.png` and `grounded_palette.csv` are arranged by test colour and language like section 4's grid (test colours that share a name each get their own entry); `grounded_sdg_<language>_<vision>.png` reproduces section 6 for each language with its arcs and notes, for typical vision plus the vision chosen there; `grounded_sdg_palette.png` is an overview of all languages; `grounded_sdg_palette.csv` lists all 17 SDG colours per audience; and every preview made in section 5 is kept, with the language, colours, design and vision in its file name.

**The story.** Participants choose a test palette, *their* language (as the designer) and *their audience's* languages once, in Part B section 3, after seeing the whole palette under each condition in section 2. Everything follows from that: distinctness is measured within the test palette (the full 17-colour grid is only context), translatability compares the audience's focal colours with the designer's, and Part C builds one grounded palette per language.

**SDG wheel.** Part B opens with the UN's SDG wheel (`images/sdg_wheel.png`), from the UN's communications materials; the link and the statement the UN's guidelines require are in the credits cell at the end of Part B. The UN also asks for permission when SDG branding appears on materials for events with registration fees (sdgpermissions@un.org); confirm this applies, or has been granted, before the session.

**Name matching.** Names are compared ignoring case, accents, punctuation and Greek final sigma, so Κόκκινο matches κοκκινο. They are also compared across scripts. The Spinner gives a Latin spelling for each name (e.g. *kokkino*), and an AI answer counts as the same name if it gives the native word, that spelling, or both, as in "κόκκινο (kokkino)". Synonyms such as *Sky blue* and *Light blue* still count as different names, which is the point of the comparison.

## Reproducing or editing the notebooks

The notebooks are edited directly: in Colab, then saved to GitHub with *File ▸ Save a copy in GitHub*. Only the toolkit is generated: `source/toolkit.py` and `source/helpers.py` are combined into `colour_masterclass.py` by running `python build_toolkit.py` inside `source/`.
