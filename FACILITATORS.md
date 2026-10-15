# Facilitator guide

Notes for running the masterclass notebooks. Participants start from [README.md](README.md).

## Before the session

### 1. How the notebooks talk to the Colour Spinner

The Colour Spinner has no separate API. The notebooks make the same requests as its web page, read from the browser's Network panel in October 2026:

| Request | Used for | Status |
|---|---|---|
| `POST /spinner/color_picker` with `r`, `g`, `b`, `language-select`, `translation-select` | Namer name and confidence, Mapper best example of that name, Translator result | Reply format checked against a saved response |
| `POST /spinner/simulate` with JSON `{r, g, b, simulation_type}` | Colour vision deficiency and age simulations (`protanopia`, `deuteranopia`, `tritanopia`, `16-40`, `40-60`, `60-80`) | Request format seen; reply read flexibly (24-byte JSON, not yet seen) |
| `GET /spinner/mobile_panels?language_id=…&colour_id=…` | Best example of a name the Spinner hasn't already returned: the AI's names in Part B, section 6, and names typed in by hand in Part C | Reply format not yet seen; the notebook searches it for colours |

All of this lives in `SPINNER_API` and `ApiSpinner` in `colour_masterclass.py`. If the site changes, only those need updating.

**Two assumptions to check:**
- Languages are numbered in the order of the Spinner's language menu: 1 American English, 2 British English, 3 French, 4 Greek, 5 OtjiHimba ("Himba" in the menu), 6 Spanish. The saved reply confirms 4 is Greek.
- These are the website's internal requests, not a published API. Confirm with the Colour Spinner team that 10–25 people using them at once is fine, and run both notebooks on the venue network beforehand.

### 2. Hosting

Everything lives in the public repository [akvilesinkeviciute/colour-brings-people-together](https://github.com/akvilesinkeviciute/colour-brings-people-together).

- The notebooks load `colour_masterclass.py` from the **`v1.0` release**, so the links work only once that release exists (Releases ▸ Create a new release ▸ tag `v1.0`).
- The "Open in Colab" buttons in `README.md` also point at `v1.0`, so everyone gets the same version.
- To fix something later, edit the files, publish a `v1.1` release, and change `v1.0` to `v1.1` in `TOOLKIT_URL` (first cell of each notebook, or `source/nbtools.py` before rebuilding) and in the README buttons. Copies people already saved keep working on `v1.0`.

If GitHub can't be reached, participants can upload `colour_masterclass.py` in Colab's Files panel (folder icon, left) and rerun the first cell, which skips the download when the file is already there.

Participants should use **File ▸ Save a copy in Drive** so their edits persist.

### 3. Run through once on the venue network

Run both notebooks on the venue network if you can. In Part B, the *Ask the Colour Spinner* cell should show the multilingual names table without the "Some lookups did not work" box.

## Fallbacks during the session

| Problem | What participants do |
|---|---|
| Spinner API unreachable | Set `BACKEND` to *Copied from the browser*, then fill in the `SHEET` table from the Spinner's Namer and Mapper panels. Mapper RGB can be typed exactly as shown, e.g. `232 21 56`. |
| Spinner simulations unreachable | Sections 3 and 5 of Part B and the re-check in Part C show a message. Once the connection is back, run `retry_simulations()` and rerun those cells. Meanwhile, use the Spinner's "Simulate age/CVD" menu in the browser. |
| Colab's built-in Gemini busy or unavailable | Choose *Paste the reply from my own AI*: the notebook shows the prompt to copy into any AI chat and a box to paste the reply into. Surrounding chatter and code fences are ignored. |
| Lost `partB_results.csv` | In Part C, choose *Type them in* and list `language, name` rows. |
| No Google account | Do the whole audit in the Spinner's browser interface. The notebooks are optional enrichment, as stated in the proposal. |

### The AI system

The notebooks demonstrate with the Gemini models built into Colab (`google.colab.ai`), for convenience rather than as a recommendation. Participants are told they can use any AI they regularly use instead. Anyone signed in to Colab can use them free, with no API key or set-up. Leave `AI_MODEL` empty for Colab's default model. `from google.colab import ai; ai.list_models()` lists the others.

Ways to use a different AI system:
- **Any chat tool:** choose *Paste the reply from my own AI*. The notebook shows the prompt and a box to paste the reply into; lines, tables and JSON are all read.
- **Their own API key:** choose *Anthropic* or *OpenAI-compatible*, add `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` under Colab ▸ Secrets (the key icon), and enter a model name.

Colab limits how much free AI use each person gets. Two short prompts per pair should fit comfortably, but test it on a free account before the session.

## Methods and their limits

**Colour science.** All colour conversions and colour differences use the [colour-science](https://www.colour-science.org) library: `colour.sRGB_to_XYZ`, `colour.XYZ_to_Lab`, `colour.delta_E(..., method="CIE 2000")` and `colour.XYZ_to_CAM16UCS` with its default sRGB viewing conditions. Colab doesn't include the library, so the toolkit installs it the first time it loads, which adds about 15 seconds to the first cell. On loading, the toolkit also checks three reference values (sRGB red in CIELAB, Sharma et al.'s first CIEDE2000 test pair, and the CAM16-UCS lightness of white). If a future version of the library changes its units, participants get a clear message instead of wrong results.

**Thresholds.** The default ΔE00 threshold of 10 is a teaching heuristic for "reads as a different colour at a glance", adjustable with a slider. The maps use CAM16-UCS (Li et al. 2017), the space the Colour Spinner works in.

**Colour vision deficiency and ageing.** Every simulation comes from the Colour Spinner's own models, the ones behind its "Simulate age/CVD" menu, so the notebooks match what participants see in the browser. The toolkit contains no simulation models of its own. Protanopia, deuteranopia and tritanopia use linear transformation matrices (Viénot, Brettel and Mollon 1999). Ageing is lens yellowing at ages 40, 60 and 80 (Pokorny, Smith and Lutze 1987): the colour is matched to the nearest of 606 measured spectral chips and integrated against age-adjusted cone fundamentals in CIE LMS 2006. The menu sends the ages as the codes `16-40`, `40-60` and `60-80`. As the Colour Spinner paper notes, these simulations show how colours appear; the naming models were trained on observers with typical colour vision, so they don't show how people with CVD or older eyes name colours.

Each notebook asks for the simulations of all its colours at once, six requests at a time, and caches them. Part B needs about 100 small requests per participant, and Part C a few dozen.

If the Spinner's simulations can't be reached, the perceptual cells show a message instead of results, and the rest of the notebook still runs. Once the connection is back, run `retry_simulations()` and rerun those cells. Meanwhile, participants can use the "Simulate age/CVD" menu in the browser.

**SDG colours.** The SDG hex values are the commonly published ones from the UN communications guidelines. Verify them against the current guidelines.

**Comparing the AI with speakers.** Part B checks each AI name against the speakers' top name, and for names that differ, asks the Mapper where the AI's name is centred and how far that is from the SDG colour. Part C sends each AI colour back to the Namer (a round trip) to see whether speakers would call it by the intended name. By default, a colour is replaced if it is too far from the name's best example or speakers would call it something else; the form in section 4 switches between the two rules.

**Naming confidence.** The Namer's confidence is the probability of the top-ranked term in the Spin Colour Forest naming distribution, read from a pre-classified grid of 7,072 CAM16-UCS points. It is not the Profiler's *Agreement* measure, and the notebooks call it confidence throughout. The best example of a name is its focal colour, the Mapper's top-left tile.

**Name matching.** Names are compared ignoring case, accents, punctuation and Greek final sigma, so Κόκκινο matches κοκκινο. They are also compared across scripts. The Spinner gives a Latin spelling for each name (e.g. *kokkino*), and an AI answer counts as the same name if it gives the native word, that spelling, or both, as in "κόκκινο (kokkino)". Synonyms such as *Sky blue* and *Light blue* still count as different names, which is the point of the comparison.

## Reproducing or editing the notebooks

The notebooks are generated from the Python scripts in `source/`:

- `toolkit.py` and `helpers.py`: the two halves of `colour_masterclass.py` (colour maths and connections; charts and tables)
- `nb_part_b.py` and `nb_part_c.py`: the cells of each notebook
- `nbtools.py` and `build_all.py`: build helpers

To rebuild the notebooks and `colour_masterclass.py`, run `python build_all.py` inside `source/`, then move the files it writes up to the top level of the repository. You can also edit `colour_masterclass.py` and the notebooks directly.
