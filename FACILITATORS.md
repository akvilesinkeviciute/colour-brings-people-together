# Facilitator guide

Notes for running the masterclass notebooks. Participants start from [README.md](README.md).

## Before the session

### 1. How the notebooks talk to the Colour Spinner

The Colour Spinner has no separate API. The notebooks make the same requests as its web page, read from the browser's Network panel in October 2026:

| Request | Used for | Status |
|---|---|---|
| `POST /spinner/color_picker` with `r`, `g`, `b`, `language-select`, `translation-select` | Namer name and confidence, Mapper best example of that name, Translator result | Reply format checked against a saved response |
| `POST /spinner/simulate` with JSON `{r, g, b, simulation_type}` | Colour vision deficiency and age simulations (`protanopia`, `deuteranopia`, `tritanopia`, `16-40`, `40-60`, `60-80`) | Request format seen; reply read flexibly (24-byte JSON, not yet seen) |
| `GET /spinner/mobile_panels?language_id=…&colour_id=…` | Best example of a name typed in by hand (Part C, *Type them in* only) | Reply format not yet seen; the notebook searches it for colours |

All of this lives in `SPINNER_API` and `ApiSpinner` in `colour_masterclass.py`. If the site changes, only those need updating.

**Two assumptions to check:**
- Languages are numbered in the order of the Spinner's language menu: 1 American English, 2 British English, 3 French, 4 Greek, 5 Himba, 6 Spanish. The saved reply confirms 4 is Greek.
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
| Colab's built-in Gemini busy or unavailable | Choose *Paste the reply myself*, copy the printed prompt into any AI chat and paste the reply into `AI_REPLY`. Surrounding chatter and code fences are ignored. |
| Lost `partB_results.csv` | In Part C, choose *Type them in* and list `language, name` rows. |
| No Google account | Do the whole audit in the Spinner's browser interface. The notebooks are optional enrichment, as stated in the proposal. |

### The AI system

The notebooks demonstrate with the Gemini models built into Colab (`google.colab.ai`), for convenience rather than as a recommendation. Participants are told they can use any AI they regularly use instead. Anyone signed in to Colab can use them free, with no API key or set-up. Leave `AI_MODEL` empty for Colab's default model. `from google.colab import ai; ai.list_models()` lists the others.

Ways to use a different AI system:
- **Any chat tool:** choose *Paste the reply myself*.
- **Their own API key:** choose *Anthropic* or *OpenAI-compatible*, add `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` under Colab ▸ Secrets (the key icon), and enter a model name.

Colab limits how much free AI use each person gets. Two short prompts per pair should fit comfortably, but test it on a free account before the session.

## Methods and their limits

**Colour difference.** Colour difference is CIEDE2000 (ΔE00), checked against Sharma, Wu and Dalal's (2005) test pairs. The default threshold of 10 is a teaching heuristic for "reads as a different colour at a glance". It is adjustable with a slider.

**Colour space.** CAM16-UCS follows Li et al. (2017), with sRGB viewing conditions (L_A = 64/π × 0.2, Y_b = 20, average surround). It reproduces the standard CAM16 worked example. It is used for the maps and for `de_ucs`.

**Colour vision deficiency and ageing.** Both are simulated by the Colour Spinner's own models, the same ones behind its "Simulate age/CVD" menu, so the notebook matches what participants see in the browser. The age options are age bands: the menu's "Age 40" sends `16-40`, "Age 60" sends `40-60` and "Age 80" sends `60-80`. The notebooks show both the label and the band. Check with the Spinner's authors that "Age 40" really means the 16–40 group.

Each notebook asks for the simulations of all colours at once, six requests at a time, and caches them. Part B needs about 100 small requests per participant, and Part C a few dozen.

If the Spinner cannot be reached, the notebook prints a warning and switches to local approximations for the rest of the session, and every note under the simulations says so:
- **Colour vision deficiency:** Machado, Oliveira and Fernandes (2009) at severity 1.0.
- **Ageing:** lens yellowing only, after van de Kraats and van Norren (2007), at ages 40, 60 and 80 rather than the Spinner's bands. Treat these as rough.

**SDG colours.** The SDG hex values are the commonly published ones from the UN communications guidelines. Verify them against the current guidelines.

**Name matching.** Names are compared ignoring case, accents and spacing, so Κόκκινο matches κοκκινο. Synonyms such as *Sky blue* and *Light blue* count as different names, which is the point of the comparison.

## Reproducing or editing the notebooks

The notebooks are generated from the Python scripts in `source/`:

- `toolkit.py` and `helpers.py`: the two halves of `colour_masterclass.py` (colour maths and connections; charts and tables)
- `nb_part_b.py` and `nb_part_c.py`: the cells of each notebook
- `nbtools.py` and `build_all.py`: build helpers

To rebuild the notebooks and `colour_masterclass.py`, run `python build_all.py` inside `source/`, then move the files it writes up to the top level of the repository. You can also edit `colour_masterclass.py` and the notebooks directly.
