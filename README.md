# Colour Brings People Together

Notebooks for the masterclass **Colour Brings People Together: Cross-Cultural Colour Communication for Inclusive Design** at BCS HCI 2026, run by [Dimitris Mylonas](https://www.linkedin.com/in/dimitrismylonas/) and [Akvile Sinkeviciute](https://www.linkedin.com/in/akvile-sinkeviciute-265069152/) (Northeastern University London).

You'll use the [Colour Spinner](https://colornaming.net/spinner) to test how well a real palette, the 17 colours of the UN Sustainable Development Goals, communicates across languages, colour vision deficiencies and ageing. You'll then compare an AI system's colour names and colours with how speakers actually use them.

## Open the notebooks

| Part | What you'll do | Time | |
|---|---|---|---|
| **B** | Audit the UN SDG palette on two axes: can people tell the colours apart, and do they agree on what they're called? | 45 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akvilesinkeviciute/colour-brings-people-together/blob/v1.0/Part_B_Audit_the_SDG_palette.ipynb) |
| **C** | Ask an AI for the colour of each name, check it against speakers' data, and build a palette grounded in evidence | 45 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akvilesinkeviciute/colour-brings-people-together/blob/v1.0/Part_C_Ground_an_AI_palette.ipynb) |

Part A, the *Colours of Babel* card game, needs no laptop.

## What you need

- A laptop with a modern web browser. A phone works for the Colour Spinner, but the notebooks are easier on a laptop.
- A Google account to run the notebooks in Google Colab. Without one, you can do every activity in the [Colour Spinner](https://colornaming.net/spinner) in your browser.
- No installation and no prior knowledge of colour science or Python.

## How to use the notebooks

1. Click **Open in Colab** for the part you're doing.
2. Choose **File ▸ Save a copy in Drive**, so your changes are kept.
3. Run each cell in order with **Shift + Enter**. You only need to change the grey form fields on the right of some cells.
4. At the end of Part B, download `partB_results.csv`; Part C starts from it.

For the AI steps, use the AI you use day to day. The notebook shows the prompt with **Copy prompt** and buttons for ChatGPT, Claude and Gemini; each button copies the prompt and opens the tool. You then paste the AI's whole reply into one box, and the notebook reads the answers itself.

## What's in this repository

| File | Contents |
|---|---|
| `Part_B_Audit_the_SDG_palette.ipynb` | Part B notebook |
| `Part_C_Ground_an_AI_palette.ipynb` | Part C notebook |
| `colour_masterclass.py` | The shared toolkit the notebooks load: colour science through the [colour-science](https://www.colour-science.org) library (CIELAB and CIEDE2000), the Colour Spinner connection, AI helpers, charts and tables |
| `images/` | The SDG wheel (from the [UN SDG communications materials](https://www.un.org/sustainabledevelopment/news/communications-material/)) and the Colour Spinner logo |
| `FACILITATORS.md` | Notes for running the session, fallbacks, and methods and their limits |
| `data/` | `spinner_similarity.json`: the Colour Spinner's name-similarity matrices (the data behind its Thesaurus), used by the scorecard |
| `source/` | The toolkit's source (`toolkit.py`, `helpers.py`) and the script that builds `colour_masterclass.py` from them |

Educators are welcome to reuse the activities in their own teaching.

## References

- Sinkeviciute, A., Ahmed, R., Stutters, J., Koliousis, A. and Mylonas, D. (2026). Colour Spinner: Making cross-linguistic colour naming diversity visible. Under review.
- Mylonas, D., Ahmed, R., Sinkeviciute, A. and Koliousis, A. (2026). Colour in translation: Data, models, and benchmarking for cross-linguistic colour naming. *CHI '26*. https://doi.org/10.1145/3772318.3791626
- United Nations (2015). Sustainable Development Goals. https://sdgs.un.org/goals

---
*The SDG wheel is reproduced from the United Nations' [communications materials](https://www.un.org/sustainabledevelopment/news/communications-material/); see [un.org/sustainabledevelopment](https://www.un.org/sustainabledevelopment). The content of this publication has not been approved by the United Nations and does not reflect the views of the United Nations or its officials or Member States.*
