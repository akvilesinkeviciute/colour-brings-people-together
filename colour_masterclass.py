"""Colour Brings People Together: toolkit for the BCS HCI 2026 masterclass notebooks.

Colour maths (CIEDE2000, CAM16-UCS), the Colour Spinner connection, AI helpers and the
charts and tables used in Parts B and C. Imported by the notebooks with
`from colour_masterclass import *`.
"""
# ════════════════════════════════════════════════════════════════════════
#  Colour Brings People Together — masterclass toolkit
#  Shared by every notebook. Pure numpy/pandas/matplotlib: nothing to install.
# ════════════════════════════════════════════════════════════════════════
import json, re, unicodedata, html as _html, io, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

try:
    from IPython.display import display, HTML, Markdown
except ImportError:                                    # outside Jupyter
    def display(x): print(x)
    HTML = Markdown = str

try:
    import google.colab                                # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ── The six languages currently covered by the Colour Spinner ────────────
SPINNER_LANGUAGES = ["American English", "British English", "French",
                     "Greek", "OtjiHimba", "Spanish"]

# ── UN Sustainable Development Goal palette ──────────────────────────────
#    Hex values from the UN SDG communications guidelines.
SDG = pd.DataFrame([
    ( 1, "No Poverty",                              "#E5243B"),
    ( 2, "Zero Hunger",                             "#DDA63A"),
    ( 3, "Good Health and Well-being",              "#4C9F38"),
    ( 4, "Quality Education",                       "#C5192D"),
    ( 5, "Gender Equality",                         "#FF3A21"),
    ( 6, "Clean Water and Sanitation",              "#26BDE2"),
    ( 7, "Affordable and Clean Energy",             "#FCC30B"),
    ( 8, "Decent Work and Economic Growth",         "#A21942"),
    ( 9, "Industry, Innovation and Infrastructure", "#FD6925"),
    (10, "Reduced Inequalities",                    "#DD1367"),
    (11, "Sustainable Cities and Communities",      "#FD9D24"),
    (12, "Responsible Consumption and Production",  "#BF8B2E"),
    (13, "Climate Action",                          "#3F7E44"),
    (14, "Life Below Water",                        "#0A97D9"),
    (15, "Life on Land",                            "#56C02B"),
    (16, "Peace, Justice and Strong Institutions",  "#00689D"),
    (17, "Partnerships for the Goals",              "#19486A"),
], columns=["sdg", "goal", "hex"]).set_index("sdg")


# ════════════════════════════════════════════════════════════════════════
#  1. Colour science, via the colour-science library (colour.readthedocs.io)
# ════════════════════════════════════════════════════════════════════════
try:
    import colour
except ImportError:                                    # not preinstalled in Colab
    import subprocess, sys
    print("Installing the colour-science library (once per session)…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "colour-science"])
    import colour
warnings.filterwarnings("ignore", module="colour")

#  Colours are passed around as '#RRGGBB' strings or (r, g, b) in 0–255, as the
#  Colour Spinner shows them. colour-science works in 0–1, so these helpers convert.
def hex_to_rgb(h):
    """'#E5243B' -> array([229., 36., 59.])"""
    h = h.strip()
    return np.asarray(colour.notation.HEX_to_RGB(h if h.startswith("#") else "#" + h), float) * 255

def rgb_to_hex(rgb):
    """(229, 36, 59) -> '#E5243B'"""
    rgb = np.clip(np.round(np.asarray(rgb, float)), 0, 255)
    return colour.notation.RGB_to_HEX(rgb / 255).upper()

def _srgb(c):
    """sRGB in 0–1 from '#hex' or (r, g, b) in 0–255."""
    return (hex_to_rgb(c) if isinstance(c, str) else np.asarray(c, float)) / 255

def rgb_to_xyz(c):
    """CIE XYZ (Y in 0–1) of an sRGB colour."""
    return colour.sRGB_to_XYZ(_srgb(c))

def rgb_to_lab(c):
    """CIELAB (D65) of an sRGB colour."""
    return colour.XYZ_to_Lab(rgb_to_xyz(c))

def de00(c1, c2):
    """CIEDE2000 colour difference between two colours given as hex or RGB."""
    return float(colour.delta_E(rgb_to_lab(c1), rgb_to_lab(c2), method="CIE 2000"))

def spread(hexes):
    """Largest ΔE2000 between any two colours in a list (NaN if fewer than two)."""
    from itertools import combinations
    hexes = [h for h in hexes if isinstance(h, str)]
    return max((de00(a, b) for a, b in combinations(hexes, 2)), default=np.nan)

def rgb_to_cam16ucs(c):
    """CAM16-UCS J′a′b′ (Li et al. 2017) under sRGB viewing conditions, the space the
    Colour Spinner works in. J′ runs 0–100."""
    return np.asarray(colour.XYZ_to_CAM16UCS(rgb_to_xyz(c)), float)

def de_ucs(c1, c2):
    """Euclidean distance in CAM16-UCS between two colours (hex or RGB)."""
    return float(np.linalg.norm(rgb_to_cam16ucs(c1) - rgb_to_cam16ucs(c2)))

def _check_colour_library():
    """Guard against unit or API changes in colour-science with known reference values."""
    checks = [
        ("sRGB red in CIELAB", rgb_to_lab("#FF0000"), [53.24, 80.09, 67.20], 0.05),
        ("CIEDE2000, Sharma et al. (2005) pair 1",
         colour.delta_E([50, 2.6772, -79.7751], [50, 0, -82.7485], method="CIE 2000"), 2.0425, 1e-3),
        ("CAM16-UCS lightness of white", rgb_to_cam16ucs("#FFFFFF")[0], 100.0, 0.5),
    ]
    bad = [n for n, got, want, tol in checks if not np.allclose(got, want, atol=tol)]
    if bad:
        raise RuntimeError(f"colour-science {colour.__version__} gave unexpected values for: "
                           f"{', '.join(bad)}. Please tell the organisers.")
_check_colour_library()


# ════════════════════════════════════════════════════════════════════════
#  2. Vision simulations — the perceptual axis
# ════════════════════════════════════════════════════════════════════════
#  All colour vision deficiency and age simulations come from the Colour Spinner
#  (see simulate() in section 3). This section only names the conditions.
CONDITIONS = ["Typical", "Protanopia", "Deuteranopia", "Tritanopia",
              "Age 40", "Age 60", "Age 80"]
# Ageing is lens yellowing at ages 40, 60 and 80 (Sinkeviciute et al. 2026). The Spinner's menu sends
# these as the codes 16-40, 40-60 and 60-80 (see SPINNER_API).
def condition_label(c, sep="\n"):
    return c

# The Spinner's language menu says "Himba"; the language is OtjiHimba.
LANGUAGE_ALIASES = {"himba": "OtjiHimba", "otjihimba": "OtjiHimba"}
def canonical_language(name):
    return LANGUAGE_ALIASES.get(str(name).strip().casefold(), str(name).strip())


# ════════════════════════════════════════════════════════════════════════
#  3. Colour Spinner access
# ════════════════════════════════════════════════════════════════════════
#  Two interchangeable back-ends with the same three methods:
#     name(hex, language)          -> {"name", "confidence"}
#     best_example(name, language) -> "#RRGGBB"   (Mapper, top-left)
#     translate(name, src, tgt)    -> {"name", "confidence"}
#
#  ApiSpinner  talks to colornaming.net/spinner the way its web page does.
#  SheetSpinner reads values you copied from the browser interface at
#               https://colornaming.net/spinner into two small tables.

SPINNER_API = {
    # The Colour Spinner has no separate API: these are the requests its own web
    # page makes (read from the browser's Network panel, October 2026). If the
    # site changes, these are the only lines that should need updating.
    "base_url":    "https://colornaming.net/spinner",
    "color_picker": "/color_picker",     # POST form: r, g, b, language-select, translation-select, …
    "mobile_panels": "/mobile_panels",   # GET ?language_id=…&colour_id=…  (JSON)
    "simulate":    "/simulate",          # POST JSON {r, g, b, simulation_type}
    "simulation_types": {"Protanopia": "protanopia", "Deuteranopia": "deuteranopia",
                         "Tritanopia": "tritanopia", "Age 40": "16-40",
                         "Age 60": "40-60", "Age 80": "60-80"},
    # Language numbers follow the order of the Spinner's language menu.
    "language_ids": {"American English": 1, "British English": 2, "French": 3,
                     "Greek": 4, "OtjiHimba": 5, "Spanish": 6},
}

def _strip(html_text):
    return _html.unescape(re.sub(r"<[^>]+>", "", html_text)).strip()

def parse_color_picker(page):
    """Read the Namer, Mapper and Translator results out of a color_picker response."""
    def grab(pattern):
        m = re.search(pattern, page, re.S)
        return m.group(1) if m else None
    name = grab(r'id="most-likely-color-name"[^>]*>(.*?)</h2>') or grab(r'id="mobile-namer-name"[^>]*>(.*?)</div>')
    conf = grab(r'id="mobile-namer-conf"[^>]*>\s*([\d.]+)\s*%') or grab(r'id="most-likely-confidence"[^>]*>[^\d]*([\d.]+)\s*%')
    rgb = re.findall(r'class="rgb-value"[^>]*>\s*(\d+)\s*<', page)
    tr_name = grab(r'data-translated-name="([^"]*)"')
    tr_title = grab(r'data-viz="translator".*?<h1[^>]*>(.*?)</h1>')
    tr_conf = grab(r'data-translation-confidence="([\d.eE+-]+)"')
    roman = grab(r'id="most-likely-color-name"[^>]*data-romanized-name="([^"]*)"') or \
            grab(r'data-romanized-name="([^"]*)"[^>]*id="most-likely-color-name"')
    tr_roman = grab(r'data-translated-romanized="([^"]*)"')
    options = {_strip(n): int(v) for v, n in re.findall(r'<option value="(\d+)"[^>]*>(.*?)</option>', page)}
    if not name or conf is None:
        raise RuntimeError("The Colour Spinner's reply did not contain a Namer result; the site may have changed.")
    return {
        "name": _strip(name),
        "romanized": _html.unescape(roman) if roman else None,
        "confidence": float(conf) / 100,
        "best_example": rgb_to_hex([int(x) for x in rgb[:3]]) if len(rgb) >= 3 else None,
        "translation": _strip(tr_title) if tr_title else (_html.unescape(tr_name) if tr_name else None),
        "translation_confidence": float(tr_conf) if tr_conf else None,
        "translation_romanized": _html.unescape(tr_roman) if tr_roman else None,
        "name_ids": options,
    }

class ApiSpinner:
    """Talks to the Colour Spinner the same way its web page does."""
    def __init__(self, cfg=SPINNER_API, timeout=30):
        import requests
        self.http = requests.Session()
        self.http.headers.update({"HX-Request": "true",
                                  "User-Agent": "colour-masterclass-notebook (BCS HCI 2026)"})
        self.cfg, self.timeout = cfg, timeout
        self._pages, self._best, self._ids = {}, {}, {}

    def _lang(self, language):
        try:
            return self.cfg["language_ids"][canonical_language(language)]
        except KeyError:
            raise KeyError(f"The Colour Spinner has no language called '{language}'.")

    def lookup(self, hex_, language, target=None):
        """Everything the Spinner shows for one colour: name, confidence, best example, translation."""
        target = target or language
        key = (hex_.upper(), language, target)
        if key not in self._pages:
            r, g, b = (int(v) for v in hex_to_rgb(hex_))
            form = {"language-select": self._lang(language), "translation-select": self._lang(target),
                    "age-cvd-simulation-select": "none", "synonyms": "on", "antonyms": "on",
                    "r": r, "g": g, "b": b, "exact_colour_id": ""}
            resp = self.http.post(self.cfg["base_url"] + self.cfg["color_picker"], data=form,
                                  timeout=self.timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"The Colour Spinner returned {resp.status_code}. Check your internet "
                                   "connection, or set BACKEND to 'Copied from the browser'.")
            res = parse_color_picker(resp.text)
            # The Namer gives a Latin spelling only for some languages. When the Translator is set to
            # the same language it names the same word, so its Latin spelling can stand in.
            if (not res.get("romanized") or norm(res["romanized"]) == norm(res["name"])) and target == language \
                    and res.get("translation_romanized") and same_name(res.get("translation") or "", res["name"]):
                res["romanized"] = res["translation_romanized"]
            self._pages[key] = res
            self._ids.setdefault(language, {}).update({norm(n): i for n, i in res["name_ids"].items()})
            if res["best_example"]:
                self._best[(language, norm(res["name"]))] = res["best_example"]
                if res.get("romanized"):
                    self._best.setdefault((language, norm(res["romanized"])), res["best_example"])
        return self._pages[key]

    def name(self, hex_, language):
        res = self.lookup(hex_, language)
        return {"name": res["name"], "confidence": res["confidence"], "romanized": res.get("romanized")}

    def translate(self, name_or_hex, src, tgt):
        """Translate a colour (given as hex) from one language to another, as the Translator panel does."""
        res = self.lookup(name_or_hex, src, tgt)
        return {"name": res["translation"], "confidence": res["translation_confidence"]}

    def simulate(self, hex_, condition):
        """The colour as the Spinner simulates it under a CVD or age condition."""
        r, g, b = (int(v) for v in hex_to_rgb(hex_))
        resp = self.http.post(self.cfg["base_url"] + self.cfg["simulate"], timeout=self.timeout,
                              json={"r": r, "g": g, "b": b,
                                    "simulation_type": self.cfg["simulation_types"][condition]})
        resp.raise_for_status()
        j = resp.json()
        d = j if isinstance(j, dict) else {}
        for k in ("simulated", "result", "rgb", "colour", "color"):
            if isinstance(d.get(k), dict): d = d[k]
        if all(k in d for k in "rgb"):
            return rgb_to_hex(_to_255([float(d["r"]), float(d["g"]), float(d["b"])]))
        found = find_colours(j)
        if not found:
            raise RuntimeError(f"Could not read a colour from the Spinner's simulation reply: {j}")
        return found[0]

    def best_example(self, name, language):
        key = (language, norm(name))
        if key in self._best:               # already seen as the top name of a colour
            return self._best[key]
        if language not in self._ids:       # learn this language's name list
            self.lookup("#808080", language)
        cid = self._ids[language].get(norm(name))
        if cid is None:
            raise KeyError(f"'{name}' is not one of the Colour Spinner's {language} names.")
        resp = self.http.get(self.cfg["base_url"] + self.cfg["mobile_panels"],
                             params={"language_id": self._lang(language), "colour_id": cid},
                             timeout=self.timeout)
        resp.raise_for_status()
        hexes = find_colours(resp.json())
        if not hexes:
            raise RuntimeError(f"Could not find a best example for '{name}' in the Spinner's reply.")
        self._best[key] = hexes[0]
        return hexes[0]

def _to_255(v):
    """RGB in 0–255, whether the Spinner sent 0–255 or 0–1 values."""
    v = [float(x) for x in v]
    if all(x <= 1 for x in v) and any(x != int(x) for x in v):
        v = [x * 255 for x in v]
    return v

def find_colours(obj):
    """Collect colours from a JSON reply in the order they appear (hex strings or [r, g, b] lists)."""
    out = []
    def walk(o):
        if isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            if len(o) == 3 and all(isinstance(x, (int, float)) for x in o) and all(0 <= x <= 255 for x in o):
                out.append(rgb_to_hex(_to_255(o)))
            else:
                for v in o: walk(v)
        elif isinstance(o, str):
            m = re.fullmatch(r"\s*#?([0-9A-Fa-f]{6})\s*", o)
            m2 = re.fullmatch(r"\s*rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*", o)
            if m: out.append("#" + m.group(1).upper())
            elif m2: out.append(rgb_to_hex([int(x) for x in m2.groups()]))
    walk(obj)
    return out


# ── Vision simulations from the Colour Spinner ─────────────────────────────
_SIM = {"client": None, "available": True, "error": "", "cache": {}}

class SimulationUnavailable(RuntimeError):
    """The Colour Spinner's simulation could not be reached."""

def _sim_client():
    if _SIM["client"] is None:
        _SIM["client"] = ApiSpinner(timeout=15)
    return _SIM["client"]

def simulate(c, condition="Typical"):
    """The hex of colour `c` as seen under `condition`, from the Colour Spinner's own
    simulation models (the ones behind its "Simulate age/CVD" menu)."""
    hx = (c if isinstance(c, str) else rgb_to_hex(c)).upper()
    if not hx.startswith("#"): hx = "#" + hx
    if condition == "Typical":
        return hx
    key = (hx, condition)
    if key not in _SIM["cache"]:
        if not _SIM["available"]:
            raise SimulationUnavailable(_SIM["error"])
        try:
            _SIM["cache"][key] = _sim_client().simulate(hx, condition)
        except Exception as e:
            _SIM["available"] = False
            _SIM["error"] = str(e)[:120]
            raise SimulationUnavailable(_SIM["error"]) from e
    return _SIM["cache"][key]

def retry_simulations():
    """Try the Colour Spinner's simulations again after a connection problem."""
    _SIM.update(available=True, error="")

def prefetch_simulations(hexes, conditions=CONDITIONS):
    """Fetch the Spinner's simulations for many colours at once (six requests in parallel)."""
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(h, c) for h in dict.fromkeys(hexes) for c in conditions if c != "Typical"]
    if not jobs or not _SIM["available"]:
        return
    try:                                  # one request first, so an outage fails fast
        simulate(*jobs[0])
    except SimulationUnavailable:
        return
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda job: try_call(simulate, *job), jobs[1:]))

def simulation_source_note():
    return ("Simulations from the Colour Spinner: colour vision deficiencies as dichromacy (Viénot et al. 1999); "
            "ageing as lens yellowing at ages 40, 60 and 80 (Pokorny et al. 1987).")

def needs_simulations(default=None):
    """Decorator: if the Spinner's simulations are unavailable, explain instead of crashing."""
    import functools
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*a, **k):
            try:
                return fn(*a, **k)
            except SimulationUnavailable as e:
                show_simulation_problem(str(e))
                return default
        return inner
    return wrap

def show_simulation_problem(err=""):
    show('<div class="card"><b>The Colour Spinner\'s vision simulations could not be reached</b>'
         '<p style="margin:6px 0 0">This part of the audit uses the Spinner\'s own colour vision deficiency '
         'and age models, so it needs the Spinner online. Check your internet connection, then run '
         '<code>retry_simulations()</code> and this cell again. Meanwhile, use the Spinner\'s '
         '"Simulate age/CVD" menu in your browser.</p>'
         + (f'<div class="note">{_h(err)}</div>' if err else "") + '</div>')


def _read_sheet(text):
    rows = [r for r in text.strip().splitlines() if r.strip() and not r.lstrip().startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(rows)), skipinitialspace=True,
                     dtype=str).apply(lambda s: s.str.strip())
    return df[df.iloc[:, 0] != df.columns[0]].reset_index(drop=True)   # drop pasted-in headers

def parse_colour(s):
    """'#E8153A', 'E8153A' or '232 21 56' (as the Mapper shows RGB) -> '#E8153A'."""
    s = str(s).strip()
    nums = re.findall(r"\d+", s)
    if len(nums) == 3 and not re.fullmatch(r"#?[0-9A-Fa-f]{6}", s):
        return rgb_to_hex([int(n) for n in nums])
    return "#" + s.lstrip("#").upper()

class SheetSpinner:
    """Values copied by hand from the Colour Spinner's Namer and Mapper panels.
    One row per (colour, language): hex, language, name, confidence, best_example."""
    def __init__(self, sheet=""):
        cols = ["hex", "language", "name", "confidence", "best_example"]
        self.rows = _read_sheet(sheet) if sheet.strip() else pd.DataFrame(columns=cols)
        missing = set(cols) - set(self.rows.columns)
        if missing:
            raise ValueError(f"The sheet needs the columns {cols}; missing {sorted(missing)}.")
        if len(self.rows):
            self.rows["language"] = self.rows["language"].map(canonical_language)
            self.rows["hex"] = self.rows["hex"].map(parse_colour)
            self.rows["best_example"] = self.rows["best_example"].map(
                lambda v: parse_colour(v) if isinstance(v, str) and v else None)
            self.rows["confidence"] = self.rows["confidence"].str.rstrip("%").astype(float).map(
                lambda v: v / 100 if v > 1 else v)

    def name(self, hex_, language):
        m = self.rows[(self.rows.hex == hex_.upper()) & (self.rows.language == canonical_language(language))]
        if m.empty:
            raise KeyError(f"no row for {hex_} in {language}")
        return {"name": m.iloc[0]["name"], "confidence": float(m.iloc[0]["confidence"])}

    def best_example(self, name, language):
        m = self.rows[(self.rows.language == canonical_language(language)) & (self.rows.name.map(norm) == norm(name))
                      & self.rows.best_example.notna()]
        if m.empty:
            raise KeyError(f"no best example for '{name}' in {language}")
        return m.iloc[0]["best_example"]

    def translate(self, name, src, tgt):
        raise NotImplementedError("Read translations from the Translator panel instead.")


def norm(s):
    """Compare colour names ignoring case, accents, spacing and punctuation
    (Κόκκινο == κοκκινο; ς == σ)."""
    s = unicodedata.normalize("NFD", str(s).strip().casefold()).replace("ς", "σ")
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", "", s)).strip()

def name_variants(text):
    """'kokkino (κόκκινο)' -> ['kokkino', 'κόκκινο']: the name and anything in brackets,
    so a reply in either script, or both, can be matched."""
    text = str(text).strip().strip('"')
    inner = re.findall(r"\(([^)]*)\)", text)
    main = re.sub(r"\s*\([^)]*\)", "", text).strip()
    parts = [main] + [p.strip() for chunk in inner for p in re.split(r"[,/;]", chunk)]
    return [p for p in dict.fromkeys(parts) if p]

def same_name(candidate, name, romanized=None):
    """True if `candidate` (in any script, with or without a bracketed romanisation)
    is the Spinner's `name` or its romanised spelling."""
    if not isinstance(candidate, str) or not isinstance(name, str):
        return False
    targets = {norm(name)} | ({norm(romanized)} if isinstance(romanized, str) and romanized else set())
    return any(norm(v) in targets for v in name_variants(candidate))

def try_call(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:
        return e


# ════════════════════════════════════════════════════════════════════════
#  4. Asking an AI system
# ════════════════════════════════════════════════════════════════════════
COLAB_AI = "Gemini built into Colab"

def ask_ai(prompt, provider=COLAB_AI, model="", base_url=""):
    """Send `prompt` to an AI system.
    provider: COLAB_AI uses google.colab.ai, free in Colab with no key;
              'Anthropic' (secret ANTHROPIC_API_KEY) or
              'OpenAI-compatible' (secret OPENAI_API_KEY; set base_url for
              Gemini, Mistral, local servers, …)."""
    import requests
    def secret(name):
        try:
            from google.colab import userdata
            return userdata.get(name)
        except Exception:
            v = os.environ.get(name)
            if not v: raise RuntimeError(f"Add {name} under Colab ▸ Secrets (key icon, left).")
            return v
    if provider == COLAB_AI:
        try:
            from google.colab import ai
        except ImportError:
            raise RuntimeError("Colab's built-in Gemini only works inside Google Colab. "
                               "Choose another provider, or paste the reply yourself.")
        try:
            return ai.generate_text(prompt, model_name=model) if model else ai.generate_text(prompt)
        except Exception as e:
            raise RuntimeError(f"Colab's built-in Gemini did not answer ({e}). Wait a minute and run "
                               "the cell again, or paste the reply yourself.") from e
    if provider == "Anthropic":
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=120, headers={
                "x-api-key": secret("ANTHROPIC_API_KEY"),
                "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": model, "max_tokens": 4000,
                  "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])
    if provider == "OpenAI-compatible":
        url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        r = requests.post(url, timeout=120,
            headers={"Authorization": f"Bearer {secret('OPENAI_API_KEY')}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    raise ValueError("Choose a provider in the form, or 'Paste the reply from my own AI'.")

def parse_json_reply(text):
    """Pull the first JSON array/object out of an AI reply (ignores ``` fences and chatter)."""
    t = re.sub(r"```(?:json)?", "", text)
    start = min([i for i in (t.find("["), t.find("{")) if i >= 0], default=-1)
    if start < 0: raise ValueError("No JSON found in the AI reply.")
    obj, _ = json.JSONDecoder().raw_decode(t[start:])
    return obj


# ════════════════════════════════════════════════════════════════════════
#  5. Look and feel
# ════════════════════════════════════════════════════════════════════════
#  Swatches sit on a neutral mid-grey surround (≈ Munsell N5), the convention
#  for judging colour: a white or black page shifts how a colour looks.
#  Atkinson Hyperlegible was designed by the Braille Institute for low-vision
#  readers.
SURROUND, INK, MUTED, PAPER = "#777777", "#1F1F1F", "#5E5E5E", "#FFFFFF"

_CSS = f"""
<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
<style>
.cbt {{ font-family:'Atkinson Hyperlegible',system-ui,sans-serif; color:{INK}; line-height:1.45; }}
.cbt .surround {{ background:{SURROUND}; padding:14px; border-radius:6px; display:flex; flex-wrap:wrap; gap:6px; }}
.cbt .tile {{ width:86px; border-radius:3px; overflow:hidden; background:{PAPER}; }}
.cbt .chip {{ height:62px; display:flex; align-items:flex-end; padding:4px 6px; font-weight:700; font-size:15px; }}
.cbt .cap {{ padding:5px 6px 6px; font-size:11.5px; line-height:1.25; }}
.cbt .cap .hex {{ color:{MUTED}; font-size:10.5px; letter-spacing:.02em; }}
.cbt .row {{ display:flex; align-items:center; gap:6px; }}
.cbt .rowlab {{ width:96px; flex:none; color:#fff; font-size:12.5px; text-align:right; padding-right:6px; }}
.cbt .mini {{ width:40px; height:30px; border-radius:2px; flex:none; }}
.cbt table.babel {{ border-collapse:separate; border-spacing:0 6px; font-size:13.5px; }}
.cbt table.babel td, .cbt table.babel th {{ padding:4px 12px 4px 0; vertical-align:middle; text-align:left; }}
.cbt table.babel th {{ font-weight:400; color:{MUTED}; font-size:12px; }}
.cbt .bar {{ height:5px; background:#E4E4E0; border-radius:3px; width:90px; margin-top:3px; }}
.cbt .bar > i {{ display:block; height:100%; border-radius:3px; background:{INK}; }}
.cbt .pair {{ display:flex; border-radius:4px; overflow:hidden; width:150px; height:54px; border:5px solid {SURROUND}; }}
.cbt .pair > div {{ flex:1; display:flex; align-items:flex-end; padding:3px 5px; font-size:10.5px; }}
.cbt .badge {{ display:inline-block; min-width:44px; text-align:center; padding:2px 7px;
               border-radius:10px; font-weight:700; font-size:12.5px; }}
.cbt .card {{ background:{PAPER}; border:1px solid #DDDDD8; border-radius:8px; padding:18px 22px; max-width:640px; }}
.cbt .card h3 {{ margin:0 0 6px; font-size:19px; }}
.cbt .note {{ color:{MUTED}; font-size:12.5px; margin-top:6px; }}
.cbt code {{ font-size:12px; }}
</style>"""

def _ink_on(hex_):
    """Black or white text, whichever contrasts more with the background (WCAG)."""
    Y = rgb_to_xyz(hex_)[1]
    return INK if (Y + 0.05) / 0.05 > 1.05 / (Y + 0.05) else "#FFFFFF"

def _h(x): return _html.escape(str(x))

def show(html_body):
    display(HTML(_CSS + f'<div class="cbt">{html_body}</div>'))

def show_palette(hexes, labels=None, captions=None, note=None):
    labels = labels or [""] * len(hexes)
    captions = captions or [""] * len(hexes)
    tiles = "".join(
        f'<div class="tile"><div class="chip" style="background:{h};color:{_ink_on(h)}">{_h(l)}</div>'
        f'<div class="cap">{_h(c)}<div class="hex">{h}</div></div></div>'
        for h, l, c in zip(hexes, labels, captions))
    show(f'<div class="surround">{tiles}</div>' + (f'<div class="note">{note}</div>' if note else ""))

@needs_simulations()
def show_conditions(hexes, labels=None, conditions=CONDITIONS, width=40, note=True):
    """One row per vision condition, so you can scan down a column for change."""
    labels = labels or [""] * len(hexes)
    w = f"width:{width}px"
    head = '<div class="row" style="align-items:flex-end"><div class="rowlab"></div>' + "".join(
        f'<div class="mini" style="{w};height:auto;color:#fff;font-size:11px;text-align:center;'
        f'line-height:1.2;padding-bottom:3px;overflow-wrap:anywhere">{_h(l)}</div>' for l in labels) + "</div>"
    rows = "".join(
        f'<div class="row"><div class="rowlab">{_h(c)}'
        + '</div>' + "".join(
            f'<div class="mini" style="{w};background:{simulate(h, c)}" title="{simulate(h, c)}"></div>'
            for h in hexes) + "</div>" for c in conditions)
    show(f'<div class="surround" style="flex-direction:column;gap:4px">{head}{rows}</div>' +
         (f'<div class="note">{simulation_source_note()}</div>'
          if note else ""))

def show_babel(table, languages):
    """table: rows with sdg, hex, and per-language '<lang>' name and '<lang>|conf' columns."""
    head = "<tr><th></th>" + "".join(f"<th>{_h(l)}</th>" for l in languages) + "</tr>"
    body = ""
    for _, r in table.iterrows():
        cells = ""
        for l in languages:
            n, c, rom = r.get(l), r.get(f"{l}|conf"), r.get(f"{l}|rom")
            if isinstance(n, str):
                w = 0 if pd.isna(c) else int(round(100 * c))
                extra = (f' <span style="color:{MUTED};font-size:11px">{_h(rom)}</span>'
                         if isinstance(rom, str) and rom and norm(rom) != norm(n) else "")
                cells += (f'<td><b>{_h(n)}</b>{extra}<div class="bar"><i style="width:{w}%"></i></div>'
                          f'<span style="color:{MUTED};font-size:11px">{w}% confidence</span></td>')
            else:
                cells += f'<td style="color:{MUTED}">—</td>'
        body += (f'<tr><td><div class="pair" style="width:96px"><div style="background:{r.hex};'
                 f'color:{_ink_on(r.hex)};font-weight:700;font-size:13px">SDG {r.name}</div></div></td>{cells}</tr>')
    show(f'<table class="babel">{head}{body}</table>')

_GOOD, _MID, _BAD = ("#DDEFE3", "#14532D"), ("#FBF0D0", "#6B4E00"), ("#F9DADA", "#7F1D1D")
def de_badge(d, threshold, small_is_good=True):
    """ΔE00 badge. small_is_good=True for 'how far off' (agreement);
    False for 'how far apart' (distinctiveness), where small is the risk."""
    if pd.isna(d): return '<span class="badge" style="background:#E4E4E0">n/a</span>'
    if small_is_good:
        bg, fg = _GOOD if d <= threshold / 2 else _MID if d <= threshold else _BAD
    else:
        bg, fg = _BAD if d < threshold / 2 else _MID if d < threshold else _GOOD
    return f'<span class="badge" style="background:{bg};color:{fg}">{d:.1f}</span>'

def show_pairs(df, left="ai_hex", right="spinner_hex", lab_left="AI", lab_right="Spinner",
               threshold=10, extra_cols=()):
    head = ("<tr><th>Language</th><th>Name</th><th>" + f"{lab_left} | {lab_right}" +
            "</th><th>ΔE00</th>" + "".join(f"<th>{_h(c)}</th>" for c in extra_cols) + "</tr>")
    body = ""
    for _, r in df.iterrows():
        def half(h, t):
            return (f'<div style="background:{h};color:{_ink_on(h)}">{t}<br>{h}</div>'
                    if isinstance(h, str) else f'<div style="background:#E4E4E0">{t}<br>missing</div>')
        body += (f"<tr><td>{_h(r.language)}</td><td><b>{_h(r['name'])}</b></td>"
                 f'<td><div class="pair">{half(r[left], lab_left)}{half(r[right], lab_right)}</div></td>'
                 f"<td>{de_badge(r.get('de00'), threshold)}</td>" +
                 "".join(f"<td>{_h(r.get(c, ''))}</td>" for c in extra_cols) + "</tr>")
    show(f'<table class="babel">{head}{body}</table>'
         f'<div class="note">ΔE00 ≈ 1 is a just-noticeable difference side by side. '
         f'Above {threshold:g} the two read as different colours.</div>')

# ── matplotlib house style ───────────────────────────────────────────────
def _install_font():
    import matplotlib.font_manager as fm
    if any("Atkinson Hyperlegible" in f.name for f in fm.fontManager.ttflist): return True
    if not IN_COLAB: return False
    try:
        import urllib.request
        base = "https://github.com/google/fonts/raw/main/ofl/atkinsonhyperlegible/"
        for f in ("AtkinsonHyperlegible-Regular.ttf", "AtkinsonHyperlegible-Bold.ttf"):
            p = f"/tmp/{f}"
            if not os.path.exists(p): urllib.request.urlretrieve(base + f, p)
            fm.fontManager.addfont(p)
        return True
    except Exception:
        return False
mpl.rcParams.update({
    "font.family": ["Atkinson Hyperlegible", "DejaVu Sans"] if _install_font() else ["DejaVu Sans"],
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": "#BDBDB8", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 13,
    "axes.titlelocation": "left", "font.size": 11, "figure.dpi": 110,
})

# Red below the threshold (shown at the middle of the scale), calm neutrals above it.
RISK_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "risk", [(0, "#8E1B1B"), (0.4999, "#E7B3A3"), (0.5, "#E3E8DF"), (1, PAPER)])

def chip(ax, y, hex_, label, height=0.72, gap_in=0.12, width_in=0.66):
    """A rounded colour chip with its label, drawn just left of the plot area at data height y.
    The same chip is used on every chart, so test colours look the same everywhere."""
    fig = ax.figure
    ax_w_in = ax.get_position().width * fig.get_figwidth()
    w, g = width_in / ax_w_in, gap_in / ax_w_in
    blend = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.add_patch(mpl.patches.FancyBboxPatch((-g - w, y - height / 2), w, height,
                 boxstyle="round,pad=0,rounding_size=0.012", color=hex_, transform=blend, clip_on=False))
    ax.text(-g - w / 2, y, label, transform=blend, ha="center", va="center", color=_ink_on(hex_),
            fontsize=10, fontweight="bold", clip_on=False)

def row_chips(ax, hexes, labels):
    """Replace y tick labels with colour chips carrying the label."""
    ax.set_yticks(range(len(hexes)), [""] * len(hexes))
    ax.tick_params(axis="y", length=0, pad=0)
    for r, (h, l) in enumerate(zip(hexes, labels)):
        chip(ax, r, h, l)

def download(path):
    """Offer a file for download in Colab; elsewhere just report where it is."""
    if IN_COLAB:
        from google.colab import files
        files.download(path)
    else:
        print(f"Saved {os.path.abspath(path)}")



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
@needs_simulations(default=(None, None))
def nearest_neighbours(pick, conditions=CONDITIONS):
    """For each colour in `pick` and each condition: the nearest other SDG colour and its ΔE00.
    Returns two tables with the same shape: distances and neighbour ids."""
    def nearest(i, c):
        me = simulate(SDG.hex[i], c)
        return min((de00(me, simulate(h, c)), j) for j, h in SDG.hex.items() if j != i)
    nn = pd.DataFrame({c: {i: nearest(i, c) for i in pick} for c in conditions})
    return nn.map(lambda t: t[0]), nn.map(lambda t: t[1])

def plot_nearest_neighbours(nn_de, nn_id, threshold):
    if nn_de is None: return
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
    ax.set_title(f"Test colour to nearest other SDG colour (ΔE00); red below {threshold}")
    plt.tight_layout(); plt.show()

@needs_simulations()
def show_at_risk(nn_de, nn_id, threshold):
    if nn_de is None: return
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
    table = SDG.loc[wide.index, ["hex"]].join(wide).join(conf)
    if "romanized" in naming:
        table = table.join(naming.pivot(index="sdg", columns="language", values="romanized").add_suffix("|rom"))
    show_babel(table, [l for l in languages if l in wide.columns])

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
            chip(ax, (min(g) + max(g)) / 2, SDG.hex[i], f"SDG {i}",
                 height=max(g) - min(g) + 0.72, gap_in=1.45)
    ax.set_yticks(ys, [l for _, l in rows]); ax.tick_params(axis="y", length=0)
    ax.invert_yaxis(); ax.set_xlim(0, 1.18); ax.spines["left"].set_visible(False)
    ax.set_xticks([0, .25, .5, .75, 1]); ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    ax.set_xlabel("Spinner confidence in its top name")
    ax.set_title("Nameability: how confident is the Spinner's top name?")
    plt.tight_layout(); fig.subplots_adjust(left=0.30); plt.show()

def show_translatability(naming, pick, languages, threshold):
    td = 'style="width:132px;padding:4px 10px 4px 0"'
    rows = ""
    for i in pick:
        sub = naming[naming.sdg == i]
        cells = ""
        for lang in languages:
            r = sub[sub.language == lang]
            if r.empty or not isinstance(r.best_example.iloc[0], str):
                cells += f'<td {td}><span style="color:{MUTED}">—</span></td>'; continue
            b = r.best_example.iloc[0]
            cells += (f'<td {td}><div class="pair" style="width:118px"><div style="background:{b};color:{ink_on(b)}">'
                      f'{esc(r.name.iloc[0])}<br>{b}</div></div><div style="margin-top:3px">'
                      f'{de_badge(r.drift.iloc[0], threshold)}</div></td>')
        sp = spread(sub.best_example.tolist())
        rows += (f'<tr><td style="width:110px"><div class="pair" style="width:96px"><div style="background:{SDG.hex[i]};'
                 f'color:{ink_on(SDG.hex[i])};font-weight:700;font-size:13px">SDG {i}</div></div></td>'
                 f'<td style="width:36px"></td>{cells}'
                 f'<td style="color:{MUTED};white-space:nowrap">{"" if pd.isna(sp) else "spread " + format(sp, ".1f")}</td></tr>')
    show('<table class="babel" style="table-layout:fixed"><tr><th>Test colour</th><th></th>' +
         "".join(f'<th {td}>{esc(l)}</th>' for l in languages) +
         f'<th></th></tr>{rows}</table><div class="note">Badges: ΔE00 between the test colour and each '
         "language's best example (focal colour). Spread: the largest ΔE00 between two languages' best examples.</div>")

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
ALL_LANGUAGES = "Average of my languages"
WORST_CASE = "Worst case of all conditions"
SUBSET_MAX = "Largest value in my subset"
SDG_RANGE = "Range across all 17 SDG colours"

def _distances(ids, condition):
    conds = CONDITIONS if condition == WORST_CASE else [condition]
    de, _ = nearest_neighbours(ids, conds)
    if de is None:
        return None
    return de.min(axis=1) if condition == WORST_CASE else de[condition]

def scorecard(pick, naming, language=ALL_LANGUAGES, condition="Typical", normalise=SUBSET_MAX):
    """One row per test colour: its perceptual distance (ΔE00 to the nearest other SDG colour, under
    `condition`) and naming confidence (in `language`, or averaged), both on a 0–1 scale."""
    de = _distances(pick, condition)
    if de is None:
        return None
    if normalise == SDG_RANGE:
        all_de = _distances(list(SDG.index), condition)
        lo, hi = all_de.min(), all_de.max()
    else:
        lo, hi = 0.0, de.max()
    conf = naming.pivot(index="sdg", columns="language", values="confidence")
    names = naming.pivot(index="sdg", columns="language", values="name")
    y = conf.mean(axis=1) if language == ALL_LANGUAGES else conf.get(language, pd.Series(np.nan, index=conf.index))
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
    ax.set_xlabel(f"Perceptual distance (ΔE00 to nearest SDG colour, {vision}; {scale})")
    ax.set_ylabel("Naming confidence" + ("" if card.attrs["language"] == ALL_LANGUAGES else f" ({card.attrs['language']})")
                  + (" (average of languages)" if card.attrs["language"] == ALL_LANGUAGES else ""))
    ax.set_title("Naming confidence × perceptual distance")
    if not 0 <= x_cut <= 1:
        ax.text(0.5, -0.16, f"The ΔE00 {confusable_below} line falls outside this scale.", transform=ax.transAxes,
                ha="center", color=MUTED, fontsize=9)
    plt.tight_layout(); plt.show()

def show_scorecard(card):
    if card is None:
        return
    langs = [c for c in card.columns if c not in ("dE00", "perceptual_0to1", "confidence_0to1")]
    head = ("<tr><th>Test colour</th><th>ΔE00</th><th>Perceptual (0–1)</th><th>Confidence (0–1)</th>" +
            "".join(f"<th>{esc(l)}</th>" for l in langs) + "</tr>")
    body = ""
    for i, r in card.iterrows():
        conf = "" if pd.isna(r.confidence_0to1) else f"{r.confidence_0to1:.2f}"
        body += (f'<tr><td><div class="pair" style="width:96px;height:40px"><div style="background:{SDG.hex[i]};'
                 f'color:{ink_on(SDG.hex[i])};font-weight:700;font-size:12px">SDG {i}</div></div></td>'
                 f"<td>{r.dE00:.1f}</td><td>{r.perceptual_0to1:.2f}</td><td><b>{conf}</b></td>" +
                 "".join(f"<td>{esc(r[l])}</td>" for l in langs) + "</tr>")
    show(f'<table class="babel">{head}{body}</table>'
         '<div class="note">Language columns: the Spinner\'s top name and its confidence for each test colour.</div>')


# ── Part B: the AI's names ────────────────────────────────────────────────
# ── Asking the AI, and reading its reply ─────────────────────────────────
PASTE = "Paste answers from my own AI"
_ANS = {"boxes": [], "rows": None, "kind": None}

def answer_boxes(rows, kind):
    """One box per answer. kind='name': rows have hex and language, and you type the AI's name.
    kind='colour': rows have language and name, and you type the AI's colour code."""
    try:
        import ipywidgets as w
    except ImportError:
        print("Answer boxes need ipywidgets, which Colab includes.")
        return
    rows = rows.reset_index(drop=True)
    items, boxes = [], []
    for _, r in rows.iterrows():
        if kind == "name":
            left = (f'<span style="display:inline-block;width:34px;height:22px;border-radius:3px;'
                    f'vertical-align:middle;background:{r.hex}"></span>&nbsp; <b>{r.hex}</b> · {_h(r.language)}')
            hint = "name your AI gave"
        else:
            left = f'{_h(r.language)}: <b>{_h(plain_with_roman(r))}</b>'
            hint = "#RRGGBB your AI gave"
        box = w.Text(placeholder=hint, layout=w.Layout(width="280px"))
        boxes.append(box)
        items.append(w.HBox([w.HTML(left, layout=w.Layout(width="320px")), box]))
    _ANS.update(boxes=boxes, rows=rows, kind=kind)
    display(w.VBox(items))

def collected_answers():
    """What was typed into the answer boxes, as a table."""
    rows, kind = _ANS["rows"], _ANS["kind"]
    if rows is None:
        raise ValueError("Run the cell above first: it shows the boxes to fill in.")
    values = [b.value.strip() for b in _ANS["boxes"]]
    if not any(values):
        raise ValueError("The boxes are empty: copy your AI's answers into them, then run this cell again.")
    t = rows.copy()
    if kind == "name":
        t["ai_name"] = [v or None for v in values]
        return t[["hex", "language", "ai_name"]]
    t["ai_hex"] = [parse_colour(v) if v and (_HEX.search(v) or len(re.findall(r"\d+", v)) == 3) else None
                   for v in values]
    return t[["language", "name", "ai_hex"]]

def show_prompt(prompt, provider, rows=None, kind="name"):
    """Show the prompt; for your own AI, also one box per answer."""
    print(prompt)
    if provider == PASTE:
        what = "the name" if kind == "name" else "the colour code"
        show(f"<b>1.</b> Copy the prompt above into your AI tool.<br>"
             f"<b>2.</b> For each line below, copy {what} it gave into the box, then run the next cell.")
        answer_boxes(rows, kind)
    else:
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
    same = "<span class='badge' style='background:#DDEFE3;color:#14532D'>same</span>"
    diff = "<span class='badge' style='background:#F9DADA;color:#7F1D1D'>differs</span>"
    def where(r):
        if r.agrees or not isinstance(r.get("ai_name"), str):
            return ""
        best = r.get("ai_best")
        if isinstance(best, str):
            return (f'<div class="row"><div class="mini" style="background:{best}" title="{best}"></div>'
                    f'{de_badge(r.ai_drift, threshold)}</div>')
        return f'<span style="color:{MUTED};font-size:12px">{esc(r.get("ai_status", ""))}</span>'
    rows = "".join(
        f'<tr><td><div class="mini" style="background:{r.hex}"></div></td><td>SDG {r.sdg}</td>'
        f'<td>{esc(r.language)}</td><td><b>{esc(r["name"])}</b> '
        f'<span style="color:{MUTED};font-size:11px">{r.confidence:.0%}</span></td>'
        f'<td><b>{esc(r.ai_name if isinstance(r.ai_name, str) else "—")}</b></td>'
        f'<td>{same if r.agrees else diff}</td><td>{where(r)}</td></tr>' for _, r in cmp.iterrows())
    show('<table class="babel"><tr><th></th><th></th><th>Language</th><th>Speakers (Spinner)</th>'
         f'<th>AI system</th><th></th><th>Where the AI\'s name points</th></tr>{rows}</table>'
         '<div class="note">For names that differ: the Spinner\'s best example of the AI\'s name in that language, '
         'and its ΔE00 from the test colour. Small: a plausible neighbouring name. Large: the name means a '
         'different colour to speakers.</div>')

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
                 [[c for c in ["sdg", "hex", "language", "name", "romanized", "best_example"] if c in partb]]
                 .rename(columns={"hex": "sdg_hex", "best_example": "spinner_hex"}))
    else:
        names = read_sheet(typed).assign(sdg=np.nan, sdg_hex=None, spinner_hex=None)[
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
    langs = list(dict.fromkeys(names.language))
    show('<table class="babel"><tr><th>Language</th><th>Names to test</th></tr>' + "".join(
         f"<tr><td>{esc(l)}</td><td><b>{', '.join(_with_roman(r) for _, r in names[names.language == l].iterrows())}</b></td></tr>"
         for l in langs) + "</table>")

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


# ── Part C: round trip ────────────────────────────────────────────────────
def show_round_trip(pal):
    rows = ""
    for _, r in pal.iterrows():
        if not isinstance(r.ai_hex, str): continue
        if isinstance(r.get("rt_name"), str):
            ok = same_name(r.rt_name, r["name"], r.get("romanized"))
            badge = ("<span class='badge' style='background:#DDEFE3;color:#14532D'>same name</span>" if ok else
                     "<span class='badge' style='background:#F9DADA;color:#7F1D1D'>different name</span>")
            said = f'<b>{esc(r.rt_name)}</b> <span style="color:{MUTED};font-size:11px">{r.rt_conf:.0%}</span>'
        else:
            badge, said = "", f'<span style="color:{MUTED}">not looked up</span>'
        rows += (f'<tr><td>{esc(r.language)}</td><td><b>{esc(r["name"])}</b></td>'
                 f'<td><div class="mini" style="background:{r.ai_hex}" title="{r.ai_hex}"></div></td>'
                 f'<td>{said}</td><td>{badge}</td></tr>')
    show('<table class="babel"><tr><th>Language</th><th>Name given to the AI</th><th>AI\'s colour</th>'
         f'<th>What speakers call that colour (Spinner)</th><th></th></tr>{rows}</table>'
         '<div class="note">The percentage is the Spinner\'s confidence in that name for the AI\'s colour.</div>')


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
         "replaced with the Colour Spinner's best example. Top strip: the AI's colour.</div>")

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

@needs_simulations()
def show_grounded_conditions(pal, languages):
    if not _SIM["available"]: return          # the problem was already explained above
    langs = [l for l in languages if pal[pal.language == l].grounded_hex.notna().sum() > 1]
    for k, lang in enumerate(langs):
        p = pal[(pal.language == lang) & pal.grounded_hex.notna()]
        show(f'<div style="font-weight:700;margin:10px 0 4px">{esc(lang)}, grounded</div>')
        show_conditions(p.grounded_hex.tolist(), labels=p.name.tolist(), width=64, note=(k == len(langs) - 1))


# ── Part C: export and finding ────────────────────────────────────────────
def export_palette(pal, languages, stem="grounded_palette"):
    """Write CSV, JSON and CSS versions of the grounded palette, zip them and offer the download."""
    import zipfile
    cols = ["language", "name", "grounded_hex", "ai_hex", "spinner_hex", "de00", "rt_name", "rt_conf",
            "replaced", "sdg", "sdg_hex"]
    export = pal[[c for c in cols if c in pal.columns]]
    export.round(2).to_csv(f"{stem}.csv", index=False)
    data = {lang: [{"name": r["name"], "hex": r.grounded_hex, "ai_hex": r.ai_hex, "spinner_hex": r.spinner_hex,
                    "dE00_ai_vs_spinner": None if pd.isna(r.de00) else round(float(r.de00), 2),
                    "speakers_call_ai_colour": r.get("rt_name") if isinstance(r.get("rt_name"), str) else None,
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

print("Toolkit ready ✓  " + ("(running in Colab)" if IN_COLAB else "(running locally)"))
