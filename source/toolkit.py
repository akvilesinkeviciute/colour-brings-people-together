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
                     "Greek", "Himba", "Spanish"]

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
# The Spinner simulates age bands; its menu labels them by the upper age.
AGE_BANDS = {"Age 40": "16–40", "Age 60": "40–60", "Age 80": "60–80"}
def condition_label(c, sep="\n"):
    return f"{c}{sep}({AGE_BANDS[c]})" if c in AGE_BANDS else c


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
                     "Greek": 4, "Himba": 5, "Spanish": 6},
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
            return self.cfg["language_ids"][language]
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
            return rgb_to_hex([float(d["r"]), float(d["g"]), float(d["b"])])
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

def find_colours(obj):
    """Collect colours from a JSON reply in the order they appear (hex strings or [r, g, b] lists)."""
    out = []
    def walk(o):
        if isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            if len(o) == 3 and all(isinstance(x, (int, float)) for x in o) and all(0 <= x <= 255 for x in o):
                out.append(rgb_to_hex(o))
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
    return "Simulations from the Colour Spinner (colornaming.net/spinner); age rows are its age bands."

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
            self.rows["hex"] = self.rows["hex"].map(parse_colour)
            self.rows["best_example"] = self.rows["best_example"].map(
                lambda v: parse_colour(v) if isinstance(v, str) and v else None)
            self.rows["confidence"] = self.rows["confidence"].str.rstrip("%").astype(float).map(
                lambda v: v / 100 if v > 1 else v)

    def name(self, hex_, language):
        m = self.rows[(self.rows.hex == hex_.upper()) & (self.rows.language == language)]
        if m.empty:
            raise KeyError(f"no row for {hex_} in {language}")
        return {"name": m.iloc[0]["name"], "confidence": float(m.iloc[0]["confidence"])}

    def best_example(self, name, language):
        m = self.rows[(self.rows.language == language) & (self.rows.name.map(norm) == norm(name))
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
COLAB_AI = "Gemini built into Colab (no key needed)"

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
SURROUND, INK, MUTED, PAPER = "#777777", "#1F1F1F", "#5E5E5E", "#FAFAF8"

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
        + (f'<br><span style="font-size:10px;opacity:.8">{AGE_BANDS[c]}</span>' if c in AGE_BANDS else "")
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
                          f'<span style="color:{MUTED};font-size:11px">{w}% agree</span></td>')
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

def row_chips(ax, hexes, labels):
    """Replace y tick labels with colour chips carrying the label."""
    ax.set_yticks(range(len(hexes)), [""] * len(hexes))
    ax.tick_params(axis="y", length=0, pad=0)
    for r, (h, l) in enumerate(zip(hexes, labels)):
        ax.add_patch(mpl.patches.FancyBboxPatch((-1.12, r - 0.36), 0.56, 0.72, boxstyle="round,pad=0,rounding_size=0.06",
                     color=h, clip_on=False, transform=ax.transData))
        ax.text(-0.84, r, l, ha="center", va="center", color=_ink_on(h), fontsize=10,
                fontweight="bold", clip_on=False)
    ax.set_xlim(-1.2, ax.get_xlim()[1])

def download(path):
    """Offer a file for download in Colab; elsewhere just report where it is."""
    if IN_COLAB:
        from google.colab import files
        files.download(path)
    else:
        print(f"Saved {os.path.abspath(path)}")

print("Toolkit ready ✓  " + ("(running in Colab)" if IN_COLAB else "(running locally)"))
