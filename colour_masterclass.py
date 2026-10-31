"""Colour Brings People Together: toolkit for the BCS HCI 2026 masterclass notebooks.

Colour maths (CIELAB and CIEDE2000), the Colour Spinner connection, AI helpers and the
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

# ── The languages currently covered by the Colour Spinner ────────────────
SPINNER_LANGUAGES = ["American English", "British English", "French",
                     "Greek", "OtjiHimba", "Serbian", "Spanish"]

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
    """The colour as 0–1 values, from '#hex' or (r, g, b) in 0–255."""
    return (hex_to_rgb(c) if isinstance(c, str) else np.asarray(c, float)) / 255

def rgb_to_xyz(c):
    """Intermediate step for CIELAB (the library's standard conversion from hex colours)."""
    return colour.sRGB_to_XYZ(_srgb(c))

def rgb_to_lab(c):
    """CIELAB (D65) of a hex colour: the space ΔE₀₀ is computed in."""
    return colour.XYZ_to_Lab(rgb_to_xyz(c))

def de00(c1, c2):
    """CIEDE2000 colour difference between two colours given as hex or RGB."""
    return float(colour.delta_E(rgb_to_lab(c1), rgb_to_lab(c2), method="CIE 2000"))

def spread(hexes):
    """Largest ΔE2000 between any two colours in a list (NaN if fewer than two)."""
    from itertools import combinations
    hexes = [h for h in hexes if isinstance(h, str)]
    return max((de00(a, b) for a, b in combinations(hexes, 2)), default=np.nan)

def _check_colour_library():
    """Guard against unit or API changes in colour-science with known reference values."""
    checks = [
        ("#FF0000 in CIELAB", rgb_to_lab("#FF0000"), [53.24, 80.09, 67.20], 0.05),
        ("CIEDE2000, Sharma et al. (2005) pair 1",
         colour.delta_E([50, 2.6772, -79.7751], [50, 0, -82.7485], method="CIE 2000"), 2.0425, 1e-3),
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
                     "Greek": 4, "OtjiHimba": 5, "Serbian": 6,
                     "Spanish": 7},                    # the menu values, alphabetical (checked 26 October 2026)
}

def _strip(html_text):
    return _html.unescape(re.sub(r"<[^>]+>", "", html_text)).strip()

def _mapper_members(page):
    """The Mapper's tiles for the top name, in its order: focal colour first, then towards the periphery."""
    i = page.find('id="mapper-plot"')
    if i < 0:
        return []
    seg = page[i:]
    seg = seg[:seg.find("</svg>")]
    fills = [f"#{h.upper()}" for h in re.findall(r"fill:\s*#([0-9a-fA-F]{6})", seg)]
    return [f for f in dict.fromkeys(fills) if f != "#FFFFFF"]

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
        "members": _mapper_members(page),
        "mapper_name": _strip(re.sub(r"<[^>]+>", "", (re.search(r"Best example of\s*(.*?)\s*\(top left\)", page, re.S) or [None, ""])[1] or "")),
    }

_NEARBY_SHOWN = set()       # colours already reported as looked up via a neighbour

class ApiSpinner:
    """Talks to the Colour Spinner the same way its web page does."""
    def __init__(self, cfg=SPINNER_API, timeout=30):
        import requests
        self.http = requests.Session()
        self.http.headers.update({"HX-Request": "true",
                                  "User-Agent": "colour-masterclass-notebook (BCS HCI 2026)"})
        self.cfg, self.timeout = cfg, timeout
        self._pages, self._best, self._ids, self._lang_ids, self._lang_names = {}, {}, {}, {}, {}

    def _lang(self, language):
        try:
            return self.cfg["language_ids"][canonical_language(language)]
        except KeyError:
            raise KeyError(f"The Colour Spinner has no language called '{language}'.")

    def _post_colour(self, hex_, language, target):
        """POST one colour to the Picker. The Spinner's server occasionally fails (HTTP 500) for particular colour
        values: retry once, then try the nearest neighbouring colours (±1 or ±2 per channel, far below anything
        visible) and use the first that works. Returns the response and the colour actually looked up."""
        import time, itertools
        r0, g0, b0 = (int(v) for v in hex_to_rgb(hex_))
        def post(r, g, b):
            form = {"language-select": self._lang(language), "translation-select": self._lang(target),
                    "age-cvd-simulation-select": "none", "synonyms": "on", "antonyms": "on",
                    "r": r, "g": g, "b": b, "exact_colour_id": ""}
            return self.http.post(self.cfg["base_url"] + self.cfg["color_picker"], data=form, timeout=self.timeout)
        resp = post(r0, g0, b0)
        if resp.status_code >= 500:
            time.sleep(1.0); resp = post(r0, g0, b0)
        if resp.status_code == 200:
            return resp, hex_.upper()
        if resp.status_code >= 500:
            steps = [d for d in itertools.product((-1, 0, 1), repeat=3) if any(d)] + \
                    [d for d in itertools.product((-2, 0, 2), repeat=3) if any(d)]
            steps.sort(key=lambda d: sum(abs(x) for x in d))
            for dr, dg, db in steps[:12]:
                r, g, b = (min(255, max(0, v + dv)) for v, dv in ((r0, dr), (g0, dg), (b0, db)))
                alt = post(r, g, b)
                if alt.status_code == 200:
                    return alt, rgb_to_hex((r, g, b)).upper()
            raise RuntimeError(f"The Colour Spinner's server returned an error ({resp.status_code}) for this colour, "
                               "even for its nearest neighbours. This is a problem on the Spinner's side, not your "
                               "connection: try again in a minute, or ask the facilitators to switch to copying from the browser.")
        raise RuntimeError(f"The Colour Spinner returned {resp.status_code}. Check your internet connection, "
                           "or ask the facilitators to switch to copying from the browser.")

    def lookup(self, hex_, language, target=None):
        """Everything the Spinner shows for one colour: name, confidence, best example, translation."""
        target = target or language
        key = (hex_.upper(), language, target)
        if key not in self._pages:
            resp, used = self._post_colour(hex_, language, target)
            res = parse_color_picker(resp.text)
            if used != hex_.upper():
                res["looked_up_as"] = used
                if hex_.upper() not in _NEARBY_SHOWN:
                    _NEARBY_SHOWN.add(hex_.upper())
                    show(f'<div class="note">The Colour Spinner had a server error for {hex_.upper()}, so it was looked up as '
                         f'the indistinguishable {used} (ΔE₀₀ = {de00(hex_, used):.1f}).</div>')
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

    def category_members(self, name, language):
        """Colours speakers call `name`, from most to least typical: every tile of the Mapper, read from the
        Picker's reply. First the page's way, asking for the name itself by its ID; otherwise via its best example."""
        key = ("members", language, norm(name))
        if key in self._pages:
            return self._pages[key]
        members = []
        cid = self._language_ids(language).get(norm(name))
        if cid is not None:
            try:
                best = self.best_example(name, language)
                r, g, b = (int(v) for v in hex_to_rgb(best))
            except Exception:
                r, g, b = 128, 128, 128
            form = {"language-select": self._lang(language), "translation-select": self._lang(language),
                    "age-cvd-simulation-select": "none", "synonyms": "on", "antonyms": "on",
                    "r": r, "g": g, "b": b, "exact_colour_id": cid}
            try:
                resp = self.http.post(self.cfg["base_url"] + self.cfg["color_picker"], data=form, timeout=self.timeout)
                if resp.status_code == 200:
                    res = parse_color_picker(resp.text)
                    if same_name(res.get("mapper_name") or res.get("name", ""), name):
                        members = res.get("members", [])
            except Exception:
                members = []
        if not members:                     # otherwise: the Mapper of the best example, if the Namer gives it that name
            best = self.best_example(name, language)
            res = self.lookup(best, language)
            members = res.get("members", []) if same_name(res.get("mapper_name") or res["name"], name) else []
        self._pages[key] = members
        return members

    def _language_ids(self, language):
        """This language's own name list, {name: ID}, as the page loads it when the language menu changes
        (GET /colour). IDs are per language: British English Blue is 50, American English Blue is 5."""
        if language not in self._lang_ids:
            ids = {}
            try:
                resp = self.http.get(self.cfg["base_url"] + "/colour", params={"language-select": self._lang(language)},
                                     timeout=self.timeout)
                if resp.status_code == 200:
                    for v, n in re.findall(r'<option value="(\d+)"[^>]*>(.*?)</option>', resp.text, re.S):
                        ids[norm(_strip(n))] = int(v)
                        self._lang_names.setdefault(language, {})[norm(_strip(n))] = _strip(n)
            except Exception:
                ids = {}
            self._lang_ids[language] = ids
        return self._lang_ids[language]

    def best_example(self, name, language):
        key = (language, norm(name))
        if key in self._best:               # already seen as the top name of a colour
            return self._best[key]
        cid = self._language_ids(language).get(norm(name))
        if cid is None:                     # older way of learning the names, if the name list can't be fetched
            if language not in self._ids:
                self.lookup("#808080", language)
            cid = self._ids[language].get(norm(name))
        if cid is None:
            raise KeyError(f"'{name}' is not one of the Colour Spinner's {language} names.")
        hexes = []
        try:                                # the page's own request for a name: its first swatch is the best example
            resp = self.http.get(self.cfg["base_url"] + self.cfg["mobile_panels"],
                                 params={"language_id": self._lang(language), "colour_id": cid}, timeout=self.timeout)
            resp.raise_for_status()
            sw = resp.json().get("swatches") or []
            if sw and all(k in sw[0] for k in "rgb"):
                hexes = [rgb_to_hex([int(sw[0]["r"]), int(sw[0]["g"]), int(sw[0]["b"])])]
        except Exception:
            hexes = []
        if not hexes:                       # otherwise the Picker request, asked for that name directly
            form = {"language-select": self._lang(language), "translation-select": self._lang(language),
                    "age-cvd-simulation-select": "none", "synonyms": "on", "antonyms": "on",
                    "r": 128, "g": 128, "b": 128, "exact_colour_id": cid}
            try:
                resp = self.http.post(self.cfg["base_url"] + self.cfg["color_picker"], data=form, timeout=self.timeout)
                if resp.status_code == 200:
                    res = parse_color_picker(resp.text)
                    if res.get("best_example") and same_name(res.get("name", ""), name):
                        hexes = [res["best_example"]]
            except Exception:
                hexes = []
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
_SIM = {"client": None, "available": True, "error": "", "cache": {}, "failures": 0, "retry_at": 0.0, "local": set()}

OFFLINE_URL = ("https://raw.githubusercontent.com/akvilesinkeviciute/colour-brings-people-together/v1.0/"
               "data/spinner_offline.json")
_OFFLINE = {"data": None, "checked": 0.0}
_SPINNER = {"failures": 0, "down_until": 0.0}      # after repeated failures, go straight to the saved copy for a minute

def _offline():
    """The saved copy of the Colour Spinner's answers for the default settings (data/spinner_offline.json). It is
    used only for answers the Spinner doesn't give. It isn't published unless the Spinner is down on the day, so if it
    isn't there, it's looked for again a minute later. Returns {} if there is none."""
    import os, time, urllib.request
    if _OFFLINE["data"] or time.time() - _OFFLINE["checked"] < 60:
        return _OFFLINE["data"] or {}
    _OFFLINE["checked"] = time.time()
    data = {}
    for src in ("spinner_offline.json", "data/spinner_offline.json"):
        if os.path.exists(src):
            data = json.load(open(src, encoding="utf-8")); break
    else:
        try:
            with urllib.request.urlopen(OFFLINE_URL, timeout=20) as f:
                data = json.load(f)
        except Exception:
            data = {}
    if data:
        data["_pages"] = {(h.upper(), lang, tgt): res for h, lang, tgt, res in data.get("pages", [])}
        data["_best"] = {(lang, key): h for lang, key, h in data.get("best", [])}
        data["_members"] = {(lang, key): tiles for lang, key, tiles in data.get("members", [])}
        data["_sims"] = {(h.upper(), c): out.upper() for h, c, out in data.get("sims", [])}
        try:
            show(small(f"Using the saved copy of the Colour Spinner's answers (made {data.get('made', '')}) for anything the "
                       f"Spinner can't answer right now."))
        except Exception:
            pass
    _OFFLINE["data"] = data
    return data

def use_saved_copy():
    """Look for the saved copy straight away (facilitators: after uploading it on the day)."""
    _OFFLINE.update(data=None, checked=0.0)
    return bool(_offline())

def _spinner_down():
    import time
    return time.time() < _SPINNER["down_until"]

def _spinner_failed():
    import time
    _SPINNER["failures"] += 1
    if _SPINNER["failures"] >= 2:                     # twice in a row: skip the Spinner for a minute
        _SPINNER["down_until"] = time.time() + 60; _SPINNER["failures"] = 0

def _with_saved_copy(live, saved_key):
    """Ask the Spinner first; only if it doesn't answer (an error, or an empty answer), use the saved copy."""
    import functools
    @functools.wraps(live)
    def inner(self, *a, **k):
        err, res, asked = None, None, False
        if not _spinner_down():
            asked = True
            try:
                res = live(self, *a, **k)
                if res not in (None, [], {}):
                    _SPINNER["failures"] = 0
                    return res
            except KeyError:
                raise                                  # not one of the Spinner's names: no saved answer either
            except Exception as e:
                err = e
                _spinner_failed()
        saved = saved_key(self, *a, **k)
        if saved not in (None, [], {}):
            return saved
        if err is not None:
            raise err
        if not asked:
            raise RuntimeError("The Colour Spinner isn't answering right now; try again in a minute.")
        return res
    return inner

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
        import time
        if not _SIM["available"] and time.time() >= _SIM["retry_at"]:
            _SIM["available"] = True                      # try the Spinner again after a pause
        if _SIM["available"]:
            try:
                _SIM["cache"][key] = _sim_client().simulate(hx, condition)
                _SIM["failures"] = 0
                return _SIM["cache"][key]
            except Exception as e:
                _SIM["failures"] += 1
                _SIM["error"] = str(e)[:120]
                if _SIM["failures"] >= 2:                 # twice in a row: pause for a minute
                    _SIM["available"] = False; _SIM["retry_at"] = time.time() + 60
        saved = _offline().get("_sims", {}).get(key)      # the Spinner's own answer, saved earlier
        if saved:
            return saved
        raise SimulationUnavailable(_SIM["error"] or "The Colour Spinner didn't answer.")
    return _SIM["cache"][key]

def retry_simulations():
    """Try the Colour Spinner's simulations again after a connection problem."""
    _SIM.update(available=True, error="")

def prefetch_simulations(hexes, conditions=CONDITIONS):
    """Fetch the Spinner's simulations for many colours at once (six requests in parallel)."""
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(h.upper() if h.startswith("#") else "#" + h.upper(), c) for h in dict.fromkeys(hexes) for c in conditions
            if c != "Typical"]
    jobs = [j for j in jobs if j not in _SIM["cache"]]
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
    show('<div class="card"><b>The Colour Spinner didn\'t answer in time</b>'
         '<p style="margin:6px 0 0">This view needs the Spinner\'s simulations for colours that aren\'t in the saved '
         'copy, such as the AI\'s own colours. Wait a minute and run the cell again; it usually works on a second try.</p>'
         + (f'<div class="note">{_h(err)}</div>' if err else "") + '</div>')


def _read_sheet(text):
    rows = [r for r in text.strip().splitlines() if r.strip() and not r.lstrip().startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(rows)), skipinitialspace=True,
                     dtype=str).apply(lambda s: s.str.strip())
    return df[df.iloc[:, 0] != df.columns[0]].reset_index(drop=True)   # drop pasted-in headers

def parse_colour(s):
    """'#E8153A', 'E8153A' or '232 21 56' (as the Mapper shows RGB) -> '#E8153A'."""
    s = str(s).strip()
    if not s:
        return ""
    nums = re.findall(r"\d+", s)
    if len(nums) == 3 and not re.fullmatch(r"#?[0-9A-Fa-f]{6}", s):
        return rgb_to_hex([int(n) for n in nums])
    return "#" + s.lstrip("#").upper()

ApiSpinner.lookup = _with_saved_copy(ApiSpinner.lookup,
    lambda self, h, language, target=None: _offline().get("_pages", {}).get((h.upper(), language, target or language)))
ApiSpinner.best_example = _with_saved_copy(ApiSpinner.best_example,
    lambda self, name, language: _offline().get("_best", {}).get((language, norm(name))))
ApiSpinner.category_members = _with_saved_copy(ApiSpinner.category_members,
    lambda self, name, language: _offline().get("_members", {}).get((language, norm(name))))
def _saved_names(self, language):
    off = _offline()
    if language in off.get("lang_ids", {}):
        self._lang_names[language] = off.get("lang_names", {}).get(language, {})
        return off["lang_ids"][language]
    return None
ApiSpinner._language_ids = _with_saved_copy(ApiSpinner._language_ids, _saved_names)

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

HIMBA_TERMS = {"burou", "dumbu", "grine", "serandu", "vapa", "vinde", "zoozu"}
HIMBA_PREFIXES = ("otji", "ozo", "oru", "omu", "oma", "oka", "otu", "on", "om", "o", "e")

def _himba_core(word):
    """An OtjiHimba colour term without the noun-class prefix an AI often adds (otjiserandu, ondumbu → serandu,
    dumbu). Only the Spinner's seven OtjiHimba terms are recognised, so other languages are unaffected."""
    w = norm(word).replace("-", "").replace(" ", "")
    if w in HIMBA_TERMS:
        return w
    for p in HIMBA_PREFIXES:
        if w.startswith(p) and w[len(p):] in HIMBA_TERMS:
            return w[len(p):]
    return None

def same_name(candidate, name, romanized=None):
    """True if `candidate` (in any script, with or without a bracketed romanisation)
    is the Spinner's `name` or its romanised spelling. For OtjiHimba's terms, a noun-class prefix the AI adds
    (otjiserandu for serandu) doesn't count as a different name."""
    if not isinstance(candidate, str) or not isinstance(name, str):
        return False
    targets = {norm(name)} | ({norm(romanized)} if isinstance(romanized, str) and romanized else set())
    if any(norm(v) in targets for v in name_variants(candidate)):
        return True
    core = _himba_core(name)
    return core is not None and any(_himba_core(v) == core for v in name_variants(candidate))

def try_call(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:
        return e


# ════════════════════════════════════════════════════════════════════════
#  4. Asking an AI system
# ════════════════════════════════════════════════════════════════════════
COLAB_AI = "Gemini built into Colab"

COLAB_MAX_TOKENS = 2048     # the answers are short; a small limit keeps each request within Colab's free quota

def _colab_generate(ai, prompt, model=""):
    """Ask Colab's built-in AI with a small output limit. If the quota still refuses, try the
    cheapest model Colab lists (a 'lite' or Gemma model) before giving up."""
    def call(name):
        kw = {"model_name": name} if name else {}
        for limit in ({"max_output_tokens": COLAB_MAX_TOKENS}, {"max_tokens": COLAB_MAX_TOKENS}, {}):
            try:
                return ai.generate_text(prompt, **kw, **limit)
            except TypeError:            # this version of the library doesn't take that argument
                continue
        return ai.generate_text(prompt, **kw)
    try:
        return call(model)
    except Exception as first:
        quota = any(k in str(first).lower() for k in ("quota", "cost", "403", "429", "rate"))
        if quota and not model:
            try:
                names = [str(m) for m in ai.list_models()]
                cheap = next((n for k in ("lite", "gemma", "flash") for n in names if k in n.lower()), None)
                if cheap:
                    return call(cheap)
            except Exception:
                pass
        raise RuntimeError(
            "Colab's built-in Gemini couldn't answer: its free usage limit has been reached for now. "
            "Choose 'Paste answers from my own AI' in the form, copy the prompt into any AI and paste "
            f"its answers into the boxes. (Details: {str(first)[:160]})") from first

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
                               "Choose 'Paste answers from my own AI' in the form instead.")
        return _colab_generate(ai, prompt, model)
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
.cbt table.babel td, .cbt table.babel th {{ padding:5px 16px 5px 0; vertical-align:middle; text-align:left; }}
.cbt table.babel th {{ font-weight:400; color:{MUTED}; font-size:12px; padding-bottom:8px; }}
.cbt .bar {{ height:5px; background:#E4E4E0; border-radius:3px; width:90px; margin:4px 0 3px; }}
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

def framed(hex_, label="", width=96, height=54, size=15):
    """The framed test-colour swatch used in every table: grey frame, colour, label bottom-left."""
    return (f'<div class="pair" style="width:{width}px;height:{height}px"><div style="background:{hex_};'
            f'color:{_ink_on(hex_)};font-weight:700;font-size:{size}px">{_h(label)}</div></div>')

def swatch(hex_, width=40, height=26, title=""):
    """A plain swatch with a thin grey frame, for colours inside table cells."""
    return (f'<span title="{_h(title or hex_)}" style="display:inline-block;width:{width}px;height:{height}px;'
            f'vertical-align:middle;border-radius:3px;background:{hex_};box-shadow:0 0 0 3px {SURROUND}"></span>')

def bar(frac, colour=None, width=90):
    """A thin horizontal bar, as in the names table. frac in 0–1."""
    w = int(round(100 * max(0.0, min(1.0, 0 if pd.isna(frac) else frac))))
    return (f'<div class="bar" style="width:{width}px"><i style="width:{w}%;'
            f'{"background:" + colour if colour else ""}"></i></div>')

def small(text, colour=None):
    return f'<span style="color:{colour or MUTED};font-size:11.5px">{text}</span>'

def table(head, rows, note=None, valign="middle"):
    """A table in the house style: muted headers, airy rows, no borders."""
    h = "".join(f"<th>{c}</th>" for c in head)
    v = f' style="vertical-align:{valign}"'
    b = "".join("<tr>" + "".join(f"<td{v}>{c}</td>" for c in r) + "</tr>" for r in rows)
    show(f'<table class="babel"><tr>{h}</tr>{b}</table>' + (f'<div class="note">{note}</div>' if note else ""))

def pill(text, kind="neutral"):
    bg, fg = {"good": _GOOD, "mid": _MID, "bad": _BAD}.get(kind, ("#ECECE8", INK))
    return f'<span class="badge" style="background:{bg};color:{fg}">{text}</span>'

def _fit_to_window(*args, **kwargs):
    """Show every open figure as a picture that shrinks to fit the window (Colab otherwise keeps plots at a fixed
    size, so wide ones are cut off or scroll sideways on a small screen). Sharp on high-resolution screens too."""
    import base64
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        png = buf.getvalue()
        w_px = int.from_bytes(png[16:20], "big") * 100 // 200       # its natural width, at Colab's usual 100 dpi
        display(HTML(f'<img src="data:image/png;base64,{base64.b64encode(png).decode()}" '
                     f'style="width:{w_px}px;max-width:100%;height:auto" alt="">'))

try:                                                   # in a notebook, every plt.show() fits the window
    from IPython import get_ipython
    if get_ipython() is not None:
        plt.show = _fit_to_window
except Exception:
    pass

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
def show_conditions(hexes, labels=None, conditions=CONDITIONS, width=40, note=False):   # references: credits cell
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

def show_babel(table, languages, labels=None, first=""):
    """table: rows with sdg, hex, and per-language '<lang>' name and '<lang>|conf' columns."""
    labels = labels or {}
    head = f"<tr><th>{first}</th>" + "".join(f"<th>{labels.get(l, _h(l))}</th>" for l in languages) + "</tr>"
    body = ""
    for _, r in table.iterrows():
        cells = ""
        for l in languages:
            n, c, rom = r.get(l), r.get(f"{l}|conf"), r.get(f"{l}|rom")
            if isinstance(n, str):
                w = 0 if pd.isna(c) else int(round(100 * c))
                extra = (f' <span style="color:{MUTED};font-size:11px">{_h(rom)}</span>'
                         if isinstance(rom, str) and rom and norm(rom) != norm(n) else "")
                light = ";background:#BDBDB8" if w < LOW_CONFIDENCE * 100 else ""      # contested names stand out
                cells += (f'<td><b>{_h(n)}</b>{extra}<div class="bar"><i style="width:{w}%{light}"></i></div>'
                          f'<span style="color:{MUTED};font-size:11px">{w}% confidence</span></td>')
            else:
                cells += f'<td style="color:{MUTED}">—</td>'
        body += (f'<tr><td><div class="pair" style="width:96px"><div style="background:{r.hex};'
                 f'color:{_ink_on(r.hex)};font-weight:700;font-size:15px">{r.name}</div></div></td>{cells}</tr>')
    show(f'<table class="babel">{head}{body}</table>'
         f'<div class="note">Lighter bars: below {LOW_CONFIDENCE:.0%} confidence, where speakers often use other names too.</div>')

_GOOD, _MID, _BAD = ("#DDEFE3", "#14532D"), ("#FBF0D0", "#6B4E00"), ("#F9DADA", "#7F1D1D")
DELTA_E_THRESHOLD = 10      # ΔE₀₀: the threshold used throughout the masterclass, in Part B and Part C
HARD_TO_NAME = 0.5          # naming confidence below this counts as hard to name
LOW_CONFIDENCE = 0.5        # in the names table, bars below this are drawn lighter

def sim_phrase(condition):
    """'Deuteranopia' -> 'deuteranopia-simulated colours'; 'Age 80' -> 'colours simulated for age 80'."""
    if condition == "Typical": return "typical colours"
    if condition.startswith("Age"): return f"colours simulated for {condition.lower()}"
    return f"{condition.lower()}-simulated colours"

def de_badge(d, threshold, small_is_good=True, prefix="ΔE₀₀ = "):
    """ΔE₀₀ badge. small_is_good=True for 'how far off' (agreement);
    False for 'how far apart' (distinctiveness), where small is the risk."""
    if pd.isna(d): return '<span class="badge" style="background:#E4E4E0">n/a</span>'
    if small_is_good:
        bg, fg = _GOOD if d <= threshold / 2 else _MID if d <= threshold else _BAD
    else:
        bg, fg = _BAD if d < threshold / 2 else _MID if d < threshold else _GOOD
    return f'<span class="badge" style="background:{bg};color:{fg}">{prefix}{d:.1f}</span>'

def show_pairs(df, left="ai_hex", right="spinner_hex", lab_left="AI", lab_right="Speakers",
               threshold=DELTA_E_THRESHOLD, extra_cols=()):
    """For each name: the AI's colour, the speakers' best example, and the ΔE₀₀ between them as a bar."""
    top = max(2 * threshold, float(np.nanmax(df["de00"])) if df["de00"].notna().any() else 1)
    rows, last = [], None
    for _, r in df.iterrows():
        lang = f"<b>{_h(r.language)}</b>" if r.language != last else ""
        last = r.language
        a = swatch(r[left]) if isinstance(r[left], str) else small("missing")
        b = swatch(r[right]) if isinstance(r[right], str) else small("missing")
        d = r.get("de00")
        if pd.isna(d):
            dist = small("n/a")
        else:
            col = "#B4342D" if d > threshold else ("#C9A227" if d > threshold / 2 else "#3E8E5E")
            dist = f'<b>ΔE₀₀ = {d:.1f}</b>{bar(d / top, col, 160)}'
        rows.append([lang, f"<b>{_h(r['name'])}</b>", a, b, dist])
    table(["Language", "Name", lab_left, lab_right, "ΔE₀₀ between them"], rows,
          f"ΔE₀₀ ≈ 1 is just noticeable side by side; above {threshold:g} (red) the two read as different "
          "colours and the AI's colour will be replaced.")

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
    "axes.grid": False, "axes.linewidth": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False, "legend.fontsize": 10,
})

# Red below the threshold (shown at the middle of the scale), calm neutrals above it.
RISK_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "risk", [(0, "#8E1B1B"), (0.4999, "#E7B3A3"), (0.5, "#E3E8DF"), (1, PAPER)])

def chip(ax, y, hex_, label, height=0.72, gap_in=0.12, width_in=1.0, frame_in=0.06, radius_in=0.06, height_in=None):
    """A test-colour chip drawn just left of the plot area at data height y, styled like the
    framed swatches in the tables: a grey frame, the colour inside, its number bottom-left."""
    fig = ax.figure
    pos = ax.get_position()
    ax_w_in, ax_h_in = pos.width * fig.get_figwidth(), pos.height * fig.get_figheight()
    y_per_in = abs(np.diff(ax.get_ylim())[0]) / ax_h_in
    blend = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    aspect = y_per_in * ax_w_in
    if height_in is not None:                 # a fixed size in inches, like the swatches in the tables
        height = height_in * y_per_in
    def box(x_in, w_in, h_data, colour, r_in):
        ax.add_patch(mpl.patches.FancyBboxPatch((x_in / ax_w_in, y - h_data / 2), w_in / ax_w_in, h_data,
                     boxstyle=f"round,pad=0,rounding_size={r_in / ax_w_in}", mutation_aspect=aspect,
                     color=colour, transform=blend, clip_on=False))
    x0 = -(gap_in + width_in)
    box(x0, width_in, height, SURROUND, radius_in)                                   # grey frame
    box(x0 + frame_in, width_in - 2 * frame_in, height - 2 * frame_in * y_per_in, hex_, radius_in / 3)
    ax.text((x0 + frame_in + 0.07) / ax_w_in, y + height / 2 - frame_in * y_per_in - 0.06 * y_per_in, label,
            transform=blend, ha="left", va="bottom", color=_ink_on(hex_), fontsize=12, fontweight="bold",
            clip_on=False)

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
            ax.text(c, r - (0.09 if neighbours is not None else 0), f"ΔE₀₀ = {v:.1f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=ink)
            if neighbours is not None:
                n = neighbours.iat[r, c]
                seen = simulate(SDG.hex[n], cols[c])            # the neighbour as it looks under this condition
                ax.add_patch(mpl.patches.Rectangle((c - 0.30, r + 0.13), 0.13, 0.15, fc=seen, ec="white", lw=0.8))
                ax.text(c - 0.12, r + 0.205, f"vs #{n}", ha="left", va="center", fontsize=8.5, color=ink)
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
    """Each cell: the test colour and its nearest neighbour as that viewer sees them, side by side on the grey frame,
    with the ΔE₀₀ between them as a badge underneath (green, amber, red as in the translatability grid). The colours
    and the numbers are kept apart, so the ΔE₀₀ can't be mistaken for a test colour's number."""
    if nn_de is None: return
    pick, cols = list(nn_de.index), list(nn_de.columns)
    nr, nc = nn_de.shape
    below = int((nn_de.values < threshold).sum())
    CW, RH = 1.55, 1.25                                    # a cell's width and a row's height, in inches
    W, H = CW * nc + 1.5, RH * nr + 1.5
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, nc); ax.set_ylim(nr, -0.55); ax.axis("off")
    for c, name in enumerate(cols):
        ax.text(c + 0.5, -0.3, condition_label(name), ha="center", va="center", fontsize=10, color=INK)
    for r, i in enumerate(pick):
        for c, cond in enumerate(cols):
            v, n = nn_de.iat[r, c], nn_id.iat[r, c]
            if np.isnan(v):
                continue
            # the two colours, as this viewer sees them, in one grey frame
            x0, y0, fw, fh = c + 0.08, r + 0.06, 0.84, 0.46
            ax.add_patch(mpl.patches.FancyBboxPatch((x0, y0), fw, fh, boxstyle="round,pad=0,rounding_size=0.05", fc=SURROUND, ec="none"))
            for k, h in enumerate((SDG.hex[i], SDG.hex[n])):
                ax.add_patch(mpl.patches.Rectangle((x0 + 0.05 + k * (fw - 0.1) / 2, y0 + 0.06), (fw - 0.1) / 2 - 0.01, fh - 0.12,
                                                   fc=simulate(h, cond), ec="none"))
            # the ΔE₀₀ underneath, as a badge, and which neighbour it is
            # green: 10 or more; yellow: 5 to 10; red: below 5
            fc, ink = (("#DCEFE3", "#1E5A34") if v >= threshold else ("#FFF1A8", "#6B5600") if v >= threshold / 2
                       else ("#F6D5D3", "#8E1B1B"))
            ax.add_patch(mpl.patches.FancyBboxPatch((c + 0.04, r + 0.6), 0.6, 0.28, boxstyle="round,pad=0,rounding_size=0.12",
                                                    fc=fc, ec="none"))
            ax.text(c + 0.34, r + 0.745, f"ΔE₀₀ = {v:.1f}", ha="center", va="center", fontsize=9.5, fontweight="bold", color=ink)
            ax.text(c + 0.68, r + 0.745, f"vs #{n}", ha="left", va="center", fontsize=9, color=MUTED)
    import textwrap
    sub = textwrap.fill(f"Each cell: the test colour (left) and its nearest other test colour (right), as that viewer sees them, "
                        f"and the ΔE₀₀ between them. Below {threshold:g}, they're hard to tell apart at a glance "
                        f"({below} of {nn_de.size} cells). Green: ΔE₀₀ {threshold:g} or more; yellow: {threshold / 2:g} to {threshold:g}; red: below {threshold / 2:g}. "
                        f"#: the neighbour's test colour number.",
                        width=int(W * 12.5))
    ax.set_title("How close is each test colour to its nearest neighbour in your palette?", loc="left", pad=14 + 13 * (sub.count("\n") + 1))
    ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 8), textcoords="offset points", ha="left", va="bottom",
                fontsize=9.5, color=MUTED, linespacing=1.3)
    fig.tight_layout()
    fig.subplots_adjust(left=max(fig.subplotpars.left, 1.3 / W))
    ax.set_ylim(nr, -0.55)
    for r, i in enumerate(pick):                           # the test colours' own numbers, on the left only
        chip(ax, r + 0.29, SDG.hex[i], str(i), height_in=0.62)
    plt.show()

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
                     f'<span style="white-space:nowrap"><b>ΔE₀₀ = {typical:.1f} → {d:.1f}</b></span><br>'
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
                badge = f'<div style="margin-top:5px">{de_badge(de00(best[source], b), threshold, prefix="ΔE₀₀ = ")}</div>'
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
    """A point inside the panel, or the place where its line leaves the panel (and whether it was moved).
    A margin keeps every shape whole: points closer to the edge than that are treated as lying outside."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    mx, my = 0.06 * (x1 - x0), 0.06 * (y1 - y0)
    x0, x1, y0, y1 = x0 + mx, x1 - mx, y0 + my, y1 - my
    p, c = np.asarray(p, float), np.asarray(centre, float)
    if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
        return p, False
    d = p - c
    t = min([((x1 if d[0] > 0 else x0) - c[0]) / d[0] if d[0] else np.inf,
             ((y1 if d[1] > 0 else y0) - c[1]) / d[1] if d[1] else np.inf])
    return c + d * t, True

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
        groups.append((f"Test colour #{i}", o, pts))
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

SIMILARITY_URL = ("https://raw.githubusercontent.com/akvilesinkeviciute/colour-brings-people-together/v1.0/"
                  "data/spinner_similarity.json")
NAME_DISTINCT = 0.7         # name distinctiveness below this: as alike as a #1 synonym in the Spinner's Thesaurus
_NAME_SIMILARITY = None

def _similarity_data():
    """The Colour Spinner's name-similarity matrices (the JSD-based data behind its Thesaurus), per language."""
    global _NAME_SIMILARITY
    if _NAME_SIMILARITY is None:
        import os, urllib.request
        for src in ("spinner_similarity.json", "data/spinner_similarity.json"):
            if os.path.exists(src):
                _NAME_SIMILARITY = json.load(open(src, encoding="utf-8")); break
        else:
            try:
                with urllib.request.urlopen(SIMILARITY_URL, timeout=20) as f:
                    _NAME_SIMILARITY = json.load(f)
            except Exception:
                _NAME_SIMILARITY = {}
        for key, m in _NAME_SIMILARITY.items():              # look names up by their Latin spelling, ignoring accents
            m["index"] = {norm(n.replace("_", " ")): k for k, n in enumerate(m["names"])}
    return _NAME_SIMILARITY

def _sim_key(language):
    return {"OtjiHimba": "himba"}.get(language, language.lower().replace(" ", "_"))

def name_similarity(language, a, b, rom_a=None, rom_b=None):
    """How alike two names are used in `language` (1 = the same name), or NaN if either isn't in the data."""
    if norm(a) == norm(b):
        return 1.0
    m = _similarity_data().get(_sim_key(language))
    if not m:
        return np.nan
    def idx(n, r):
        for c in (n, r):
            if isinstance(c, str) and norm(c.replace("_", " ")) in m["index"]:
                return m["index"][norm(c.replace("_", " "))]
        return None
    i, j = idx(a, rom_a), idx(b, rom_b)
    return float(m["similarity"][i][j]) if i is not None and j is not None else np.nan

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
    # name distinctiveness: 1 − the highest similarity between this colour's name and another test colour's name
    if language not in (ALL_LANGUAGES, AUDIENCE_AVG):
        rows = naming[naming.language == language].drop_duplicates("sdg").set_index("sdg")
        def nm(i): return rows.at[i, "name"] if i in rows.index else None
        def rm(i): return rows.at[i, "romanized"] if i in rows.index and "romanized" in rows else None
        dist = {}
        for i in pick:
            sims = [name_similarity(language, nm(i), nm(j), rm(i), rm(j)) for j in pick
                    if j != i and isinstance(nm(i), str) and isinstance(nm(j), str)]
            sims = [v for v in sims if not pd.isna(v)]
            dist[i] = 1 - max(sims) if sims else np.nan
        card["distinct_0to1"] = pd.Series(dist)
    for l in conf.columns:
        card[l] = [f"{names.at[i, l]} ({conf.at[i, l]:.0%})" if i in conf.index and not pd.isna(conf.at[i, l]) else ""
                   for i in pick]
    card.attrs.update(lo=lo, hi=hi, language=language, condition=condition, normalise=normalise)
    card.index.name = "SDG"
    return card

def plot_two_axes(card, confusable_below=DELTA_E_THRESHOLD, hard_below=HARD_TO_NAME, top=None):
    """Colour identity scorecard: name distinctiveness (across; from the Spinner's name similarity, the data
    behind its Thesaurus) against nearest-colour ΔE₀₀ (up), the ΔE₀₀ axis reaching just past the largest
    distance. Falls back to naming confidence across if the similarity data can't be loaded."""
    if card is None:
        return
    if top is None:              # the axis reaches past the largest distance, leaving room for the corner labels
        biggest = max(float(card["dE00"].max()) if card["dE00"].notna().any() else 0, confusable_below) + 1
        top = confusable_below + (biggest - confusable_below) / 0.8
    by_name = "distinct_0to1" in card and card["distinct_0to1"].notna().any()
    xcol = "distinct_0to1" if by_name else "confidence_0to1"
    if by_name:
        hard_below = NAME_DISTINCT
    elif "distinct_0to1" in card:
        show(small("The Spinner's name-similarity data couldn't be loaded, so this shows naming confidence across."))
    if card["dE00"].isna().all():
        show("<b>The scorecard needs at least two test colours</b> " + small("to measure how far each is from the nearest other one. Add more in section 3."))
        return
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    labels = (["Distinct to the eye, alike by name", "Distinct to the eye and by name",
               "Alike to the eye and by name", "Alike to the eye, distinct by name"] if by_name else
              ["Distinct, hard to name", "Distinct + nameable", "Confusable + hard to name", "Nameable, confusable"])
    # Each axis is stretched in two straight pieces so that its threshold sits in the middle: four equal squares.
    def fx(v): return 0.5 * v / hard_below if v <= hard_below else 0.5 + 0.5 * (v - hard_below) / (1 - hard_below)
    def fy(v): return 0.5 * v / confusable_below if v <= confusable_below else 0.5 + 0.5 * (v - confusable_below) / (top - confusable_below)
    zones = [((0, 0.5), "#F4F4F2", labels[0], "left", "top"), ((0.5, 0.5), "#E2EEE7", labels[1], "right", "top"),
             ((0, 0), "#F6E9E2", labels[2], "left", "bottom"), ((0.5, 0), "#F4F4F2", labels[3], "right", "bottom")]
    for (x, y), col, label, ha, va in zones:
        ax.add_patch(mpl.patches.Rectangle((x, y), 0.5, 0.5, fc=col, ec="none", zorder=0))
        ax.text(x + (0.015 if ha == "left" else 0.485), y + (0.488 if va == "top" else 0.012), label,
                ha=ha, va=va, fontsize=9, color="#7A7A75", zorder=1)
    ax.axvline(0.5, color="#9A9A95", ls=(0, (4, 3)), lw=1.2, zorder=1)
    ax.axhline(0.5, color="#9A9A95", ls=(0, (4, 3)), lw=1.2, zorder=1)
    pts = [(i, fx(r[xcol]), fy(min(r.dE00, top)), r.dE00) for i, r in card.iterrows()
           if not (pd.isna(r[xcol]) or pd.isna(r.dE00))]
    # Two test colours that are each other's nearest neighbour share the same ΔE₀₀, so their discs can land on
    # top of each other: nudge any overlapping discs apart sideways so every colour stays visible.
    groups = []
    for p_ in pts:
        g = next((g for g in groups if abs(g[0][1] - p_[1]) < 0.06 and abs(g[0][2] - p_[2]) < 0.05), None)
        (g.append(p_) if g else groups.append([p_]))
    nudged = False
    for g in groups:
        g = sorted(g, key=lambda t: t[1])                  # left to right, spread evenly around their middle
        centre = sum(t[1] for t in g) / len(g)
        offs = [(k - (len(g) - 1) / 2) * 0.075 for k in range(len(g))]
        shift = max(0.0, 0.045 - (centre + min(offs))) - max(0.0, centre + max(offs) - 0.955)   # keep discs inside
        for k, (i, x, y, raw) in enumerate(g):
            dx = centre + offs[k] + shift - x
            nudged |= len(g) > 1
            ax.scatter(x + dx, y, s=900, color=SDG.hex[i], edgecolor=SURROUND, lw=1.5, zorder=3)
            ax.text(x + dx, y, str(i), ha="center", va="center", color=ink_on(SDG.hex[i]), fontweight="bold",
                    fontsize=12, zorder=4)
            if raw > top:
                ax.text(x + dx, y + 0.035, f"↑ ΔE₀₀ = {raw:.0f}", ha="center", va="bottom", fontsize=8.5, color=MUTED, zorder=4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if by_name:
        ax.set_xticks([fx(v) for v in (0, 0.35, hard_below, 0.85, 1)], ["0\nsame name", "0.35", f"{hard_below:g}", "0.85", "1\nunrelated"])
    else:
        ax.set_xticks([fx(0), fx(hard_below), fx(1)], ["Low", f"{hard_below:.0%}", "High"])
    from matplotlib.ticker import MaxNLocator
    ticks = [t for t in MaxNLocator(4, integer=True).tick_values(confusable_below, top) if confusable_below < t <= top]
    ys = sorted(set(ticks) | {0, confusable_below / 2, confusable_below})
    ax.set_yticks([fy(t) for t in ys], [f"{t:g}" for t in ys])
    ax.spines["left"].set_color(INK); ax.spines["bottom"].set_color(INK)
    lab = card.attrs["language"]
    who = ("your audience on average" if lab == AUDIENCE_AVG else "all your languages on average" if lab == ALL_LANGUAGES else lab)
    ax.set_xlabel("Name distinctiveness (1 − similarity to the nearest other test colour's name)" if by_name
                  else "Naming confidence")
    ax.set_ylabel("ΔE₀₀ to the nearest other test colour")
    cond = card.attrs["condition"]
    seen = ("with typical colour vision" if cond == "Typical" else
            f"as seen at {cond.lower()}" if cond.startswith("Age") else f"as seen with {cond.lower()}")
    _titles(ax, "Colour identity scorecard",
            (f"Names: {who}, compared with the Colour Spinner's name similarity (the data behind its Thesaurus). "
             f"ΔE₀₀: from each test colour to the nearest other test colour, {seen}. Dashed lines: ΔE₀₀ {confusable_below:g} and name distinctiveness "
             f"{hard_below:g}, as alike as a top synonym; each axis is stretched so its threshold sits in the middle." if by_name else
             f"Confidence: {who}. ΔE₀₀: from each test colour to the nearest other test colour, {seen}. Dashed lines: ΔE₀₀ {confusable_below:g} and "
             f"{hard_below:.0%} confidence.") + (" Discs at the same spot are set side by side." if nudged else ""))
    plt.tight_layout(); plt.show()

def scorecard_pairs(pick, naming, language, condition="Typical"):
    """One row per pair of test colours: the ΔE₀₀ between the two colours (as seen with `condition`, the same
    numbers as section 4's 'Pairs at risk') and the name distinctiveness between their names in `language`
    (1 − the Colour Spinner's name similarity, the data behind its Thesaurus)."""
    from itertools import combinations
    pick = list(dict.fromkeys(pick))
    if len(pick) < 2:
        return None
    prefetch_simulations([SDG.hex[i] for i in pick])
    rows = naming[naming.language == language].drop_duplicates("sdg").set_index("sdg")
    def nm(i): return rows.at[i, "name"] if i in rows.index else None
    def rm(i): return rows.at[i, "romanized"] if i in rows.index and "romanized" in rows else None
    out = []
    for i, j in combinations(pick, 2):
        d = de00(simulate(SDG.hex[i], condition), simulate(SDG.hex[j], condition))
        sim = name_similarity(language, nm(i), nm(j), rm(i), rm(j)) if isinstance(nm(i), str) and isinstance(nm(j), str) else np.nan
        out.append(dict(a=i, b=j, dE00=d, distinct=1 - sim if not pd.isna(sim) else np.nan, name_a=nm(i), name_b=nm(j)))
    card = pd.DataFrame(out)
    card.attrs.update(language=language, condition=condition)
    return card

PART_B_THRESHOLD = DELTA_E_THRESHOLD   # ΔE₀₀ 10, the same in Part B and Part C
SCORECARD_DE = PART_B_THRESHOLD   # the scorecard's lines: ΔE₀₀ 10 between the two colours,
SCORECARD_SIMILARITY = 0.5  # and name similarity 0.5 (name distinctiveness 0.5)

def plot_pair_scorecard(card, confusable_below=SCORECARD_DE, name_line=1 - SCORECARD_SIMILARITY):
    """The colour identity scorecard, one point per pair of test colours: name distinctiveness between their
    names (across) against the ΔE₀₀ between the colours (up). Each axis is stretched in two straight pieces
    so that its threshold sits in the middle; every point sits at its true value."""
    if card is None or card.empty:
        show("<b>The scorecard needs at least two test colours.</b> " + small("Add more in section 3.")); return
    ok = card.dropna(subset=["dE00", "distinct"])
    if ok.empty:
        show(small("The Spinner's name-similarity data couldn't be loaded for these names, so the scorecard can't be drawn.")); return
    biggest = max(float(ok.dE00.max()), confusable_below) + 1
    top = confusable_below + (biggest - confusable_below) / 0.8                  # room above for the corner labels
    def fx(v): return 0.5 * v / name_line if v <= name_line else 0.5 + 0.5 * (v - name_line) / (1 - name_line)
    def fy(v): return 0.5 * v / confusable_below if v <= confusable_below else 0.5 + 0.5 * (v - confusable_below) / (top - confusable_below)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    zones = [((0, 0.5), "#F4F4F2", "Distinct to the eye, alike by name", "left", "top"),
             ((0.5, 0.5), "#E2EEE7", "Distinct to the eye and by name", "right", "top"),
             ((0, 0), "#F6E9E2", "Alike to the eye and by name", "left", "bottom"),
             ((0.5, 0), "#F4F4F2", "Alike to the eye, distinct by name", "right", "bottom")]
    for (x, y), col, label, ha, va in zones:
        ax.add_patch(mpl.patches.Rectangle((x, y), 0.5, 0.5, fc=col, ec="none", zorder=0))
        ax.text(x + (0.015 if ha == "left" else 0.485), y + (0.488 if va == "top" else 0.012), label,
                ha=ha, va=va, fontsize=10.5, fontweight="bold", color=INK, zorder=1)
    ax.axvline(0.5, color="#9A9A95", ls=(0, (4, 3)), lw=1.2, zorder=1)
    ax.axhline(0.5, color="#9A9A95", ls=(0, (4, 3)), lw=1.2, zorder=1)
    # each pair: a disc split in two halves, one in each colour, labelled with both numbers
    pts = [[fx(r.distinct), fy(min(r.dE00, top)), int(r.a), int(r.b)] for _, r in ok.iterrows()]
    ax_w, ax_h = fig.get_figwidth() * 0.80, fig.get_figheight() * 0.70        # the plotting area, in inches
    rx = 0.039; ry = rx * ax_w / ax_h                                          # round on screen
    W, H = 2 * rx * 1.12, 2 * ry * 1.08                                         # a point, its numbers inside
    orig = [p_[0] for p_ in pts]
    for _ in range(300):                       # push overlapping points apart sideways; heights stay true
        moved = False
        for m in range(len(pts)):
            for n in range(m + 1, len(pts)):
                dx, dy = pts[n][0] - pts[m][0], pts[n][1] - pts[m][1]
                if abs(dx) < W and abs(dy) < H:
                    push = (W - abs(dx)) / 2 + 1e-4
                    sgn = 1 if (dx > 0 or (dx == 0 and orig[n] >= orig[m])) else -1
                    pts[m][0] -= sgn * push; pts[n][0] += sgn * push; moved = True
        for p_, o in zip(pts, orig):                   # never across the dashed line, never off the plot
            lo, hi = (rx + 0.012, 0.5 - rx * 0.6) if o < 0.5 else (0.5 + rx * 0.6, 1 - rx - 0.012)
            p_[0] = min(max(p_[0], lo), hi)
        if not moved:
            break
    nudged = any(abs(p_[0] - o) > 1e-3 for p_, o in zip(pts, orig))
    for cx, y, i, j in pts:
        for col, t1, t2 in ((SDG.hex[i], 90, 270), (SDG.hex[j], -90, 90)):
            th = np.radians(np.linspace(t1, t2, 40))
            ax.add_patch(mpl.patches.Polygon(np.column_stack([cx + rx * np.cos(th), y + ry * np.sin(th)]),
                                             closed=True, fc=col, ec="none", zorder=3))
        ax.add_patch(mpl.patches.Ellipse((cx, y), 2 * rx, 2 * ry, fc="none", ec=SURROUND, lw=1.2, zorder=4))
        for num, col, side in ((i, SDG.hex[i], -1), (j, SDG.hex[j], 1)):     # each colour's number in its own half
            ax.text(cx + side * rx * 0.46, y, str(num), ha="center", va="center", fontsize=9.5 if num < 10 else 8.5,
                    fontweight="bold", color=_ink_on(col), zorder=5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    from matplotlib.ticker import MaxNLocator
    ticks = [t for t in MaxNLocator(4, integer=True).tick_values(confusable_below, top) if confusable_below < t <= top]
    ys = sorted(set(ticks) | {0, confusable_below / 2, confusable_below})
    ax.set_yticks([fy(t) for t in ys], [f"{t:g}" for t in ys])
    ax.set_xticks([fx(v) for v in (0, name_line / 2, name_line, (1 + name_line) / 2, 1)],
                  ["0\nsame name", f"{name_line / 2:g}", f"{name_line:g}", f"{(1 + name_line) / 2:g}", "1\nunrelated"])
    ax.spines["left"].set_color(INK); ax.spines["bottom"].set_color(INK)
    ax.set_xlabel("Name distinctiveness between the two names (1 − the Spinner's name similarity)")
    ax.set_ylabel("ΔE₀₀ between the two colours")
    cond = card.attrs["condition"]
    seen = ("with typical colour vision" if cond == "Typical" else
            f"as seen at {cond.lower()}" if cond.startswith("Age") else f"as seen with {cond.lower()}")
    _titles(ax, "Colour identity scorecard",
            f"Each disc is a pair of your test colours: its left half is one colour, its right half the other, with their numbers. "
            f"Up: how different the two look (ΔE₀₀ between them, {seen.replace('with ', 'with ').replace('as seen ', '')}). "
            f"Across: how differently their names are used in {card.attrs['language']} (0: the same name; 1: unrelated names). "
            f"Dashed lines: ΔE₀₀ = {confusable_below:g} and name distinctiveness {name_line:g}; each axis is stretched so its line sits in the middle."
            + (" Points that would overlap are moved sideways; their heights are exact." if nudged else ""))
    plt.tight_layout(); plt.show()

def show_scorecard(card, top=40):
    if card is None:
        return
    langs = [c for c in card.columns if c not in ("dE00", "perceptual_0to1", "confidence_0to1")]
    rows = []
    for i, r in card.iterrows():
        perc = f'<b>ΔE₀₀ = {r.dE00:.1f}</b>{bar(r.dE00 / top)}{small("to the nearest test colour")}'
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

def show_name_comparison(cmp, threshold=DELTA_E_THRESHOLD, you=None):
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
                    where = f' {swatch(r.ai_best, 22, 14)} {small(f"ΔE₀₀ = {r.ai_drift:.1f}")}'
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
            if not up:
                raise ValueError("No file was uploaded. Run this cell again and choose partB_results.csv, "
                                 "or set NamesFrom to 'Type them in'.")
            partb = pd.read_csv(io.BytesIO(next(iter(up.values()))), encoding="utf-8-sig")
        else:
            partb = pd.read_csv("partB_results.csv", encoding="utf-8-sig")
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
        d = de_badge(r.de00, threshold, prefix="ΔE₀₀ = ") if not pd.isna(r.get("de00", np.nan)) else small("n/a")
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
    ok = per_test_colour(pal).dropna(subset=["ai_hex", "spinner_hex"])     # names shared by test colours: in each panel
    if ok.empty: return
    by_sdg = "sdg" in ok and ok.sdg.notna().all()
    order = list(dict.fromkeys(_BY_SDG.sdg.astype(int))) if _BY_SDG is not None and len(_BY_SDG) else []
    keys = ([k for k in order if k in set(ok.sdg.astype(int))] or list(dict.fromkeys(ok.sdg.astype(int)))) if by_sdg \
           else [l for l in languages if l in set(ok.language)]
    groups = []
    for k in keys:
        g = ok[ok.sdg == k] if by_sdg else ok[ok.language == k]
        pts = [rgb_to_lab(h)[1:] for h in pd.concat([g.ai_hex, g.spinner_hex])]
        centre = rgb_to_lab(SDG.hex[k])[1:] if by_sdg else np.mean(pts, axis=0)
        groups.append((f"Names for test colour #{k}" if by_sdg else k, centre, pts + [centre]))
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
    """Per language, and for all languages together: how far the AI's colours were from speakers' best
    examples, as the mean ΔE₀₀ and its standard deviation (SD) across names, and the worst."""
    ok = pal.dropna(subset=["ai_hex", "spinner_hex", "de00"])
    top = max(2 * threshold, float(ok.de00.max()) if len(ok) else 1)
    def row(label, g, bold=False):
        n_bad = int((g.de00 > threshold).sum())
        sd = f"SD = {g.de00.std(ddof=1):.1f}" if len(g) > 1 else "—"          # sample SD; one name has no spread
        name = f"<b>{label}</b>" if not bold else f"<b>{label}</b>"
        return [name, f'<b>ΔE₀₀ = {g.de00.mean():.1f}</b>{bar(g.de00.mean() / top, None, 160)}', f"<b>{sd}</b>",
                f"<b>ΔE₀₀ = {g.de00.max():.1f}</b>"]
    rows = [row(_h(l), ok[ok.language == l]) for l in languages if not ok[ok.language == l].empty]
    if len(rows) > 1:
        rows.append(row("All languages", ok, True))
    show("<b>Errors by language</b>")
    table(["Language", "Mean ΔE₀₀", "SD", "Worst ΔE₀₀"], rows,
          note="Mean and SD: the average ΔE₀₀ between the AI's colours and speakers' best examples, and how much it "
               "varies from name to name (standard deviation). A small SD means the AI is consistently close or "
               "consistently off; a large one means it gets some names right and others badly wrong.")

# ── Part C: grounding ─────────────────────────────────────────────────────
def show_grounded(pal, languages, title="Your grounded palette"):
    """The grounded palette. From Part B: one row per test colour, like the other grids; otherwise per language."""
    def status(r):
        return (pill(f"renamed from {r.renamed_from}", "mid") if isinstance(r.get("renamed_from"), str) else
                pill("optimised", "mid") if isinstance(r.get("repaired_from"), str) else
                pill("replaced", "mid") if r.replaced else pill("kept", "good"))
    def cell(r):
        ai = swatch(r.ai_hex, 26, 16, "AI's colour") if isinstance(r.ai_hex, str) else small("—")
        return (f'{framed(r.grounded_hex, r["name"], 112, 54, 13)}'
                f'<div style="margin-top:6px">{status(r)} {small("AI:")} {ai}</div>')
    ok = pal[pal.grounded_hex.notna()]
    note = (f"{int(ok.replaced.sum())} of {len(ok)} colours differ from the AI's. kept: the AI's colour matched · "
            "replaced: the speakers' best example · renamed: a close synonym, because two test colours shared a name · "
            "optimised: another colour of the same name, to tell it apart from the palette's other colours. The small swatch is the AI's original colour.")
    if _BY_SDG is not None and len(_BY_SDG):
        get = _grounded_lookup(ok)
        role = dict(zip(pal.language, pal.get("role", pd.Series("", index=pal.index))))
        langs = [l for l in languages if l in set(ok.language)]
        rows = []
        for i in dict.fromkeys(_BY_SDG.sdg.astype(int)):
            cells = []
            for l in langs:
                g = _BY_SDG[(_BY_SDG.sdg == i) & (_BY_SDG.language == l)]
                r = get(l, i, g["name"].iloc[0]) if not g.empty else None
                cells.append(cell(r) if r is not None else small("—"))
            rows.append([framed(SDG.hex[i], str(i))] + cells)
        head = ["Your test palette"] + [(f"<b style='color:{INK}'>{_h(l)}</b> (you)" if role.get(l) == "your language" else _h(l))
                                        for l in langs]
        show(f"<b>{_h(title)}</b>")
        table(head, rows, valign="top", note=note)
        return
    rows = []
    for lang in languages:
        p = ok[ok.language == lang]
        if p.empty: continue
        cells = "".join(f'<div style="display:inline-block;margin:0 16px 10px 0;vertical-align:top">{cell(r)}</div>'
                        for _, r in p.iterrows())
        rows.append([f"<b>{_h(lang)}</b>", cells])
    show(f"<b>{_h(title)}</b>")
    table(["Language", "Grounded palette"], rows, note=note)

NOTICEABLE = 1.0       # ΔE₀₀ below this: two colours aren't noticeably different

def repair_palettes(pal, spinner, languages, threshold, candidates=None, conditions=CONDITIONS):
    """Repair, one language's palette at a time, for every viewer: typical colour vision, the three colour vision
    deficiencies and the three ages.
    Where two colours of the palette are not noticeably different (ΔE₀₀ below 1) for some viewer, one of them is
    swapped for another tile of its name in the Spinner's Mapper that is noticeably different from the name's best
    example but still close to it (ΔE₀₀ above 1 and below `threshold`, 5, with typical vision, so the name keeps
    its meaning). Among those tiles, it takes the one that best separates the palette for the worst-off viewer
    (the largest smallest ΔE₀₀ over all pairs and conditions); on a tie, the more typical tile."""
    from itertools import combinations
    pal = pal.copy()
    pal["repaired_from"] = None
    log = []
    def worst(hexes):
        """The closest pair over all viewing conditions: (ΔE₀₀, a, b, condition)."""
        best = (np.inf, None, None, None)
        for c in conditions:
            sims = [simulate(h, c) for h in hexes]
            for a, b in combinations(range(len(hexes)), 2):
                d = de00(sims[a], sims[b])
                if d < best[0]:
                    best = (d, a, b, c)
        return best
    for lang in languages:
        idx = list(pal.index[(pal.language == lang) & pal.grounded_hex.notna()])
        if len(idx) < 2:
            continue
        prefetch_simulations([pal.at[k, "grounded_hex"] for k in idx])
        for _ in range(len(idx)):                       # at most one swap per colour
            hexes = [pal.at[k, "grounded_hex"] for k in idx]
            d, a, b, cond = worst(hexes)
            if d >= NOTICEABLE:
                break
            best = None
            for pos in (b, a):                          # the later colour first; the earlier keeps its best example
                k = idx[pos]
                if pal.at[k, "repaired_from"]:
                    continue
                focal = pal.at[k, "spinner_hex"] if isinstance(pal.at[k, "spinner_hex"], str) else pal.at[k, "grounded_hex"]
                members = list(dict.fromkeys(safe(spinner.category_members, pal.at[k, "name"], lang) or []))
                if candidates:
                    members = members[:candidates]
                allowed = [m for m in members if m != pal.at[k, "grounded_hex"] and NOTICEABLE < de00(m, focal) < threshold]
                prefetch_simulations(allowed)
                for rank, m in enumerate(allowed):      # the Mapper's order: most typical first
                    trial = hexes.copy(); trial[pos] = m
                    score = worst(trial)[0]
                    if best is None or score > best[0] + 1e-9:
                        best = (score, k, pos, m, de00(m, focal))
                if best:
                    break
            if not best or best[0] <= d:
                log.append(dict(language=lang, pair=(pal.at[idx[a], "name"], pal.at[idx[b], "name"]),
                                before=d, condition=cond, fixed=False))
                break
            score, k, pos, m, from_focal = best
            trial = hexes.copy(); trial[pos] = m
            after, _, _, after_cond = worst(trial)
            log.append(dict(language=lang, name=pal.at[k, "name"], old=pal.at[k, "grounded_hex"], new=m,
                            pair=(pal.at[idx[a], "name"], pal.at[idx[b], "name"]), before=d, condition=cond,
                            after=after, after_condition=after_cond, from_focal=from_focal, fixed=after >= NOTICEABLE))
            pal.at[k, "repaired_from"] = pal.at[k, "grounded_hex"]
            pal.at[k, "grounded_hex"] = m
    pal.attrs["repairs"] = log
    return pal

def show_repairs(pal, threshold):
    log = pal.attrs.get("repairs", [])
    if not log:
        return
    joint = pal.attrs.get("joint", False)
    rows = []
    for e in log:
        if "new" in e:
            where = (f"still {_h(e['name'])}: tile {e['tile']} of {e.get('of', 21)} in its Mapper, from the best to the worst example; "
                     f"ΔE₀₀ = {e['from_focal']:.1f} from the best example"
                     if "tile" in e else f"ΔE₀₀ = {e['from_focal']:.1f} from the best example")
            change = f'{swatch(e["old"])} <span style="color:{MUTED}">→</span> {swatch(e["new"])}<br>{small(where)}'
            good = e["after"] >= (threshold if joint else NOTICEABLE)
            result = (pill(f'ΔE₀₀ = {e["before"]:.1f} → {e["after"]:.1f}', "good" if good else "mid") + " " +
                      small("told apart at a glance" if e["after"] >= threshold else
                            "noticeably different, but not at a glance" if e["after"] >= NOTICEABLE else
                            "still not noticeably different"))
            what = f'<b>{_h(e["name"])}</b>'
        else:
            change, what = small("no colour of these names can be told apart at a glance"), "—"
            result = pill(f'ΔE₀₀ = {e["before"]:.1f}', "bad") + " " + small("consider a second cue: a pattern, label or icon")
        rows.append([f"<b>{_h(e['language'])}</b>",
                     f'{_h(e["pair"][0])} & {_h(e["pair"][1])}<br>{small("closest pair, " + condition_label(e["condition"], " ").lower())}',
                     what, change, result])
    show("<b>Optimised colours</b>")
    who = vision_phrase(pal.attrs["vision"]) if pal.attrs.get("vision") else "the worst-off viewer"
    table(["Language", "Pair too close", "Colour changed", "Before → after", f"Closest ΔE₀₀, {who}"], rows,
          note=(f"The colours are chosen by trying every combination of allowed colours: each test colour may "
                f"keep its colour, move to another colour of its name in the Spinner's Mapper (any within ΔE₀₀ {threshold:g} of the "
                f"best example, or further if speakers would still give it that name), or take a synonym's best example if it shares "
                f"a name. The best palette has every pair at least ΔE₀₀ {threshold:g} apart for the kind of vision you chose, if "
                f"possible, and then the fewest and smallest changes.") if joint else
               (f"Each replacement is another tile of the same name in the Colour Spinner's Mapper, noticeably different from the name's "
                f"best example but still close to it (ΔE₀₀ {NOTICEABLE:g}–{threshold:g}), chosen to separate the palette best for the "
                f"worst-off viewer. (The combined search wasn't possible here, so these were chosen step by step.)"))


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
def show_closest_pairs(pal, languages, threshold=DELTA_E_THRESHOLD, conditions=CONDITIONS, before=None):
    """For each language: the two closest colours of the palette, under the viewing condition where they are
    closest, as they look there, with the ΔE₀₀ between the simulated colours. With `before` (the palette before
    the repair), the columns compare before and after the repair; otherwise the AI's palette and the grounded one."""
    from itertools import combinations
    cols = ([("Before optimising", before, "grounded_hex"), ("After optimising", pal, "grounded_hex")] if before is not None else
            [("AI palette", pal, "ai_hex"), ("Grounded palette", pal, "grounded_hex")])
    prefetch_simulations([h for _, d, c in cols for h in d[c] if isinstance(h, str)])
    rows = []
    for lang in languages:
        cells = []
        for label, d, col in cols:
            hexes = [h for h in d[d.language == lang][col] if isinstance(h, str)]
            if len(hexes) < 2:
                cells.append(small("fewer than two colours")); continue
            d, c, a, b = min((de00(simulate(x, cc), simulate(y, cc)), cc, x, y)
                             for cc in conditions for x, y in combinations(hexes, 2))
            sw = (f'<div style="display:flex;gap:4px;margin-bottom:4px">{swatch(simulate(a, c), 34, 24)}'
                  f'{swatch(simulate(b, c), 34, 24)}</div>')
            cells.append(f'{sw}{pill(f"ΔE₀₀ = {d:.1f}", "bad" if d < threshold else "good")} {small(_h(c))}')
        rows.append([f"<b>{_h(lang)}</b>"] + cells)
    show("<b>" + (f"With {vision_phrase(conditions[0])}, can you tell each palette's colours apart?" if len(conditions) == 1
                  else "Can every viewer still tell each palette's colours apart?") + "</b>")
    table(["Language"] + [label for label, _, _ in cols], rows, valign="top",
          note=f"For each palette: its two closest colours under the viewing condition where they are closest, as they "
               f"look there, and the ΔE₀₀ between the simulated colours. Red: below {threshold:g}. A pair still red after "
               f"optimising can't be separated within its names: give it a second cue, such as a pattern, label or icon.")

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
    if isinstance(r.get("renamed_from"), str): return "renamed"
    if isinstance(r.get("repaired_from"), str): return "optimised"
    return "replaced" if r.get("replaced") else "kept"

def palette_card(pal, languages, path=None, title="Your grounded palette", display_it=True):
    """A picture of the grounded palette as on screen: one column per test colour, one row per language
    (test colours that share a name appear in both columns). Without Part B's names: one row per language."""
    import textwrap
    tab = _by_test_colour(pal, languages)
    if tab is not None and len(tab):
        tests = list(dict.fromkeys(_BY_SDG.sdg.astype(int)))
        langs = [l for l in languages if (tab.language == l).any()]
        cells = {(r.language, r.sdg): r for _, r in tab.iterrows()}
    else:
        tests, langs = None, [l for l in languages if pal[(pal.language == l) & pal.grounded_hex.notna()].shape[0]]
    n = len(tests) if tests else max([pal[(pal.language == l) & pal.grounded_hex.notna()].shape[0] for l in langs] or [1])
    rows_n = len(langs) + (1 if tests else 0)
    W, Hd = n * 1.55 + 2.0, rows_n * 1.45 + 0.1
    fig, ax = plt.subplots(figsize=(W * 0.95, Hd * 0.95 + 1.0))
    ax.set_xlim(0, W); ax.set_ylim(Hd, -0.2); ax.set_aspect("equal"); ax.axis("off")
    def tile(x, y, hex_, label, line1="", line2=""):
        ax.add_patch(mpl.patches.FancyBboxPatch((x, y), 1.35, 0.78, boxstyle="round,pad=0,rounding_size=0.06", fc=SURROUND, ec="none"))
        ax.add_patch(mpl.patches.FancyBboxPatch((x + 0.06, y + 0.06), 1.23, 0.66, boxstyle="round,pad=0,rounding_size=0.03", fc=hex_, ec="none"))
        ax.text(x + 0.14, y + 0.62, label, fontsize=9.5, fontweight="bold", color=_ink_on(hex_))
        if line1: ax.text(x, y + 0.98, line1, fontsize=8.5, color=INK, family="monospace")
        if line2: ax.text(x, y + 1.18, line2, fontsize=8, color=MUTED)
    row = 0
    if tests:
        ax.text(0, 0.42, "Your test palette", fontsize=11, color=MUTED, va="center")
        for k, i in enumerate(tests):
            tile(2.0 + k * 1.55, 0, SDG.hex[i], str(i), SDG.hex[i])
        row = 1
    for lang in langs:
        y = row * 1.45
        ax.text(0, y + 0.42, lang, fontsize=11.5, fontweight="bold", va="center", color=INK)
        if tests:
            for k, i in enumerate(tests):
                r = cells.get((lang, i))
                if r is not None:
                    tile(2.0 + k * 1.55, y, r.grounded_hex, r["name"], r.grounded_hex, r.status)
        else:
            for k, (_, r) in enumerate(pal[(pal.language == lang) & pal.grounded_hex.notna()].iterrows()):
                tile(2.0 + k * 1.55, y, r.grounded_hex, r["name"], r.grounded_hex, _status(r))
        row += 1
    note = textwrap.fill("Each colour shows what speakers of that language mean by the name, from the Colour Spinner. "
                         "kept: the AI's colour matched · replaced: the speakers' best example · "
                         "renamed: a close synonym, because two test colours shared a name · optimised: another colour of the same name, "
                         "to tell it apart from the palette's other colours.", width=int(W * 0.95 * 15))
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
            short = n > 8 and nm.split(" ", 1)[0].isdigit()          # many colours: the SDG number, then the name below
            ax.text(k + 0.5, 1.04, nm.split(" ", 1)[0] if short else nm, ha="center", va="center",
                    fontsize=14 if n <= 4 else 12 if n <= 6 else 10 if n <= 8 else 9,
                    color=INK, fontweight="bold")
    elif kind == "Bar chart":
        for k, (c, v) in enumerate(zip(cols, bar)):
            ax.add_patch(mpl.patches.FancyBboxPatch((k - 0.36, 0), 0.72, v, boxstyle="round,pad=0,rounding_size=0.08",
                         fc=c, ec="none", mutation_aspect=60))
        pad = (max(n, 4) - n) / 2                    # at least four slots, so few bars stay bar-shaped, centred
        ax.set_xlim(-0.7 - pad, n - 0.3 + pad); ax.set_ylim(0, 100)
        ticks = [nm.split(" ", 1)[0] for nm in names] if n > 8 and all(nm.split(" ", 1)[0].isdigit() for nm in names) \
                else [chr(65 + k) for k in range(n)]
        ax.set_xticks(range(n), ticks); ax.set_yticks(range(0, 101, 20))
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

def full_sdg_palette(pal, language):
    """All 17 SDG colours for one language: your test colours replaced by their grounded colours (with their
    names), the others as the UN's. Returns (labels, hexes, grounded flags)."""
    get = _grounded_lookup(pal[(pal.language == language) & pal.grounded_hex.notna()])
    by = {}
    if _BY_SDG is not None:
        for _, r in _BY_SDG[_BY_SDG.language == language].iterrows():
            g = get(language, r.sdg, r["name"])
            if g is not None: by[int(r.sdg)] = (g["name"], g.grounded_hex)
    labels, hexes, flags = [], [], []
    for i, h in SDG.hex.items():
        if i in by:
            labels.append(f"{i} {by[i][0]}"); hexes.append(by[i][1]); flags.append(True)
        else:
            labels.append(f"{i}"); hexes.append(h); flags.append(False)
    return labels, hexes, flags

def preview_applications(pal, language, vision="Typical", path=None, kind="Bar chart", colours="Your test colours"):
    """The grounded palette for one language applied to a chart, map or swatches, in the style of the
    palette explorer, with typical vision and under `vision` side by side. `colours`: just your grounded
    test colours, or all 17 SDG colours with your test colours grounded."""
    language = _available_language(pal, language, "PreviewLanguage")
    p = pal[(pal.language == language) & pal.grounded_hex.notna()]
    if p.empty:
        show(f"<b>No grounded palette for {_h(language)}.</b> {small('Choose one of: ' + _h(', '.join(dict.fromkeys(pal.language))))}")
        return
    full = colours.startswith("All 17")
    if full and (_BY_SDG is None or not len(_BY_SDG)):
        show(small("All 17 SDG colours needs the names from Part B, so this shows your test colours."))
        full = False
    if full:
        names, hexes, _ = full_sdg_palette(pal, language)
    else:
        names, hexes = list(p["name"]), list(p.grounded_hex)
    data = _explorer_data(len(hexes))
    if kind == "Map (world)" and _world() is None:          # no download: the US states map, built in, instead
        show(small("The world map couldn't be downloaded, so here is the US states map instead."))
        kind = "Map (US states)"
    conds = ["Typical"] + ([vision] if vision != "Typical" else [])
    many = len(hexes) > 8
    fig, axes = plt.subplots(1, len(conds), figsize=(6.2 * len(conds), 4.3 + (1.0 if many else 0)), squeeze=False)
    fig.patch.set_facecolor("white")
    for ax, cond in zip(axes[0], conds):
        cols = [simulate(h, cond) for h in hexes]
        _draw_application(ax, kind, cols, names, data)
        ax.set_title(condition_label(cond, " ") if cond != "Typical" else "Typical colour vision", fontsize=11, loc="left", pad=10)
        if kind != "Colour swatches" or many:  # a legend of small squares, as in the explorer
            handles = [mpl.patches.Patch(color=c, label=nm) for c, nm in zip(cols, names)]
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1 if kind not in ("Donut chart", "Map (US states)", "Map (world)") else -0.02),
                      ncol=min(len(names), 6 if many else 4), frameon=False, fontsize=8 if many else 9,
                      handlelength=0.8, handleheight=0.8, labelcolor=MUTED)
    what = {"Colour swatches": "as swatches", "Map (US states)": "applied to a map of the US states",
            "Map (world)": "applied to a world map"}.get(kind, f"applied to a {kind.lower()}")
    fig.suptitle(f"The {language} grounded palette, {what}" + (" (all 17 SDG colours)" if full else ""), x=0.01, ha="left",
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
    get = _grounded_lookup(pal[pal.grounded_hex.notna()])
    rows = [("The UN's original palette", dict(SDG.hex), set())]
    for lang in languages:
        corr, marked = dict(SDG.hex), set()
        for _, r in _BY_SDG[_BY_SDG.language == lang].iterrows():
            g = get(lang, r.sdg, r["name"]); h = g.grounded_hex if g is not None else None
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

def full_palette_picture(pal, language, vision, path, threshold=DELTA_E_THRESHOLD):
    """Section 6 as a picture: the UN's palette, the grounded one for `language`, the arcs and the notes."""
    p = pal[(pal.language == language) & pal.grounded_hex.notna() & pal.sdg.notna()]
    if p.empty:
        return False
    original, corrected, named, changed, arcs, old, notes, old_notes = _full_check(pal, p, language, vision, threshold)
    sim = lambda h: simulate(h, vision)
    clean = lambda t: re.sub(r"<[^>]+>", "", t)
    allarcs = sorted([(i, j, True, a) for i, j, d, a in arcs] + [(i, j, False, False) for i, j, d, a in old],
                     key=lambda t: (not t[2], abs(t[0] - t[1])))
    lines = ([("New, made by grounding (red arcs)", "#B4342D", True)] + [(clean(n), INK, False) for n in notes] if notes else []) + \
            ([("Already close in the UN palette (grey dashed arcs)", "#5E5E5E", True), (", ".join(old_notes), INK, False)] if old_notes else [])
    if not lines:
        lines = [(f"No colour pair involving your test colours is closer than ΔE₀₀ {threshold:g}.", INK, False)]
    T = 0.62
    arc_h = 0.25 + 0.16 * len(allarcs)
    H = 0.6 + 2 * 1.0 + arc_h + 0.3 + 0.27 * len(lines) + 0.5
    fig, ax = plt.subplots(figsize=(17 * T + 0.4, H)); ax.axis("off")
    ax.set_xlim(0, 17 * T + 0.4); ax.set_ylim(H, 0)
    view = "typical colour vision" if vision == "Typical" else condition_label(vision, " ").lower()
    ax.text(0.15, 0.3, f"Your test colours, grounded for {language}, in the full SDG palette", fontsize=12.5, fontweight="bold", color=INK)
    ax.text(0.15, 0.55, f"as seen with {view}", fontsize=9, color=MUTED)
    def tiles(colours, y, label, mark=()):
        ax.text(0.15, y, label, fontsize=9, color=MUTED, va="center")
        for c, (i, h) in enumerate(colours.items()):
            x = 0.15 + c * T; v = sim(h)
            ax.add_patch(mpl.patches.FancyBboxPatch((x, y + 0.15), 0.54, 0.5, boxstyle="round,pad=0,rounding_size=0.04", fc=SURROUND, ec="none"))
            ax.add_patch(mpl.patches.Rectangle((x + 0.05, y + 0.2), 0.44, 0.4, fc=v, ec="none"))
            ax.text(x + 0.08, y + 0.56, str(i), fontsize=7.5, fontweight="bold", color=_ink_on(v), va="bottom")
            if i in mark:
                ax.add_patch(mpl.patches.Rectangle((x, y + 0.7), 0.54, 0.05, fc=INK, ec="none"))
    tiles(original, 0.85, "The UN's original palette")
    tiles(corrected, 1.85, f"Grounded for {language} (marked: changed from the UN colour)", changed)
    y0 = 1.85 + 0.8
    xc = lambda i: 0.15 + (i - 1) * T + 0.27
    for k_, (i, j, new, arrow) in enumerate(allarcs):
        depth = 0.12 + 0.16 * k_
        xs = np.linspace(xc(i), xc(j), 60); t = np.linspace(0, 1, 60)
        ys = y0 + 4 * depth * t * (1 - t)
        ax.plot(xs, ys, color="#B4342D" if new else "#8A8A85", lw=1.6 if new else 1.1, ls="-" if new else (0, (4, 3)))
        if new and arrow:
            ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-4], ys[-4]), arrowprops=dict(arrowstyle="-|>", color="#B4342D", lw=1.4, mutation_scale=10))
    y = y0 + arc_h + 0.15
    for text, col, bold in lines:
        ax.text(0.15, y, text, fontsize=8.8, color=col, fontweight="bold" if bold else "normal", va="top"); y += 0.27
    ax.text(0.15, y + 0.1, "Only the test colours were grounded; the others are the UN's. The SDG goals also have numbers, icons and "
            "names: an example, not a proposal to change the UN's colours.", fontsize=7.8, color=MUTED, va="top")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return True

def _by_test_colour(pal, languages):
    """Every (test colour, language) with its grounded colour and the evidence, including test colours that
    share a name (each gets its own row). Without Part B's names: one row per name, as before."""
    if _BY_SDG is None or not len(_BY_SDG):
        return None
    get = _grounded_lookup(pal)
    rows = []
    for _, t in _BY_SDG.iterrows():
        if t.language not in languages: continue
        r = get(t.language, t.sdg, t["name"])
        if r is None: continue
        rows.append({"sdg": int(t.sdg), "test_colour": SDG.hex[int(t.sdg)], "goal": SDG.goal[int(t.sdg)],
                     "language": t.language, "name": r["name"], "grounded_hex": r.grounded_hex, "status": _status(r),
                     "ai_hex": r.ai_hex, "speakers_best_example": r.spinner_hex, "de00_ai_vs_speakers": r.get("de00"),
                     "speakers_call_the_ai_colour": r.get("rt_name"), "round_trip_confidence": r.get("rt_conf"),
                     "repaired_from": r.get("repaired_from")})
    out = pd.DataFrame(rows)
    order = {l: k for k, l in enumerate(languages)}
    return out.sort_values(["language", "sdg"], key=lambda c: c.map(order) if c.name == "language" else c).reset_index(drop=True)

def export_palette(pal, languages, stem="grounded_palette"):
    """Pictures of the palettes and the data behind them, zipped for one download."""
    import zipfile, glob
    cols = ["language", "name", "grounded_hex", "ai_hex", "spinner_hex", "de00", "rt_name", "rt_conf",
            "replaced", "repaired_from", "sdg", "sdg_hex"]
    export = _by_test_colour(pal, languages)
    if export is None:
        export = pal[[c for c in cols if c in pal.columns]]
    files, rows = [], []
    palette_card(pal, languages, path=f"{stem}_card.png", display_it=False)
    files.append(f"{stem}_card.png"); rows.append([f"<code>{stem}_card.png</code>", "Your grounded palette, by test colour and language, as on screen"])
    if corrected_sdg_card(pal, languages, "grounded_sdg_palette.png"):
        files.append("grounded_sdg_palette.png")
        rows.append(["<code>grounded_sdg_palette.png</code>", "Overview: the full SDG palette grounded for every language"])
        made = 0
        for lang in languages:
            for v in dict.fromkeys(["Typical", _LAST_FULL_VIEW.get(lang, "Typical")]):
                fn = f"grounded_sdg_{lang.replace(' ', '_')}_{v.replace(' ', '_').lower()}.png"
                if try_call(full_palette_picture, pal, lang, v, fn):
                    files.append(fn); made += 1
        if made:
            rows.append(["<code>grounded_sdg_…png</code>", "Section 6 for each language, with its arcs and notes: typical vision, plus the vision you chose there"])
    previews = sorted(glob.glob("preview_*.png"))
    if previews:
        files += previews; rows.append(["<code>preview_….png</code>", "The designs you previewed in section 5"])
    export.round(2).to_csv(f"{stem}.csv", index=False, encoding="utf-8-sig")
    files.append(f"{stem}.csv"); rows.append([f"<code>{stem}.csv</code>", "Every test colour in every language: grounded hex code, status, and the evidence"])
    if _BY_SDG is not None and len(_BY_SDG):
        full_rows = []
        for lang in languages:
            if pal[(pal.language == lang) & pal.grounded_hex.notna()].empty: continue
            labels, hexes, flags = full_sdg_palette(pal, lang)
            for i, (lab, h, f) in enumerate(zip(labels, hexes, flags), 1):
                full_rows.append({"language": lang, "sdg": i, "goal": SDG.goal[i], "hex": h,
                                  "source": "grounded" if f else "UN", "name": lab.split(" ", 1)[1] if f else ""})
        pd.DataFrame(full_rows).to_csv("grounded_sdg_palette.csv", index=False, encoding="utf-8-sig")
        files.append("grounded_sdg_palette.csv")
        rows.append(["<code>grounded_sdg_palette.csv</code>", "All 17 SDG colours for each audience, with your test colours grounded"])
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
  <div style="color:{MUTED};font-size:14px">ΔE₀₀ =<br><b style="font-size:34px;color:{INK}">{row.de00:.1f}</b></div>
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
    ax.text(4.65, 1.3, "ΔE₀₀ =", fontsize=10, color="white"); ax.text(4.65, 1.95, f"{row.de00:.1f}", fontsize=24, fontweight="bold", color="white")
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
_LAST_FULL_VIEW = {}

def _full_check(pal, p, language, vision, threshold):
    """Section 6's checks for one language: the grounded full palette, and the close pairs it creates (new) or
    keeps (already in the UN palette), as seen with `vision`."""
    original = dict(SDG.hex)
    corrected, named = dict(original), {}
    pairs = (_BY_SDG[_BY_SDG.language == language] if _BY_SDG is not None else p[["sdg", "name"]])
    get = _grounded_lookup(p)
    for _, r in pairs.iterrows():
        g = get(language, r.sdg, r["name"]); h = g.grounded_hex if g is not None else None
        if isinstance(h, str):
            corrected[int(r.sdg)] = h; named[int(r.sdg)] = g["name"]
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
                notes.append(f"<b>#{i} and #{j}</b>: {why}, ΔE₀₀ = {d:.1f} apart, though the originals were distinct.")
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
        ids = " and ".join(f"#{i}" for i, _ in hits)
        verb = "is" if len(hits) == 1 else "are"
        notes.append(f"<b>{ids}</b> (<i>{_h(named[hits[0][0]])}</i>) {verb} now ΔE₀₀ = {hits[0][1]:.1f} from the unchanged SDG colour <b>#{j}</b>.")
    old, old_notes, seen = [], [], set()          # already close in the UN palette, and still close now
    for i in tests:
        for j in original:
            if j == i or tuple(sorted((i, j))) in seen: continue
            d_now = de00(sim(corrected[i]), sim(corrected[j]))
            if d_now < threshold and de00(sim(original[i]), sim(original[j])) < threshold:
                seen.add(tuple(sorted((i, j))))
                old.append((i, j, d_now, False))
                old_notes.append(f"#{min(i, j)} and #{max(i, j)} (ΔE₀₀ = {d_now:.1f})")
    return original, corrected, named, changed, arcs, old, notes, old_notes

@needs_simulations()
def show_corrected_sdg(pal, language, vision="Typical", threshold=DELTA_E_THRESHOLD):
    """The UN's 17 colours next to the same palette with the test colours replaced by their grounded colours
    for one language, under a chosen kind of vision. Arcs under the corrected row point to the pairs that the
    correction brought too close, so they can be checked again."""
    if "sdg" not in pal or pal.sdg.isna().all():
        show("This step needs the names from Part B (uploaded <code>partB_results.csv</code>), because it has to know "
             "which SDG colour each name belongs to.")
        return
    language = _available_language(pal, language, "Grounded for")
    p = pal[(pal.language == language) & pal.grounded_hex.notna() & pal.sdg.notna()]
    if p.empty:
        show(f"<b>No grounded colours for {_h(language)}.</b> " + small("Choose one of: " + _h(", ".join(dict.fromkeys(pal.language)))))
        return
    original, corrected, named, changed, arcs, old, notes, old_notes = _full_check(pal, p, language, vision, threshold)
    sim = lambda h: simulate(h, vision)
    _LAST_FULL_VIEW[language] = vision                   # remembered for the download
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
            + arcs_svg()
            + ((f'<div style="font-size:12px;color:{MUTED};margin:0 0 4px">'
                f'<span style="color:#B4342D;font-weight:700">⟶ red arc</span>: a new problem, made by grounding · '
                f'<span style="color:#8A8A85;font-weight:700">- - grey dashed arc</span>: already close in the UN palette</div>')
               if (arcs or old) else ""))
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
         f'Click to open the Colour Spinner in a new tab. It opens in American English: choose your language in its menu first.</div>')


# ── Menus that list only what the pair chose, and redraw straight away ─────
VISIONS = ["Typical", "Protanopia", "Deuteranopia", "Tritanopia", "Age 40", "Age 60", "Age 80"]

def _safe_render(render, choices):
    """Draw one choice of the menus; if the Spinner doesn't answer, say so instead of showing a traceback."""
    try:
        render(**choices)
    except SimulationUnavailable:
        show("<b>The Colour Spinner didn't answer in time.</b> " + small("Its simulations are needed for this view. "
             "Wait a minute and choose again; typical vision always works."))
    except Exception as e:
        show("<b>This view couldn't be drawn.</b> " + small(_h(f"{type(e).__name__}: {e}")[:300]))

def live_controls(controls, render):
    """Dropdown menus built when the cell runs (so they list only the pair's own languages), with the output
    redrawn whenever a choice changes. controls: {key: (label, options, default)}; render(**choices)."""
    try:
        import ipywidgets as w
    except ImportError:
        _safe_render(render, {k: d for k, (_, _, d) in controls.items()}); return
    menus = {k: w.Dropdown(options=list(opts), value=d if d in opts else list(opts)[0], description=label,
                           style={"description_width": "initial"}, layout=w.Layout(width="auto", margin="0 18px 0 0"))
             for k, (label, opts, d) in controls.items()}
    out = w.Output()
    def draw(*_):
        out.clear_output(wait=True)
        with out:
            _safe_render(render, {k: m.value for k, m in menus.items()})
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

def my_language(df):
    """The pair's own language (role 'your language' in Part B's results), or else the first one listed."""
    if "role" in df:
        mine = df[df.role == "your language"].language
        if len(mine):
            return mine.iloc[0]
    langs = list(dict.fromkeys(df.language))
    return langs[0] if langs else None


def per_test_colour(pal):
    """One row per test colour and language, before grounding and repair. Part C asks about each name once,
    so two test colours that share a name (6 and 14, both Burou) would otherwise be a single row, and the
    repair could never separate them. Safe to run again."""
    if "per_test_colour" in pal.columns or _BY_SDG is None or not len(_BY_SDG):
        out = pal.copy()
        if "per_test_colour" not in out.columns:
            out["per_test_colour"] = False
        return out
    by_name = {}
    for _, r in pal.iterrows():
        by_name.setdefault((r.language, norm(r["name"])), r)
    rows = []
    for _, t in _BY_SDG.iterrows():
        r = by_name.get((t.language, norm(t["name"])))
        if r is None:
            continue
        r = r.copy(); r["sdg"] = int(t.sdg)
        if "sdg_hex" in r.index: r["sdg_hex"] = SDG.hex[int(t.sdg)]
        rows.append(r)
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["per_test_colour"] = True
    return out

def _grounded_lookup(df):
    """Find the row for a test colour: by (language, test colour) once each test colour has its own row,
    otherwise by (language, name)."""
    per_test = "per_test_colour" in df.columns and bool(df.per_test_colour.any())
    by_sdg, by_name = {}, {}
    for _, r in df.iterrows():
        if per_test and not pd.isna(r.get("sdg")):
            by_sdg[(r.language, int(r.sdg))] = r
        by_name.setdefault((r.language, norm(r["name"])), r)
    def get(language, sdg, name):
        if per_test:
            return by_sdg.get((language, int(sdg)))
        return by_name.get((language, norm(name)))
    return get

def show_repair_journey(pal, language, path=None):
    """One language, step by step: each test colour → the AI's colour → speakers' best example → the grounded
    colour → after repair, with the ΔE₀₀ and name at each step. Every value comes from the run in this notebook."""
    get = _grounded_lookup(pal)
    tests = list(dict.fromkeys(_BY_SDG.sdg.astype(int))) if _BY_SDG is not None and len(_BY_SDG) else []
    rows = []
    for i in tests:
        t = _BY_SDG[(_BY_SDG.sdg == i) & (_BY_SDG.language == language)]
        if t.empty: continue
        r = get(language, i, t["name"].iloc[0])
        if r is None: continue
        before = r.repaired_from if isinstance(r.get("repaired_from"), str) else r.grounded_hex
        rows.append((i, r, before))
    if not rows:
        show(small(f"No grounded colours for {_h(language)} yet: run section 4 first.")); return
    steps = ["Test colour", "AI's colour", "Speakers' best example", "Grounded", "Optimised"]
    fig, ax = plt.subplots(figsize=(12.5, 1.15 * len(rows) + 1.4)); ax.axis("off")
    ax.set_xlim(0, 12.5); ax.set_ylim(1.15 * len(rows) + 0.9, -0.2)
    xs = [1.3 + k * 2.35 for k in range(5)]
    for x, lab in zip(xs, steps):
        ax.text(x + 0.75, 0.1, lab, ha="center", fontsize=10, fontweight="bold", color=INK)
    def chip(x, y, h, top, bottom, edge=None):
        ax.add_patch(mpl.patches.FancyBboxPatch((x, y), 1.5, 0.7, boxstyle="round,pad=0,rounding_size=0.05",
                     fc=SURROUND if edge is None else edge, ec="none"))
        ax.add_patch(mpl.patches.Rectangle((x + 0.06, y + 0.06), 1.38, 0.58, fc=h, ec="none"))
        ax.text(x + 0.12, y + 0.56, top, fontsize=8.5, fontweight="bold", color=_ink_on(h))
        ax.text(x + 0.75, y + 0.86, bottom, fontsize=7.8, color=MUTED, ha="center", va="top")
    for k, (i, r, before) in enumerate(rows):
        y = 0.45 + k * 1.15
        ax.text(0.1, y + 0.35, r["name"], fontsize=10, fontweight="bold", va="center", color=INK)
        d_ai = de00(SDG.hex[i], r.ai_hex) if isinstance(r.ai_hex, str) else np.nan
        chip(xs[0], y, SDG.hex[i], str(i), SDG.hex[i])
        if isinstance(r.ai_hex, str):
            rt = f" · speakers: {r.rt_name}" if isinstance(r.get("rt_name"), str) else ""
            chip(xs[1], y, r.ai_hex, "AI", f"{r.ai_hex}{rt}")
        if isinstance(r.spinner_hex, str):
            chip(xs[2], y, r.spinner_hex, "Speakers", f"{r.spinner_hex} · ΔE₀₀ = {r.de00:.1f} from the AI" if not pd.isna(r.get("de00")) else r.spinner_hex)
        chip(xs[3], y, before, "replaced" if before != r.ai_hex else "kept", before)
        repaired = isinstance(r.get("repaired_from"), str)
        chip(xs[4], y, r.grounded_hex, "optimised" if repaired else "unchanged",
             f"{r.grounded_hex} · ΔE₀₀ = {de00(r.grounded_hex, r.spinner_hex):.1f} from the best example" if repaired else r.grounded_hex,
             edge=INK if repaired else None)
        for x in xs[:-1]:
            ax.annotate("", xy=(x + 2.32, y + 0.35), xytext=(x + 1.55, y + 0.35),
                        arrowprops=dict(arrowstyle="-|>", color="#9A9A95", lw=1, mutation_scale=9))
    log = [e for e in pal.attrs.get("repairs", []) if e.get("language") == language]
    note = ("Optimised: " + "; ".join(f"{e['name']} {e['old']} → {e['new']}: closest pair ΔE₀₀ = {e['before']:.1f} → {e['after']:.1f} "
                                    f"({condition_label(e['condition'], ' ').lower()})" for e in log if "new" in e)) if log else "No colours needed optimising."
    ax.text(0.1, 1.15 * len(rows) + 0.55, note, fontsize=8.5, color=MUTED, va="top", wrap=True)
    fig.suptitle(f"From test colours to the final palette · {language}", x=0.02, ha="left", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if path: fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.show()


# ── Naming repair: two test colours that share a name get different names ──
SYNONYMS_TO_TRY = 6         # as many synonyms as the Spinner's Thesaurus shows

def _latin_variants(name):
    """Latin spellings of a Greek name in the ways the Spinner's similarity data writes them (μπ as mp or b, …)."""
    t = norm(name)
    if not re.search("[α-ω]", t):
        return {t}
    pairs = [("ου", ["ou"]), ("ευ", ["eu", "ef"]), ("αυ", ["au", "af"]), ("μπ", ["mp", "b"]), ("ντ", ["nt", "d"]),
             ("γκ", ["gk", "g"]), ("γγ", ["gg", "ng"]), ("θ", ["th"]), ("χ", ["ch", "x"]), ("ψ", ["ps"]), ("ξ", ["x", "ks"])]
    single = dict(zip("αβγδεζηικλμνοπρσςτυφω", ["a", "v", "g", "d", "e", "z", "i", "i", "k", "l", "m", "n", "o", "p", "r", "s", "s", "t", "y", "f", "o"]))
    out = {""}
    k = 0
    while k < len(t):
        two = t[k:k + 2]
        hit = next((opts for g, opts in pairs if two == g or t[k] == g), None)
        if hit is not None and any(two == g for g, _ in pairs):
            out = {o + v for o in out for v in hit}; k += 2; continue
        if hit is not None:
            out = {o + v for o in out for v in hit}; k += 1; continue
        ch = t[k]
        opts = [single.get(ch, ch)] + (["b"] if ch == "β" else [])
        out = {o + v for o in out for v in opts}; k += 1
        if len(out) > 64: out = set(list(out)[:64])
    return out

def _spinner_name(spinner, language, latin):
    """The Spinner's own spelling of a name that the similarity data writes in Latin letters, or None."""
    try:
        spinner._language_ids(language)
        names = spinner._lang_names.get(language, {})
    except Exception:
        return None
    if norm(latin) in names:
        return names[norm(latin)]
    for key, shown in names.items():
        if norm(latin) in _latin_variants(key):
            return shown
    return None

def rename_shared_names(pal, spinner, languages, top=SYNONYMS_TO_TRY, conditions=CONDITIONS):
    """Where two test colours share a name in a language (6 and 14, both Burou), the later one is renamed.
    Its synonyms (its top Thesaurus synonyms, from the Spinner's name similarity) are tried from the most similar
    down, and it takes the first whose best example is at least the threshold (ΔE₀₀ 10) from the palette's other colours for the
    worst-off viewer; if none is, the one that comes closest. Its colour becomes that synonym's best example.
    Names already in the palette aren't used again."""
    pal = pal.copy()
    pal["renamed_from"] = None
    log = []
    data = _similarity_data()
    for lang in languages:
        m = data.get(_sim_key(lang))
        idx = list(pal.index[(pal.language == lang) & pal.grounded_hex.notna()])
        seen = {}
        for k in idx:
            key = norm(pal.at[k, "name"])
            if key not in seen:
                seen[key] = k; continue
            entry = dict(language=lang, sdg=int(pal.at[k, "sdg"]), old=pal.at[k, "name"], shared_with=int(pal.at[seen[key], "sdg"]))
            if not m:
                log.append({**entry, "new": None, "why": "no similarity data for this language"}); continue
            rom = pal.at[k, "romanized"] if "romanized" in pal and isinstance(pal.at[k, "romanized"], str) else None
            row = next((m["index"][norm(c.replace("_", " "))] for c in (pal.at[k, "name"], rom)
                        if isinstance(c, str) and norm(c.replace("_", " ")) in m["index"]), None)
            if row is None:
                log.append({**entry, "new": None, "why": "name not found in the similarity data"}); continue
            used = {norm(pal.at[j, "name"]) for j in idx} | {norm(pal.at[j, "romanized"]) for j in idx
                                                             if "romanized" in pal and isinstance(pal.at[j, "romanized"], str)}
            order = sorted(((v, c) for c, v in enumerate(m["similarity"][row]) if c != row), reverse=True)
            others = [pal.at[j, "grounded_hex"] for j in idx if j != k]
            other_ids = [(int(pal.at[j, "sdg"]), pal.at[j, "name"]) for j in idx if j != k]
            best = None
            tried = 0
            for sim, c in order:
                if tried >= top: break
                latin = m["names"][c]
                if norm(latin) in used: continue
                tried += 1
                shown = _spinner_name(spinner, lang, latin)
                if not shown or norm(shown) in used: continue
                hexb = safe(spinner.best_example, shown, lang)
                if not isinstance(hexb, str): continue
                prefetch_simulations([hexb] + others)
                dists = [(de00(simulate(hexb, cc), simulate(o, cc)), oid) for cc in conditions for o, oid in zip(others, other_ids)]
                score, near = min(dists, key=lambda t: t[0]) if dists else (np.inf, None)
                if score >= DELTA_E_THRESHOLD:          # the most similar synonym that can be told apart at a glance
                    best = (score, shown, latin, hexb, sim, near); break
                if best is None or score > best[0]:     # otherwise, remember the most distinct one
                    best = (score, shown, latin, hexb, sim, near)
            if not best:
                log.append({**entry, "new": None, "why": f"none of its top {top} synonyms could be used"}); continue
            score, shown, latin, hexb, sim, near = best
            log.append({**entry, "new": shown, "similarity": sim, "hex": hexb, "nearest": score,
                        "nearest_sdg": near[0] if near else None, "nearest_name": near[1] if near else None,
                        "vision": conditions[0] if len(conditions) == 1 else None})
            pal.at[k, "renamed_from"] = pal.at[k, "name"]
            pal.at[k, "orig_spinner_hex"] = pal.at[k, "spinner_hex"]
            pal.at[k, "orig_romanized"] = rom
            pal.at[k, "name"] = shown
            if "romanized" in pal: pal.at[k, "romanized"] = latin
            pal.at[k, "spinner_hex"] = hexb
            pal.at[k, "grounded_hex"] = hexb
            pal.at[k, "replaced"] = True
            used.add(norm(shown))
    pal.attrs["renames"] = log
    return pal

def show_renames(pal):
    log = pal.attrs.get("renames", [])
    if not log:
        return
    rows = []
    for e in log:
        if e.get("new"):
            rows.append([f"<b>{_h(e['language'])}</b>", framed(SDG.hex[e["sdg"]], str(e["sdg"])),
                         f"<i>{_h(e['old'])}</i>, like #{e['shared_with']}",
                         f"<b>{_h(e['new'])}</b> {swatch(e['hex'])}<br>{small('a synonym of ' + _h(e['old']) + ', similarity ' + format(e['similarity'], '.2f'))}",
                         pill(f"ΔE₀₀ = {e['nearest']:.1f}", "good" if e["nearest"] >= DELTA_E_THRESHOLD else "mid") + " " +
                         small((f"to #{e['nearest_sdg']} ({_h(e['nearest_name'])}), the nearest"
                                if e.get("nearest_sdg") is not None else "to the nearest other colour") + ", " +
                               ("with " + vision_phrase(e["vision"]) if e.get("vision") else "for the worst-off viewer"))])
        else:
            rows.append([f"<b>{_h(e['language'])}</b>", framed(SDG.hex[e["sdg"]], str(e["sdg"])),
                         f"<i>{_h(e['old'])}</i>, like #{e['shared_with']}", small("not renamed: " + e["why"]),
                         pill("same name", "bad") + " " + small("consider a second cue: a pattern, label or icon")])
    show("<b>Renamed: test colours that shared a name</b>")
    table(["Language", "Test colour", "Name it shared", "Renamed to", "Now"], rows,
          note=f"Two test colours with the same name would be confused when people talk about them, however different they look. "
               f"The later one gets a different name: the most similar synonym of the shared name (among its top {SYNONYMS_TO_TRY} in the "
               f"Spinner's Thesaurus) whose best example can be told apart at a glance from your other colours (ΔE₀₀ {DELTA_E_THRESHOLD:g} "
               f"or more with typical colour vision); if none can, the one that comes closest. Step 3 then checks other kinds of vision.")


def undo_renames(pal):
    """Back to the names and best examples from section 3, so that rerunning 'Ground and repair' starts afresh."""
    pal = pal.copy()
    if "renamed_from" in pal:
        for k in pal.index[pal.renamed_from.apply(lambda v: isinstance(v, str))]:
            pal.at[k, "name"] = pal.at[k, "renamed_from"]
            pal.at[k, "spinner_hex"] = pal.at[k, "orig_spinner_hex"]
            if "romanized" in pal: pal.at[k, "romanized"] = pal.at[k, "orig_romanized"]
        pal = pal.drop(columns=["renamed_from", "orig_spinner_hex", "orig_romanized"], errors="ignore")
    return pal


# ── Renaming and repair chosen together ───────────────────────────────────
MAX_COMBINATIONS = 200_000      # beyond this, the step-by-step repair is used instead

def optimise_palettes(pal, spinner, languages, threshold=DELTA_E_THRESHOLD, conditions=CONDITIONS,
                      tiles_per_colour=8, synonyms=SYNONYMS_TO_TRY):
    """Renaming and perceptual repair chosen together, one language at a time, by trying every combination of
    allowed colours. Each test colour may keep its grounded colour, move to another tile of its name (within
    ΔE₀₀ 5 of the name's best example, or further if the Spinner's Namer still gives that name for the tile, so
    boundary colours aren't used), or, if it shares a name with another test colour, take a synonym's best example.
    The best palette has: no shared names; every pair at least ΔE₀₀ `threshold` apart for the worst-off viewer, if
    any combination allows it, otherwise the largest such distance; then the fewest renamings and changes, and the
    most typical colours."""
    from itertools import combinations, product
    pal = pal.copy()
    for col in ("repaired_from", "renamed_from", "orig_spinner_hex", "orig_romanized"):
        if col not in pal: pal[col] = None
    repairs, renames = [], []
    data = _similarity_data()
    for lang in languages:
        idx = list(pal.index[(pal.language == lang) & pal.grounded_hex.notna()])
        if len(idx) < 2:
            continue
        names = [pal.at[k, "name"] for k in idx]
        shared = {norm(n) for n in names if sum(norm(x) == norm(n) for x in names) > 1}
        opts = []                                       # per colour: (hex, name, kind, cost, extra)
        for k in idx:
            name, g = pal.at[k, "name"], pal.at[k, "grounded_hex"]
            focal = pal.at[k, "spinner_hex"] if isinstance(pal.at[k, "spinner_hex"], str) else g
            o = [(g, name, "keep", 0.0, {})]
            mapper = list(dict.fromkeys(safe(spinner.category_members, name, lang) or []))   # best to worst example
            tiles = [t for t in mapper if t != g]
            ranked = sorted(tiles, key=lambda t: de00(t, focal))
            for t in ranked:
                d = de00(t, focal)
                if d < NOTICEABLE:
                    continue
                if d >= threshold:                      # outside the core: only if speakers still call it this
                    namer = safe(spinner.name, t, lang)
                    if not namer or not same_name(namer.get("name", ""), name):
                        continue
                o.append((t, name, "repair", d, {"from_focal": d, "tile": mapper.index(t) + 1, "of": len(mapper)}))
                if len(o) > tiles_per_colour: break
            if norm(name) in shared:
                m = data.get(_sim_key(lang))
                rom = pal.at[k, "romanized"] if "romanized" in pal and isinstance(pal.at[k, "romanized"], str) else None
                row = None
                if m:
                    row = next((m["index"][norm(c.replace("_", " "))] for c in (name, rom)
                                if isinstance(c, str) and norm(c.replace("_", " ")) in m["index"]), None)
                if row is not None:
                    taken = {norm(n) for n in names}
                    order = sorted(((v, c) for c, v in enumerate(m["similarity"][row]) if c != row), reverse=True)
                    tried = 0
                    for sim, c in order:
                        if tried >= synonyms: break
                        latin = m["names"][c]
                        if norm(latin) in taken: continue
                        tried += 1
                        shown = _spinner_name(spinner, lang, latin)
                        hexb = safe(spinner.best_example, shown, lang) if shown else None
                        if isinstance(hexb, str):
                            o.append((hexb, shown, "rename", 20 + 20 * (1 - sim), {"similarity": sim, "latin": latin}))
            opts.append(o)
        n_combos = int(np.prod([len(o) for o in opts]))
        if n_combos > MAX_COMBINATIONS:
            raise RuntimeError(f"{lang}: {n_combos} combinations, too many to try")
        hexes = sorted({h for o in opts for h, *_ in o})
        prefetch_simulations(hexes)
        S = {(h, c): simulate(h, c) for h in hexes for c in conditions}
        cache = {}
        def dist(a, b):
            key = (a, b) if a <= b else (b, a)
            if key not in cache:
                cache[key] = min(de00(S[(a, c)], S[(b, c)]) for c in conditions)
            return cache[key]
        pairs = list(combinations(range(len(idx)), 2))
        def worst(choice):
            return min(dist(choice[a][0], choice[b][0]) for a, b in pairs)
        start = [o[0] for o in opts]
        before = worst(start)
        best_key, best_choice = None, None
        for choice in product(*opts):
            nm = [norm(c[1]) for c in choice]
            dupes = len(nm) - len(set(nm))
            w = worst(choice)
            renames_n = sum(c[2] == "rename" for c in choice)
            changes = sum(c[2] != "keep" for c in choice)
            cost = sum(c[3] for c in choice)
            reach = w >= threshold
            key = (-dupes, reach, 0 if reach else w, -renames_n, -changes, -cost)
            if best_key is None or key > best_key:
                best_key, best_choice = key, choice
        after = worst(best_choice)
        # the closest pair before and after, and for whom, for the report
        def closest_info(choice):
            best = (np.inf, None, None, None)
            for a, b in pairs:
                for c in conditions:
                    d = de00(S[(choice[a][0], c)], S[(choice[b][0], c)])
                    if d < best[0]: best = (d, a, b, c)
            return best
        d0, a0, b0, c0 = closest_info(start)
        d1, a1, b1, c1 = closest_info(best_choice)
        for k, (h, name, kind, cost, extra) in zip(idx, best_choice):
            old_name, old_hex = pal.at[k, "name"], pal.at[k, "grounded_hex"]
            if kind == "rename":
                renames.append(dict(language=lang, sdg=int(pal.at[k, "sdg"]), old=old_name,
                                    shared_with=int(next(pal.at[j, "sdg"] for j in idx if j != k and norm(pal.at[j, "name"]) == norm(old_name))),
                                    new=name, similarity=extra["similarity"], hex=h, nearest=min(dist(h, c[0]) for j, c in zip(idx, best_choice) if j != k)))
                pal.at[k, "renamed_from"] = old_name
                pal.at[k, "orig_spinner_hex"] = pal.at[k, "spinner_hex"]
                pal.at[k, "orig_romanized"] = pal.at[k, "romanized"] if "romanized" in pal else None
                pal.at[k, "name"] = name
                if "romanized" in pal: pal.at[k, "romanized"] = extra["latin"]
                pal.at[k, "spinner_hex"] = h
                pal.at[k, "grounded_hex"] = h
                pal.at[k, "replaced"] = True
            elif kind == "repair":
                repairs.append(dict(language=lang, name=old_name, old=old_hex, new=h, from_focal=extra["from_focal"],
                                    tile=extra["tile"], of=extra["of"], pair=(names[a0], names[b0]), before=d0, condition=c0, after=after,
                                    after_condition=c1, fixed=after >= threshold))
                pal.at[k, "repaired_from"] = old_hex
                pal.at[k, "grounded_hex"] = h
        if not any(r["language"] == lang for r in repairs) and before < threshold and after < threshold \
                and not any(r["language"] == lang for r in renames):
            repairs.append(dict(language=lang, pair=(names[a0], names[b0]), before=d0, condition=c0, fixed=False))
    pal.attrs["repairs"] = repairs
    pal.attrs["renames"] = renames
    pal.attrs["joint"] = True
    return pal


# ── Part C, section 4 in three steps ──────────────────────────────────────
def vision_phrase(v):
    """'Typical' → 'typical colour vision', 'Deuteranopia' → 'deuteranopia', 'Age 60' → 'age 60'."""
    return "typical colour vision" if v == "Typical" else v.lower()

def ground_palette(pal, replace_when="Either", threshold=DELTA_E_THRESHOLD):
    """Step 1: replace the AI's colour by speakers' best example wherever it's wrong. Starts afresh from
    section 3's names every time, so the steps can be rerun in any order."""
    pal = per_test_colour(undo_renames(pal))
    pal = pal.drop(columns=["repaired_from"], errors="ignore")
    too_far = [pd.isna(d) or d > threshold for d in pal.de00]
    renamed = [isinstance(n, str) and not same_name(n, m, r)
               for n, m, r in zip(pal.get("rt_name", pd.Series(None, index=pal.index)), pal.name,
                                  pal.get("romanized", pd.Series(None, index=pal.index)))]
    rule = {"Either": [a or b for a, b in zip(too_far, renamed)],
            "Too far from the best example": too_far,
            "Speakers would use a different name": renamed}[replace_when]
    pal["grounded_hex"] = [best if isinstance(best, str) and r else ai
                           for ai, best, r in zip(pal.ai_hex, pal.spinner_hex, rule)]
    pal["replaced"] = pal.grounded_hex != pal.ai_hex
    pal.attrs.pop("repairs", None); pal.attrs.pop("renames", None)
    return pal

def name_palette(pal, spinner, languages):
    """Step 2: test colours that share a name get different names (a close synonym's best example)."""
    out = rename_shared_names(pal, spinner, languages, conditions=["Typical"])
    if out.attrs.get("renames"):
        show_renames(out)
    else:
        show("<b>No test colours share a name</b> " + small("in any of your languages, so nothing needed renaming."))
    return out

def repair_for(pal, spinner, languages, vision="Typical", threshold=DELTA_E_THRESHOLD):
    """Step 3: separate colours that look alike for one kind of vision, within their names. Only that kind of
    vision is used: the palette is corrected for the viewer you choose."""
    cond = [vision]
    try:
        out = optimise_palettes(pal, spinner, languages, threshold, conditions=cond)
    except Exception as err:
        show(small(f"Choosing step by step instead ({err})."))
        out = repair_palettes(pal, spinner, languages, threshold, conditions=cond)
    out.attrs["vision"] = vision
    if out.attrs.get("repairs"):
        show_repairs(out, threshold)
    else:
        show(f"<b>No colours needed optimising</b> " + small(f"for {vision_phrase(vision)}: "
             f"every pair is already ΔE₀₀ = {threshold:g} or more apart, or no colour of the same name would help."))
    show_grounded(out, languages, title=f"Your final palette, corrected for {vision_phrase(vision)}")
    return out


# ── Making the saved copy of the Spinner's answers (run once, while the Spinner is up) ──
DEFAULT_SUBSET = [3, 6, 10, 14, 15]
ALL_LANGUAGES_7 = ["American English", "British English", "French", "Greek", "OtjiHimba", "Serbian", "Spanish"]

def build_offline_cache(subset=DEFAULT_SUBSET, languages=ALL_LANGUAGES_7, path="spinner_offline.json"):
    """Ask the Colour Spinner, once, for everything the notebooks need with the default test colours, and save it:
    the name of each test colour in every language; every name's best example; the Mapper tiles of the test colours'
    names and of their Thesaurus synonyms, and what speakers call the tiles further than ΔE₀₀ 5 from the best
    example; and the simulations of the 17 SDG colours, every best example and those tiles. Saves as it goes,
    so it can be rerun if interrupted. Takes a while: about 20–40 minutes, depending on the Spinner."""
    import time
    sp = ApiSpinner()
    out = {"made": time.strftime("%Y-%m-%d %H:%M"), "subset": list(subset), "pages": [], "best": [], "members": [],
           "lang_ids": {}, "lang_names": {}, "sims": []}
    def save():
        out["pages"] = [[k[0], k[1], k[2], v] for k, v in sp._pages.items() if k[0] != "members"]
        out["members"] = [[k[1], k[2], v] for k, v in sp._pages.items() if k[0] == "members"]
        out["best"] = [[k[0], k[1], v] for k, v in sp._best.items()]
        out["lang_ids"], out["lang_names"] = sp._lang_ids, sp._lang_names
        out["sims"] = [[k[0], k[1], v] for k, v in _SIM["cache"].items() if k not in _SIM["local"]]
        json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    step = lambda t: print(f"{time.strftime('%H:%M:%S')}  {t}", flush=True)
    # 1 · every language's names, and each name's best example
    for lang in languages:
        names = sp._language_ids(lang)
        step(f"{lang}: {len(names)} names")
        for key in names:
            shown = sp._lang_names.get(lang, {}).get(key, key)
            try_call(sp.best_example, shown, lang)
        save()
    # 2 · the test colours' names in every language, and their Mapper tiles (and those of their synonyms)
    data = _similarity_data()
    for i in subset:
        for lang in languages:
            res = try_call(sp.lookup, SDG.hex[i], lang)
            if not res: step(f"  SDG {i} in {lang}: no answer"); continue
            names = [res["name"]]
            m = data.get(_sim_key(lang))
            if m:                                    # its synonyms too, for the renaming step
                row = next((m["index"][norm(c)] for c in (res["name"], res.get("romanized")) if isinstance(c, str) and norm(c) in m["index"]), None)
                if row is not None:
                    order = sorted(((v, c) for c, v in enumerate(m["similarity"][row]) if c != row), reverse=True)[:SYNONYMS_TO_TRY]
                    names += [n for n in (_spinner_name(sp, lang, m["names"][c]) for _, c in order) if n]
            for n in names:
                tiles = try_call(sp.category_members, n, lang) or []
                best = try_call(sp.best_example, n, lang)
                for t in tiles:                       # what speakers call the tiles outside the core
                    if best and de00(t, best) >= DELTA_E_THRESHOLD:
                        try_call(sp.name, t, lang)
            step(f"  SDG {i} in {lang}: {res['name']} and {len(names) - 1} synonyms")
        save()
    # 3 · the simulations
    hexes = list(SDG.hex) + list(dict.fromkeys(v for v in sp._best.values() if isinstance(v, str)))
    hexes += [t for k, v in sp._pages.items() if k[0] == "members" for t in v]
    hexes = list(dict.fromkeys(h.upper() for h in hexes))
    step(f"simulating {len(hexes)} colours under 6 conditions")
    for k in range(0, len(hexes), 60):
        prefetch_simulations(hexes[k:k + 60]); save()
        step(f"  {min(k + 60, len(hexes))} of {len(hexes)}")
    save()
    step(f"done: {path}, {len(out['pages'])} lookups, {len(out['best'])} best examples, {len(out['members'])} Mapper lists, "
         f"{len(out['sims'])} simulations")
    return path


def show_nearest_sdg(pal, languages, threshold=DELTA_E_THRESHOLD):
    """Step 4: each final colour matched to its nearest of the UN's 17 SDG colours (ΔE₀₀, typical vision). Grounding
    and optimising can move a colour so far that it now looks more like another goal's colour than its own."""
    get = _grounded_lookup(pal)
    tests = list(dict.fromkeys(_BY_SDG.sdg.astype(int))) if _BY_SDG is not None and len(_BY_SDG) else []
    langs = [l for l in languages if (pal.language == l).any()]
    if not tests or not langs:
        show(small("This step needs the names from Part B.")); return
    rows, moved = [], 0
    for i in tests:
        cells = []
        for l in langs:
            t = _BY_SDG[(_BY_SDG.sdg == i) & (_BY_SDG.language == l)]
            r = get(l, i, t["name"].iloc[0]) if not t.empty else None
            if r is None or not isinstance(r.grounded_hex, str):
                cells.append(small("—")); continue
            d = {k: de00(r.grounded_hex, h) for k, h in SDG.hex.items()}
            k = min(d, key=d.get)
            own = k == i
            moved += not own
            badge = pill(f"#{k} · ΔE₀₀ = {d[k]:.1f}", "good" if own else "mid")
            note = small("still closest to its own goal") if own else small(f"now closer to #{k} than to #{i} (ΔE₀₀ = {d[i]:.1f})")
            cells.append(f'{framed(r.grounded_hex, r["name"], 104, 50, 12)}<div style="margin-top:6px">{badge}<br>{note}</div>')
        rows.append([framed(SDG.hex[i], f"#{i}", 84, 50, 12)] + cells)
    show("<b>Which SDG colour is each of your colours closest to?</b>")
    table(["Your test colour"] + [_h(l) for l in langs], rows, valign="top",
          note=f"Each final colour, matched to the nearest of the UN's 17 SDG colours (ΔE₀₀, typical colour vision). "
               f"{moved} of your colours now look more like another goal's colour than their own: in a design, check that "
               f"they can't be mistaken for that goal." if moved else
               "Every final colour is still closest to its own goal's colour.")

TOOLKIT_VERSION = "2026-10-31 13:16"
print(f"Toolkit ready ✓  version {TOOLKIT_VERSION}  " + ("(running in Colab)" if IN_COLAB else "(running locally)"))
