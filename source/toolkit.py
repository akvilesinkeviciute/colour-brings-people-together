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
#  1. Colour conversions
# ════════════════════════════════════════════════════════════════════════
def hex_to_rgb(h):
    h = h.strip().lstrip("#")
    if len(h) == 3: h = "".join(c * 2 for c in h)
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)

def rgb_to_hex(rgb):
    r, g, b = np.clip(np.round(np.asarray(rgb, float)), 0, 255).astype(int)
    return f"#{r:02X}{g:02X}{b:02X}"

def _as_rgb(c):
    """Accept '#hex', (r,g,b) in 0–255, or arrays of those."""
    if isinstance(c, str): return hex_to_rgb(c)
    return np.asarray(c, float)

def srgb_to_linear(rgb255):
    c = np.asarray(rgb255, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(lin):
    lin = np.clip(lin, 0, 1)
    c = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return c * 255.0

_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
WHITE_D65 = np.array([95.047, 100.0, 108.883])

def rgb_to_xyz(rgb255):
    return srgb_to_linear(rgb255) @ _M_RGB2XYZ.T * 100.0

def xyz_to_lab(xyz, white=WHITE_D65):
    t = np.asarray(xyz, float) / white
    d = 6 / 29
    f = np.where(t > d ** 3, np.cbrt(t), t / (3 * d ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)

def rgb_to_lab(c):
    return xyz_to_lab(rgb_to_xyz(_as_rgb(c)))

def delta_e2000(lab1, lab2):
    """CIEDE2000 colour difference (Sharma, Wu & Dalal 2005)."""
    L1, a1, b1 = np.moveaxis(np.asarray(lab1, float), -1, 0)
    L2, a2, b2 = np.moveaxis(np.asarray(lab2, float), -1, 0)
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(C1p * C2p == 0, 0, np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh)))
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1p + h2p
    hbp = np.where(C1p * C2p == 0, hs,
          np.where(np.abs(h1p - h2p) <= 180, hs / 2,
          np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dtheta = 30 * np.exp(-((hbp - 275) / 25) ** 2)
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                   + Rt * (dCp / Sc) * (dHp / Sh))

def de00(c1, c2):
    """ΔE2000 between two colours given as hex or RGB."""
    return float(delta_e2000(rgb_to_lab(c1), rgb_to_lab(c2)))

def spread(hexes):
    """Largest ΔE2000 between any two colours in a list (NaN if fewer than two)."""
    from itertools import combinations
    hexes = [h for h in hexes if isinstance(h, str)]
    return max((de00(a, b) for a, b in combinations(hexes, 2)), default=np.nan)

# ── CAM16-UCS (Li et al. 2017), the space the Colour Spinner works in ────
_M16 = np.array([[ 0.401288, 0.650173, -0.051461],
                 [-0.250268, 1.204414,  0.045854],
                 [-0.002079, 0.048952,  0.953127]])

def xyz_to_cam16ucs(xyz, white=WHITE_D65, L_A=64 / np.pi * 0.2, Y_b=20.0,
                    F=1.0, c=0.69, Nc=1.0):
    xyz = np.asarray(xyz, float)
    RGB_w = _M16 @ white
    D = np.clip(F * (1 - (1 / 3.6) * np.exp((-L_A - 42) / 92)), 0, 1)
    D_RGB = D * white[1] / RGB_w + 1 - D
    k = 1 / (5 * L_A + 1)
    F_L = 0.2 * k ** 4 * (5 * L_A) + 0.1 * (1 - k ** 4) ** 2 * (5 * L_A) ** (1 / 3)
    n = Y_b / white[1]
    z = 1.48 + np.sqrt(n)
    N_bb = 0.725 * n ** -0.2
    def adapt(x):
        p = (F_L * np.abs(x) / 100) ** 0.42
        return 400 * np.sign(x) * p / (p + 27.13) + 0.1
    Rw, Gw, Bw = adapt(D_RGB * RGB_w)
    A_w = (2 * Rw + Gw + 0.05 * Bw - 0.305) * N_bb
    R, G, B = np.moveaxis(adapt(D_RGB * (xyz @ _M16.T)), -1, 0)
    a = R - 12 * G / 11 + B / 11
    b = (R + G - 2 * B) / 9
    h = np.arctan2(b, a)
    e_t = 0.25 * (np.cos(h + 2) + 3.8)
    A = (2 * R + G + 0.05 * B - 0.305) * N_bb
    J = 100 * np.maximum(A / A_w, 0) ** (c * z)
    t = (50000 / 13 * Nc * N_bb * e_t * np.hypot(a, b)) / (R + G + 21 * B / 20)
    C = np.maximum(t, 0) ** 0.9 * np.sqrt(J / 100) * (1.64 - 0.29 ** n) ** 0.73
    M = C * F_L ** 0.25
    Jp = 1.7 * J / (1 + 0.007 * J)
    Mp = np.log1p(0.0228 * M) / 0.0228
    return np.stack([Jp, Mp * np.cos(h), Mp * np.sin(h)], -1)

def rgb_to_cam16ucs(c):
    return xyz_to_cam16ucs(rgb_to_xyz(_as_rgb(c)))

def de_ucs(c1, c2):
    """Euclidean distance in CAM16-UCS between two colours (hex or RGB)."""
    return float(np.linalg.norm(rgb_to_cam16ucs(c1) - rgb_to_cam16ucs(c2)))


# ════════════════════════════════════════════════════════════════════════
#  2. Vision simulations — the perceptual axis
# ════════════════════════════════════════════════════════════════════════
#  Colour vision deficiency: Machado, Oliveira & Fernandes (2009), severity 1.0,
#  applied in linear sRGB.
_MACHADO = {
    "Protanopia":   np.array([[ 0.152286, 1.052583, -0.204868],
                              [ 0.114503, 0.786281,  0.099216],
                              [-0.003882,-0.048116,  1.051998]]),
    "Deuteranopia": np.array([[ 0.367322, 0.860646, -0.227968],
                              [ 0.280085, 0.672501,  0.047413],
                              [-0.011820, 0.042940,  0.968881]]),
    "Tritanopia":   np.array([[ 1.255528,-0.076749, -0.178779],
                              [-0.078411, 0.930809,  0.147602],
                              [ 0.004733, 0.691367,  0.303900]]),
}

#  The notebooks use the Colour Spinner's own simulations (see simulate() below).
#  The local models here are a fallback for when the Spinner cannot be reached.
#  Ageing fallback: lens yellowing only, a first-order approximation.
#  Optical density of the crystalline lens after van de Kraats & van Norren
#  (2007), evaluated at the dominant wavelengths of the sRGB primaries and
#  expressed relative to a 20-year-old observer. It ignores pupil size, scatter
#  and chromatic adaptation; treat the Colour Spinner's age simulation as the
#  reference and use this only to explore trends.
def _lens_density(lam, age):
    A = age
    return ((0.446 + 0.000031 * A**2) * (400 / lam) ** 4
            + 14.19 * 10.68 * np.exp(-(0.057 * (lam - 273)) ** 2)
            + (0.998 - 0.000063 * A**2) * 2.13 * np.exp(-(0.029 * (lam - 370)) ** 2)
            + (0.059 + 0.000186 * A**2) * 11.95 * np.exp(-(0.021 * (lam - 325)) ** 2)
            + (0.016 + 0.000132 * A**2) * 1.43 * np.exp(-(0.008 * (lam - 325)) ** 2)
            + 0.111)

_PRIMARY_NM = np.array([611.0, 549.0, 464.0])
def _age_gain(age, ref=20):
    return 10 ** -(_lens_density(_PRIMARY_NM, age) - _lens_density(_PRIMARY_NM, ref))

CONDITIONS = ["Typical", "Protanopia", "Deuteranopia", "Tritanopia",
              "Age 40", "Age 60", "Age 80"]
# The Spinner simulates age bands; its menu labels them by the upper age.
AGE_BANDS = {"Age 40": "16–40", "Age 60": "40–60", "Age 80": "60–80"}
def condition_label(c, sep="\n"):
    return f"{c}{sep}({AGE_BANDS[c]})" if c in AGE_BANDS else c

def simulate_local(c, condition="Typical"):
    """Local fallback: the hex of colour `c` under `condition` (used only if the Spinner is unreachable)."""
    lin = srgb_to_linear(_as_rgb(c))
    if condition in _MACHADO:
        lin = _MACHADO[condition] @ lin
    elif condition.startswith("Age"):
        lin = lin * _age_gain(int(condition.split()[1]))
    return rgb_to_hex(linear_to_srgb(lin))


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
    options = {_strip(n): int(v) for v, n in re.findall(r'<option value="(\d+)"[^>]*>(.*?)</option>', page)}
    if not name or conf is None:
        raise RuntimeError("The Colour Spinner's reply did not contain a Namer result; the site may have changed.")
    return {
        "name": _strip(name),
        "confidence": float(conf) / 100,
        "best_example": rgb_to_hex([int(x) for x in rgb[:3]]) if len(rgb) >= 3 else None,
        "translation": _strip(tr_title) if tr_title else (_html.unescape(tr_name) if tr_name else None),
        "translation_confidence": float(tr_conf) if tr_conf else None,
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
        return self._pages[key]

    def name(self, hex_, language):
        res = self.lookup(hex_, language)
        return {"name": res["name"], "confidence": res["confidence"]}

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
_SIM = {"client": None, "use_spinner": True, "cache": {}, "fallbacks": 0}

def _sim_client():
    if _SIM["use_spinner"] and _SIM["client"] is None:
        try:
            _SIM["client"] = ApiSpinner(timeout=10)
        except Exception:
            _SIM["use_spinner"] = False
    return _SIM["client"] if _SIM["use_spinner"] else None

def simulate(c, condition="Typical"):
    """The hex of colour `c` as seen under `condition`, using the Colour Spinner's
    simulation models. If the Spinner cannot be reached, the notebook switches to
    local models for the rest of the session and says so in its notes."""
    hx = (c if isinstance(c, str) else rgb_to_hex(c)).upper()
    if not hx.startswith("#"): hx = "#" + hx
    if condition == "Typical":
        return hx
    key = (hx, condition)
    if key not in _SIM["cache"]:
        client = _sim_client()
        if client is not None:
            try:
                _SIM["cache"][key] = (client.simulate(hx, condition), "spinner")
            except Exception as e:
                _SIM["use_spinner"] = False
                print(f"⚠ The Colour Spinner's simulation is unavailable ({str(e)[:80]}). "
                      "Using local approximations for the rest of this session.")
        if key not in _SIM["cache"]:
            _SIM["cache"][key] = (simulate_local(hx, condition), "local")
            _SIM["fallbacks"] += 1
    return _SIM["cache"][key][0]

def prefetch_simulations(hexes, conditions=CONDITIONS):
    """Fetch the Spinner's simulations for many colours at once (a few requests in parallel)."""
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(h, c) for h in dict.fromkeys(hexes) for c in conditions if c != "Typical"]
    client = _sim_client()
    if client is None or not jobs:
        return
    try:                                  # one request first, so an outage fails fast
        simulate(*jobs[0])
    except Exception:
        return
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda job: simulate(*job), jobs[1:]))

def simulation_source_note():
    srcs = {src for _, src in _SIM["cache"].values()}
    if srcs == {"spinner"} or not srcs:
        return ("Simulations from the Colour Spinner (colornaming.net/spinner); "
                "age rows are its age bands.")
    if srcs == {"local"}:
        return ("The Colour Spinner could not be reached, so these are local approximations: "
                "CVD after Machado et al. (2009); ageing as lens yellowing only.")
    return ("Mostly the Colour Spinner's simulations; some colours fell back to local "
            "approximations because the Spinner stopped responding.")


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
    """Compare colour names ignoring case, accents and spacing (Κόκκινο == κοκκινο)."""
    s = unicodedata.normalize("NFD", str(s).strip().lower())
    return re.sub(r"\s+", " ", "".join(ch for ch in s if unicodedata.category(ch) != "Mn"))

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
    raise ValueError("Choose a provider, or paste the AI's reply into AI_REPLY.")

def parse_json_reply(text):
    """Pull the first JSON array/object out of an AI reply (ignores ``` fences and chatter)."""
    t = re.sub(r"```(?:json)?", "", text)
    start = min([i for i in (t.find("["), t.find("{")) if i >= 0], default=-1)
    if start < 0: raise ValueError("No JSON found in the AI reply. Paste the reply into AI_REPLY, or choose a provider in the form.")
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
    Y = rgb_to_xyz(hex_to_rgb(hex_))[1] / 100
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
            n, c = r.get(l), r.get(f"{l}|conf")
            if isinstance(n, str):
                w = 0 if pd.isna(c) else int(round(100 * c))
                cells += (f'<td><b>{_h(n)}</b><div class="bar"><i style="width:{w}%"></i></div>'
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
