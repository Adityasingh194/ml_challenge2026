# ------------------------- copied from er_crossencoder_v2 §4 (cell 14) -------------------------
_LEGAL_SUFFIX_MAP = {
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "limited": "ltd", "ltd": "ltd",
    "llc": "llc", "l.l.c": "llc", "llp": "llp", "l.l.p": "llp", "lp": "lp",
    "private": "pvt", "pvt": "pvt", "plc": "plc", "gmbh": "gmbh",
    "sa": "sa", "sarl": "sarl", "sas": "sas", "bv": "bv", "nv": "nv",
    "pty": "pty", "group": "group", "holdings": "holdings", "trust": "trust",
    "foundation": "fdn", "enterprises": "ent", "industries": "ind",
    "international": "intl", "solutions": "sol", "services": "svc",
}
_LEGAL_SUFFIX_TOKENS = set(_LEGAL_SUFFIX_MAP)


def strip_legal_suffix(name_clean):
    """Return (core_name, canonical_suffix_or_None). Ampersand normalized to 'and'."""
    if not name_clean:
        return "", None
    text = re.sub(r"[^a-z0-9&\s]", " ", name_clean)
    text = text.replace("&", " and ")
    tokens = text.split()
    suffix = None
    # only the single trailing token -- stripping repeatedly would eat real words ("Trust Bank Co")
    if len(tokens) > 1 and tokens[-1] in _LEGAL_SUFFIX_TOKENS:
        suffix = _LEGAL_SUFFIX_MAP[tokens.pop()]
    return " ".join(tokens), suffix


def name_sorted_signature(core_tokens_str):
    """Order-invariant signature: sorted tokens joined. Catches word transpositions."""
    return " ".join(sorted(core_tokens_str.split()))


def name_acronym(core_tokens_str):
    toks = core_tokens_str.split()
    if len(toks) < 2:
        return ""
    return "".join(t[0] for t in toks if t)


_SOUNDEX_CODES = {
    **{c: "1" for c in "bfpv"}, **{c: "2" for c in "cgjkqsxz"},
    **{c: "3" for c in "dt"}, "l": "4", **{c: "5" for c in "mn"}, "r": "6",
}

def soundex(token):
    """Classic Soundex phonetic code (4 chars). Pure-stdlib, no extra dependency."""
    token = re.sub(r"[^a-z]", "", token.lower())
    if not token:
        return ""
    first = token[0]
    out = []
    prev = _SOUNDEX_CODES.get(first, "")
    for c in token[1:]:
        code_c = _SOUNDEX_CODES.get(c, "")
        if code_c and code_c != prev:
            out.append(code_c)
        if c not in "hw":  # h/w don't separate duplicate codes; vowels do
            prev = code_c
    return (first.upper() + "".join(out) + "000")[:4]


def name_phonetic_key(core_tokens_str):
    toks = core_tokens_str.split()
    if not toks:
        return ""
    return soundex(toks[0]) + (soundex(toks[1]) if len(toks) > 1 else "")


def normalize_names(df):
    core_suffix = df["name_clean"].map(strip_legal_suffix)
    df["name_core"] = [c for c, _ in core_suffix]
    df["name_suffix"] = [s or "" for _, s in core_suffix]
    df["name_sorted"] = df["name_core"].map(name_sorted_signature)
    df["name_acronym"] = df["name_core"].map(name_acronym)
    df["name_phonetic"] = df["name_core"].map(name_phonetic_key)
    return df


_COUNTRY_ALIASES = {
    "us": "US", "usa": "US", "u.s.": "US", "u.s.a.": "US", "united states": "US",
    "united states of america": "US",
    "india": "India", "in": "India", "bharat": "India",
    "france": "France", "fr": "France",
}

def canonicalize_country(raw):
    """Soft canonicalization only -- unrecognized strings pass through as-is (title-cased),
    never filtered or mapped to null. Keeps `country` an open set end to end."""
    if raw is None:
        return ""
    key = str(raw).strip().lower()
    if key in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[key]
    return str(raw).strip().title() if raw else ""

# ------------------------- copied from er_crossencoder_v2 §4 (cell 15) -------------------------
# Region registry keyed by COUNTRY -> {lowercase name/code/alias: canonical name}. Keyed by country
# because short codes collide across countries ("tn" = Tennessee or Tamil Nadu, "ga" = Georgia or
# Goa). It is an open dict: a country with no entry simply gets no region canonicalization.
_REGION_REGISTRY = {}

def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _add_regions(country, spec):
    """spec: 'Name:code1/code2, Name2:code, ...' -- codes/aliases optional."""
    d = _REGION_REGISTRY.setdefault(country, {})
    for item in spec.split(","):
        name, _, alts = item.strip().partition(":")
        for k in [name] + [a for a in alts.split("/") if a]:
            d[_strip_accents(k).lower().strip()] = name

_add_regions("US", "Alabama:al, Alaska:ak, Arizona:az, Arkansas:ar, California:ca/calif, Colorado:co, "
    "Connecticut:ct, Delaware:de, District of Columbia:dc/washington dc, Florida:fl, Georgia:ga, Hawaii:hi, "
    "Idaho:id, Illinois:il, Indiana:in, Iowa:ia, Kansas:ks, Kentucky:ky, Louisiana:la, Maine:me, "
    "Maryland:md, Massachusetts:ma, Michigan:mi, Minnesota:mn, Mississippi:ms, Missouri:mo, Montana:mt, "
    "Nebraska:ne, Nevada:nv, New Hampshire:nh, New Jersey:nj, New Mexico:nm, New York:ny, "
    "North Carolina:nc, North Dakota:nd, Ohio:oh, Oklahoma:ok, Oregon:or, Pennsylvania:pa, "
    "Rhode Island:ri, South Carolina:sc, South Dakota:sd, Tennessee:tn, Texas:tx, Utah:ut, Vermont:vt, "
    "Virginia:va, Washington:wa, West Virginia:wv, Wisconsin:wi, Wyoming:wy, Puerto Rico:pr")
_add_regions("India", "Andhra Pradesh:ap, Arunachal Pradesh:ar, Assam:as, Bihar:br, Chhattisgarh:cg/ct/chattisgarh, "
    "Goa:ga, Gujarat:gj/gujrat, Haryana:hr, Himachal Pradesh:hp, Jharkhand:jh, Karnataka:ka/karnatka, "
    "Kerala:kl, Madhya Pradesh:mp, Maharashtra:mh/maharastra, Manipur:mn, Meghalaya:ml, Mizoram:mz, "
    "Nagaland:nl, Odisha:od/or/orissa, Punjab:pb, Rajasthan:rj, Sikkim:sk, Tamil Nadu:tn/tamilnadu, "
    "Telangana:tg/ts, Tripura:tr, Uttar Pradesh:up, Uttarakhand:uk/uttaranchal, West Bengal:wb, "
    "Delhi:dl/new delhi, Jammu and Kashmir:jk, Ladakh:la, Chandigarh:ch, Puducherry:py/pondicherry, "
    "Andaman and Nicobar Islands:an, Dadra and Nagar Haveli and Daman and Diu:dd, Lakshadweep:ld")
_add_regions("France", "Ile-de-France:idf/ile de france, Auvergne-Rhone-Alpes:ara/rhone alpes, "
    "Bourgogne-Franche-Comte:bfc/franche comte/bourgogne, Bretagne:bzh/brittany, Centre-Val de Loire:cvl, "
    "Corse:corsica, Grand Est:alsace/lorraine/champagne ardenne, Hauts-de-France:hdf/nord pas de calais/picardie, "
    "Normandie:normandy, Nouvelle-Aquitaine:aquitaine, Occitanie:languedoc roussillon/midi pyrenees, "
    "Pays de la Loire:pdl, Provence-Alpes-Cote d'Azur:paca/provence alpes cote d azur, "
    "Guadeloupe, Martinique, Guyane:french guiana, La Reunion:reunion, Mayotte")

# unambiguous long names/aliases usable for ANY country (e.g. a mislabeled record)
_REGION_LONG_ANY = {k: v for d in _REGION_REGISTRY.values() for k, v in d.items() if len(k) > 3}


def canonical_region(value, country):
    v = _strip_accents(str(value)).strip().lower().rstrip(".")
    if not v:
        return ""
    d = _REGION_REGISTRY.get(country, {})
    if v in d:
        return d[v].lower()
    return _REGION_LONG_ANY.get(v, v).lower()


def extract_region_fallback(addr_clean, country):
    """Country-aware region lookup from the address; used only when extracted_state is blank.
    Short codes (<=3 chars) only count when they are an entire comma-separated segment --
    otherwise they'd false-match ordinary address tokens."""
    if not addr_clean:
        return ""
    d = _REGION_REGISTRY.get(country, {})
    text = re.sub(r"[^a-z\s,]", " ", _strip_accents(addr_clean).lower())
    segments = [seg.split() for seg in text.split(",")]
    for seg in segments:
        if len(seg) == 1 and len(seg[0]) <= 3 and seg[0] in d:
            return d[seg[0]]
    tokens = [t for seg in segments for t in seg]
    n = len(tokens)
    for width in (5, 4, 3, 2, 1):  # longest first, so "west bengal" beats "bengal"
        for start in range(n - width + 1):
            cand = " ".join(tokens[start:start + width])
            if len(cand) > 3 and (cand in d or cand in _REGION_LONG_ANY):
                return d.get(cand) or _REGION_LONG_ANY[cand]
    return ""


_DIGIT_TOKEN_RE = re.compile(r"\d+")

def digit_tokens(addr_clean):
    return frozenset(_DIGIT_TOKEN_RE.findall(addr_clean or ""))


def normalize_all(df):
    """copied; the addr_digits / name_tokens frozenset columns moved to add_set_columns()."""
    df = normalize_names(df)
    df["country_norm"] = df["country"].map(canonicalize_country)
    blank = df["extracted_state"].astype(str).str.strip() == ""
    if blank.any():
        df.loc[blank, "extracted_state"] = [
            extract_region_fallback(a, c) for a, c in zip(df.loc[blank, "addr_clean"], df.loc[blank, "country_norm"])]
    df["state_norm"] = [canonical_region(s, c) for s, c in zip(df["extracted_state"], df["country_norm"])]
    df["city_norm"] = [_strip_accents(str(c)).strip().lower() for c in df["extracted_city"]]
    return df

# ------------------------- new in this notebook -------------------------
_ADDR_WORD_RE = re.compile(r"[a-z]{3,}")


def add_set_columns(df):
    """The frozenset columns the copied _features_chunk expects, built only for the rows a chunk needs."""
    df["addr_digits"] = df["addr_clean"].map(digit_tokens)
    df["name_tokens"] = df["name_clean"].astype(str).map(lambda s: frozenset(s.split()))
    df["addr_words"] = df["addr_clean"].map(lambda s: frozenset(_ADDR_WORD_RE.findall(s or "")))
    return df


_POSTCODE_RE = {
    "India": re.compile(r"(?<!\d)([1-9]\d{2})\s?(\d{3})(?!\d)"),
    "US": re.compile(r"(?<!\d)(\d{5})(?:-\d{4})?(?!\d)"),
    "France": re.compile(r"(?<!\d)(\d{5})(?!\d)"),
}


def extract_postcode(addr_clean, country_norm):
    """Country-aware ZIP/PIN. '' when absent -- a missing postcode never counts as agreement."""
    rx = _POSTCODE_RE.get(country_norm)
    if rx is None or not addr_clean:
        return ""
    ms = list(rx.finditer(addr_clean))
    if not ms:
        return ""
    m = ms[-1]
    if country_norm == "US" and len(ms) == 1 and _DIGIT_TOKEN_RE.search(addr_clean).start() == m.start():
        return ""   # a lone leading 5-digit number is a house number ("12345 main st"), not a ZIP
    return "".join(g for g in m.groups() if g)


def primary_house_number(addr_clean, postcode):
    """First number in the address that is not (part of) the postcode; '' when there is none."""
    for tok in _DIGIT_TOKEN_RE.findall(addr_clean or ""):
        if postcode and (tok == postcode or tok in (postcode[:3], postcode[3:])):
            continue
        return tok
    return ""


def is_non_latin(text):
    return any(ch.isalpha() and ord(ch) > 0x024F for ch in (text or ""))


NORM_KEEP_COLS = ["entity_id", "name_clean", "addr_clean", "name_core", "name_suffix", "name_sorted",
                  "name_acronym", "name_phonetic", "country_norm", "state_norm", "city_norm",
                  "postcode", "house_no", "non_latin"]


def normalize_frame(df, min_parallel_rows=200_000):
    """Raw columns -> the lean normalized frame used by retrieval and features. Large frames are cut into
    row chunks normalized in NUM_WORKERS processes (the functions are pure per row, so the result is identical)."""
    if len(df) >= min_parallel_rows and cfg.NUM_WORKERS > 1:
        step = math.ceil(len(df) / (4 * cfg.NUM_WORKERS))
        parts = joblib.Parallel(n_jobs=cfg.NUM_WORKERS, backend="loky")(
            joblib.delayed(_normalize_frame_serial)(df.iloc[a:a + step].reset_index(drop=True))
            for a in range(0, len(df), step))
        return pd.concat(parts, ignore_index=True)
    return _normalize_frame_serial(df)


def _normalize_frame_serial(df):
    df = normalize_all(df)
    df["postcode"] = [extract_postcode(a, c) for a, c in zip(df["addr_clean"], df["country_norm"])]
    df["house_no"] = [primary_house_number(a, p) for a, p in zip(df["addr_clean"], df["postcode"])]
    df["non_latin"] = np.fromiter((is_non_latin(x) for x in df["business_name"]), dtype=np.int8, count=len(df))
    keep = NORM_KEEP_COLS + [c for c in ("pool_source", "samp_match", "samp_dist", "samp_hard") if c in df.columns]
    return df[keep].reset_index(drop=True)