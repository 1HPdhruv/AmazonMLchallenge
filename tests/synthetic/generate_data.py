"""Synthetic entity-resolution benchmark in the OFFICIAL competition layout (DATASET UNAVAILABLE MODE).

SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE.
The distributions are invented to exercise every pipeline component; they are NOT estimates of
the real competition data. The output schema matches the official spec exactly: tab-separated files
with entity_id (S1-/S2-/S3- prefixes), business_name, business_address (single free-text field),
country. Train countries are {US, India}; test adds France (open-set country check).

Covered phenomena: exact matches; case/punctuation/whitespace/spelling noise; token reordering;
abbreviations; dropped tokens; address-format variation incl. landmark references and component
reordering; missing name/address; common vs rare names; chain branches (same name, different
address); shared buildings; near-miss decoys; no-match / single / multi-match entities; duplicate
source records; numeric address noise; a candidate-explosion chain; partially missing records.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import unicodedata
import zlib

import numpy as np
import pandas as pd

COUNTRIES = {
    "US": dict(aliases=["US", "USA", "United States"],
               legal=["Inc", "LLC", "Corp", "Co"],
               street_types=[("Street", "St"), ("Avenue", "Ave"), ("Road", "Rd"), ("Boulevard", "Blvd")],
               cities=[("New York", "NY"), ("Chicago", "IL"), ("Houston", "TX"), ("Seattle", "WA"),
                       ("Denver", "CO"), ("Boston", "MA"), ("Austin", "TX"), ("Miami", "FL")],
               postal=lambda r: f"{r.integers(10000, 99999)}", fmt="us"),
    "DE": dict(aliases=["DE", "Germany", "Deutschland"],
               legal=["GmbH", "AG", "KG"],
               street_types=[("Straße", "Str."), ("Weg", "Weg"), ("Allee", "Allee")],
               cities=[("Berlin", "BE"), ("München", "BY"), ("Hamburg", "HH"), ("Köln", "NW"),
                       ("Frankfurt", "HE"), ("Düsseldorf", "NW")],
               postal=lambda r: f"{r.integers(10000, 99999)}", fmt="de"),
    "FR": dict(aliases=["FR", "France"],
               legal=["SARL", "SA", "SAS"],
               street_types=[("Rue", "R."), ("Avenue", "Av."), ("Boulevard", "Bd")],
               cities=[("Paris", "IDF"), ("Lyon", "ARA"), ("Marseille", "PAC"), ("Nantes", "PDL"),
                       ("Lille", "HDF"), ("Toulouse", "OCC")],
               postal=lambda r: f"{r.integers(10000, 99999)}", fmt="fr"),
    "IN": dict(aliases=["IN", "India"],
               legal=["Pvt Ltd", "Ltd", "Private Limited"],
               street_types=[("Road", "Rd"), ("Marg", "Marg"), ("Nagar", "Ngr")],
               cities=[("Mumbai", "MH"), ("Bengaluru", "KA"), ("Chennai", "TN"), ("Delhi", "DL"),
                       ("Pune", "MH"), ("Hyderabad", "TG")],
               postal=lambda r: f"{r.integers(110000, 699999)}", fmt="in"),
    "BR": dict(aliases=["BR", "Brazil", "Brasil"],
               legal=["Ltda", "SA", "ME"],
               street_types=[("Rua", "R."), ("Avenida", "Av."), ("Travessa", "Tv.")],
               cities=[("São Paulo", "SP"), ("Rio de Janeiro", "RJ"), ("Curitiba", "PR"),
                       ("Recife", "PE"), ("Salvador", "BA")],
               postal=lambda r: f"{r.integers(10000, 99999)}-{r.integers(100, 999)}", fmt="br"),
    "JP": dict(aliases=["JP", "Japan"],
               legal=["KK", "Co Ltd"],
               street_types=[("Dori", "Dori"), ("Machi", "Machi")],
               cities=[("Tokyo", "13"), ("Osaka", "27"), ("Nagoya", "23"), ("Sapporo", "01")],
               postal=lambda r: f"{r.integers(100, 999)}-{r.integers(1000, 9999)}", fmt="jp"),
}
LABEL = {"US": "US", "IN": "India", "FR": "France", "DE": "Germany", "BR": "Brazil", "JP": "Japan"}
TRAIN_COUNTRY_P = {"US": 0.55, "IN": 0.45}             # official: train = {US, India}
TEST_COUNTRY_P = {"US": 0.45, "IN": 0.35, "FR": 0.20}  # official: test adds France (unseen in train)
LANDMARKS = ["Near SBI ATM", "Opp. Bus Stand", "Behind City Mall", "Near Railway Station",
             "Next to Apollo Pharmacy"]

COMMON_WORDS = ["Blue", "River", "Star", "Golden", "Green", "Royal", "Summit", "Pacific", "Eagle",
                "Silver", "North", "Sun", "Oak", "Crown", "Prime", "Metro", "City", "Alpha", "Delta",
                "Pioneer", "Liberty", "Atlas", "Apex", "Harbor", "Vista", "Bright", "First", "Union"]
INDUSTRY = ["Logistics", "Bakery", "Consulting", "Systems", "Foods", "Motors", "Pharma", "Textiles",
            "Electronics", "Construction", "Software", "Trading", "Hotels", "Dental", "Media",
            "Solutions", "Engineering", "Imports", "Coffee", "Furniture", "Clinic", "Auto Parts"]
GENERIC = ["Group", "International", "Holdings", "Enterprises", "Services", "Company"]
STREET_WORDS = ["Main", "Oak", "Maple", "Park", "Lake", "Hill", "Market", "Church", "Station",
                "Garden", "Mill", "Bridge", "King", "Queen", "Victoria", "Linden", "Rosen", "Berg",
                "Paix", "Fleurs", "Liberdade", "Gandhi", "Nehru", "Sakura", "Ginza", "Harbor",
                "Cedar", "Elm", "Pine", "Sunset", "Lincoln", "Washington", "Mozart", "Goethe"]
ABBREV = {"International": "Intl", "Company": "Co", "Corporation": "Corp", "Enterprises": "Ent",
          "Services": "Svcs", "Holdings": "Hldgs", "Engineering": "Eng", "Electronics": "Elec",
          "Construction": "Constr", "Private Limited": "Pvt Ltd", "Brothers": "Bros"}
SYL = ["ka", "ze", "tri", "vo", "lan", "mar", "qu", "en", "ra", "dor", "vex", "li", "no", "sta",
       "bel", "cor", "fin", "gal", "hel", "jun", "kor", "lum", "nex", "or", "pax", "rin", "sol",
       "tor", "ul", "ven", "wyn", "xa", "yor", "zel", "ami", "bri", "cas", "dru", "eli", "fal"]


def _brand(r) -> str:
    n = r.choice([2, 2, 3, 3, 4])
    return "".join(r.choice(SYL, n)).capitalize()


def _fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


class Gen:
    def __init__(self, n_entities: int, seed: int):
        self.r = np.random.default_rng(seed)
        self.n = n_entities
        self.brands = list({_brand(self.r) for _ in range(n_entities * 3)})
        self.bi = 0

    # ---------- clean entity construction ----------
    def new_brand(self):
        self.bi = (self.bi + 1) % len(self.brands)
        return self.brands[self.bi]

    def make_name(self, cc, core=None):
        r = self.r
        c = COUNTRIES[cc]
        if core is None:
            u = r.random()
            if u < 0.55:
                core = self.new_brand()
            elif u < 0.85:
                core = f"{r.choice(COMMON_WORDS)} {self.new_brand()}"
            else:  # common-words-only name: generic, collision-prone
                core = f"{r.choice(COMMON_WORDS)} {r.choice(COMMON_WORDS)}"
            if cc in ("FR", "DE", "BR") and r.random() < 0.25:
                core += " " + {"FR": "Société", "DE": "Bäckerei", "BR": "Comércio"}[cc]
        parts = []
        if r.random() < 0.08:
            parts.append("The")
        parts.append(core)
        parts.append(str(r.choice(INDUSTRY)))
        if r.random() < 0.3:
            parts.append(str(r.choice(GENERIC)))
        parts.append(str(r.choice(c["legal"])))
        return " ".join(parts), core

    def make_address(self, cc, city=None):
        r = self.r
        c = COUNTRIES[cc]
        if city is None:
            city = c["cities"][r.integers(len(c["cities"]))]
        # small house numbers are common -> weak identifiers on their own
        hn = int(r.choice([r.integers(1, 30), r.integers(1, 300), r.integers(1, 9999)], p=[.4, .35, .25]))
        sw = str(r.choice(STREET_WORDS))
        st_full, _ = c["street_types"][r.integers(len(c["street_types"]))]
        unit = f"Suite {r.integers(1, 40) * 10}" if (cc == "US" and r.random() < 0.3) else ""
        # postal codes drawn from a small per-city pool -> postal alone is a weak key
        pool_seed = zlib.crc32(f"{cc}|{city[0]}".encode("utf-8"))
        pr = np.random.default_rng(pool_seed)
        postal_pool = [c["postal"](pr) for _ in range(12)]
        return dict(house=str(hn), street_word=sw, street_type=st_full, unit=unit, city=city[0],
                    region=city[1], postal=str(r.choice(postal_pool)), country=cc)

    # ---------- rendering ----------
    @staticmethod
    def street_line(a, abbrev=False, unit_style=0):
        cc = a["country"]
        st = a["street_type"]
        if abbrev:
            st = dict(COUNTRIES[cc]["street_types"]).get(st, st)
        fmt = COUNTRIES[cc]["fmt"]
        if fmt == "us":
            s = f"{a['house']} {a['street_word']} {st}"
            if a["unit"]:
                num = a["unit"].split()[-1]
                s += [f", {a['unit']}", f" Ste {num}", f" #{num}"][unit_style % 3]
        elif fmt == "de":
            s = f"{a['street_word']}{st.lower() if st in ('Straße', 'Str.') else ' ' + st} {a['house']}"
        elif fmt == "fr":
            s = f"{a['house']} {st} de {a['street_word']}"
        elif fmt == "br":
            s = f"{st} {a['street_word']}, {a['house']}"
        elif fmt == "in":
            s = f"{a['house']} {a['street_word']} {st}"
        else:  # jp
            s = f"{a['house']}-{(int(a['house']) % 7) + 1} {a['street_word']} {st}"
        return s

    # ---------- noise ----------
    def noisy_name(self, name, level):
        r = self.r
        if name is None or r.random() < 0.03:
            return None
        s = name
        if r.random() < 0.03:  # acronym: lexically unrecoverable (non-lexical miss by design)
            toks = [t for t in name.split() if t[0].isupper()]
            return "".join(t[0] for t in toks[:-1]) + " " + toks[-1]
        if r.random() < 0.3 * level:
            for k, v in ABBREV.items():
                if k in s and r.random() < 0.7:
                    s = s.replace(k, v)
        toks = s.split()
        if len(toks) > 2 and r.random() < 0.35 * level:  # drop legal suffix / generic
            toks = toks[:-1]
        if len(toks) > 2 and r.random() < 0.10 * level:  # drop a random non-first token
            del toks[r.integers(1, len(toks))]
        if len(toks) > 2 and r.random() < 0.12 * level:  # reorder
            i = r.integers(0, len(toks) - 1)
            toks[i], toks[i + 1] = toks[i + 1], toks[i]
        s = " ".join(toks)
        if r.random() < 0.35 * level:
            s = self.typo(s)
        if r.random() < 0.15 * level:
            s = self.typo(s)
        if r.random() < 0.25:
            s = _fold(s)
        u = r.random()
        if u < 0.25:
            s = s.upper()
        elif u < 0.35:
            s = s.lower()
        if r.random() < 0.2:
            s = s.replace(" ", "  ", 1) + ("." if r.random() < 0.5 else ",")
        return s

    def typo(self, s):
        r = self.r
        if len(s) < 4:
            return s
        i = int(r.integers(1, len(s) - 1))
        op = r.integers(4)
        if op == 0:
            return s[:i] + s[i + 1:]
        if op == 1:
            return s[:i] + s[i + 1] + s[i] + s[i + 2:]
        if op == 2:
            return s[:i] + str(r.choice(list("aeiourstnl"))) + s[i:]
        return s[:i] + str(r.choice(list("aeiourstnl"))) + s[i + 1:]

    def noisy_address(self, a, level):
        r = self.r
        a = dict(a)
        if r.random() < 0.08 * level:  # house-number noise
            a["house"] = str(max(1, int(a["house"]) + int(r.choice([-2, -1, 1, 2, 10]))))
        if r.random() < 0.3:
            a["unit"] = ""
        return a

    # ---------- records (competition schema: entity_id, business_name, business_address, country) ----------
    def render_address(self, a, level, clean=False):
        """Single free-text business_address: the only address field in the competition schema."""
        r = self.r
        cc = a["country"]
        street = self.street_line(a, abbrev=(not clean and r.random() < 0.5), unit_style=int(r.integers(3)))
        parts = [street]
        if cc == "IN" and r.random() < (0.1 if clean else 0.3):  # landmark-based reference
            parts.insert(int(r.integers(0, 2)), str(r.choice(LANDMARKS)))
        city = a["city"] if (clean or r.random() >= 0.3) else _fold(a["city"])
        keep_city = clean or r.random() < 0.85
        pc = a["postal"] if (r.random() < (0.95 if clean else 0.75)) else None
        if cc == "US":
            parts += ([city] if keep_city else []) + [f"{a['region']} {pc}" if pc else a["region"]]
        elif cc == "IN":
            parts += ([f"{city} - {pc}" if pc else city] if keep_city else ([pc] if pc else []))
            if clean or r.random() < 0.5:
                parts.append(a["region"])
        elif cc == "FR":
            parts += [f"{pc} {city}" if pc else city] if keep_city else ([pc] if pc else [])
        else:
            parts += ([city] if keep_city else []) + ([pc] if pc else [])
        s = ", ".join(p for p in parts if p)
        if not clean:
            if r.random() < 0.15 * level:
                s = self.typo(s)
            if r.random() < 0.05 * level:  # component reordering
                p = s.split(", ")
                r.shuffle(p)
                s = ", ".join(p)
            if r.random() < 0.2:
                s = s.upper()
        return s

    def render_record(self, rid, name, a, level, missing_addr=False):
        return dict(entity_id=rid, business_name=name or "",
                    business_address="" if missing_addr else self.render_address(a, level),
                    country=LABEL[a["country"]])

    # ---------- main ----------
    def run(self, country_p):
        r = self.r
        s1, s2, s3, gt = [], [], [], {}
        ctr = {"S2": 0, "S3": 0}

        def emit(eid, name, addr, level):
            src = "S2" if r.random() < 0.55 else "S3"
            ctr[src] += 1
            rec_id = f"{src}-{ctr[src]:06d}"
            rec = self.render_record(rec_id, self.noisy_name(name, level), self.noisy_address(addr, level), level,
                                     r.random() < 0.10)
            (s2 if src == "S2" else s3).append(rec)
            if eid is not None:
                gt[eid].append(rec_id)
            return rec_id

        ccs = list(country_p)
        pcs = np.array([country_p[c] for c in ccs])
        chains = []
        for _ in range(max(1, self.n // 150)):
            cc = str(r.choice(ccs, p=pcs))
            chains.append((cc, f"{r.choice(COMMON_WORDS)} {self.new_brand()}", str(r.choice(INDUSTRY))))
        explosion = ("US", "Star", "Coffee")  # candidate-explosion chain
        buildings = []  # shared addresses (same address, different name)

        for i in range(self.n):
            eid = f"S1-{i + 1:06d}"
            gt[eid] = []
            cc = str(r.choice(ccs, p=pcs))
            kind = r.random()
            if kind < 0.12:  # chain branch
                cc, core, ind = chains[r.integers(len(chains))]
                name = f"{core} {ind} {r.choice(COUNTRIES[cc]['legal'])}"
            elif kind < 0.15 and "US" in country_p:
                cc, core, ind = explosion
                name = f"{core} {ind} {r.choice(COUNTRIES[cc]['legal'])}"
            else:
                name, core = self.make_name(cc)
            if buildings and r.random() < 0.06 and buildings[-1]["country"] == cc:
                addr = dict(buildings[-1])
                addr["unit"] = f"Suite {r.integers(1, 40) * 10}" if cc == "US" else ""
            else:
                addr = self.make_address(cc)
                if r.random() < 0.05:
                    buildings.append(addr)
            s1.append(dict(entity_id=eid, business_name=name,
                           business_address="" if r.random() < 0.03 else self.render_address(addr, 0.0, clean=True),
                           country=LABEL[cc]))
            u = r.random()
            n_match = 0 if u < 0.35 else (1 if u < 0.80 else int(r.choice([2, 2, 3, 4])))
            level = float(r.choice([0.5, 1.2, 2.0], p=[0.3, 0.45, 0.25]))
            for _ in range(n_match):
                emit(eid, name, addr, level)
            # distractors (records of other, unlisted businesses)
            if r.random() < 0.45:
                v = r.random()
                city = (addr["city"], addr["region"])
                if v < 0.35:    # same core brand, different industry, same city
                    emit(None, self.make_name(cc, core=core)[0], self.make_address(cc, city=city), 0.5)
                elif v < 0.6:   # pluralised / near-identical name, same city
                    emit(None, name.replace(core, core + "s") if core in name else name,
                         self.make_address(cc, city=city), 0.5)
                elif v < 0.8:   # identical-ish name, same street, different house number
                    a2 = dict(addr)
                    a2["house"] = str(int(addr["house"]) + int(r.integers(1, 40)))
                    emit(None, self.make_name(cc, core=core)[0] if r.random() < 0.5 else name, a2, 0.5)
                else:           # same name, different city
                    emit(None, name, self.make_address(cc), 0.8)
            if r.random() < 0.08:  # same address, different name
                emit(None, self.make_name(cc)[0], addr, 0.5)
            if kind < 0.15 and r.random() < 0.6:  # unlisted branch of the same chain
                emit(None, name, self.make_address(cc), 0.5)
        for _ in range(int(self.n * 0.4)):  # pure orphan records
            cc = str(r.choice(ccs, p=pcs))
            emit(None, self.make_name(cc)[0], self.make_address(cc), 0.5)
        gdf = pd.DataFrame({"source1_entity_id": list(gt), "matched_entity_ids": [",".join(v) for v in gt.values()]})
        return pd.DataFrame(s1), pd.DataFrame(s2), pd.DataFrame(s3), gdf


def _write_tsv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for c in df.columns:
        assert not df[c].astype(str).str.contains("[\t\n\"]", regex=True).any(), f"unsafe char in {path}:{c}"
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE, lineterminator="\n", encoding="utf-8")


def generate(n_entities: int = 6000, seed: int = 7, out_dir: str = "data/synthetic", n_test: int | None = None):
    """Writes the OFFICIAL competition layout under out_dir:
      dataset/train/train_source{1,2,3}.tsv, dataset/train/train_ground_truth.tsv   (countries: US, India)
      dataset/test/test_source{1,2,3}.tsv                                          (adds France)
      _hidden/test_ground_truth.tsv   <- synthetic-only; used solely to score output mechanics
    """
    n_test = n_test or max(10, n_entities // 3)
    tr = Gen(n_entities, seed).run(TRAIN_COUNTRY_P)
    te = Gen(n_test, seed + 1000).run(TEST_COUNTRY_P)
    for split, (s1, s2, s3, gt) in (("train", tr), ("test", te)):
        d = os.path.join(out_dir, "dataset", split)
        _write_tsv(s1, os.path.join(d, f"{split}_source1.tsv"))
        _write_tsv(s2, os.path.join(d, f"{split}_source2.tsv"))
        _write_tsv(s3, os.path.join(d, f"{split}_source3.tsv"))
        _write_tsv(gt, os.path.join(d, "train_ground_truth.tsv") if split == "train"
                   else os.path.join(out_dir, "_hidden", "test_ground_truth.tsv"))
    meta = dict(label="SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE", seed=seed,
                train=dict(n_s1=len(tr[0]), n_s2=len(tr[1]), n_s3=len(tr[2])),
                test=dict(n_s1=len(te[0]), n_s2=len(te[1]), n_s3=len(te[2])))
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/synthetic")
    a = ap.parse_args()
    print(generate(a.n, a.seed, a.out))


def generate_test_tree(n_test: int, seed: int, out_dir: str):
    """A fresh, independent synthetic TEST tree (with hidden labels) for final holdout reporting.
    SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE."""
    s1, s2, s3, gt = Gen(n_test, seed).run(TEST_COUNTRY_P)
    d = os.path.join(out_dir, "dataset", "test")
    _write_tsv(s1, os.path.join(d, "test_source1.tsv"))
    _write_tsv(s2, os.path.join(d, "test_source2.tsv"))
    _write_tsv(s3, os.path.join(d, "test_source3.tsv"))
    _write_tsv(gt, os.path.join(out_dir, "_hidden", "test_ground_truth.tsv"))
    return dict(n_s1=len(s1), n_s2=len(s2), n_s3=len(s3))
