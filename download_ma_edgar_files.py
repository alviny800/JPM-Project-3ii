#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_ma_edgar_files.py

Batch-download SEC EDGAR filings for a Bloomberg/MA-exported merger universe.

Designed for corporate-action election / proration research:
- Reads a Bloomberg M&A export CSV.
- Filters candidate deals by Payment Type / Deal Status / date.
- Maps Target Name and Acquirer Name to SEC CIKs using SEC company_tickers.json.
- Pulls filings from SEC submissions API.
- Filters around each announcement date and form type.
- Downloads primary documents and optionally filing exhibits from SEC Archives.
- Scores each downloaded document for election/proration keywords.
- Writes a manifest CSV and unresolved-name diagnostics.

Important:
SEC requires an identifying User-Agent header. Pass a real name/email:
    --user-agent "Your Name your.email@domain.edu"

Example:
    python download_ma_edgar_files.py \
        --input ma_export_33248147_212700.csv \
        --output-dir ma_edgar_docs \
        --user-agent "Xiangyu Wang xiangyuwang@berkeley.edu" \
        --payment-types "Cash or Stock" "Cash and Stock" \
        --deal-status Completed \
        --pre-days 60 \
        --post-days 730 \
        --download-exhibits

No SEC API key is required for this official-SEC version.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import itertools
import json
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, quote

import pandas as pd
import requests

from event_csv_adapter import normalize_event_input

try:
    from rapidfuzz import fuzz, process as rf_process  # type: ignore
    HAS_RAPIDFUZZ = True
except Exception:
    import difflib
    HAS_RAPIDFUZZ = False


SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/"
SEC_ARCHIVES_INDEX = SEC_ARCHIVES_BASE + "index.json"

DEFAULT_FORMS = [
    # Core merger/election documents
    "S-4", "S-4/A", "F-4", "F-4/A",
    "424B3", "424B4", "424B5", "424B2", "424B7",
    "DEFM14A", "PREM14A", "DEFA14A",
    "8-K", "8-K/A",
    "425",
    "SC TO-T", "SC TO-T/A",
    "SC TO-I", "SC TO-I/A",
    "SC 14D9", "SC 14D9/A",
    "SC 13E3", "SC 13E3/A",
]

DEFAULT_KEYWORDS = [
    "cash election",
    "stock election",
    "mixed election",
    "election deadline",
    "election form",
    "letter of transmittal",
    "form of election",
    "election procedures",
    "proration",
    "proration factor",
    "proration procedures",
    "non-election",
    "non-electing",
    "oversubscribed",
    "oversubscription",
    "allocation procedures",
    "exchange ratio",
    "aggregate cash consideration",
    "aggregate stock consideration",
    "cash consideration",
    "stock consideration",
    "merger consideration",
    "final proration",
    "preliminary proration",
    "election results",
]

# Field-level locator specs.  These drive the research workflow: the script no longer
# treats filings as useful just because they are merger filings; it tries to locate
# the specific economic state variables needed for the proration model.
DEFAULT_FIELD_SPECS: Dict[str, Dict[str, Any]] = {
    "cash_consideration": {
        "description": "Cash amount or cash election consideration per target share.",
        "timing_bucket": "pre_election_trade_entry",
        "preferred_forms": ["S-4", "S-4/A", "F-4", "F-4/A", "424B3", "424B5", "DEFM14A", "PREM14A"],
        "keywords": [
            "cash consideration", "cash election", "per share in cash", "cash amount",
            "cash component", "aggregate cash consideration", "cash merger consideration"
        ],
    },
    "stock_consideration": {
        "description": "Stock election consideration or acquirer-share component per target share.",
        "timing_bucket": "pre_election_trade_entry",
        "preferred_forms": ["S-4", "S-4/A", "F-4", "F-4/A", "424B3", "424B5", "DEFM14A", "PREM14A"],
        "keywords": [
            "stock consideration", "stock election", "share consideration", "stock component",
            "aggregate stock consideration", "stock merger consideration", "common stock"
        ],
    },
    "exchange_ratio": {
        "description": "Exchange ratio used to convert target shares into acquirer shares.",
        "timing_bucket": "pre_election_trade_entry",
        "preferred_forms": ["S-4", "S-4/A", "F-4", "F-4/A", "424B3", "424B5", "DEFM14A", "PREM14A"],
        "keywords": [
            "exchange ratio", "exchange rate", "fixed exchange ratio", "floating exchange ratio",
            "shares of common stock for each share", "shares for each share"
        ],
    },
    "cash_stock_caps": {
        "description": "Aggregate cash/stock caps, maximum cash election amount, maximum stock election amount, and allocation limits.",
        "timing_bucket": "pre_election_trade_entry",
        "preferred_forms": ["S-4", "S-4/A", "F-4", "F-4/A", "424B3", "424B5", "DEFM14A", "PREM14A", "EX-99"],
        "keywords": [
            "cash cap", "stock cap", "aggregate cash consideration", "aggregate stock consideration",
            "maximum cash", "maximum stock", "cash election number", "stock election number",
            "cash fraction", "stock fraction", "allocation procedures", "oversubscribed", "undersubscribed"
        ],
    },
    "proration_formula": {
        "description": "Proration mechanics/formula when cash or stock elections exceed the cap.",
        "timing_bucket": "pre_election_trade_entry",
        "preferred_forms": ["S-4", "S-4/A", "F-4", "F-4/A", "424B3", "424B5", "DEFM14A", "PREM14A", "EX-99"],
        "keywords": [
            "proration", "proration procedures", "proration factor", "prorated", "pro rata",
            "oversubscribed", "oversubscription", "allocation procedures", "election procedures",
            "cash elections are oversubscribed", "stock elections are oversubscribed"
        ],
    },
    "non_election_treatment": {
        "description": "Default treatment for holders who fail to make a valid election.",
        "timing_bucket": "pre_election_trade_entry",
        "preferred_forms": ["DEFM14A", "PREM14A", "S-4", "S-4/A", "F-4", "F-4/A", "424B3", "EX-99"],
        "keywords": [
            "non-election", "non-electing", "non electing", "non-election shares", "non-electing shares",
            "deemed to have elected", "no election", "fail to make an election", "failed to make an election",
            "valid election", "election form"
        ],
    },
    "election_deadline": {
        "description": "Deadline/expiration time for making a cash, stock, or mixed election.",
        "timing_bucket": "pre_election_trade_entry",
        "preferred_forms": ["DEFM14A", "PREM14A", "S-4", "S-4/A", "F-4", "F-4/A", "424B3", "EX-99"],
        "keywords": [
            "election deadline", "deadline for making", "election form", "election procedures",
            "expiration date", "expiration time", "properly completed", "notice of guaranteed delivery",
            "letter of transmittal", "form of election"
        ],
    },
    "record_date": {
        "description": "Record date for voting/election eligibility, if stated.",
        "timing_bucket": "pre_election_trade_entry",
        "preferred_forms": ["DEFM14A", "PREM14A", "S-4", "S-4/A", "F-4", "F-4/A", "424B3"],
        "keywords": ["record date", "close of business on", "holders of record", "stockholders of record"],
    },
    "odd_lot_priority": {
        "description": "Odd-lot priority or special treatment for small holders, if present.",
        "timing_bucket": "pre_election_trade_entry",
        "preferred_forms": ["S-4", "S-4/A", "F-4", "F-4/A", "424B3", "DEFM14A", "EX-99"],
        "keywords": ["odd lot", "odd-lot", "less than 100 shares", "priority", "small holder"],
    },
    "guaranteed_delivery_window": {
        "description": "Guaranteed-delivery mechanics and timing window for election forms or tender documents.",
        "timing_bucket": "pre_election_trade_entry",
        "preferred_forms": ["SC TO-T", "SC TO-T/A", "SC 14D9", "SC 14D9/A", "S-4", "424B3", "EX-99"],
        "keywords": ["guaranteed delivery", "notice of guaranteed delivery", "letter of transmittal", "election form"],
    },
    "preliminary_proration": {
        "description": "Preliminary election result or preliminary proration factor announced after election deadline.",
        "timing_bucket": "post_election_label",
        "preferred_forms": ["8-K", "8-K/A", "425", "EX-99"],
        "keywords": [
            "preliminary proration", "preliminary election results", "preliminary results",
            "preliminary proration factor", "election results", "proration factor"
        ],
    },
    "final_proration": {
        "description": "Final election result / final proration factor, the realized fill-rate label.",
        "timing_bucket": "post_election_label",
        "preferred_forms": ["8-K", "8-K/A", "425", "EX-99"],
        "keywords": [
            "final proration", "final election results", "final results", "final proration factor",
            "election results", "proration factor", "proration rate"
        ],
    },
    "deal_completion_or_break": {
        "description": "Whether and when the merger was completed or terminated.",
        "timing_bucket": "post_election_label",
        "preferred_forms": ["8-K", "8-K/A", "425", "DEFA14A", "EX-99"],
        "keywords": [
            "completed the merger", "completion of the merger", "effective time", "closed the transaction",
            "terminated the merger agreement", "termination of the merger agreement"
        ],
    },
}

FIELD_EXTRACTION_SCHEMA = {
    "event_id": "string",
    "target_name": "string",
    "acquirer_name": "string",
    "fields": {
        "<field_name>": {
            "value": "directly extracted value/summary, or null",
            "basis": "direct/inferred/not_found",
            "timing_bucket": "pre_election_trade_entry/post_election_label/external_model_input",
            "source_doc_ids": ["doc_id"],
            "source_form_types": ["DEFM14A"],
            "source_filing_dates": ["YYYY-MM-DD"],
            "evidence_quotes": ["short exact quotes"],
            "confidence": "high/medium/low",
            "notes": "ambiguities, conflicts, or missing information"
        }
    },
    "recommended_follow_up_documents": [
        {"doc_id": "doc_id", "reason": "why it should be manually reviewed or uploaded"}
    ],
    "timing_notes": "State clearly which fields were available before the election deadline and which are realized labels.",
}

FIELD_LLM_SYSTEM_PROMPT = """You extract field-level data from SEC merger filings for a corporate-action election/proration trading dataset.

The goal is not to summarize documents. The goal is to fill the requested fields with values and evidence.
Rules:
- Return one valid JSON object only. No markdown.
- For every requested field, provide value, basis, timing_bucket, source_doc_ids, source_form_types, source_filing_dates, evidence_quotes, confidence, and notes.
- Use null and basis='not_found' when a field is not found in the supplied evidence. Never omit a requested field.
- Do not invent dates, ratios, caps, proration factors, deadlines, or default rules.
- Distinguish pre-election trade-entry fields from post-election realized-label fields.
- Prefer final prospectus/proxy/election-form documents for trade-entry mechanics. Prefer post-deadline 8-K/press release exhibits for realized proration labels.
- Realized election-demand labels (realized_cash_election_demand, realized_stock_election_demand, preliminary_proration_results, final_proration_results) are frequently reported NOT as a clean percentage but as: (a) raw share counts electing each option, (b) an aggregate dollar amount of cash paid, or (c) a proration/allocation factor or an "oversubscribed"/"undersubscribed" statement. Capture whichever form is disclosed — do NOT leave the field null when the underlying counts, dollar amounts, or factor are present in the evidence.
- When the evidence also supplies a base (shares outstanding, total shares electing, or shares deemed outstanding for the election), DERIVE the percentage from the raw counts, put the resulting % in value, show the arithmetic in notes, and set basis='derived'. If only the raw figure is available, report it with basis='direct'.
- CRITICAL — election DEMAND vs. post-proration OUTCOME. `realized_cash_election_demand` / `realized_stock_election_demand` are the percentage of shares whose holders CHOSE / ELECTED that option at the deadline, BEFORE proration — this is DEMAND. Do NOT place a post-proration allocation there: the percentage of shares that RECEIVED, were CONVERTED to, or were ALLOCATED an option AFTER proration is a different quantity and belongs in `final_proration_results` (with the proration/fill factor). Cues: 'elected' / 'made a cash (stock) election' / 'shares electing' = demand; 'converted into' / 'received the consideration' / 'allocated after proration' = outcome. When a filing reports both (e.g. "~96% elected stock; after the 50/50 proration ~47.9% were converted to cash"), put 96% in `realized_stock_election_demand` and the 47.9% / proration factor in `final_proration_results` — never swap them, and never report the same number for both meanings.
- A "completion"/"effective time" 8-K that merely states the merger closed and restates the consideration mechanics is NOT a results filing. Look elsewhere in the supplied evidence for the election-results / proration press release before concluding a realized field is absent. Only if no election breakdown is disclosed anywhere in the evidence, set the realized-demand fields to null with basis='not_found' and note that results were not disclosed.
"""

LLM_EXTRACTION_SCHEMA = {
    "event_id": "string",
    "target_name": "string",
    "acquirer_name": "string",
    "announce_date": "YYYY-MM-DD or null",
    "deal_currency_mix": {
        "cash_component": "direct quote or concise description, or null",
        "stock_component": "direct quote or concise description, or null",
        "mixed_election_allowed": "yes/no/unclear",
        "source_doc_ids": ["doc_id"],
    },
    "election_mechanics": {
        "cash_election_available": "yes/no/unclear",
        "stock_election_available": "yes/no/unclear",
        "election_deadline": {
            "value": "date/time as stated, or null",
            "estimated": False,
            "basis": "direct/estimated/not_found",
            "source_doc_ids": ["doc_id"],
            "evidence": "short quote or null",
        },
        "non_election_treatment": {
            "value": "summary or null",
            "estimated": False,
            "basis": "direct/estimated/not_found",
            "source_doc_ids": ["doc_id"],
            "evidence": "short quote or null",
        },
    },
    "proration": {
        "proration_applicable": "yes/no/unclear",
        "proration_formula_or_limits": {
            "value": "summary or null",
            "estimated": False,
            "basis": "direct/estimated/not_found",
            "source_doc_ids": ["doc_id"],
            "evidence": "short quote or null",
        },
        "preliminary_proration": {
            "value": "summary/numeric value or null",
            "estimated": False,
            "basis": "direct/estimated/not_found",
            "source_doc_ids": ["doc_id"],
            "evidence": "short quote or null",
        },
        "final_proration": {
            "value": "summary/numeric value or null",
            "estimated": False,
            "basis": "direct/estimated/not_found",
            "source_doc_ids": ["doc_id"],
            "evidence": "short quote or null",
        },
    },
    "important_dates": [
        {
            "label": "election_deadline/effective_time/expiration/closing/other",
            "value": "date/time or null",
            "estimated": False,
            "basis": "direct/estimated",
            "source_doc_ids": ["doc_id"],
            "evidence": "short quote",
        }
    ],
    "recommended_manual_review": [
        {
            "doc_id": "doc_id",
            "reason": "why this document matters",
        }
    ],
    "confidence": {
        "overall": "high/medium/low",
        "notes": "short explanation of gaps or ambiguity",
    },
}

LLM_SYSTEM_PROMPT = """You extract structured data from SEC merger filings for corporate-action election and proration research.

Rules:
- Return one valid JSON object only. No markdown.
- Separate directly stated facts from estimates. Use estimated=true only when inferring from context.
- Preserve short evidence quotes for each extracted value when available.
- Prefer final/prospectus/proxy/election result documents over early 8-K press releases when they conflict.
- If a field is not found, use null or "unclear" and explain the gap in confidence.notes.
- Do not invent dates, ratios, caps, or proration factors.
"""

# File extensions worth downloading from filing folders.
# Exhibits in SEC filing folders are often .htm/.html/.txt. PDFs occasionally appear for election forms.
DOWNLOADABLE_EXTS = {".htm", ".html", ".txt", ".pdf"}
EXCLUDED_FILE_PATTERNS = [
    r"\.xml$", r"\.xsd$", r"\.jpg$", r"\.jpeg$", r"\.png$", r"\.gif$",
    r"\.css$", r"\.js$", r"\.zip$", r"\.xlsx?$", r"FilingSummary\.xml$",
    r"R[0-9]+\.htm$",  # XBRL report fragments
]


LEGAL_SUFFIXES = [
    "incorporated", "inc", "corp", "corporation", "co", "company", "ltd", "limited",
    "plc", "lp", "llp", "llc", "holdings", "holding", "group", "the", "sa", "nv",
    "ag", "spa", "sarl", "bv", "gmbh", "pte", "partners", "trust", "fund",
    "technologies", "technology", "systems", "international",
]


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class SecClient:
    def __init__(
        self,
        user_agent: str,
        sleep_seconds: float = 0.13,
        retries: int = 4,
        timeout: int = 30,
        cache_dir: Optional[Path] = None,
    ) -> None:
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC requires an identifying User-Agent. "
                "Use e.g. --user-agent 'Your Name your.email@domain.edu'"
            )
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
        })
        self.data_session = requests.Session()
        self.data_session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        })
        self.sleep_seconds = sleep_seconds
        self.retries = retries
        self.timeout = timeout
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, url: str) -> Path:
        assert self.cache_dir is not None
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.json"

    def get_json(self, url: str, host: str = "www") -> Dict[str, Any]:
        cache_path = self._cache_key(url) if self.cache_dir else None
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        sess = self.data_session if "data.sec.gov" in url else self.session
        last_err: Optional[Exception] = None
        for attempt in range(self.retries):
            time.sleep(self.sleep_seconds + random.uniform(0, self.sleep_seconds / 2))
            try:
                r = sess.get(url, timeout=self.timeout)
                if r.status_code in {429, 403, 503}:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    print(f"[SEC throttle] {r.status_code} for {url}; sleeping {wait:.1f}s", file=sys.stderr)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                data = r.json()
                if cache_path:
                    cache_path.write_text(json.dumps(data), encoding="utf-8")
                return data
            except Exception as e:
                last_err = e
                time.sleep((2 ** attempt) + random.uniform(0, 1))
        raise RuntimeError(f"Failed JSON GET after retries: {url}; last error={last_err}")

    def fetch_file(self, url: str) -> Tuple[bool, Optional[str], bytes]:
        """Return (ok, error, content)."""
        sess = self.session
        last_err: Optional[Exception] = None
        for attempt in range(self.retries):
            time.sleep(self.sleep_seconds + random.uniform(0, self.sleep_seconds / 2))
            try:
                r = sess.get(url, timeout=self.timeout)
                if r.status_code in {429, 403, 503}:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    print(f"[SEC throttle] {r.status_code} for {url}; sleeping {wait:.1f}s", file=sys.stderr)
                    time.sleep(wait)
                    continue
                if r.status_code == 404:
                    return False, "404", b""
                r.raise_for_status()
                return True, None, r.content
            except Exception as e:
                last_err = e
                time.sleep((2 ** attempt) + random.uniform(0, 1))
        return False, str(last_err), b""

    def download_file(self, url: str, dest: Path, resume: bool = True) -> Tuple[bool, Optional[str], int]:
        """Return (downloaded_or_exists, error, bytes)."""
        if resume and dest.exists() and dest.stat().st_size > 0:
            return True, None, dest.stat().st_size

        dest.parent.mkdir(parents=True, exist_ok=True)
        ok, err, content = self.fetch_file(url)
        if not ok:
            return False, err, 0
        dest.write_bytes(content)
        return True, None, len(content)


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = html.unescape(name).lower()
    name = re.sub(r"\([^)]*\)", " ", name)
    name = name.replace("&", " and ")
    name = re.sub(r"[^a-z0-9 ]+", " ", name)
    parts = [p for p in name.split() if p not in LEGAL_SUFFIXES]
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def safe_filename(s: Any, max_len: int = 90) -> str:
    s = str(s) if s is not None else ""
    s = html.unescape(s)
    s = re.sub(r"[^A-Za-z0-9._ -]+", "_", s)
    s = re.sub(r"\s+", "_", s).strip("._- ")
    return s[:max_len] or "NA"


def parse_date(x: Any) -> Optional[pd.Timestamp]:
    if pd.isna(x):
        return None
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return None


def make_event_id(event_idx: int, announce_date: Any, target_name: Any, acquirer_name: Any) -> str:
    """Canonical event id used across manifest, coverage, Claude payloads, and packages."""
    ann = parse_date(announce_date)
    ann_token = ann.strftime("%Y%m%d") if ann is not None and not pd.isna(ann) else "nodate"
    return (
        f"E{int(event_idx):06d}_{ann_token}_"
        f"{safe_filename(str(target_name), 45)}__{safe_filename(str(acquirer_name), 45)}"
    )


def load_company_tickers(client: SecClient, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        # company_tickers is on www.sec.gov; use manual request because SEC serves this under /files.
        r = client.session.get(SEC_COMPANY_TICKERS_URL, timeout=30)
        r.raise_for_status()
        data = r.json()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data), encoding="utf-8")

    rows = []
    for _, item in data.items():
        cik_int = int(item["cik_str"])
        rows.append({
            "cik_int": cik_int,
            "cik10": f"{cik_int:010d}",
            "ticker": item.get("ticker", ""),
            "title": item.get("title", ""),
            "norm_title": normalize_name(item.get("title", "")),
        })
    return pd.DataFrame(rows)


def fuzzy_match_company(name: str, cik_df: pd.DataFrame, min_score: int = 84) -> Dict[str, Any]:
    norm = normalize_name(name)
    if not norm:
        return {"query_name": name, "query_norm": norm, "matched": False, "score": 0}

    # Exact-ish substring first.
    exact = cik_df[cik_df["norm_title"].eq(norm)]
    if len(exact) > 0:
        row = exact.iloc[0]
        return {
            "query_name": name, "query_norm": norm, "matched": True, "score": 100,
            "cik_int": int(row["cik_int"]), "cik10": row["cik10"], "ticker": row["ticker"],
            "sec_title": row["title"], "match_method": "exact_norm",
        }

    choices = cik_df["norm_title"].fillna("").tolist()
    if HAS_RAPIDFUZZ:
        # token_set_ratio handles legal suffixes and word order well.
        match = rf_process.extractOne(norm, choices, scorer=fuzz.token_set_ratio)
        if not match:
            return {"query_name": name, "query_norm": norm, "matched": False, "score": 0}
        match_norm, score, idx = match
    else:
        # Fallback: slower and less robust, but no external dependency.
        scores = [(difflib.SequenceMatcher(None, norm, c).ratio() * 100, i, c) for i, c in enumerate(choices)]
        score, idx, match_norm = max(scores, key=lambda z: z[0])

    row = cik_df.iloc[int(idx)]
    return {
        "query_name": name,
        "query_norm": norm,
        "matched": bool(score >= min_score),
        "score": float(score),
        "cik_int": int(row["cik_int"]),
        "cik10": row["cik10"],
        "ticker": row["ticker"],
        "sec_title": row["title"],
        "sec_norm": row["norm_title"],
        "match_method": "rapidfuzz_token_set" if HAS_RAPIDFUZZ else "difflib",
    }


SEC_EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
_EFTS_CACHE: Dict[str, Dict[str, Any]] = {}

# Hand-verified name -> CIK map for the recoverable resolver tail (delisted micro-caps
# and renamed targets that efts/company_tickers miss or match to the wrong entity).
# Keyed by normalize_name(target_name). Populated by load_cik_overrides().
_CIK_OVERRIDES: Dict[str, Dict[str, Any]] = {}


def load_cik_overrides(path: Path) -> int:
    """Load a hand-verified target-name -> CIK override table (see cik_manual_overrides.csv).

    Each override is trusted absolutely (match_method='manual_override', score=100) and
    consulted BEFORE efts/company_tickers, so a verified CIK can't be re-broken by fuzzy
    matching. Returns the number of overrides loaded."""
    if not path or not Path(path).exists():
        return 0
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for _, r in df.iterrows():
        name = str(r.get("target_name", "")).strip()
        cik10 = str(r.get("cik10", "")).strip()
        if not name or not cik10:
            continue
        cik10 = cik10.zfill(10)
        _CIK_OVERRIDES[normalize_name(name)] = {
            "query_name": name, "query_norm": normalize_name(name),
            "matched": True, "score": 100.0,
            "cik_int": int(cik10), "cik10": cik10,
            "ticker": "", "sec_title": str(r.get("resolved_name", "")),
            "match_method": "manual_override",
        }
    return len(_CIK_OVERRIDES)


def override_match(name: str) -> Optional[Dict[str, Any]]:
    """Return a copy of the manual override for `name`, or None."""
    hit = _CIK_OVERRIDES.get(normalize_name(name))
    return dict(hit) if hit else None


def resolve_cik_via_efts(client: SecClient, name: str, min_score: int = 84) -> Dict[str, Any]:
    """Resolve a company name to a SEC CIK via EDGAR full-text search (efts).

    company_tickers.json lists only CURRENT registrants, so it misses the ~96% of
    merger targets that are delisted post-acquisition. efts indexes filing text back to
    2001 (our universe is 2006+), so a delisted target's own proxies/8-Ks still resolve.
    We collect the filer entities of filings matching the name and pick the entity whose
    own name best matches the query (guards against ticker/name reuse and co-filers like
    the acquirer or institutional 13G holders)."""
    norm = normalize_name(name)
    base = {"query_name": name, "query_norm": norm, "matched": False, "score": 0, "match_method": "efts"}
    if not norm:
        return base
    if name in _EFTS_CACHE:
        return dict(_EFTS_CACHE[name])
    try:
        # NOTE: use a plain request with only the UA — do NOT reuse client.session, which
        # hardcodes Host: www.sec.gov and makes efts.sec.gov return a non-JSON error page.
        ua = client.session.headers.get("User-Agent", "")
        time.sleep(getattr(client, "sleep_seconds", 0.13))
        r = requests.get(
            SEC_EFTS_URL + '?q=%22' + quote(name) + '%22',
            headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"},
            timeout=30,
        )
        r.raise_for_status()
        hits = r.json().get("hits", {}).get("hits", [])
    except Exception:
        return base

    cand: Dict[str, Dict[str, Any]] = {}
    for h in hits:
        for dn in h.get("_source", {}).get("display_names", []):
            m = re.search(r"\(CIK (\d{10})\)", dn)
            if not m:
                continue
            cik10 = m.group(1)
            disp = dn.split("  (")[0].strip()
            tm = re.search(r"\(([A-Z][A-Z0-9.\-]{0,6})\)\s*\(CIK", dn)
            e = cand.setdefault(cik10, {"name": disp, "ticker": tm.group(1) if tm else "", "freq": 0})
            e["freq"] += 1
    if not cand:
        _EFTS_CACHE[name] = base
        return dict(base)

    best_cik, best_score = None, -1.0
    for cik10, e in cand.items():
        cand_norm = normalize_name(e["name"])
        if HAS_RAPIDFUZZ:
            s = float(fuzz.token_set_ratio(norm, cand_norm))
        else:
            s = difflib.SequenceMatcher(None, norm, cand_norm).ratio() * 100.0
        if s > best_score:
            best_cik, best_score = cik10, s
    e = cand[best_cik]
    out = {
        "query_name": name, "query_norm": norm,
        "matched": bool(best_score >= min_score), "score": float(round(best_score, 1)),
        "cik_int": int(best_cik), "cik10": best_cik,
        "ticker": e["ticker"], "sec_title": e["name"], "match_method": "efts",
    }
    _EFTS_CACHE[name] = out
    return dict(out)


def flatten_submissions(sub: Dict[str, Any], client: SecClient) -> pd.DataFrame:
    """
    Convert SEC submissions JSON into a dataframe.

    The top-level 'filings.recent' covers recent filings.
    Older filings can be paginated via filenames in 'filings.files'.
    """
    frames: List[pd.DataFrame] = []

    def recent_to_df(recent: Dict[str, List[Any]]) -> pd.DataFrame:
        if not recent:
            return pd.DataFrame()
        # All arrays are parallel.
        return pd.DataFrame(recent)

    recent = sub.get("filings", {}).get("recent", {})
    df_recent = recent_to_df(recent)
    if not df_recent.empty:
        frames.append(df_recent)

    # Older paginated files live under the same data.sec.gov/submissions path.
    files = sub.get("filings", {}).get("files", []) or []
    for f in files:
        name = f.get("name")
        if not name:
            continue
        url = "https://data.sec.gov/submissions/" + name
        try:
            data = client.get_json(url)
            df_old = recent_to_df(data)
            if not df_old.empty:
                frames.append(df_old)
        except Exception as e:
            print(f"[WARN] Could not fetch old submissions file {url}: {e}", file=sys.stderr)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    # Normalize column names expected from SEC.
    if "filingDate" in out.columns:
        out["filingDate"] = pd.to_datetime(out["filingDate"], errors="coerce")
    if "reportDate" in out.columns:
        out["reportDate"] = pd.to_datetime(out["reportDate"], errors="coerce")
    return out


def filing_urls(cik_int: int, accession: str, primary_doc: str) -> Dict[str, str]:
    acc_nodash = accession.replace("-", "")
    base = SEC_ARCHIVES_BASE.format(cik_int=int(cik_int), acc_nodash=acc_nodash)
    return {
        "base_url": base,
        "primary_url": urljoin(base, primary_doc),
        "complete_txt_url": urljoin(base, f"{accession}.txt"),
        "index_json_url": SEC_ARCHIVES_INDEX.format(cik_int=int(cik_int), acc_nodash=acc_nodash),
    }


def should_download_item(name: str) -> bool:
    lname = name.lower()
    for pat in EXCLUDED_FILE_PATTERNS:
        if re.search(pat, lname):
            return False
    return Path(lname).suffix in DOWNLOADABLE_EXTS


def strip_html_for_scoring(raw: bytes, max_chars: int = 4_000_000) -> str:
    # Lightweight enough for lots of filings; avoids requiring BeautifulSoup.
    text = raw[:max_chars].decode("utf-8", errors="ignore")
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def score_text(text: str, keywords: List[str]) -> Tuple[int, List[str], Dict[str, int]]:
    counts: Dict[str, int] = {}
    total = 0
    found: List[str] = []
    for kw in keywords:
        pattern = re.escape(kw.lower())
        c = len(re.findall(pattern, text))
        if c:
            counts[kw] = c
            found.append(kw)
            # Weight core keywords higher.
            weight = 4 if kw in {"cash election", "stock election", "proration", "election deadline", "non-election", "non-electing"} else 1
            total += c * weight
    return total, found, counts


def make_snippets(text: str, keywords: List[str], max_snips: int = 3, window: int = 260) -> str:
    snippets = []
    lowered = text.lower()
    for kw in keywords:
        idx = lowered.find(kw.lower())
        if idx >= 0:
            start = max(0, idx - window)
            end = min(len(text), idx + len(kw) + window)
            snippet = re.sub(r"\s+", " ", text[start:end]).strip()
            snippets.append(snippet)
            if len(snippets) >= max_snips:
                break
    return " ||| ".join(snippets)


def default_field_specs_path() -> Optional[Path]:
    """Return a nearby field_specs.json if present, otherwise None."""
    here = Path(__file__).resolve().parent / "field_specs.json"
    cwd = Path.cwd() / "field_specs.json"
    if here.exists():
        return here
    if cwd.exists():
        return cwd
    return None


def load_field_specs(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Load field locator specs from JSON, falling back to built-in defaults."""
    chosen: Optional[Path] = Path(path) if path else default_field_specs_path()
    if chosen and chosen.exists():
        data = json.loads(chosen.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"field specs JSON must be an object: {chosen}")
        # Accept either {field: spec} or {"fields": {field: spec}}.
        if "fields" in data and isinstance(data["fields"], dict):
            data = data["fields"]
        return data
    return DEFAULT_FIELD_SPECS


def normalize_form_for_match(form: str) -> str:
    form = str(form or "").upper().strip()
    # Exhibit rows often have the parent form in `form` while document_name carries EX-99.*.
    return form.replace("/A", "")


def preferred_form_score(form: str, document_name: str, preferred_forms: List[str]) -> int:
    form_u = str(form or "").upper().strip()
    form_base = normalize_form_for_match(form_u)
    doc_u = str(document_name or "").upper()
    preferred = {str(x).upper().strip() for x in preferred_forms}
    preferred_base = {normalize_form_for_match(x) for x in preferred}
    score = 0
    if form_u in preferred:
        score += 12
    elif form_base in preferred_base:
        score += 9
    # Exhibit document names are often ex99-1.htm, exhibit2_1.htm, ex-2.1.htm, etc.
    for pf in preferred:
        if pf.startswith("EX-"):
            ex_num = re.sub(r"[^0-9]", "", pf)
            if ex_num and re.search(rf"EX(?:HIBIT)?[-_ ]?{ex_num}", doc_u):
                score += 8
                break
    if any(x.startswith("EX-99") for x in preferred) and ("EX-99" in doc_u or re.search(r"EX[-_]?99", doc_u)):
        score += 8
    if any("LETTER OF TRANSMITTAL" in x for x in preferred) and "TRANSMITTAL" in doc_u:
        score += 8
    return score


# Strong "realized RESULTS" signal phrases — their presence means a document REPORTS
# election outcomes (share counts / proration / oversubscription), as opposed to merely
# describing the election mechanics ex-ante or announcing that the merger became effective.
# Used to lift a genuine election-results press release above a bare "completion" 8-K when
# both file at close and would otherwise tie on the form bonus alone (TFCF/NYSE/Andeavor).
RESULTS_SIGNAL_PHRASES = [
    "elected to receive cash", "elected to receive stock", "elected to receive the cash stock",
    "shares elected", "were prorated", "was prorated", "prorated at", "proration factor",
    "allocation factor", "oversubscribed", "undersubscribed", "election results",
    "results of the election", "final proration", "preliminary proration",
    "cash elections", "stock elections", "percent elected", "% elected",
    "cash election shares", "stock election shares", "shares electing",
]


def score_field_text(
    text: str,
    form: str,
    document_name: str,
    spec: Dict[str, Any],
) -> Tuple[int, List[str], Dict[str, int], str, int]:
    """Return field_score, keywords_found, counts, snippet, form_bonus."""
    keywords = [str(k) for k in spec.get("keywords", []) if str(k).strip()]
    counts: Dict[str, int] = {}
    hits: List[str] = []
    keyword_score = 0
    lowered = text.lower()
    for kw in keywords:
        # Treat whitespace/hyphen variants flexibly, but keep exact phrase intent.
        parts = [re.escape(p) for p in kw.lower().replace("-", " ").split()]
        if not parts:
            continue
        pattern = r"\b" + r"[\s\-]+".join(parts) + r"\b"
        c = len(re.findall(pattern, lowered))
        if c:
            counts[kw] = c
            hits.append(kw)
            # Multi-word exact economic terms get heavier weight.
            keyword_score += c * (4 if len(kw.split()) >= 2 else 1)
    form_bonus = preferred_form_score(form, document_name, list(spec.get("preferred_forms", [])))
    # Realized-results ("post_election_label") fields live in announcement forms (8-K/
    # EX-99/425) that present results as numeric tables with SPARSE keyword matches — a
    # keyword-dense S-4 that merely *describes* the election mechanics would otherwise
    # out-score the actual results 8-K. For these fields, let the form/announcement bonus
    # carry a candidate even with zero keyword hits so the results filing surfaces.
    is_label = str(spec.get("timing_bucket", "")) == "post_election_label"
    # Results-signal bonus: for realized-label fields, reward documents that actually REPORT
    # outcomes so a results press release beats a same-day bare completion 8-K. Capped so it
    # augments — never dominates — the keyword/form signal.
    results_bonus = 0
    if is_label:
        sig = 0
        for ph in RESULTS_SIGNAL_PHRASES:
            parts = [re.escape(p) for p in ph.lower().replace("-", " ").split()]
            if parts and re.search(r"[\s\-]+".join(parts), lowered):
                sig += 1
        results_bonus = min(sig, 5) * 6
    field_score = keyword_score + (form_bonus if (keyword_score > 0 or is_label) else 0) + results_bonus
    snippet = make_snippets(text, hits or keywords, max_snips=4, window=380)
    return field_score, hits, counts, snippet, form_bonus


def field_locator_rows_for_doc(
    manifest_row: "ManifestRow",
    text: str,
    field_specs: Dict[str, Dict[str, Any]],
    min_field_score: int,
) -> List["FieldLocatorRow"]:
    rows: List[FieldLocatorRow] = []
    for field_name, spec in field_specs.items():
        field_score, hits, counts, snippet, form_bonus = score_field_text(
            text=text,
            form=manifest_row.form,
            document_name=manifest_row.document_name,
            spec=spec,
        )
        if field_score < min_field_score:
            continue
        # Realized-results fields must come from post-close announcement forms, never a
        # pre-close registration/proxy (S-4, 424B*, DEF 14A). Those score form_bonus==0
        # against the label fields' preferred forms (8-K/8-K-A/425/EX-99), so gate on it —
        # this removes the logically-impossible "2018 S-4 as evidence for 2019 results" case.
        if str(spec.get("timing_bucket", "")) == "post_election_label" and form_bonus == 0:
            continue
        rows.append(FieldLocatorRow(
            event_id=manifest_row.event_id,
            event_idx=manifest_row.event_idx,
            target_name=manifest_row.target_name,
            acquirer_name=manifest_row.acquirer_name,
            announce_date=manifest_row.announce_date,
            payment_type=manifest_row.payment_type,
            deal_status=manifest_row.deal_status,
            field_name=field_name,
            field_description=str(spec.get("description", "")),
            timing_bucket=str(spec.get("timing_bucket", "")),
            side=manifest_row.side,
            candidate_form=manifest_row.form,
            filing_date=manifest_row.filing_date,
            accession_number=manifest_row.accession_number,
            document_name=manifest_row.document_name,
            doc_id=f"{manifest_row.event_id}:{manifest_row.side}:{manifest_row.accession_number.replace('-', '')}:{safe_filename(manifest_row.document_name, 60)}",
            sec_url=manifest_row.sec_url,
            local_path=manifest_row.local_path,
            field_score=field_score,
            form_bonus=form_bonus,
            keyword_hits=";".join(hits),
            keyword_counts_json=json.dumps(counts, ensure_ascii=False),
            evidence_snippet=snippet[:2000],
            upload_to_claude=field_score > 0,
            reason=(
                f"Matched field keywords for {field_name}"
                + (f"; preferred form/document bonus={form_bonus}" if form_bonus else "")
            ),
        ))
    return rows


def estimate_close_date(manifest_rows: List["ManifestRow"]) -> Optional[pd.Timestamp]:
    """Infer the deal's close date ≈ the TARGET company's last SEC filing.

    The input CSV has no close date, but a merger target deregisters (Form 15) and goes
    dark within days of closing, so the latest target-side filing is a reliable close
    proxy — computed purely from filings we already downloaded, no extra data. Used to
    anchor realized-results evidence selection (the results 8-K is filed at close, so it
    beats both unrelated later 8-Ks and the keyword-heavy pre-close proxies)."""
    dts = [
        pd.to_datetime(getattr(r, "filing_date", None), errors="coerce")
        for r in manifest_rows
        if str(getattr(r, "side", "")).lower() == "target"
    ]
    dts = [d for d in dts if pd.notna(d)]
    return max(dts) if dts else None


def _cusip8(v: Any) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", str(v or "")).upper()[:8]


def load_close_dates(path: Optional[str]) -> Dict[str, pd.Timestamp]:
    """Load authoritative deal-close dates (CRSP delisting) keyed by 8-char target CUSIP.

    Produced by build_close_dates.py from the target CUSIP + CRSP stocknames nameenddt.
    This is the reliable close anchor; estimate_close_date() is the fallback when a
    target has no CRSP coverage (foreign CINS / OTC)."""
    if not path or not Path(path).exists():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    out: Dict[str, pd.Timestamp] = {}
    for _, r in df.iterrows():
        c8 = _cusip8(r.get("target_cusip8") or r.get("target_cusip"))
        d = pd.to_datetime(r.get("close_date", ""), errors="coerce")
        if c8 and pd.notna(d):
            out[c8] = d
    return out


def _close_proximity_bonus(days_from_close: Optional[float]) -> int:
    """Boost for a realized-results filing near the inferred close date."""
    if days_from_close is None:
        return 0
    d = abs(days_from_close)
    if d <= 45:
        return 25
    if d <= 120:
        return 10
    if d <= 270:
        return 3
    return 0


def trim_field_locator(
    locator_df: pd.DataFrame,
    top_k: int,
    close_est_by_event: Optional[Dict[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    if locator_df.empty or top_k <= 0:
        return locator_df
    df = locator_df.copy()
    df["_fdate"] = pd.to_datetime(df["filing_date"], errors="coerce")
    is_label = df["timing_bucket"].astype(str) == "post_election_label"

    # Timing gate (needs a reliable close date): a realized-results field cannot be
    # satisfied by a filing made well BEFORE close — the results don't exist yet. This
    # drops the deal-ANNOUNCEMENT 8-K/425 (right form, describes the election mechanics,
    # high keyword score, but filed months pre-close) which otherwise out-scores the terse
    # results 8-K. Only gate when we actually know close (CRSP); otherwise leave as-is.
    if close_est_by_event:
        PRE_CLOSE_BUFFER = 30  # results are announced around close; allow a small lead
        drop = pd.Series(False, index=df.index)
        for i in df.index[is_label]:
            ce = close_est_by_event.get(str(df.at[i, "event_id"]))
            fd = df.at[i, "_fdate"]
            if ce is not None and pd.notna(fd) and (ce - fd).days > PRE_CLOSE_BUFFER:
                drop.at[i] = True
        if drop.any():
            df = df[~drop]
            is_label = df["timing_bucket"].astype(str) == "post_election_label"

    # Close-date anchor: for realized-results fields, add a proximity-to-close bonus so the
    # results announcement (filed at close) outranks unrelated later 8-Ks (earnings) and
    # keyword-heavy pre-close proxies. Keyword score stays primary; proximity is the booster.
    prox = pd.Series(0, index=df.index, dtype=int)
    if close_est_by_event:
        for i in df.index[is_label]:
            ce = close_est_by_event.get(str(df.at[i, "event_id"]))
            fd = df.at[i, "_fdate"]
            if ce is not None and pd.notna(fd):
                prox.at[i] = _close_proximity_bonus((fd - ce).days)
    df["_eff"] = df["field_score"] + prox

    parts = []
    # Label fields: rank by (score + proximity), latest-first among ties.
    lab = df[is_label].sort_values(
        ["event_id", "field_name", "_eff", "_fdate"], ascending=[True, True, False, False]
    )
    # Pre-close term fields: rank by score, earliest authoritative filing among ties.
    oth = df[~is_label].sort_values(
        ["event_id", "field_name", "field_score", "_fdate"], ascending=[True, True, False, True]
    )
    for sub in (lab, oth):
        if not sub.empty:
            parts.append(sub.groupby(["event_id", "field_name"], dropna=False).head(top_k))
    return pd.concat(parts).drop(columns=["_fdate", "_eff"]).reset_index(drop=True)


def build_event_field_coverage(
    candidate_events: pd.DataFrame,
    locator_df: pd.DataFrame,
    field_specs: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """One row per event-field, including missing required fields.

    This is the audit gate that answers: for every required canonical field, did the
    SEC API retrieval locate at least one candidate filing/snippet that can be shown
    to Claude?  Missing rows are intentional and should be investigated rather than
    silently dropped.
    """
    best = pd.DataFrame()
    if locator_df is not None and not locator_df.empty:
        best = (
            locator_df.sort_values(["event_id", "field_name", "field_score"], ascending=[True, True, False])
            .groupby(["event_id", "field_name"], dropna=False)
            .head(1)
            .copy()
        )

    rows: List[Dict[str, Any]] = []
    for _, ev in candidate_events.iterrows():
        event_idx = int(ev.get("orig_row_idx", ev.name if ev.name is not None else -1))
        target = str(ev.get("Target Name", ""))
        acquirer = str(ev.get("Acquirer Name", ""))
        ann = parse_date(ev.get("Announce Date", ""))
        announce_date = str(ann.date()) if ann is not None and not pd.isna(ann) else str(ev.get("Announce Date", ""))
        event_id = make_event_id(event_idx, ev.get("Announce Date", ""), target, acquirer)
        for field_name, spec in field_specs.items():
            required = bool(spec.get("required", True))
            expected_document_keys = ";".join(map(str, spec.get("document_keys", [])))
            preferred_forms = ";".join(map(str, spec.get("preferred_forms", [])))
            release_timing = str(spec.get("release_timing", ""))
            timing_bucket = str(spec.get("timing_bucket", ""))
            hit = pd.DataFrame()
            if not best.empty:
                hit = best[(best["event_id"].astype(str) == str(event_id)) & (best["field_name"].astype(str) == str(field_name))]
            if hit.empty:
                rows.append({
                    "event_id": event_id,
                    "event_idx": event_idx,
                    "target_name": target,
                    "acquirer_name": acquirer,
                    "announce_date": announce_date,
                    "field_name": field_name,
                    "field_description": spec.get("description", ""),
                    "timing_bucket": timing_bucket,
                    "release_timing": release_timing,
                    "required": required,
                    "covered": False,
                    "missing_required": required,
                    "expected_document_keys": expected_document_keys,
                    "preferred_forms": preferred_forms,
                    "best_field_score": 0,
                    "best_form": "",
                    "best_filing_date": "",
                    "best_document_name": "",
                    "best_sec_url": "",
                    "best_local_path": "",
                    "best_evidence_snippet": "",
                })
            else:
                r = hit.iloc[0]
                rows.append({
                    "event_id": r.get("event_id", event_id),
                    "event_idx": r.get("event_idx", event_idx),
                    "target_name": r.get("target_name", target),
                    "acquirer_name": r.get("acquirer_name", acquirer),
                    "announce_date": r.get("announce_date", announce_date),
                    "field_name": r.get("field_name", field_name),
                    "field_description": r.get("field_description", spec.get("description", "")),
                    "timing_bucket": r.get("timing_bucket", timing_bucket),
                    "release_timing": release_timing,
                    "required": required,
                    "covered": True,
                    "missing_required": False,
                    "expected_document_keys": expected_document_keys,
                    "preferred_forms": preferred_forms,
                    "best_field_score": r.get("field_score", 0),
                    "best_form": r.get("candidate_form", ""),
                    "best_filing_date": r.get("filing_date", ""),
                    "best_document_name": r.get("document_name", ""),
                    "best_sec_url": r.get("sec_url", ""),
                    "best_local_path": r.get("local_path", ""),
                    "best_evidence_snippet": r.get("evidence_snippet", ""),
                })
    return pd.DataFrame(rows)


def build_selected_upload_docs(locator_df: pd.DataFrame, max_docs_per_event: int) -> pd.DataFrame:
    if locator_df.empty:
        return pd.DataFrame()
    work = locator_df[locator_df["upload_to_claude"].astype(bool)].copy()
    if work.empty:
        return pd.DataFrame()
    grouped = (
        work.groupby(["event_id", "doc_id", "local_path", "sec_url", "candidate_form", "filing_date", "document_name"], dropna=False)
        .agg(
            total_field_score=("field_score", "sum"),
            max_field_score=("field_score", "max"),
            matched_fields=("field_name", lambda x: ";".join(sorted(set(map(str, x))))),
            timing_buckets=("timing_bucket", lambda x: ";".join(sorted(set(map(str, x))))),
            reasons=("reason", lambda x: " ||| ".join(sorted(set(map(str, x)))[:5])),
        )
        .reset_index()
        .sort_values(["event_id", "total_field_score", "max_field_score"], ascending=[True, False, False])
    )
    if max_docs_per_event > 0:
        # Reserve slots for realized-results evidence. Ranking purely by total_field_score
        # lets the term-heavy proxy (S-4/424B matches ~all term fields) consume every slot
        # and crowd out the sparse-keyword results 8-K — which serves only the 2-3 label
        # fields. Guarantee the top few label-evidence docs survive, then fill by score.
        reserve = min(3, max_docs_per_event)
        grouped["_is_label"] = grouped["timing_buckets"].astype(str).str.contains("post_election_label", na=False)
        parts = []
        for _ev, g in grouped.groupby("event_id", dropna=False):
            lab = g[g["_is_label"]].sort_values(["max_field_score", "total_field_score"], ascending=False).head(reserve)
            rest = g[~g.index.isin(lab.index)]
            keep = pd.concat([lab, rest]).head(max_docs_per_event)
            # restore the total-score ordering for the surviving set
            parts.append(keep.sort_values(["total_field_score", "max_field_score"], ascending=False))
        grouped = pd.concat(parts).drop(columns=["_is_label"]).reset_index(drop=True)
    return grouped


def build_claude_field_payloads(
    locator_df: pd.DataFrame,
    selected_docs_df: pd.DataFrame,
    field_specs: Dict[str, Dict[str, Any]],
    max_fields_per_event_field: int = 3,
    docs: Optional[List[LlmDocument]] = None,
) -> List[Dict[str, Any]]:
    """Build field-level Claude payloads.

    Important: requested_fields contains every required field from field_specs, even
    when no evidence was found.  Claude must return a value or basis='not_found' for
    every requested field.  When docs are supplied, selected document text excerpts
    are included so --llm-stage send can actually extract values rather than seeing
    only local file paths.
    """
    payloads: List[Dict[str, Any]] = []
    if locator_df is None or locator_df.empty:
        return payloads

    selected_doc_ids_by_event: Dict[str, set] = {}
    if selected_docs_df is not None and not selected_docs_df.empty:
        for event_id, g in selected_docs_df.groupby("event_id"):
            selected_doc_ids_by_event[str(event_id)] = set(map(str, g["doc_id"].tolist()))

    docs_by_id: Dict[str, LlmDocument] = {d.doc_id: d for d in (docs or [])}

    for event_id, g0 in locator_df.groupby("event_id", dropna=False):
        g = g0.copy()
        if str(event_id) in selected_doc_ids_by_event:
            doc_ids = selected_doc_ids_by_event[str(event_id)]
            g = g[g["doc_id"].astype(str).isin(doc_ids)]
        if g.empty:
            continue
        first = g.iloc[0]
        evidence_by_field: Dict[str, List[Dict[str, Any]]] = {name: [] for name in field_specs.keys()}
        for field_name, gf in g.sort_values(["field_name", "field_score"], ascending=[True, False]).groupby("field_name", dropna=False):
            field_name_s = str(field_name)
            evidence_by_field.setdefault(field_name_s, [])
            for _, r in gf.head(max_fields_per_event_field).iterrows():
                evidence_by_field[field_name_s].append({
                    "doc_id": r.get("doc_id", ""),
                    "field_score": int(r.get("field_score", 0)),
                    "form": r.get("candidate_form", ""),
                    "filing_date": r.get("filing_date", ""),
                    "document_name": r.get("document_name", ""),
                    "sec_url": r.get("sec_url", ""),
                    "local_path": r.get("local_path", ""),
                    "keyword_hits": r.get("keyword_hits", ""),
                    "evidence_snippet": r.get("evidence_snippet", ""),
                })

        requested_fields = {
            name: {
                "description": spec.get("description", ""),
                "timing_bucket": spec.get("timing_bucket", ""),
                "release_timing": spec.get("release_timing", ""),
                "document_keys": spec.get("document_keys", []),
                "preferred_forms": spec.get("preferred_forms", []),
                "required": bool(spec.get("required", True)),
            }
            for name, spec in field_specs.items()
        }

        selected_doc_ids = set()
        for vals in evidence_by_field.values():
            selected_doc_ids.update(str(v.get("doc_id", "")) for v in vals if v.get("doc_id"))
        documents = []
        for doc_id in sorted(selected_doc_ids):
            d = docs_by_id.get(doc_id)
            if d is not None:
                documents.append(asdict(d))

        payloads.append({
            "task": "extract_requested_fields_from_sec_merger_filings",
            "system_prompt": FIELD_LLM_SYSTEM_PROMPT,
            "expected_json_schema": FIELD_EXTRACTION_SCHEMA,
            "event": {
                "event_id": str(event_id),
                "event_idx": int(first.get("event_idx", -1)),
                "target_name": str(first.get("target_name", "")),
                "acquirer_name": str(first.get("acquirer_name", "")),
                "announce_date": str(first.get("announce_date", "")),
                "payment_type": str(first.get("payment_type", "")),
                "deal_status": str(first.get("deal_status", "")),
            },
            "requested_fields": requested_fields,
            "candidate_evidence_by_field": evidence_by_field,
            "documents": documents,
            "instruction": (
                "Use the evidence snippets and document excerpts to extract every requested field. "
                "Do not summarize filings. Return one JSON object only. For every field in requested_fields, "
                "return an entry in fields; use value=null and basis='not_found' when evidence is insufficient."
            ),
        })
    return payloads


def write_claude_upload_packages(
    out_dir: Path,
    payloads: List[Dict[str, Any]],
    selected_docs_df: pd.DataFrame,
) -> None:
    if not payloads:
        return
    pkg_root = out_dir / "claude_upload_packages"
    pkg_root.mkdir(parents=True, exist_ok=True)
    selected_by_event = {eid: g for eid, g in selected_docs_df.groupby("event_id")} if not selected_docs_df.empty else {}
    for payload in payloads:
        event_id = payload.get("event", {}).get("event_id", "event")
        safe_event = safe_filename(event_id, 120)
        event_dir = pkg_root / safe_event
        docs_dir = event_dir / "selected_docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        # Write prompt and evidence index even if local docs were not saved.
        (event_dir / "evidence_index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        prompt_text = (
            payload.get("system_prompt", "")
            + "\n\nUSER PAYLOAD JSON:\n"
            + json.dumps({k: v for k, v in payload.items() if k != "system_prompt"}, ensure_ascii=False, indent=2)
        )
        (event_dir / "claude_prompt.txt").write_text(prompt_text, encoding="utf-8")
        copied = []
        if event_id in selected_by_event:
            for _, r in selected_by_event[event_id].iterrows():
                local_path = str(r.get("local_path", ""))
                if not local_path:
                    continue
                src = Path(local_path)
                if src.exists() and src.is_file():
                    dest = docs_dir / f"{safe_filename(str(r.get('doc_id', 'doc')), 80)}__{safe_filename(src.name, 100)}"
                    try:
                        shutil.copy2(src, dest)
                        copied.append({"source": local_path, "copied_to": str(dest)})
                    except Exception as e:
                        copied.append({"source": local_path, "copy_error": str(e)})
        (event_dir / "copied_docs.json").write_text(json.dumps(copied, ensure_ascii=False, indent=2), encoding="utf-8")


def select_llm_documents(docs: List[LlmDocument], max_docs: int) -> List[LlmDocument]:
    """Keep the most likely useful documents while preserving enough variety."""
    if max_docs <= 0 or len(docs) <= max_docs:
        return docs
    return sorted(
        docs,
        key=lambda d: (d.keyword_score, d.form in {"S-4", "S-4/A", "424B3", "DEFM14A"}, d.filing_date),
        reverse=True,
    )[:max_docs]


def build_llm_event_payload(event: pd.Series, rows: List[ManifestRow], docs: List[LlmDocument], max_docs: int) -> Dict[str, Any]:
    selected_docs = select_llm_documents(docs, max_docs=max_docs)
    event_id = rows[0].event_id if rows else ""
    return {
        "task": "extract_merger_election_and_proration_fields",
        "system_prompt": LLM_SYSTEM_PROMPT,
        "expected_json_schema": LLM_EXTRACTION_SCHEMA,
        "event": {
            "event_id": event_id,
            "event_idx": int(event.get("orig_row_idx", -1)),
            "target_name": str(event.get("Target Name", "")),
            "acquirer_name": str(event.get("Acquirer Name", "")),
            "announce_date": str(event.get("Announce Date", "")),
            "payment_type": str(event.get("Payment Type", "")),
            "deal_status": str(event.get("Deal Status", "")),
        },
        "document_selection": {
            "documents_in_manifest": len(rows),
            "documents_sent_or_prepared": len(selected_docs),
            "selection_note": "Documents are ranked by local election/proration keyword score and merger-form priority.",
        },
        "documents": [asdict(d) for d in selected_docs],
    }


def anthropic_messages_payload(llm_payload: Dict[str, Any], model: str, max_tokens: int) -> Dict[str, Any]:
    """Create a Claude Messages API request for either legacy or field-level payloads."""
    user_payload = {k: v for k, v in llm_payload.items() if k != "system_prompt"}
    task = str(llm_payload.get("task", "extract_sec_merger_fields"))
    if task == "extract_requested_fields_from_sec_merger_filings":
        lead = (
            "Extract every requested canonical field from the supplied SEC filing evidence. "
            "Return JSON only and include a fields object with one key per requested field.\n\n"
        )
    else:
        lead = "Extract the requested merger election/proration data from these SEC filing excerpts. Return JSON only.\n\n"
    return {
        "model": model,
        "max_tokens": max_tokens,
        # NOTE: temperature is intentionally omitted — Opus 4.8 / Sonnet 5 reject
        # sampling params (400). Extraction determinism is handled via prompting.
        "system": llm_payload["system_prompt"],
        "messages": [
            {
                "role": "user",
                "content": lead + json.dumps(user_payload, ensure_ascii=False),
            }
        ],
    }


def call_anthropic(llm_payload: Dict[str, Any], api_key: str, model: str, max_tokens: int) -> Dict[str, Any]:
    if not api_key:
        raise ValueError("Anthropic API key is required for --llm-stage send. Pass --anthropic-api-key or set ANTHROPIC_API_KEY.")
    req = anthropic_messages_payload(llm_payload, model=model, max_tokens=max_tokens)
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        data=json.dumps(req),
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    text_blocks = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    text = "\n".join(text_blocks).strip()
    try:
        parsed = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
            except Exception:
                parsed = {"parse_error": True, "raw_text": text, "stop_reason": data.get("stop_reason", "")}
        else:
            parsed = {"parse_error": True, "raw_text": text, "stop_reason": data.get("stop_reason", "")}
    return {
        "provider": "anthropic",
        "model": model,
        "request": req,
        "response": data,
        "parsed": parsed,
    }


# (input, output) USD per 1M tokens. The Message Batches API applies a further -50%.
MODEL_PRICES_USD_PER_MTOK = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),   # intro pricing through 2026-08-31
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def _parse_anthropic_message(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the JSON object from a Messages API response body (same logic as call_anthropic)."""
    text_blocks = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    text = "\n".join(text_blocks).strip()
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass
    return {"parse_error": True, "raw_text": text, "stop_reason": data.get("stop_reason", "")}


def estimate_batch_cost(payloads: List[Dict[str, Any]], model: str, max_tokens: int) -> Tuple[float, int, int]:
    """Rough pre-flight cost (USD, est_input_tok, est_output_tok) for a batch, incl. the -50% batch discount.
    Output is planned at ~9k tok/deal (observed 8-11k); input from payload chars/4."""
    pin, pout = MODEL_PRICES_USD_PER_MTOK.get(model, (5.0, 25.0))
    tin = tout = 0.0
    # SEC filing text tokenizes denser than prose: the 12-deal Sonnet-5 run measured
    # ~256K real input tokens against ~792K payload chars/deal => ~3.05 chars/token. Use 3.0
    # (a touch conservative) so this pre-flight never UNDER-estimates the cost cap.
    for p in payloads:
        body = json.dumps({k: v for k, v in p.items() if k != "system_prompt"}, ensure_ascii=False)
        tin += (len(body) + len(str(p.get("system_prompt", "")))) / 3.0
        tout += min(max_tokens, 9000)
    cost = (tin * pin / 1e6 + tout * pout / 1e6) * 0.5
    return cost, int(tin), int(tout)


def run_anthropic_batch(
    payloads: List[Dict[str, Any]],
    api_key: str,
    model: str,
    max_tokens: int,
    max_cost_usd: Optional[float] = None,
    poll_seconds: int = 30,
    max_wait_seconds: int = 86400,
) -> List[Dict[str, Any]]:
    """Submit all field payloads as ONE Message Batch (-50% vs sync), poll to completion, and
    return records in the same shape call_anthropic produces so flatten_llm_records just works.

    A hard cost cap (max_cost_usd) is enforced BEFORE submission: if the pre-flight estimate
    exceeds it, we abort without spending — raise the cap or cut --max-events."""
    if not api_key:
        raise ValueError("Anthropic API key is required for --llm-stage batch.")
    proj, est_in, est_out = estimate_batch_cost(payloads, model, max_tokens)
    print(f"[batch] {len(payloads)} requests; est input ~{est_in:,} tok / output ~{est_out:,} tok; "
          f"projected batch cost ~${proj:.2f}")
    if max_cost_usd is not None and proj > max_cost_usd:
        raise SystemExit(
            f"[batch] ABORT (no spend): projected ${proj:.2f} exceeds --max-batch-cost-usd ${max_cost_usd:.2f}. "
            f"Raise --max-batch-cost-usd or lower --max-events. Payloads are cached; re-running is free until submit."
        )

    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    order: Dict[str, Dict[str, Any]] = {}
    requests_body: List[Dict[str, Any]] = []
    for i, p in enumerate(payloads):
        cid = f"evt-{i:05d}"
        order[cid] = p
        requests_body.append({"custom_id": cid, "params": anthropic_messages_payload(p, model=model, max_tokens=max_tokens)})

    def _post_with_retry(url: str, body: Dict[str, Any], tries: int = 5) -> Dict[str, Any]:
        last = None
        for attempt in range(tries):
            try:
                r = requests.post(url, headers=headers, data=json.dumps(body), timeout=120)
                if r.status_code in (429, 500, 502, 503, 529):
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last = e
                wait = min(60, (2 ** attempt)) + random.uniform(0, 1)
                print(f"[batch] submit retry {attempt+1}/{tries} after error: {e}; sleeping {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
        raise RuntimeError(f"[batch] submit failed after {tries} tries: {last}")

    created = _post_with_retry("https://api.anthropic.com/v1/messages/batches", {"requests": requests_body})
    batch_id = created.get("id")
    print(f"[batch] submitted batch {batch_id}; polling every {poll_seconds}s (up to {max_wait_seconds//3600}h)...")

    waited = 0
    results_url = None
    while waited < max_wait_seconds:
        try:
            st = requests.get(f"https://api.anthropic.com/v1/messages/batches/{batch_id}", headers=headers, timeout=60).json()
        except Exception as e:
            print(f"[batch] poll error (will retry): {e}", file=sys.stderr)
            time.sleep(poll_seconds); waited += poll_seconds; continue
        status = st.get("processing_status")
        counts = st.get("request_counts", {})
        print(f"[batch] {now_utc()} status={status} counts={counts}")
        if status == "ended":
            results_url = st.get("results_url")
            break
        time.sleep(poll_seconds); waited += poll_seconds
    if results_url is None:
        raise RuntimeError(f"[batch] batch {batch_id} did not finish within {max_wait_seconds}s")

    # Stream JSONL results; each line: {custom_id, result: {type, message?, error?}}
    rr = requests.get(results_url, headers=headers, timeout=300)
    rr.raise_for_status()
    records: List[Dict[str, Any]] = []
    n_ok = n_err = 0
    for line in rr.text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cid = obj.get("custom_id", "")
        payload = order.get(cid, {})
        res = obj.get("result", {})
        if res.get("type") == "succeeded":
            data = res.get("message", {})
            parsed = _parse_anthropic_message(data)
            n_ok += 1
        else:
            data = {"batch_result_type": res.get("type"), "error": res.get("error")}
            parsed = {"parse_error": True, "batch_error": res.get("error"), "result_type": res.get("type")}
            n_err += 1
        records.append({"provider": "anthropic", "model": model, "custom_id": cid,
                        "request": {"event_id": payload.get("event", {}).get("event_id", "")},
                        "response": data, "parsed": parsed})
    print(f"[batch] collected {len(records)} results: {n_ok} succeeded, {n_err} errored")
    return records


def field_value(x: Any, *path: str) -> Any:
    cur = x
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    if isinstance(cur, (dict, list)):
        return json.dumps(cur, ensure_ascii=False)
    return cur


def flatten_llm_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten Claude results. Field-level results become one row per event-field."""
    out_rows: List[Dict[str, Any]] = []
    for record in records:
        parsed = record.get("parsed", record)
        if isinstance(parsed, dict) and isinstance(parsed.get("fields"), dict):
            event_id = parsed.get("event_id")
            target_name = parsed.get("target_name")
            acquirer_name = parsed.get("acquirer_name")
            for field_name, field_obj in parsed.get("fields", {}).items():
                if not isinstance(field_obj, dict):
                    field_obj = {"value": field_obj}
                out_rows.append({
                    "event_id": event_id,
                    "target_name": target_name,
                    "acquirer_name": acquirer_name,
                    "field_name": field_name,
                    "value": field_obj.get("value"),
                    "basis": field_obj.get("basis"),
                    "timing_bucket": field_obj.get("timing_bucket"),
                    "source_doc_ids": json.dumps(field_obj.get("source_doc_ids", []), ensure_ascii=False),
                    "source_form_types": json.dumps(field_obj.get("source_form_types", []), ensure_ascii=False),
                    "source_filing_dates": json.dumps(field_obj.get("source_filing_dates", []), ensure_ascii=False),
                    "evidence_quotes": json.dumps(field_obj.get("evidence_quotes", []), ensure_ascii=False),
                    "confidence": field_obj.get("confidence"),
                    "notes": field_obj.get("notes"),
                    "parse_error": parsed.get("parse_error", False),
                    "raw_text": parsed.get("raw_text", ""),
                })
        else:
            out_rows.append(flatten_llm_extraction(record))
    return pd.DataFrame(out_rows)


def flatten_llm_extraction(record: Dict[str, Any]) -> Dict[str, Any]:
    parsed = record.get("parsed", record)
    return {
        "event_id": parsed.get("event_id"),
        "target_name": parsed.get("target_name"),
        "acquirer_name": parsed.get("acquirer_name"),
        "announce_date": parsed.get("announce_date"),
        "mixed_election_allowed": field_value(parsed, "deal_currency_mix", "mixed_election_allowed"),
        "cash_component": field_value(parsed, "deal_currency_mix", "cash_component"),
        "stock_component": field_value(parsed, "deal_currency_mix", "stock_component"),
        "cash_election_available": field_value(parsed, "election_mechanics", "cash_election_available"),
        "stock_election_available": field_value(parsed, "election_mechanics", "stock_election_available"),
        "election_deadline": field_value(parsed, "election_mechanics", "election_deadline", "value"),
        "election_deadline_estimated": field_value(parsed, "election_mechanics", "election_deadline", "estimated"),
        "election_deadline_basis": field_value(parsed, "election_mechanics", "election_deadline", "basis"),
        "non_election_treatment": field_value(parsed, "election_mechanics", "non_election_treatment", "value"),
        "proration_applicable": field_value(parsed, "proration", "proration_applicable"),
        "proration_formula_or_limits": field_value(parsed, "proration", "proration_formula_or_limits", "value"),
        "preliminary_proration": field_value(parsed, "proration", "preliminary_proration", "value"),
        "final_proration": field_value(parsed, "proration", "final_proration", "value"),
        "important_dates_json": field_value(parsed, "important_dates"),
        "recommended_manual_review_json": field_value(parsed, "recommended_manual_review"),
        "confidence_overall": field_value(parsed, "confidence", "overall"),
        "confidence_notes": field_value(parsed, "confidence", "notes"),
        "parse_error": parsed.get("parse_error", False),
        "raw_text": parsed.get("raw_text", ""),
    }


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@dataclass
class ManifestRow:
    event_id: str
    event_idx: int
    target_name: str
    acquirer_name: str
    announce_date: str
    payment_type: str
    deal_status: str
    side: str
    query_company_name: str
    cik_int: Optional[int]
    cik10: str
    ticker: str
    sec_title: str
    match_score: float
    form: str
    filing_date: str
    accession_number: str
    primary_document: str
    document_name: str
    sec_url: str
    local_path: str
    downloaded: bool
    bytes: int
    keyword_score: int
    keywords_found: str
    keyword_counts_json: str
    snippet: str
    error: str


@dataclass
class LlmDocument:
    doc_id: str
    event_id: str
    side: str
    form: str
    filing_date: str
    accession_number: str
    document_name: str
    sec_url: str
    local_path: str
    keyword_score: int
    keywords_found: str
    snippet: str
    text_excerpt: str


@dataclass
class FieldLocatorRow:
    event_id: str
    event_idx: int
    target_name: str
    acquirer_name: str
    announce_date: str
    payment_type: str
    deal_status: str
    field_name: str
    field_description: str
    timing_bucket: str
    side: str
    candidate_form: str
    filing_date: str
    accession_number: str
    document_name: str
    doc_id: str
    sec_url: str
    local_path: str
    field_score: int
    form_bonus: int
    keyword_hits: str
    keyword_counts_json: str
    evidence_snippet: str
    upload_to_claude: bool
    reason: str


@dataclass
class ProcessEventResult:
    rows: List[ManifestRow]
    llm_documents: List[LlmDocument]
    field_locator_rows: List[FieldLocatorRow]


def process_event(
    event_idx: int,
    event: pd.Series,
    cik_matches: Dict[str, Dict[str, Any]],
    client: SecClient,
    out_dir: Path,
    forms: List[str],
    keywords: List[str],
    pre_days: int,
    post_days: int,
    download_exhibits: bool,
    dry_run: bool,
    resume: bool,
    max_docs_per_event_side: int,
    save_documents: bool,
    collect_llm_documents: bool,
    llm_max_doc_chars: int,
    field_specs: Dict[str, Dict[str, Any]],
    min_field_score: int,
) -> ProcessEventResult:
    rows: List[ManifestRow] = []
    llm_documents: List[LlmDocument] = []
    field_locator_rows: List[FieldLocatorRow] = []

    ann = parse_date(event.get("Announce Date"))
    if ann is None or pd.isna(ann):
        return ProcessEventResult(rows=rows, llm_documents=llm_documents, field_locator_rows=field_locator_rows)

    date_min = ann - pd.Timedelta(days=pre_days)
    date_max = ann + pd.Timedelta(days=post_days)

    target = str(event.get("Target Name", "")).strip()
    acquirer = str(event.get("Acquirer Name", "")).strip()
    event_id = make_event_id(event_idx, ann, target, acquirer)
    event_dir = out_dir / "documents" / event_id

    for side, name in [("target", target), ("acquirer", acquirer)]:
        match = cik_matches.get(f"{event_idx}:{side}")
        if not match or not match.get("matched"):
            continue

        cik_int = int(match["cik_int"])
        cik10 = str(match["cik10"])
        try:
            sub = client.get_json(SEC_SUBMISSIONS_URL.format(cik10=cik10))
            filings = flatten_submissions(sub, client)
        except Exception as e:
            rows.append(ManifestRow(
                event_id=event_id, event_idx=event_idx, target_name=target, acquirer_name=acquirer,
                announce_date=str(ann.date()), payment_type=str(event.get("Payment Type", "")),
                deal_status=str(event.get("Deal Status", "")), side=side, query_company_name=name,
                cik_int=cik_int, cik10=cik10, ticker=str(match.get("ticker", "")),
                sec_title=str(match.get("sec_title", "")), match_score=float(match.get("score", 0)),
                form="", filing_date="", accession_number="", primary_document="", document_name="",
                sec_url="", local_path="", downloaded=False, bytes=0, keyword_score=0,
                keywords_found="", keyword_counts_json="{}", snippet="", error=f"submissions_error: {e}",
            ))
            continue

        if filings.empty:
            continue

        # Filter form and filing date.
        filings["form_norm"] = filings["form"].astype(str).str.upper().str.strip()
        form_set = {f.upper().strip() for f in forms}
        mask = (
            filings["form_norm"].isin(form_set)
            & filings["filingDate"].notna()
            & (filings["filingDate"] >= date_min)
            & (filings["filingDate"] <= date_max)
        )
        fdf = filings.loc[mask].sort_values("filingDate").copy()

        if max_docs_per_event_side and len(fdf) > max_docs_per_event_side:
            # STRATIFY, don't just take the earliest N. The deal proxy (terms) sits near
            # the announcement (start of window) but the election-RESULTS 8-K sits at close
            # (later in the window). Taking only the earliest N — when a target files
            # heavily around announcement (e.g. VMware) — truncates before close and the
            # results filing is never retrieved. Keep the earliest half AND the latest half.
            half = max_docs_per_event_side // 2
            fdf = pd.concat([fdf.head(half), fdf.tail(max_docs_per_event_side - half)]).drop_duplicates()

        for _, f in fdf.iterrows():
            form = str(f.get("form", ""))
            filing_date = f.get("filingDate")
            filing_date_str = "" if pd.isna(filing_date) else pd.Timestamp(filing_date).strftime("%Y-%m-%d")
            accession = str(f.get("accessionNumber", ""))
            primary_doc = str(f.get("primaryDocument", ""))
            urls = filing_urls(cik_int, accession, primary_doc)

            doc_items: List[Tuple[str, str]] = [(primary_doc, urls["primary_url"])]

            if download_exhibits:
                try:
                    idx_json = client.get_json(urls["index_json_url"])
                    items = idx_json.get("directory", {}).get("item", []) or []
                    for item in items:
                        fname = item.get("name", "")
                        if fname and should_download_item(fname):
                            u = urljoin(urls["base_url"], fname)
                            if (fname, u) not in doc_items:
                                doc_items.append((fname, u))
                except Exception as e:
                    print(f"[WARN] index failed for {accession}: {e}", file=sys.stderr)

            for doc_name, url in doc_items:
                rel_dir = Path(event_id) / side / f"{filing_date_str}_{form.replace('/', '-')}_{accession.replace('-', '')}"
                local = event_dir / side / f"{filing_date_str}_{form.replace('/', '-')}_{accession.replace('-', '')}" / safe_filename(doc_name, 120)
                downloaded = False
                nbytes = 0
                err = ""
                raw: Optional[bytes] = None

                if dry_run:
                    downloaded = False
                elif save_documents:
                    ok, dl_err, nbytes = client.download_file(url, local, resume=resume)
                    downloaded = ok
                    err = dl_err or ""
                    if downloaded and local.exists() and local.suffix.lower() != ".pdf":
                        try:
                            raw = local.read_bytes()
                        except Exception as e:
                            err = (err + "; " if err else "") + f"read_error: {e}"
                else:
                    ok, dl_err, raw = client.fetch_file(url)
                    downloaded = ok
                    nbytes = len(raw)
                    err = dl_err or ""

                keyword_score = 0
                found: List[str] = []
                counts: Dict[str, int] = {}
                snippet = ""
                text = ""

                if downloaded and raw is not None and Path(doc_name).suffix.lower() != ".pdf":
                    try:
                        text = strip_html_for_scoring(raw)
                        keyword_score, found, counts = score_text(text, keywords)
                        snippet = make_snippets(text, found or keywords, max_snips=3)
                    except Exception as e:
                        err = (err + "; " if err else "") + f"score_error: {e}"

                doc_id = f"{event_id}:{side}:{accession.replace('-', '')}:{safe_filename(doc_name, 60)}"
                if collect_llm_documents and downloaded and text:
                    llm_documents.append(LlmDocument(
                        doc_id=doc_id,
                        event_id=event_id,
                        side=side,
                        form=form,
                        filing_date=filing_date_str,
                        accession_number=accession,
                        document_name=doc_name,
                        sec_url=url,
                        local_path=str(local) if save_documents else "",
                        keyword_score=keyword_score,
                        keywords_found=";".join(found),
                        snippet=snippet[:1200],
                        text_excerpt=text[:llm_max_doc_chars],
                    ))

                mrow = ManifestRow(
                    event_id=event_id,
                    event_idx=event_idx,
                    target_name=target,
                    acquirer_name=acquirer,
                    announce_date=str(ann.date()),
                    payment_type=str(event.get("Payment Type", "")),
                    deal_status=str(event.get("Deal Status", "")),
                    side=side,
                    query_company_name=name,
                    cik_int=cik_int,
                    cik10=cik10,
                    ticker=str(match.get("ticker", "")),
                    sec_title=str(match.get("sec_title", "")),
                    match_score=float(match.get("score", 0)),
                    form=form,
                    filing_date=filing_date_str,
                    accession_number=accession,
                    primary_document=primary_doc,
                    document_name=doc_name,
                    sec_url=url,
                    local_path=str(local) if save_documents else "",
                    downloaded=downloaded,
                    bytes=nbytes,
                    keyword_score=keyword_score,
                    keywords_found=";".join(found),
                    keyword_counts_json=json.dumps(counts, ensure_ascii=False),
                    snippet=snippet[:1200],
                    error=err,
                )
                rows.append(mrow)

                field_text_for_locator = text
                if not field_text_for_locator and downloaded:
                    # PDF/text-unreadable exhibits can still be important when the file name
                    # itself says "letter of transmittal", "election form", "ex-99.1", etc.
                    field_text_for_locator = f"{doc_name} {form} {primary_doc}".lower()
                if downloaded and field_text_for_locator and field_specs:
                    try:
                        field_locator_rows.extend(
                            field_locator_rows_for_doc(
                                manifest_row=mrow,
                                text=field_text_for_locator,
                                field_specs=field_specs,
                                min_field_score=min_field_score,
                            )
                        )
                    except Exception as e:
                        print(f"[WARN] field locator failed for {doc_id}: {e}", file=sys.stderr)

    return ProcessEventResult(rows=rows, llm_documents=llm_documents, field_locator_rows=field_locator_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download SEC EDGAR filings for M&A events.")
    p.add_argument("--input", required=False, help="Bloomberg M&A export CSV.")
    p.add_argument("--output-dir", default="ma_edgar_docs", help="Output directory.")
    p.add_argument("--user-agent", required=False, help="Required SEC User-Agent, e.g. 'Name email@domain.com'.")
    p.add_argument("--payment-types", nargs="*", default=["Cash or Stock", "Cash and Stock"],
                   help="Payment Type values to keep. Use empty string list with --all-payment-types to disable.")
    p.add_argument("--all-payment-types", action="store_true", help="Do not filter by Payment Type.")
    p.add_argument("--deal-status", nargs="*", default=["Completed"], help="Deal Status values to keep.")
    p.add_argument("--all-status", action="store_true", help="Do not filter by Deal Status.")
    p.add_argument("--start-date", default=None, help="Optional earliest Announce Date, YYYY-MM-DD.")
    p.add_argument("--end-date", default=None, help="Optional latest Announce Date, YYYY-MM-DD.")
    p.add_argument("--forms", nargs="*", default=DEFAULT_FORMS, help="SEC form types to download.")
    p.add_argument("--keywords", nargs="*", default=DEFAULT_KEYWORDS, help="Keywords used to score documents.")
    p.add_argument("--pre-days", type=int, default=60, help="Days before announcement to include.")
    p.add_argument("--post-days", type=int, default=730, help="Days after announcement to include.")
    p.add_argument("--min-name-score", type=int, default=84, help="Minimum fuzzy score for CIK matching.")
    p.add_argument("--cik-overrides", default="cik_manual_overrides.csv", help="Optional CSV (target_name, cik10, ...) of hand-verified CIKs for the recoverable resolver tail (delisted/renamed targets). Consulted before efts/company_tickers. Build/verify with build_cik_overrides.py.")
    p.add_argument("--close-dates", default=None, help="Optional CSV (target_cusip8/target_cusip, close_date) of authoritative CRSP delisting/close dates; anchors realized-results evidence. Build with build_close_dates.py.")
    p.add_argument("--max-events", type=int, default=None, help="Limit events for testing.")
    p.add_argument("--start-event", type=int, default=0, help="Skip events whose orig_row_idx (the E###### number) is below this. Use on a --resume run to jump straight to remaining deals instead of re-scanning already-downloaded ones. Set it a few below the highest downloaded event to re-verify the boundary deal.")
    p.add_argument("--only-event-idx", default=None, help="Comma-separated orig_row_idx values (E###### numbers) to process EXCLUSIVELY. Reuses cached downloads for a targeted re-extraction of specific deals.")
    p.add_argument("--max-docs-per-event-side", type=int, default=60, help="Cap filings per event-side after filtering (stratified: earliest + latest, so both the announcement proxy and the close-date results filing survive).")
    p.add_argument("--download-exhibits", action="store_true", help="Also download filing-folder text/html/pdf exhibits.")
    p.add_argument("--no-save-documents", action="store_true",
                   help="Fetch SEC documents for scoring/LLM payloads but do not save document files locally.")
    p.add_argument("--dry-run", action="store_true", help="Build manifest candidates but do not download files.")
    p.add_argument("--resume", action="store_true", default=True, help="Skip already downloaded files.")
    p.add_argument("--sleep-seconds", type=float, default=0.13, help="Delay between SEC requests. Keep >=0.11.")
    p.add_argument("--cache-dir", default=None, help="Optional JSON cache directory.")
    p.add_argument("--llm-stage", choices=["off", "prepare", "send", "batch"], default="off",
                   help="off: no LLM work; prepare: write per-event JSONL payloads only; send: call provider synchronously per deal; batch: submit ALL deals as one Message Batch (half-price, async).")
    p.add_argument("--max-batch-cost-usd", type=float, default=75.0,
                   help="Hard pre-flight cost ceiling for --llm-stage batch. If the estimate exceeds this, abort BEFORE submitting (no spend).")
    p.add_argument("--batch-poll-seconds", type=int, default=30, help="Polling interval while waiting for the Message Batch to finish.")
    p.add_argument("--llm-provider", choices=["anthropic"], default="anthropic",
                   help="LLM provider used when --llm-stage send.")
    p.add_argument("--llm-model", default="claude-sonnet-4-6", help="Claude/LLM model name.")
    p.add_argument("--llm-max-docs-per-event", type=int, default=12,
                   help="Maximum text documents included in each event-level LLM payload.")
    p.add_argument("--llm-max-doc-chars", type=int, default=80_000,
                   help="Maximum cleaned characters retained per document for LLM payloads.")
    p.add_argument("--llm-max-tokens", type=int, default=12000, help="Max tokens requested from the LLM.")
    p.add_argument("--field-specs", default=None,
                   help="Optional field_specs.json path. Defaults to ./field_specs.json or built-in specs.")
    p.add_argument("--min-field-score", type=int, default=1,
                   help="Minimum field-level score required for a field locator row.")
    p.add_argument("--field-locator-top-k", type=int, default=3,
                   help="Keep top K evidence rows per event-field in field_locator.csv. Use 0 to keep all.")
    p.add_argument("--claude-package-max-docs-per-event", type=int, default=10,
                   help="Max unique local documents selected per event for Claude upload packages.")
    p.add_argument("--make-claude-packages", action="store_true",
                   help="Create claude_upload_packages/event_id/ with selected local docs, evidence_index.json, and claude_prompt.txt.")
    p.add_argument("--anthropic-api-key", default=None,
                   help="Anthropic API key. If omitted, ANTHROPIC_API_KEY is used for --llm-stage send.")
    p.add_argument("--aggregate-llm-results", default=None,
                   help="Read an existing LLM results JSONL file and write llm_field_extractions.csv, then exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "_cache"

    if args.aggregate_llm_results:
        records = read_jsonl(Path(args.aggregate_llm_results))
        flattened = flatten_llm_records(records)
        out_path = out_dir / "llm_field_extractions.csv"
        flattened.to_csv(out_path, index=False)
        print(f"[{now_utc()}] Aggregated LLM records: {len(flattened):,}")
        print(f"[{now_utc()}] Wrote LLM extraction summary: {out_path}")
        return

    if not args.input:
        raise ValueError("--input is required unless --aggregate-llm-results is used.")
    if not args.user_agent:
        raise ValueError("--user-agent is required when fetching SEC data.")

    field_specs = load_field_specs(args.field_specs)
    print(f"[{now_utc()}] Loaded field specs: {len(field_specs):,}")

    print(f"[{now_utc()}] Loading input: {args.input}")
    df = normalize_event_input(pd.read_csv(args.input))
    required = ["Announce Date", "Target Name", "Acquirer Name", "Payment Type", "Deal Status"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}. Columns={list(df.columns)}")

    df["Announce Date Parsed"] = pd.to_datetime(df["Announce Date"], errors="coerce")

    # Filter universe.
    udf = df.copy()
    if not args.all_payment_types:
        udf = udf[udf["Payment Type"].astype(str).isin(args.payment_types)]
    if not args.all_status:
        udf = udf[udf["Deal Status"].astype(str).isin(args.deal_status)]
    if args.start_date:
        udf = udf[udf["Announce Date Parsed"] >= pd.to_datetime(args.start_date)]
    if args.end_date:
        udf = udf[udf["Announce Date Parsed"] <= pd.to_datetime(args.end_date)]
    udf = udf.dropna(subset=["Announce Date Parsed"]).reset_index(drop=False).rename(columns={"index": "orig_row_idx"})

    if args.max_events:
        udf = udf.head(args.max_events)

    if args.start_event:
        before = len(udf)
        udf = udf[udf["orig_row_idx"].astype(int) >= int(args.start_event)]
        print(f"[{now_utc()}] --start-event {args.start_event}: skipping {before - len(udf)} already-processed "
              f"events; only events with orig_row_idx >= {args.start_event} are (re)processed. This makes a resume "
              f"jump straight to remaining deals instead of re-scanning downloaded ones.")

    if args.only_event_idx:
        ids = {int(x) for x in str(args.only_event_idx).split(",") if x.strip() != ""}
        before = len(udf)
        udf = udf[udf["orig_row_idx"].astype(int).isin(ids)]
        print(f"[{now_utc()}] --only-event-idx: processing only {len(udf)}/{before} events "
              f"(orig_row_idx in {sorted(ids)}); reuses cached downloads for a targeted re-extraction.")

    print(f"[{now_utc()}] Candidate events after filters: {len(udf):,}")
    out_dir.joinpath("candidate_events.csv").write_text(udf.to_csv(index=False), encoding="utf-8")

    client = SecClient(user_agent=args.user_agent, sleep_seconds=args.sleep_seconds, cache_dir=cache_dir)
    cik_df = load_company_tickers(client, out_dir / "_cache" / "company_tickers.json")
    print(f"[{now_utc()}] Loaded SEC company tickers: {len(cik_df):,}")
    n_ovr = load_cik_overrides(Path(args.cik_overrides)) if args.cik_overrides else 0
    if n_ovr:
        print(f"[{now_utc()}] Loaded {n_ovr} hand-verified CIK overrides from {args.cik_overrides}")

    # Map names to CIK.
    match_rows: List[Dict[str, Any]] = []
    cik_matches: Dict[str, Dict[str, Any]] = {}

    for i, event in udf.iterrows():
        event_idx = int(event["orig_row_idx"])
        for side, col in [("target", "Target Name"), ("acquirer", "Acquirer Name")]:
            name = str(event.get(col, "")).strip()
            # Hand-verified overrides win outright (recoverable delisted/renamed targets).
            # Otherwise: EDGAR full-text search (resolves delisted merger targets), then
            # fall back to the current-registrant company_tickers.json only if efts misses.
            m = override_match(name)
            if m is not None:
                m.update({"side": side})
                match_rows.append({**m,
                                   "event_idx": event_idx, "side": side,
                                   "target_name": str(event.get("Target Name", "")),
                                   "acquirer_name": str(event.get("Acquirer Name", "")),
                                   "announce_date": str(event.get("Announce Date", "")),
                                   "payment_type": str(event.get("Payment Type", "")),
                                   "deal_status": str(event.get("Deal Status", ""))})
                cik_matches[f"{event_idx}:{side}"] = match_rows[-1]
                continue
            m = resolve_cik_via_efts(client, name, min_score=args.min_name_score)
            if not m.get("matched"):
                m_ticker = fuzzy_match_company(name, cik_df, min_score=args.min_name_score)
                if m_ticker.get("matched") or float(m_ticker.get("score", 0) or 0) > float(m.get("score", 0) or 0):
                    m = m_ticker
            m.update({
                "event_idx": event_idx,
                "side": side,
                "target_name": str(event.get("Target Name", "")),
                "acquirer_name": str(event.get("Acquirer Name", "")),
                "announce_date": str(event.get("Announce Date", "")),
                "payment_type": str(event.get("Payment Type", "")),
                "deal_status": str(event.get("Deal Status", "")),
            })
            match_rows.append(m)
            cik_matches[f"{event_idx}:{side}"] = m

    matches_df = pd.DataFrame(match_rows)
    matches_df.to_csv(out_dir / "cik_name_matches.csv", index=False)
    unresolved = matches_df[~matches_df["matched"].astype(bool)]
    unresolved.to_csv(out_dir / "unresolved_names.csv", index=False)

    print(f"[{now_utc()}] Matched names: {matches_df['matched'].sum():,}/{len(matches_df):,}")
    print(f"[{now_utc()}] Unresolved names written to: {out_dir / 'unresolved_names.csv'}")

    all_manifest: List[ManifestRow] = []
    all_field_locator_rows: List[FieldLocatorRow] = []
    llm_payloads: List[Dict[str, Any]] = []
    llm_results: List[Dict[str, Any]] = []
    close_dates8 = load_close_dates(args.close_dates)
    if close_dates8:
        print(f"[{now_utc()}] Loaded {len(close_dates8):,} CRSP close dates for the realized-results anchor")
    all_close_by_event: Dict[str, pd.Timestamp] = {}
    save_documents = not args.no_save_documents
    collect_llm_documents = args.llm_stage in {"prepare", "send", "batch"}
    if args.llm_stage != "off":
        print(f"[{now_utc()}] LLM stage: {args.llm_stage}; save_documents={save_documents}")

    for j, (_, event) in enumerate(udf.iterrows(), start=1):
        event_idx = int(event["orig_row_idx"])
        if j % 20 == 1 or j == len(udf):
            print(f"[{now_utc()}] Processing event {j}/{len(udf)}: {event.get('Target Name')} / {event.get('Acquirer Name')}")

        try:
            result = process_event(
                event_idx=event_idx,
                event=event,
                cik_matches=cik_matches,
                client=client,
                out_dir=out_dir,
                forms=args.forms,
                keywords=args.keywords,
                pre_days=args.pre_days,
                post_days=args.post_days,
                download_exhibits=args.download_exhibits,
                dry_run=args.dry_run,
                resume=args.resume,
                max_docs_per_event_side=args.max_docs_per_event_side,
                save_documents=save_documents,
                collect_llm_documents=collect_llm_documents,
                llm_max_doc_chars=args.llm_max_doc_chars,
                field_specs=field_specs,
                min_field_score=args.min_field_score,
            )
            rows = result.rows
            all_manifest.extend(rows)
            all_field_locator_rows.extend(result.field_locator_rows)

            if collect_llm_documents and rows and result.field_locator_rows:
                event_locator_df = pd.DataFrame([asdict(r) for r in result.field_locator_rows])
                event_locator_df["field_score"] = pd.to_numeric(event_locator_df["field_score"], errors="coerce").fillna(0).astype(int)
                event_close = None
                if rows:
                    eid = str(rows[0].event_id)
                    # Authoritative CRSP close date keyed by target CUSIP; fall back to the
                    # target's-last-filing estimate when the target has no CRSP coverage.
                    ec = close_dates8.get(_cusip8(event.get("Target cusip"))) if close_dates8 else None
                    if ec is None:
                        ec = estimate_close_date(rows)
                    event_close = {eid: ec}
                    all_close_by_event[eid] = ec
                event_locator_df = trim_field_locator(event_locator_df, top_k=args.field_locator_top_k, close_est_by_event=event_close)
                event_selected_docs_df = build_selected_upload_docs(event_locator_df, max_docs_per_event=args.claude_package_max_docs_per_event)
                event_payloads = build_claude_field_payloads(
                    event_locator_df,
                    event_selected_docs_df,
                    field_specs,
                    max_fields_per_event_field=args.field_locator_top_k if args.field_locator_top_k > 0 else 3,
                    docs=result.llm_documents,
                )
                for llm_payload in event_payloads:
                    llm_payloads.append(llm_payload)
                    if args.llm_stage == "send":
                        if args.llm_provider != "anthropic":
                            raise ValueError(f"Unsupported LLM provider: {args.llm_provider}")
                        api_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
                        llm_result = call_anthropic(
                            llm_payload=llm_payload,
                            api_key=api_key,
                            model=args.llm_model,
                            max_tokens=args.llm_max_tokens,
                        )
                        llm_results.append(llm_result)

        except Exception as e:
            print(f"[ERROR] event {event_idx} failed: {e}", file=sys.stderr)

        # Incremental manifest every 25 events.
        if j % 25 == 0:
            pd.DataFrame([asdict(r) for r in all_manifest]).to_csv(out_dir / "manifest_sec_filings_partial.csv", index=False)
            if all_field_locator_rows:
                pd.DataFrame([asdict(r) for r in all_field_locator_rows]).to_csv(out_dir / "field_locator_partial.csv", index=False)
            if llm_payloads:
                write_jsonl(out_dir / "llm_field_payloads_partial.jsonl", llm_payloads)
            if llm_results:
                write_jsonl(out_dir / "llm_field_results_partial.jsonl", llm_results)

    manifest_df = pd.DataFrame([asdict(r) for r in all_manifest])
    manifest_path = out_dir / "manifest_sec_filings.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"[{now_utc()}] Manifest rows: {len(manifest_df):,}")
    print(f"[{now_utc()}] Wrote: {manifest_path}")

    if all_field_locator_rows:
        field_locator_df = pd.DataFrame([asdict(r) for r in all_field_locator_rows])
        field_locator_df["field_score"] = pd.to_numeric(field_locator_df["field_score"], errors="coerce").fillna(0).astype(int)
        # Reuse the per-event close dates already resolved during processing (CRSP where
        # available, filing-based estimate otherwise); backfill any gaps from the manifest.
        close_by_event: Dict[str, pd.Timestamp] = dict(all_close_by_event)
        for _eid, _rs in itertools.groupby(sorted(all_manifest, key=lambda r: r.event_id), key=lambda r: r.event_id):
            if str(_eid) not in close_by_event or close_by_event[str(_eid)] is None:
                close_by_event[str(_eid)] = estimate_close_date(list(_rs))
        field_locator_df = trim_field_locator(field_locator_df, top_k=args.field_locator_top_k, close_est_by_event=close_by_event)
        field_locator_path = out_dir / "field_locator.csv"
        field_locator_df.to_csv(field_locator_path, index=False)
        print(f"[{now_utc()}] Wrote field locator: {field_locator_path} ({len(field_locator_df):,} rows)")

        coverage_df = build_event_field_coverage(udf, field_locator_df, field_specs)
        coverage_path = out_dir / "event_field_coverage.csv"
        coverage_df.to_csv(coverage_path, index=False)
        print(f"[{now_utc()}] Wrote event-field coverage: {coverage_path}")

        selected_docs_df = build_selected_upload_docs(field_locator_df, max_docs_per_event=args.claude_package_max_docs_per_event)
        selected_docs_path = out_dir / "selected_upload_docs.csv"
        selected_docs_df.to_csv(selected_docs_path, index=False)
        print(f"[{now_utc()}] Wrote selected Claude-upload docs: {selected_docs_path}")

        field_payloads = build_claude_field_payloads(
            field_locator_df,
            selected_docs_df,
            field_specs,
            max_fields_per_event_field=args.field_locator_top_k if args.field_locator_top_k > 0 else 3,
        )
        claude_field_payload_path = out_dir / "claude_field_payloads.jsonl"
        write_jsonl(claude_field_payload_path, field_payloads)
        print(f"[{now_utc()}] Wrote Claude field payloads: {claude_field_payload_path}")

        if args.make_claude_packages:
            write_claude_upload_packages(out_dir, field_payloads, selected_docs_df)
            print(f"[{now_utc()}] Wrote Claude upload packages: {out_dir / 'claude_upload_packages'}")

    if llm_payloads:
        llm_payload_path = out_dir / "llm_field_payloads.jsonl"
        write_jsonl(llm_payload_path, llm_payloads)
        print(f"[{now_utc()}] Wrote LLM payloads: {llm_payload_path}")

    # Batch stage: all payloads are built and downloads are done — submit ONE Message Batch.
    if args.llm_stage == "batch" and llm_payloads:
        api_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        llm_results = run_anthropic_batch(
            llm_payloads, api_key=api_key, model=args.llm_model, max_tokens=args.llm_max_tokens,
            max_cost_usd=args.max_batch_cost_usd, poll_seconds=args.batch_poll_seconds,
        )

    if llm_results:
        llm_results_path = out_dir / "llm_field_results.jsonl"
        write_jsonl(llm_results_path, llm_results)
        llm_summary_path = out_dir / "llm_field_extractions.csv"
        flatten_llm_records(llm_results).to_csv(llm_summary_path, index=False)
        print(f"[{now_utc()}] Wrote LLM raw results: {llm_results_path}")
        print(f"[{now_utc()}] Wrote LLM extraction summary: {llm_summary_path}")

    if not manifest_df.empty:
        # Create a high-priority review list.
        review = manifest_df.copy()
        review["keyword_score"] = pd.to_numeric(review["keyword_score"], errors="coerce").fillna(0)
        review = review.sort_values(["event_id", "keyword_score", "filing_date"], ascending=[True, False, True])
        top_review = review[review["keyword_score"] > 0].copy()
        top_review.to_csv(out_dir / "review_priority_keyword_hits.csv", index=False)
        print(f"[{now_utc()}] Wrote keyword-hit review file: {out_dir / 'review_priority_keyword_hits.csv'}")

        # Event-level summary.
        summary = (
            manifest_df.assign(keyword_score=pd.to_numeric(manifest_df["keyword_score"], errors="coerce").fillna(0))
            .groupby(["event_id", "target_name", "acquirer_name", "announce_date", "payment_type", "deal_status"], dropna=False)
            .agg(
                docs=("document_name", "count"),
                downloaded=("downloaded", "sum"),
                max_keyword_score=("keyword_score", "max"),
                total_keyword_score=("keyword_score", "sum"),
                forms=("form", lambda x: ";".join(sorted(set(map(str, x))))),
                keywords=("keywords_found", lambda x: ";".join(sorted(set(";".join(map(str, x)).split(";")) - {""}))),
            )
            .reset_index()
            .sort_values(["max_keyword_score", "total_keyword_score"], ascending=False)
        )
        summary.to_csv(out_dir / "event_level_summary.csv", index=False)
        print(f"[{now_utc()}] Wrote event summary: {out_dir / 'event_level_summary.csv'}")


if __name__ == "__main__":
    main()
