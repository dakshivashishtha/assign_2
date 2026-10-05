#!/usr/bin/env python

from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))  # silences the Windows core-count warning

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn

__all__ = ["MarketingCampaignPipeline", "PipelineConfig", "EXAMPLE_ROWS"]

log = logging.getLogger("campaign_pipeline")

REQUIRED_NUMERIC = ["Duration", "Impressions", "Clicks", "Leads", "Conversions",
                    "Acquisition_Cost", "Engagement_Score"]
OPTIONAL_NUMERIC = ["Revenue", "ROI"]
REQUIRED_CATEGORICAL = ["Campaign_Type", "Target_Audience", "Language", "Customer_Segment", "Channel_Used"]
ALL_REQUIRED = REQUIRED_NUMERIC + REQUIRED_CATEGORICAL + ["Date"]

KNOWN_LEVELS = {
    "Campaign_Type":    ["Email", "Influencer", "Paid Ads", "SEO", "Social Media"],
    "Target_Audience":  ["College Students", "Premium Shoppers", "Tier 2 City Customers", "Working Women", "Youth"],
    "Language":         ["Bengali", "English", "Hindi", "Tamil"],
    "Customer_Segment": ["College Students", "Premium Shoppers", "Tier 2 City Customers", "Working Women", "Youth"],
}
KNOWN_CHANNELS = ["Email", "Facebook", "Google", "Instagram", "WhatsApp", "YouTube"]

TIER_LABELS = ["Low Performing", "Medium Performing", "High Performing"]
DEFAULT_TIER_EDGES = (1.374, 3.583)       # ROAS tertile cut-points printed in Notebook 5 (fallback only)
STAT_THRESHOLD = 3.5                       # Notebook 7 robust-z threshold


ANOMALY_RULES = {
    "CPC":             {"direction": "high", "label": "Sudden increase in CPC"},
    "CTR":             {"direction": "low",  "label": "Significant CTR decrease"},
    "Conversion_Rate": {"direction": "low",  "label": "Unexpected conversion drop"},
    "Spend":           {"direction": "both", "label": "Abnormal advertising spend"},
    "ROAS":            {"direction": "both", "label": "Unusual ROAS"},
}
ANOMALY_ACTIONS = {
    "Sudden increase in CPC":     "check bids, targeting and audience overlap; consider a CPC cap",
    "Significant CTR decrease":   "refresh creatives/copy and check for audience fatigue",
    "Unexpected conversion drop": "audit landing page, tracking pixels and the checkout funnel",
    "Abnormal advertising spend": "verify budget caps, pacing and billing",
    "Unusual ROAS":               "validate revenue attribution before acting on this campaign",
    "Unusual combination of metrics": "review the campaign manually; no single metric explains the flag",
}
TIER_ACTIONS = {
    "High Performing":   "protect budget and consider scaling",
    "Medium Performing": "optimise creative/targeting before scaling",
    "Low Performing":    "diagnose funnel drop-off and reduce spend until fixed",
}

F_REGRESSOR = "best_regressor_revenue.joblib"
F_CLASSIFIER = "best_classifier_performance_tier.joblib"
F_LABEL_ENCODER = "performance_tier_label_encoder.joblib"
F_SEGMENTATION = "audience_segmentation_artifact.joblib"
F_ANOMALY_MODEL = "anomaly_isolation_forest.joblib"
F_ANOMALY_FEATURES = "anomaly_detection_features.joblib"
F_REFERENCE = "pipeline_reference.joblib"


EXAMPLE_ROWS = pd.DataFrame([
    ["NY-CMP-1000", "Social Media", "College Students", 21, "WhatsApp, YouTube", 57804, 6156, 3616, 2355, 1867515, 111.03, "Hindi", 20.98, "College Students", "29-04-2025"],
    ["NY-CMP-1001", "Paid Ads", "Tier 2 City Customers", 18, "YouTube", 91801, 3321, 1971, 1357, 1046247, 180.83, "Hindi", 7.24, "College Students", "06-04-2025"],
    ["NY-CMP-1002", "Influencer", "Youth", 23, "WhatsApp, Google, YouTube", 15536, 2182, 952, 755, 197055, 90.60, "English", 25.03, "College Students", "14-01-2025"],
    ["NY-CMP-1003", "Email", "Working Women", 18, "YouTube, Facebook, Instagram", 88114, 8413, 2231, 947, 376906, 249.07, "Hindi", 13.15, "College Students", "04-06-2025"],
    ["NY-CMP-1004", "Paid Ads", "College Students", 10, "Facebook, Instagram", 96871, 3743, 2060, 1258, 518296, 228.60, "Hindi", 7.29, "Tier 2 City Customers", "29-12-2024"],
], columns=["Campaign_ID", "Campaign_Type", "Target_Audience", "Duration", "Channel_Used", "Impressions",
            "Clicks", "Leads", "Conversions", "Revenue", "Acquisition_Cost", "Language",
            "Engagement_Score", "Customer_Segment", "Date"])



@dataclass
class PipelineConfig:
    models_dir: Optional[Path] = None
    reference_path: Optional[Path] = None
    channel_encoding: str = "legacy"       # "legacy" (matches trained models) or "correct"
    stat_threshold: float = STAT_THRESHOLD
    dayfirst: bool = True                   # fallback date parsing; primary format is dd-mm-YYYY


class _Notes:
    """Collects per-row data-quality / skip notes by row position."""

    def __init__(self, n: int):
        self._rows: List[List[str]] = [[] for _ in range(n)]

    def add(self, mask, text: str) -> None:
        for i in np.flatnonzero(np.asarray(mask, dtype=bool)):
            if text not in self._rows[i]:
                self._rows[i].append(text)

    def to_series(self, index) -> pd.Series:
        return pd.Series(["; ".join(r) for r in self._rows], index=index, dtype=object)


def _ratio(num: pd.Series, den: pd.Series, scale: float = 1.0) -> pd.Series:
    """num/den*scale; non-positive or missing denominators give NaN (never inf) - same rule as Notebook 3."""
    n = num.to_numpy(dtype=float)
    d = den.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(d > 0, n / np.where(d > 0, d, 1.0), np.nan) * scale
    return pd.Series(out, index=num.index)


def _robust_params(series: pd.Series) -> Dict[str, float]:
    """Median and MAD with the same degenerate-MAD fallback used in Notebook 7."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    median = float(s.median())
    mad = float((s - median).abs().median())
    if mad == 0:
        mad = float((s - median).abs().mean())
    return {"median": median, "mad": mad}


def _robust_z(values: np.ndarray, median: float, mad: float) -> np.ndarray:
    if mad == 0:
        return np.zeros(len(values))
    return 0.6745 * (values - median) / mad


def _find_models_dir(explicit: Optional[Path]) -> Path:
    if explicit is not None:                      # an explicit path must be right - never fall back silently
        explicit = Path(explicit)
        if not (explicit / F_REGRESSOR).exists():
            raise FileNotFoundError(f"models_dir '{explicit}' does not contain {F_REGRESSOR} "
                                    f"(and the other trained .joblib files from Notebooks 4-7).")
        return explicit
    here = Path(__file__).resolve().parent
    candidates = []
    env = os.environ.get("MCI_MODELS_DIR")
    if env:
        candidates.append(Path(env))
    candidates += [Path.cwd() / "models", Path.cwd().parent / "models", here / "models", here.parent / "models"]
    for c in candidates:
        if c is not None and Path(c).is_dir() and (Path(c) / F_REGRESSOR).exists():
            return Path(c)
    tried = "\n  ".join(str(c) for c in candidates if c is not None)
    raise FileNotFoundError(
        f"Could not find the trained models ({F_REGRESSOR} etc.). Pass --models-dir / models_dir=..., "
        f"or set MCI_MODELS_DIR. Looked in:\n  {tried}")



class MarketingCampaignPipeline:

    def __init__(self, models_dir: Optional[Path] = None, reference_path: Optional[Path] = None,
                 channel_encoding: str = "legacy", config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig(models_dir=models_dir, reference_path=reference_path,
                                               channel_encoding=channel_encoding)
        if self.config.channel_encoding not in ("legacy", "correct"):
            raise ValueError("channel_encoding must be 'legacy' or 'correct'")
        self.models_dir = _find_models_dir(self.config.models_dir)
        self.reference_path = Path(self.config.reference_path) if self.config.reference_path \
            else self.models_dir / F_REFERENCE
        self.last_warnings: List[str] = []
        self._load_warnings: List[str] = []
        self._load_artifacts()

    
    def _load_artifacts(self) -> None:
        m = self.models_dir
        log.info("Loading models from %s", m)
        self.regressor = joblib.load(m / F_REGRESSOR)
        self.classifier = joblib.load(m / F_CLASSIFIER)
        self.label_encoder = joblib.load(m / F_LABEL_ENCODER)
        self.segmentation = joblib.load(m / F_SEGMENTATION)
        self.iso_forest = joblib.load(m / F_ANOMALY_MODEL)
        self.anomaly_features: List[str] = list(joblib.load(m / F_ANOMALY_FEATURES))

        trained_with = str(self.segmentation.get("sklearn_version", ""))
        if trained_with and trained_with.split(".")[:2] != sklearn.__version__.split(".")[:2]:
            msg = (f"Models were trained with scikit-learn {trained_with} but {sklearn.__version__} is installed. "
                   f"If scoring fails or results look odd, run: pip install scikit-learn=={trained_with}")
            log.warning(msg)
            self._load_warnings.append(msg)

        self.reference: Optional[dict] = None
        if self.reference_path.exists():
            self.reference = joblib.load(self.reference_path)
            missing = [f for f in self.anomaly_features if f not in self.reference.get("anomaly_reference", {})]
            if missing:
                log.warning("Reference file lacks statistics for %s - rebuild it with build-reference.", missing)
                self.reference = None
        self.tier_edges = tuple(self.reference["tier_edges"]) if self.reference else DEFAULT_TIER_EDGES

    def build_reference(self, train_features_csv: Path) -> Path:
        
        train = pd.read_csv(train_features_csv)
        missing = [c for c in self.anomaly_features + ["ROAS"] if c not in train.columns]
        if missing:
            raise ValueError(f"Training file is missing columns {missing}; use the Notebook 3 features file.")
        ref = {
            "anomaly_reference": {f: _robust_params(train[f]) for f in self.anomaly_features},
            "tier_edges": [float(x) for x in pd.qcut(train["ROAS"], q=3, retbins=True)[1][1:3]],
            "stat_threshold": STAT_THRESHOLD,
            "n_training_rows": int(len(train)),
            "built_from": str(train_features_csv),
            "built_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.reference_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(ref, self.reference_path)
        self.reference = ref
        self.tier_edges = tuple(ref["tier_edges"])
        log.info("Reference saved to %s (tier edges %.3f / %.3f)", self.reference_path, *self.tier_edges)
        return self.reference_path

    def _parse_dates(self, s: pd.Series) -> pd.Series:
        if pd.api.types.is_datetime64_any_dtype(s):
            return s
        s = s.astype(object)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")
            miss = out.isna() & s.notna()
            if miss.any():
                try:
                    alt = pd.to_datetime(s[miss], format="ISO8601", errors="coerce")
                except (ValueError, TypeError):
                    alt = pd.to_datetime(s[miss], errors="coerce", dayfirst=self.config.dayfirst)
                out = out.fillna(alt)
            still = out.isna() & s.notna()
            if still.any():
                out = out.fillna(pd.to_datetime(s[still], errors="coerce", dayfirst=self.config.dayfirst))
        return out

    def validate_input(self, raw: pd.DataFrame, notes: _Notes):
        """Clean types, flag problems, and return (clean_frame, scorable_mask)."""
        missing_cols = [c for c in ALL_REQUIRED if c not in raw.columns]
        if missing_cols:
            raise ValueError(f"Input is missing required column(s): {missing_cols}. "
                             f"Run the 'template' command to see the expected layout.")
        d = raw.reset_index(drop=True).copy()
        if "Campaign_ID" not in d.columns:
            d["Campaign_ID"] = [f"ROW-{i + 1}" for i in range(len(d))]
        valid = np.ones(len(d), dtype=bool)

        for col in REQUIRED_NUMERIC + OPTIONAL_NUMERIC:
            if col not in d.columns:
                d[col] = np.nan                       # only optional columns can be absent here
                continue
            original = d[col]
            num = pd.to_numeric(original, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
            bad_type = (original.notna() & num.isna()).to_numpy()
            notes.add(bad_type, f"Non-numeric {col}")
            d[col] = num
            if col in REQUIRED_NUMERIC:
                missing = num.isna().to_numpy()
                negative = (num < 0).to_numpy()
                notes.add(missing & ~bad_type, f"Missing {col}")
                notes.add(negative, f"Negative {col}")
                valid &= ~(missing | negative)
            elif col == "Revenue":
                neg_rev = (num < 0).to_numpy()
                notes.add(neg_rev, "Negative Revenue ignored")
                d.loc[neg_rev, "Revenue"] = np.nan

        d["Date"] = self._parse_dates(d["Date"])
        bad_date = d["Date"].isna().to_numpy()
        notes.add(bad_date, "Missing or unparseable Date (expected dd-mm-YYYY)")
        valid &= ~bad_date

        for col in REQUIRED_CATEGORICAL:
            d[col] = d[col].map(lambda v: v.strip() if isinstance(v, str) else v)
            empty = d[col].isna().to_numpy() | (d[col].astype(str).str.len() == 0).to_numpy()
            notes.add(empty, f"Missing {col}")
            valid &= ~empty
        for col, levels in KNOWN_LEVELS.items():
            unseen = (~d[col].isin(levels) & d[col].notna()).to_numpy()
            notes.add(unseen, f"Unseen {col} value - encoded like the baseline level")

        # logical consistency (scored anyway, but flagged)
        notes.add((d["Clicks"] > d["Impressions"]).to_numpy(), "Clicks > Impressions")
        notes.add((d["Leads"] > d["Clicks"]).to_numpy(), "Leads > Clicks")
        notes.add((d["Conversions"] > d["Leads"]).to_numpy(), "Conversions > Leads")
        notes.add(d["Campaign_ID"].duplicated(keep=False).to_numpy(), "Duplicate Campaign_ID")
        return d, valid

    # ------------------------------------------------------------------ 2. feature engineering
    def engineer_features(self, d: pd.DataFrame, notes: Optional[_Notes] = None) -> pd.DataFrame:
        """Reproduces Notebooks 2-3: Spend, KPIs, intensity, date, channel and category features."""
        f = d.copy()
        f["Spend"] = f["Acquisition_Cost"] * f["Conversions"]           # reconstructed, see module docstring
        f["CTR"] = _ratio(f["Clicks"], f["Impressions"], 100)
        f["Conversion_Rate"] = _ratio(f["Conversions"], f["Clicks"], 100)
        f["CPC"] = _ratio(f["Spend"], f["Clicks"])
        f["CPA"] = _ratio(f["Spend"], f["Conversions"])
        f["ROAS"] = _ratio(f["Revenue"], f["Spend"])                    # NaN when Revenue is not supplied
        f["CPL"] = _ratio(f["Spend"], f["Leads"])
        f["Lead_Rate"] = _ratio(f["Leads"], f["Clicks"], 100)
        f["Lead_to_Conversion_Rate"] = _ratio(f["Conversions"], f["Leads"], 100)
        f["Overall_Funnel_Efficiency"] = _ratio(f["Conversions"], f["Impressions"], 100)

        for col in ["Impressions", "Clicks", "Leads", "Conversions", "Spend"]:
            f[f"{col}_per_Day"] = _ratio(f[col], f["Duration"])

        f["Year"] = f["Date"].dt.year
        f["Month"] = f["Date"].dt.month
        f["Quarter"] = f["Date"].dt.quarter
        f["Day_of_Week"] = f["Date"].dt.dayofweek
        f["Is_Weekend"] = (f["Day_of_Week"] >= 5).astype(int)

        channels = f["Channel_Used"].fillna("").astype(str).str.split(",") \
            .map(lambda parts: [p.strip() for p in parts if p.strip()])
        f["Channel_Count"] = channels.map(len)
        f["Is_Multichannel"] = (f["Channel_Count"] > 1).astype(int)
        legacy = self.config.channel_encoding == "legacy"
        for ch in KNOWN_CHANNELS:
            if legacy:      # reproduces Notebook 3 as trained: flag only when the channel is NOT listed first
                f[f"Channel_{ch}"] = channels.map(lambda p, ch=ch: int(ch in p[1:]))
            else:
                f[f"Channel_{ch}"] = channels.map(lambda p, ch=ch: int(ch in p))
        if notes is not None:
            unseen = channels.map(lambda p: any(c not in KNOWN_CHANNELS for c in p)).to_numpy()
            notes.add(unseen, "Unseen channel name in Channel_Used")

        for col, levels in KNOWN_LEVELS.items():
            for lv in levels:
                f[f"{col}_{lv}"] = (f[col] == lv).astype(int)
        return f

    @staticmethod
    def _select(feats: pd.DataFrame, cols: List[str], who: str) -> pd.DataFrame:
        missing = [c for c in cols if c not in feats.columns]
        if missing:
            raise RuntimeError(f"{who} expects feature(s) the pipeline cannot build: {missing}. "
                               f"The model was probably retrained with new features - update engineer_features().")
        return feats[cols].astype(float)

    @staticmethod
    def _ok_rows(X: pd.DataFrame, valid: np.ndarray, notes: _Notes, who: str) -> np.ndarray:
        finite = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
        # one shared message so a row skipped by several models carries a single note
        notes.add(valid & ~finite, "Not scored by one or more models: non-finite KPI inputs "
                                   "(e.g. zero clicks, leads or duration)")
        return valid & finite

    def _predict_revenue(self, feats, valid, notes) -> np.ndarray:
        X = self._select(feats, list(self.regressor.feature_names_in_), "Revenue model")
        ok = self._ok_rows(X, valid, notes, "Revenue model")
        pred = np.full(len(X), np.nan)
        if ok.any():
            pred[ok] = np.clip(self.regressor.predict(X.loc[ok]), 0, None)
        return pred

    # b. performance tier
    def _classify(self, feats, valid, notes):
        X = self._select(feats, list(self.classifier.feature_names_in_), "Tier classifier")
        ok = self._ok_rows(X, valid, notes, "Tier classifier")
        classes = self.label_encoder.inverse_transform(np.asarray(self.classifier.classes_))
        labels = np.full(len(X), None, dtype=object)
        proba = np.full((len(X), len(classes)), np.nan)
        if ok.any():
            p = self.classifier.predict_proba(X.loc[ok])
            proba[ok] = p
            labels[ok] = classes[p.argmax(axis=1)]
        return labels, proba, list(classes)

    def _tier_from_roas(self, roas: pd.Series) -> pd.Series:
        e1, e2 = self.tier_edges
        tier = pd.Series(np.select([roas <= e1, roas <= e2], ["Low Performing", "Medium Performing"],
                                   default="High Performing"), index=roas.index, dtype=object)
        return tier.where(roas.notna(), None)

    
    def _segment(self, feats, valid, notes):
        art = self.segmentation
        pipe = art["pipeline"]
        X = self._select(feats, list(art["features"]), "Segmentation model")
        ok = self._ok_rows(X, valid, notes, "Segmentation model")
        cluster = np.full(len(X), np.nan)
        margin = np.full(len(X), np.nan)
        names = np.full(len(X), None, dtype=object)
        actions = np.full(len(X), None, dtype=object)
        if ok.any():
            Xo = X.loc[ok]
            try:
                c = pipe.predict(Xo)
            except AttributeError as exc:          # typical symptom of a scikit-learn version mismatch
                raise RuntimeError(
                    f"Segmentation model could not run ({exc}). It was saved with scikit-learn "
                    f"{art.get('sklearn_version', '?')}; installed is {sklearn.__version__}. "
                    f"Fix: pip install scikit-learn=={art.get('sklearn_version', '1.7.1')}") from exc
            dist = np.sort(pipe["kmeans"].transform(pipe[:-1].transform(Xo)), axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                m = 1 - dist[:, 0] / np.where(dist[:, 1] > 0, dist[:, 1], np.nan)
            cluster[ok], margin[ok] = c, m
            seg = [art["cluster_to_segment"][int(i)] for i in c]
            names[ok] = seg
            actions[ok] = [art["segment_actions"].get(s.split(" (C")[0], "") for s in seg]
        return cluster, names, actions, margin

    # ------------------------------------------------------------------ 3d. anomalies
    def _detect_anomalies(self, feats, valid, notes, predicted_revenue):
        """Statistical robust-z check + Isolation Forest, combined as in Notebook 7.

        ROAS is needed by the anomaly model. If Revenue was supplied the actual ROAS is used;
        otherwise ROAS = predicted revenue / Spend and ROAS_Source says so.
        """
        n = len(feats)
        spend = feats["Spend"]
        roas_actual = feats["ROAS"]
        roas_pred = _ratio(pd.Series(predicted_revenue, index=feats.index), spend)
        roas_used = roas_actual.where(roas_actual.notna(), roas_pred)
        source = pd.Series(np.where(roas_actual.notna(), "actual",
                                    np.where(roas_pred.notna(), "predicted", None)), index=feats.index, dtype=object)

        A = feats[self.anomaly_features].copy()
        if "ROAS" in A.columns:
            A["ROAS"] = roas_used
        X = A.astype(float)
        ok = self._ok_rows(X, valid, notes, "Anomaly detection")

        # --- Isolation Forest
        if_flag = np.zeros(n, dtype=bool)
        if_score = np.full(n, np.nan)
        if ok.any():
            Xo = X.loc[ok]
            if_flag[ok] = self.iso_forest.predict(Xo) == -1
            if_score[ok] = -self.iso_forest.score_samples(Xo)

        # --- statistical robust z-scores (need training reference)
        z = pd.DataFrame(np.nan, index=feats.index, columns=[f"{c}_z" for c in self.anomaly_features])
        stat_available = self.reference is not None
        stat_flag = np.zeros(n, dtype=bool)
        reasons = np.full(n, "None", dtype=object)
        if stat_available:
            ref = self.reference["anomaly_reference"]
            for c in self.anomaly_features:
                z[f"{c}_z"] = _robust_z(X[c].to_numpy(dtype=float), ref[c]["median"], ref[c]["mad"])
            z.loc[~ok, :] = np.nan
            thr = self.config.stat_threshold
            stat_flag = (z.abs() > thr).any(axis=1).to_numpy()
            for i in np.flatnonzero(stat_flag):
                hits = []
                for metric, rule in ANOMALY_RULES.items():
                    if metric not in self.anomaly_features:
                        continue
                    v = z.iloc[i][f"{metric}_z"]
                    if (rule["direction"] == "high" and v > thr) or (rule["direction"] == "low" and v < -thr) \
                            or (rule["direction"] == "both" and abs(v) > thr):
                        hits.append((abs(v), rule["label"]))
                if not hits:       # flagged by an |z| in a direction no rule names
                    worst = z.iloc[i].abs().idxmax().replace("_z", "")
                    hits.append((0, f"Unusual {worst}"))
                reasons[i] = "; ".join(lbl for _, lbl in sorted(hits, reverse=True))
        else:
            msg = ("No pipeline_reference.joblib found - statistical anomaly check skipped (Isolation Forest only). "
                   "Run: python marketing_campaign_pipeline.py build-reference --train-data <features.csv>")
            self.last_warnings.append(msg)
            log.warning(msg)

        only_if = if_flag & ~stat_flag
        reasons[only_if] = "Unusual combination of metrics"
        confidence = np.select([stat_flag & if_flag, stat_flag | if_flag],
                               ["High (both methods)", "Medium (one method)"], default="None")
        confidence = np.where(ok, confidence, None)
        reasons = np.where(ok, reasons, None)
        return pd.DataFrame({
            "ROAS_Used_For_Anomaly": roas_used, "ROAS_Source": source,
            "Statistical_Anomaly_Flag": np.where(ok, stat_flag, None) if stat_available else None,
            "Statistical_Max_Abs_Z": z.abs().max(axis=1) if stat_available else np.nan,
            "IF_Anomaly_Flag": np.where(ok, if_flag, None), "IF_Anomaly_Score": if_score,
            "Anomaly_Confidence": confidence, "Anomaly_Reason": reasons,
        }, index=feats.index)

    @staticmethod
    def _insight(r: pd.Series) -> str:
        if pd.isna(r.get("Predicted_Revenue")) and r.get("Predicted_Performance_Tier") is None:
            return "Not scored - see Pipeline_Notes."
        bits = []
        tier = r.get("Predicted_Performance_Tier")
        if tier:
            bits.append(f"{tier} ({r['Tier_Confidence']:.0%} confidence): {TIER_ACTIONS.get(tier, '')}.")
        seg = r.get("Audience_Segment")
        if seg:
            bits.append(f"Segment '{seg}': {r['Segment_Action']}")
        if pd.notna(r.get("Predicted_Revenue")):
            line = f"Estimated revenue {r['Predicted_Revenue']:,.0f}"
            if pd.notna(r.get("Predicted_ROAS")):
                line += f" (ROAS about {r['Predicted_ROAS']:.2f})"
            bits.append(line + ".")
        conf = r.get("Anomaly_Confidence")
        if conf and conf != "None":
            primary = str(r["Anomaly_Reason"]).split("; ")[0]
            bits.append(f"ANOMALY [{conf}] - {r['Anomaly_Reason']}: {ANOMALY_ACTIONS.get(primary, 'review manually')}.")
        return " ".join(bits)

    def run(self, raw: pd.DataFrame, full_output: bool = False) -> pd.DataFrame:
        """Score a raw campaign DataFrame. Returns one result row per input row (same index)."""
        self.last_warnings = list(self._load_warnings)
        if raw is None or len(raw) == 0:
            raise ValueError("Input has no rows.")
        notes = _Notes(len(raw))
        clean, valid = self.validate_input(raw, notes)
        feats = self.engineer_features(clean, notes)

        pred_rev = self._predict_revenue(feats, valid, notes)
        tier, proba, classes = self._classify(feats, valid, notes)
        cluster, seg_name, seg_action, seg_margin = self._segment(feats, valid, notes)
        anomalies = self._detect_anomalies(feats, valid, notes, pred_rev)

        out = pd.DataFrame(index=feats.index)
        out["Campaign_ID"] = feats["Campaign_ID"]
        for c in ["Campaign_Type", "Channel_Used", "Spend", "CTR", "Conversion_Rate", "CPC", "CPA"]:
            out[c] = feats[c]
        out["Predicted_Revenue"] = pred_rev
        out["Predicted_ROAS"] = _ratio(pd.Series(pred_rev, index=feats.index), feats["Spend"])
        out["Predicted_Performance_Tier"] = tier
        for j, cls in enumerate(classes):
            out[f"Prob_{cls.replace(' ', '_')}"] = proba[:, j]
        out["Tier_Confidence"] = np.nanmax(np.where(np.isnan(proba), -1, proba), axis=1)
        out.loc[np.isnan(proba).all(axis=1), "Tier_Confidence"] = np.nan
        if feats["Revenue"].notna().any():
            out["Actual_Revenue"] = feats["Revenue"]
            out["Actual_ROAS"] = feats["ROAS"]
            out["Actual_Performance_Tier"] = self._tier_from_roas(feats["ROAS"])
            out["Tier_Matches_Actual"] = np.where(out["Actual_Performance_Tier"].isna() | out["Predicted_Performance_Tier"].isna(),
                                                  None, out["Actual_Performance_Tier"] == out["Predicted_Performance_Tier"])
        out["Cluster"] = cluster
        out["Audience_Segment"] = seg_name
        out["Segment_Action"] = seg_action
        out["Assignment_Margin"] = np.round(seg_margin, 3)
        out = pd.concat([out, anomalies], axis=1)
        out["Marketing_Insight"] = out.apply(self._insight, axis=1)
        out["Pipeline_Notes"] = notes.to_series(feats.index)

        if full_output:
            extra = [c for c in feats.columns if c not in out.columns]
            out = pd.concat([out, feats[extra]], axis=1)
        out.index = raw.index
        return out

    def summarize(self, res: pd.DataFrame) -> str:
        """Plain-text batch summary for the console or a report."""
        n = len(res)
        scored = int(res["Predicted_Performance_Tier"].notna().sum())
        L = [f"Campaigns received: {n}   scored by all models: {scored}   not fully scored: {n - scored}"]
        if scored:
            L.append("\nPredicted performance tier:")
            L += [f"  {k:<20}{v:>6}" for k, v in res["Predicted_Performance_Tier"].value_counts().items()]
            L.append("\nAudience segments:")
            L += [f"  {k:<28}{v:>6}" for k, v in res["Audience_Segment"].value_counts().items()]
            L.append(f"\nEstimated total revenue: {res['Predicted_Revenue'].sum():,.0f}")
            conf = res["Anomaly_Confidence"].value_counts()
            L.append("\nAnomalies: " + (", ".join(f"{k}: {v}" for k, v in conf.items()) or "none"))
            flagged = res[res["Anomaly_Confidence"].isin(["High (both methods)", "Medium (one method)"])]
            if len(flagged):
                L.append("  Reasons: " + ", ".join(
                    f"{k} x{v}" for k, v in flagged["Anomaly_Reason"].str.split("; ").explode().value_counts().items()))
            if "Tier_Matches_Actual" in res:
                m = res["Tier_Matches_Actual"].dropna()
                if len(m):
                    L.append(f"\nPredicted tier matches actual ROAS tier for {m.astype(bool).mean():.0%} of {len(m)} campaigns "
                             f"(ROAS source for anomaly check: actual where Revenue given, otherwise predicted).")
        noted = int((res["Pipeline_Notes"] != "").sum())
        if noted:
            L.append(f"\n{noted} row(s) carry data-quality notes - see the Pipeline_Notes column.")
        for w in self.last_warnings:
            L.append(f"\nWARNING: {w}")
        return "\n".join(L)

    def self_check(self) -> pd.DataFrame:
        """Run the five built-in example rows through every stage; raises if any model cannot be fed."""
        res = self.run(EXAMPLE_ROWS)
        if res["Predicted_Performance_Tier"].isna().any() or res["Audience_Segment"].isna().any():
            raise RuntimeError("Self-check: example rows were not fully scored.\n" + res["Pipeline_Notes"].to_string())
        return res

def _cli(argv=None) -> int:
    p = argparse.ArgumentParser(description="Marketing Campaign Intelligence pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--models-dir", type=Path, help="folder with the .joblib models (default: ./models or ../models)")
        sp.add_argument("--reference", type=Path, help="path to pipeline_reference.joblib")
        sp.add_argument("--channel-encoding", choices=["legacy", "correct"], default="legacy",
                        help="legacy = matches the trained models (default); correct = fixed Notebook 3 logic")

    t = sub.add_parser("template", help="write a CSV with the expected input layout (5 example rows)")
    t.add_argument("--output", type=Path, default=Path("new_campaigns_template.csv"))

    b = sub.add_parser("build-reference", help="compute anomaly statistics + tier cut-points from the training features file")
    common(b)
    b.add_argument("--train-data", type=Path, required=True, help="Notebook 3 output, e.g. nykaa_campaign_features.csv")

    c = sub.add_parser("check", help="load all models and score the built-in example rows")
    common(c)

    r = sub.add_parser("run", help="score a CSV of new campaigns")
    common(r)
    r.add_argument("--input", type=Path, required=True)
    r.add_argument("--output", type=Path)
    r.add_argument("--train-data", type=Path, help="if given and no reference exists yet, build it first")
    r.add_argument("--full", action="store_true", help="also write every engineered feature column")

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if a.cmd == "template":
        EXAMPLE_ROWS.to_csv(a.output, index=False)
        print(f"Template written to {a.output} (Revenue is optional; delete the column if unknown).")
        return 0

    pipe = MarketingCampaignPipeline(models_dir=a.models_dir, reference_path=a.reference,
                                     channel_encoding=a.channel_encoding)
    if a.cmd == "build-reference":
        path = pipe.build_reference(a.train_data)
        print(f"Reference saved to {path}")
        return 0
    if a.cmd == "check":
        res = pipe.self_check()
        print(f"All models loaded from {pipe.models_dir} and scored the example rows.")
        print(f"Statistical anomaly reference: {'found' if pipe.reference else 'MISSING - run build-reference'}")
        print(res[["Campaign_ID", "Predicted_Revenue", "Predicted_Performance_Tier", "Audience_Segment",
                   "Anomaly_Confidence"]].to_string(index=False))
        return 0
    if a.cmd == "run":
        if pipe.reference is None and a.train_data:
            pipe.build_reference(a.train_data)
        res = pipe.run(pd.read_csv(a.input), full_output=a.full)
        out_path = a.output or a.input.with_name(a.input.stem + "_scored.csv")
        res.to_csv(out_path, index=False)
        print(pipe.summarize(res))
        print(f"\nResults written to {out_path}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(_cli())
