#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Journal-ready meta-analysis plots (Cochrane-style)
- Forest plots (pCR=OR from 2x2 or OR+CI; DFS/OS=HR+CI)
- Random-effects (DerSimonian–Laird)
- Weights as filled squares (area ∝ weight) + pooled diamond
- Funnel plots with Egger's test (SE vertical, inverted)
- Leave-one-out (LOO) analysis
- Subgroup summaries written to CSV (e.g., TILs vs PIK3CA)
- Clean handling of k<2 (no heterogeneity for single-study groups)
- Optional Trim-and-Fill (Duval & Tweedie) for funnel plots (python or R/metafor)
- Optional data-driven mode — read studies from a CSV/Excel extraction file

Examples:
  # reproduce built-in figures (exactly like the ones you reviewed)
  python3 meta_plots_polished.py --format pdf --dpi 600

  # load from a CSV extraction sheet (same styling)
  python3 meta_plots_polished.py --data extraction.csv --format pdf tiff --dpi 600

Outputs saved in ./figures/
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from scipy.stats import chi2, t
import statsmodels.api as sm

# ---------- Aesthetics (journal-ready) ----------
matplotlib.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})
MONO = "DejaVu Sans Mono"

# ---------- Helpers ----------

def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _safe_log(x: float) -> float:
    if x is None or np.isnan(x) or x <= 0:
        raise ValueError("Effect sizes for ratio measures must be > 0.")
    return float(np.log(x))

def _se_from_ci(low: float, high: float) -> float:
    return float((np.log(high) - np.log(low)) / (2.0 * 1.96))

def _dl_tau2(y: np.ndarray, v: np.ndarray) -> float:
    """DerSimonian–Laird tau^2 (safe for small k)."""
    k = len(y)
    if k < 2:
        return 0.0
    w = 1.0 / v
    y_fe = np.sum(w * y) / np.sum(w)
    Q = np.sum(w * (y - y_fe) ** 2)
    c = np.sum(w) - (np.sum(w ** 2) / np.sum(w))
    if c <= 0:
        return 0.0
    return float(max(0.0, (Q - (k - 1)) / c))

def _pool_random(y: np.ndarray, v: np.ndarray) -> Dict[str, float]:
    k = len(y)
    if k == 1:
        y0 = float(y[0])
        se0 = float(np.sqrt(v[0]))
        return dict(y=y0, se=se0, low=y0 - 1.96 * se0, high=y0 + 1.96 * se0,
                    tau2=0.0, Q=0.0, df=0, p_Q=np.nan, I2=0.0,
                    weights=np.array([1.0]))
    w_fe = 1.0 / v
    y_fe = float(np.sum(w_fe * y) / np.sum(w_fe))
    Q = float(np.sum(w_fe * (y - y_fe) ** 2))
    df = k - 1
    p_Q = float(1.0 - chi2.cdf(Q, df))
    tau2 = _dl_tau2(y, v)
    w_re = 1.0 / (v + tau2)
    y_re = float(np.sum(w_re * y) / np.sum(w_re))
    se_re = float(np.sqrt(1.0 / np.sum(w_re)))
    ci_low = y_re - 1.96 * se_re
    ci_high = y_re + 1.96 * se_re
    I2 = float(max(0.0, (Q - df) / Q * 100.0)) if Q > 0 else 0.0
    return dict(y=y_re, se=se_re, low=ci_low, high=ci_high, tau2=tau2, Q=Q, df=df, p_Q=p_Q, I2=I2,
                weights=(w_re / np.sum(w_re)))

def _egger_test(y: np.ndarray, se: np.ndarray) -> Dict[str, float]:
    if len(y) < 3:
        return dict(intercept=np.nan, se=np.nan, t=np.nan, p=np.nan)
    snd = y / se
    precision = 1.0 / se
    X = sm.add_constant(precision)
    model = sm.OLS(snd, X).fit()
    a = float(model.params[0]); sa = float(model.bse[0])
    tval = a / sa
    pval = 2.0 * (1.0 - t.cdf(abs(tval), df=len(y) - 2))
    return dict(intercept=a, se=sa, t=float(tval), p=float(pval))

def _diamond(xc: float, y_c: float, half_width: float, half_height: float) -> Polygon:
    return Polygon([[xc - half_width, y_c], [xc, y_c + half_height],
                    [xc + half_width, y_c], [xc, y_c - half_height]],
                   closed=True, edgecolor="black", facecolor="black", lw=1)

# Save figure in multiple formats
def _save_multi(fig: plt.Figure, path_png: str, formats: List[str], dpi: int):
    base, _ = os.path.splitext(path_png)
    for fmt in formats:
        out = f"{base}.{fmt.lower()}"
        fig.savefig(out, dpi=dpi, bbox_inches='tight')

# ---------- Effect-size constructors ----------
@dataclass
class StudyRow:
    study: str
    biomarker: str
    subgroup: str
    measure: str              # 'OR' or 'HR'
    effect: Optional[float]
    low: Optional[float]
    high: Optional[float]
    pos_ev: Optional[int]
    pos_total: Optional[int]
    neg_ev: Optional[int]
    neg_total: Optional[int]

    def to_log_effect(self) -> Tuple[float, float]:
        if self.measure == "OR" and all(v is not None for v in [self.pos_ev, self.pos_total, self.neg_ev, self.neg_total]):
            a = float(self.pos_ev); b = float(self.pos_total - self.pos_ev)
            c = float(self.neg_ev); d = float(self.neg_total - self.neg_ev)
            if min(a, b, c, d) == 0:  # continuity correction
                a += 0.5; b += 0.5; c += 0.5; d += 0.5
            yi = math.log((a * d) / (b * c))
            vi = 1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d
            return yi, vi
        # fallback to effect+CI (OR or HR)
        if None in (self.effect, self.low, self.high):
            raise ValueError(f"Study '{self.study}' missing effect or CI")
        yi = _safe_log(self.effect)
        se = _se_from_ci(self.low, self.high)
        return yi, se * se

# ---------- Trim-and-Fill (optional) ----------

def trim_and_fill_python(y: np.ndarray, v: np.ndarray, side: str = "right") -> Dict[str, object]:
    """Lightweight L0 right-side trim-and-fill approximation."""
    y = y.copy(); v = v.copy()
    pooled = _pool_random(y, v)
    y_star = pooled['y']
    if side not in {"right", "left"}:
        side = "right"
    mask_side = (y > y_star) if side == "right" else (y < y_star)
    distances = np.abs(y[mask_side] - y_star)
    if distances.size == 0:
        return dict(k0=0, y_adj=pooled['y'], idx_filled=[], y_filled=y, v_filled=v)
    order = np.argsort(-distances)
    idx_side = np.where(mask_side)[0][order]
    k0 = max(0, len(idx_side) - (len(y) - len(idx_side)))
    if k0 == 0:
        return dict(k0=0, y_adj=pooled['y'], idx_filled=[], y_filled=y, v_filled=v)
    k0 = int(min(k0, len(idx_side)))
    idx_to_reflect = idx_side[:k0]
    y_new = y.tolist(); v_new = v.tolist(); filled_idx = []
    for idx in idx_to_reflect:
        y_reflect = 2.0 * y_star - y[idx]
        y_new.append(y_reflect)
        v_new.append(v[idx])
        filled_idx.append(idx)
    y_new = np.array(y_new, dtype=float)
    v_new = np.array(v_new, dtype=float)
    pooled_adj = _pool_random(y_new, v_new)
    return dict(k0=k0, y_adj=pooled_adj['y'], idx_filled=filled_idx, y_filled=y_new, v_filled=v_new)

def trim_and_fill_r(y: np.ndarray, v: np.ndarray, side: str = "right") -> Optional[Dict[str, object]]:
    """Use R's metafor::trimfill via rpy2, if available. Returns None if not available."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()
        ro.r("library(metafor)")
        ro.globalenv['yi'] = y
        ro.globalenv['vi'] = v
        ro.globalenv['side'] = 1 if side == 'right' else -1
        res = ro.r("rma(yi=yi, vi=vi, method='DL')")
        tf = ro.r("trimfill(res, side=side)")
        y_adj = float(tf.rx2('b')[0])
        k0 = int(tf.rx2('k0')[0])
        y_filled = np.array(ro.r('tf$yi.filled')) if 'yi.filled' in tf.names else y
        v_filled = np.array(ro.r('tf$vi.filled')) if 'vi.filled' in tf.names else v
        return dict(k0=k0, y_adj=y_adj, y_filled=y_filled, v_filled=v_filled)
    except Exception:
        return None

# ---------- Plotting ----------

def forest_plot(
    df: pd.DataFrame,
    title: str,
    outfile: str,
    x_label: str,
    pooled: Dict[str, float],
    show_weights: bool = True,
    figsize=(8.6, 6.5),
    formats: List[str] = None,
    dpi: int = 300,
):
    """Cochrane-style forest with a separate right text panel (no overlap)."""
    if formats is None:
        formats = ["png"]
    _ensure_dir(os.path.dirname(outfile))
    k = df.shape[0]
    order = df.index[::-1]
    y_pos = np.arange(k) + 1

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04, hspace=0.02, wspace=0.02)
    gs = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[3.8, 2.2])
    ax = fig.add_subplot(gs[0, 0])
    ax_txt = fig.add_subplot(gs[0, 1])

    # CIs + squares
    for i, idx in enumerate(order):
        yi = df.loc[idx, "yi"]; se = float(np.sqrt(df.loc[idx, "vi"]))
        l = float(np.exp(yi - 1.96 * se)); u = float(np.exp(yi + 1.96 * se))
        ax.plot([l, u], [y_pos[i], y_pos[i]], '-', color="black", lw=1)
    max_side = 0.32
    areas = df.loc[order, "weight"].to_numpy(); areas = areas / areas.max()
    for (i, idx), area in zip(enumerate(order), areas):
        side = float(max_side * math.sqrt(area))
        x = float(np.exp(df.loc[idx, "yi"])); y = y_pos[i]
        ax.add_patch(Rectangle((x - side/2, y - side/2), width=side, height=side,
                               edgecolor='black', facecolor='black'))

    # Pooled diamond
    y_sum = 0.35
    mu = float(np.exp(pooled['y']))
    ci_l = float(np.exp(pooled['low']))
    ci_u = float(np.exp(pooled['high']))
    ax.add_patch(_diamond(mu, y_sum, half_width=(mu-ci_l), half_height=0.22))

    # Left axis formatting
    ax.set_xscale('log')
    ax.set_ylim(0, k + 1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(list(df.loc[order, 'study']))
    ax.axvline(x=1.0, linestyle='--', color='gray', lw=1)
    ax.set_xlabel(x_label, labelpad=40)
    ax.set_title(title, loc='left')

    # Header labels in figure space
    fig.text(0.08, 0.97, 'Study', fontsize=10, ha='left')

    # Right text panel
    ax_txt.set_axis_off(); ax_txt.set_ylim(0, k + 1); ax_txt.set_xlim(0, 1)
    hdr = 'Effect [95% CI]' + ('   Weight' if show_weights else '')
    ax_txt.text(1.0, k+0.95, hdr, fontsize=10, ha='right', family=MONO)
    for i, idx in enumerate(order):
        yi = df.loc[idx, 'yi']; se = float(np.sqrt(df.loc[idx, 'vi']))
        eff = float(np.exp(yi)); l = float(np.exp(yi - 1.96*se)); u = float(np.exp(yi + 1.96*se))
        w = float(df.loc[idx, 'weight']*100.0)
        label = f"{eff:>6.2f} [{l:>6.2f}, {u:>6.2f}]"
        if show_weights:
            label += f"   {w:5.1f}%"
        ax_txt.text(1.0, y_pos[i], label, va='center', ha='right', fontsize=9, family=MONO)

    # Footer stats across figure (below xlabel)
    stats_txt = (
        f"Pooled = {mu:.2f} [{ci_l:.2f}, {ci_u:.2f}]   "
        f"I² = {pooled['I2']:.1f}%   τ² = {pooled['tau2']:.3f}   "
        f"Q({pooled['df']}) = {pooled['Q']:.2f}, p = {pooled['p_Q']:.3f}"
    )
    fig.text(0.01, -0.02, stats_txt, fontsize=9, ha='left', va='bottom', family=MONO)

    _save_multi(fig, outfile, formats, dpi)
    plt.close(fig)

def funnel_plot(y: np.ndarray, se: np.ndarray, pooled: Dict[str, float], title: str, x_label: str, outfile: str,
                figsize=(5.6, 6.0), formats: List[str] = None, dpi: int = 300,
                trimfill: str = 'none', side: str = 'right'):
    if formats is None:
        formats = ["png"]
    _ensure_dir(os.path.dirname(outfile))
    fig, ax = plt.subplots(figsize=figsize)

    x_vals = np.exp(y)
    ax.scatter(x_vals, se, facecolors='none', edgecolors='black', s=24, label='Observed studies')
    ax.invert_yaxis()

    # 95% pseudo limits around pooled (RE)
    se_line = np.linspace(se.min(), se.max(), 80)
    left = np.exp(pooled['y'] - 1.96 * se_line)
    right = np.exp(pooled['y'] + 1.96 * se_line)
    ax.plot(left, se_line, '--', color='gray', lw=1)
    ax.plot(right, se_line, '--', color='gray', lw=1)
    ax.axvline(x=np.exp(pooled['y']), linestyle='-', color='black', lw=1, label='Pooled (RE)')

    # Optional: Trim-and-Fill
    tf_info = None
    if trimfill.lower() != 'none' and len(y) >= 3:
        if trimfill.lower() == 'r':
            tf_info = trim_and_fill_r(y, se**2, side=side)
        if tf_info is None and trimfill.lower() in {'python','r'}:
            tf_info = trim_and_fill_python(y, se**2, side=side)
        if tf_info is not None and tf_info.get('k0', 0) > 0:
            y_adj = tf_info['y_adj']
            ax.axvline(x=np.exp(y_adj), linestyle=':', color='black', lw=1, label='Trim-and-Fill pooled')
            extra = len(tf_info['y_filled']) - len(y)
            if extra > 0:
                y_all = tf_info['y_filled']
                se_all = np.sqrt(tf_info['v_filled'])
                ax.scatter(np.exp(y_all[-extra:]), se_all[-extra:], marker='^', s=28,
                           facecolors='white', edgecolors='black', label='Filled (imputed)')

    ax.set_xscale('log')
    ax.set_xlabel(x_label, labelpad=25)
    ax.set_ylabel('Standard error', labelpad=6)
    ax.set_title(title, loc='left')

    egger = _egger_test(y, se)
    if not np.isnan(egger['intercept']):
        egger_text = (f"Egger intercept = {egger['intercept']:.3f}\n"
                      f"p = {egger['p']:.3f}")
        ax.text(0.98, 0.04, egger_text, transform=ax.transAxes, ha='right', va='bottom', fontsize=10, family=MONO,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, boxstyle='round,pad=0.25'))

    if trimfill.lower() != 'none':
        ax.legend(loc='upper right', frameon=False)

    plt.tight_layout()
    _save_multi(fig, outfile, formats, dpi)
    plt.close(fig)

def loo_plot(y: np.ndarray, v: np.ndarray, studies: List[str], title: str, x_label: str, outfile: str,
             figsize=(6.8, 5.0), formats: List[str] = None, dpi: int = 300):
    if formats is None:
        formats = ["png"]
    _ensure_dir(os.path.dirname(outfile))
    k = len(y)
    pooled_full = _pool_random(y, v)

    est, lo, hi = [], [], []
    for i in range(k):
        mask = np.ones(k, dtype=bool); mask[i] = False
        pooled = _pool_random(y[mask], v[mask])
        est.append(float(np.exp(pooled['y'])))
        lo.append(float(np.exp(pooled['low'])))
        hi.append(float(np.exp(pooled['high'])))

    fig, ax = plt.subplots(figsize=figsize)
    ypos = np.arange(k) + 1
    for i in range(k):
        ax.plot([lo[i], hi[i]], [ypos[i], ypos[i]], '-', color='black', lw=1)
        ax.plot(est[i], ypos[i], 's', color='black', ms=4)

    ax.axvline(x=1.0, ymin=0, ymax=1, linestyle='--', color='gray', lw=1)
    ax.axvline(x=float(np.exp(pooled_full['y'])), ymin=0, ymax=1, linestyle='-', color='black', lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(studies[::-1])
    ax.set_xscale('log')
    ax.set_xlabel(x_label, labelpad=25)
    ax.set_title(title, loc='left')
    ax.set_ylim(0, k + 1)
    plt.tight_layout()
    _save_multi(fig, outfile, formats, dpi)
    plt.close(fig)

# ---------- Data from extraction sheet (optional) ----------

def _rows_from_dataframe(df: pd.DataFrame, outcome: str) -> List[StudyRow]:
    """Parse a tidy extraction table into StudyRow list for the given outcome.
    Expected flexible columns (case-insensitive, spaces/underscores ignored):
      - study, biomarker, subgroup (fallback to biomarker), outcome (pcr/dfs/os)
      - measure (OR/HR) OR effect, low, high
      - pos_ev, pos_total, neg_ev, neg_total (for 2x2 OR)
    Rows with missing essentials for the chosen outcome are skipped.
    """
    def norm(s):
        return str(s).strip().lower().replace(' ', '').replace('_', '')
    cols = {norm(c): c for c in df.columns}
    def get(row, key, default=None):
        for k in [key, key.replace('_','')]:
            if k in cols:
                return row[cols[k]]
        return default
    out_rows: List[StudyRow] = []
    for _, r in df.iterrows():
        o = str(get(r, 'outcome', '')).strip().lower()
        if o and o not in {'pcr','dfs','os'}:
            continue
        if o and o != outcome:
            continue
        study = get(r, 'study') or get(r, 'studyid')
        biomarker = get(r, 'biomarker') or ''
        subgroup = get(r, 'subgroup') or biomarker or 'All'
        measure = str(get(r, 'measure', '') or '').upper()
        # numeric fields
        effect = get(r, 'effect'); low = get(r, 'low'); high = get(r, 'high')
        pos_ev = get(r, 'pos_ev'); pos_total = get(r, 'postotal') or get(r,'pos_total')
        neg_ev = get(r, 'neg_ev'); neg_total = get(r, 'negtotal') or get(r,'neg_total')
        def num(x):
            try:
                return None if pd.isna(x) else float(x)
            except Exception:
                return None
        effect, low, high = map(num, (effect, low, high))
        pos_ev, pos_total, neg_ev, neg_total = map(num, (pos_ev, pos_total, neg_ev, neg_total))
        if not measure:
            measure = 'OR' if pos_ev is not None else 'HR'
        if measure == 'OR' and not (
            (pos_ev is not None and pos_total is not None and neg_ev is not None and neg_total is not None)
            or (effect is not None and low is not None and high is not None)
        ):
            continue
        if measure == 'HR' and not (effect is not None and low is not None and high is not None):
            continue
        out_rows.append(StudyRow(
            study=str(study), biomarker=str(biomarker), subgroup=str(subgroup), measure=measure,
            effect=effect, low=low, high=high, pos_ev=None if pos_ev is None else int(round(pos_ev)),
            pos_total=None if pos_total is None else int(round(pos_total)),
            neg_ev=None if neg_ev is None else int(round(neg_ev)),
            neg_total=None if neg_total is None else int(round(neg_total)),
        ))
    return out_rows

# ---------- Built-in data (kept to reproduce your current figures) ----------

def build_pcr_data_builtin() -> List[StudyRow]:
    return [
        StudyRow("Liu 2016", "TILs (≥30%)", "TILs", "OR", None, None, None, 23, 36, 19, 80),
        StudyRow("Inoue 2017", "TILs (≥30%)", "TILs", "OR", None, None, None, 15, 19, 35, 78),
        StudyRow("Kim 2023", "PIK3CA mutation", "PIK3CA", "OR", None, None, None, 3, 12, 16, 24),
        StudyRow("Hong 2021", "TILs", "TILs", "OR", None, None, None, 50, 162, 24, 299),
        StudyRow("Hwang 2018", "TILs", "TILs", "OR", None, None, None, 57, 127, 16, 181),
        StudyRow("Denkert 2018", "TILs", "TILs", "OR", None, None, None, 127, 262, 194, 605),
        StudyRow("Kim 2021", "Ki-67 index, TILs", "TILs", "OR", None, None, None, 35, 48, 69, 125),
    ]

def build_dfs_data_builtin() -> List[StudyRow]:
    return [
        StudyRow("Liu 2016", "TILs (≥30%)", "TILs", "HR", 0.15, 0.04, 0.67, None, None, None, None),
        StudyRow("Almekinders 2022", "TILs", "TILs", "HR", 0.54, 0.29, 1.01, None, None, None, None),
        StudyRow("Hong 2021", "TILs", "TILs", "HR", 0.50, 0.31, 0.81, None, None, None, None),
        StudyRow("Hwang 2018", "TILs", "TILs", "HR", 0.94, 0.86, 1.02, None, None, None, None),
        StudyRow("Denkert 2018", "TILs", "TILs", "HR", 0.94, 0.89, 0.99, None, None, None, None),
        StudyRow("Loibl 2014", "PIK3CA mutation", "PIK3CA", "HR", 1.065, 0.54, 2.086, None, None, None, None),
    ]

def build_os_data_builtin() -> List[StudyRow]:
    return [
        StudyRow("Prat 2021", "Tumor cellularity + TILs", "TILs", "HR", 0.43, 0.20, 0.92, None, None, None, None),
    ]

# ---------- Runners ----------

def run_meta(rows: List[StudyRow], x_label: str, title_prefix: str, out_stub: str,
             formats: List[str], dpi: int, forest_size=(8.6, 6.5), funnel_size=(5.6, 6.0),
             trimfill: str = 'none', side: str = 'right'):
    es = []
    for r in rows:
        yi, vi = r.to_log_effect()
        es.append(dict(study=r.study, biomarker=r.biomarker, subgroup=r.subgroup, yi=yi, vi=vi))
    df = pd.DataFrame(es)

    pooled = _pool_random(df["yi"].to_numpy(), df["vi"].to_numpy())
    df["weight"] = pooled["weights"]

    # Forest
    forest_plot(df=df.copy(), title=f"{title_prefix} — Random-effects",
                outfile=f"figures/forest_{out_stub}.png", x_label=x_label, pooled=pooled,
                show_weights=True, formats=formats, dpi=dpi, figsize=forest_size)

    # Funnel + Egger (+ optional trim-and-fill)
    y = df["yi"].to_numpy(); se = np.sqrt(df["vi"].to_numpy())
    funnel_plot(y=y, se=se, pooled=pooled, title=f"Funnel plot — {title_prefix}", x_label=x_label,
                outfile=f"figures/funnel_{out_stub}.png", formats=formats, dpi=dpi, figsize=funnel_size,
                trimfill=trimfill, side=side)

    # LOO
    loo_plot(y=y, v=df["vi"].to_numpy(), studies=list(df["study"]),
             title=f"Leave-one-out — {title_prefix}", x_label=x_label,
             outfile=f"figures/loo_{out_stub}.png", formats=formats, dpi=dpi)

    # Subgroups CSV
    sub_summary = []
    for g, gdf in df.groupby("subgroup"):
        pooled_g = _pool_random(gdf["yi"].to_numpy(), gdf["vi"].to_numpy())
        sub_summary.append(dict(
            subgroup=g, k=gdf.shape[0],
            pooled=float(np.exp(pooled_g["y"])), low=float(np.exp(pooled_g["low"])), high=float(np.exp(pooled_g["high"])),
            I2=float(pooled_g["I2"]), tau2=float(pooled_g["tau2"]) if gdf.shape[0] > 1 else np.nan,
        ))
    pd.DataFrame(sub_summary).sort_values("subgroup").to_csv(f"figures/{out_stub}_subgroup_summary.csv", index=False)

    # Effect sizes CSV
    out = df.copy()
    out["effect"] = np.exp(out["yi"]).astype(float)
    out["low"] = np.exp(out["yi"] - 1.96 * np.sqrt(out["vi"]))
    out["high"] = np.exp(out["yi"] + 1.96 * np.sqrt(out["vi"]))
    out[["study", "biomarker", "subgroup", "effect", "low", "high", "weight"]].to_csv(
        f"figures/{out_stub}_effect_sizes.csv", index=False
    )

    # Console summary
    pooled_str = (
        f"Pooled ({x_label}): {np.exp(pooled['y']):.2f} "
        f"[{np.exp(pooled['low']):.2f}, {np.exp(pooled['high']):.2f}]\n"
        f"I^2 = {pooled['I2']:.1f}%, tau^2 = {pooled['tau2']:.4f}, "
        f"Q({pooled['df']}) = {pooled['Q']:.2f}, p = {pooled['p_Q']:.4f}"
    )
    print(f"\n=== {title_prefix} ===\n{pooled_str}")

def run_os_single(rows: List[StudyRow], formats: List[str], dpi: int):
    r = rows[0]
    yi, vi = r.to_log_effect()
    eff, low, high = float(np.exp(yi)), float(np.exp(yi - 1.96*np.sqrt(vi))), float(np.exp(yi + 1.96*np.sqrt(vi)))

    _ensure_dir("figures")
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    ax.plot([low, high], [1, 1], '-', color='black', lw=1)
    ax.plot(eff, 1, 's', color='black', ms=5)
    ax.axvline(x=1.0, linestyle='--', color='gray', lw=1)
    ax.set_yticks([1]); ax.set_yticklabels([r.study])
    ax.set_xscale('log'); ax.set_xlabel('Hazard ratio (OS)', labelpad=25)
    ax.set_title('Overall Survival (limited) — single study', loc='left')
    ax.set_ylim(0.7, 1.3)
    plt.tight_layout()
    _save_multi(fig, 'figures/forest_os_limited.png', formats, dpi)
    plt.close(fig)
    print(f"\n=== OS (limited) ===\n{r.study}: HR = {eff:.2f} [{low:.2f}, {high:.2f}]")

# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Generate meta-analysis plots (journal-ready)")
    parser.add_argument('--pcr', action='store_true', help='Run pCR (OR)')
    parser.add_argument('--dfs', action='store_true', help='Run DFS (HR)')
    parser.add_argument('--os', action='store_true', help='Run OS (limited)')
    parser.add_argument('--format', nargs='+', default=['png'], choices=['png','pdf','tiff','svg'],
                        help='Output formats (one or more).')
    parser.add_argument('--dpi', type=int, default=300, help='Figure DPI for raster formats (PNG/TIFF).')
    parser.add_argument('--width', choices=['single','double','wide'], default=None,
                        help='Preset figure widths: single≈85mm, double≈178mm, wide≈220mm.')
    parser.add_argument('--trimfill', choices=['none','python','r'], default='none',
                        help='Apply Duval & Tweedie trim-and-fill (python or R/metafor via rpy2).')
    parser.add_argument('--tf-side', choices=['right','left'], default='right', help='Asymmetry side to fill.')
    parser.add_argument('--data', help='Path to CSV/Excel extraction sheet to load studies from.')
    args = parser.parse_args()

    # Width presets (inches). 1 inch = 25.4 mm
    if args.width == 'single':
        forest_size = (3.35, 6.0)
        funnel_size = (3.35, 4.2)
    elif args.width == 'double':
        forest_size = (7.0, 6.0)
        funnel_size = (5.6, 6.0)
    elif args.width == 'wide':
        forest_size = (8.6, 6.5)
        funnel_size = (6.5, 6.5)
    else:
        forest_size = (8.6, 6.5)
        funnel_size = (5.6, 6.0)

    print("Generating figures in ./figures ...")

    def _load_rows(outcome: str, builtin_fn, label: str) -> List[StudyRow]:
        if not args.data:
            return builtin_fn()
        # CSV or Excel
        ext = os.path.splitext(args.data)[1].lower()
        if ext in {'.xls', '.xlsx'}:
            df = pd.read_excel(args.data)
        else:
            df = pd.read_csv(args.data)
        rows = _rows_from_dataframe(df, outcome)
        if not rows:
            print(f"[WARN] No usable rows found for {label} in {args.data}. Falling back to built-in data.")
            return builtin_fn()
        return rows

    run_all = not (args.pcr or args.dfs or args.os)

    if args.pcr or run_all:
        rows = _load_rows('pcr', build_pcr_data_builtin, 'pCR')
        run_meta(rows, x_label="Odds ratio (pCR)", title_prefix="pCR (biomarker+ vs biomarker−)", out_stub="pcr",
                 formats=args.format, dpi=args.dpi, forest_size=forest_size, funnel_size=funnel_size,
                 trimfill=args.trimfill, side=args.tf_side)

    if args.dfs or run_all:
        rows = _load_rows('dfs', build_dfs_data_builtin, 'DFS')
        run_meta(rows, x_label="Hazard ratio (DFS)", title_prefix="DFS", out_stub="dfs",
                 formats=args.format, dpi=args.dpi, forest_size=forest_size, funnel_size=funnel_size,
                 trimfill=args.trimfill, side=args.tf_side)

    if args.os or run_all:
        rows = _load_rows('os', build_os_data_builtin, 'OS')
        run_os_single(rows, formats=args.format, dpi=args.dpi)

    print("Done.")

if __name__ == "__main__":
    main()
