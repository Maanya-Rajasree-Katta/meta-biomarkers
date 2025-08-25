# Meta-analysis: Biomarkers and Neoadjuvant Outcomes  

![Python](https://img.shields.io/badge/Python-3.13-blue)  
![Reproducibility](https://img.shields.io/badge/Reproducible-100%25-brightgreen)  
![OpenScience](https://img.shields.io/badge/Open%20Science-%F0%9F%8C%90-lightgrey)  

📊 **Reproducible Meta-Analysis Package** — Includes code, dummy data, and journal-ready figures for biomarkers and neoadjuvant therapy outcomes. Designed for transparency and open science practices, following Cochrane-style methods (random-effects, heterogeneity, Egger’s test, Trim-and-Fill, subgroup analyses).  

---

## Contents  
This repository contains code and minimal data needed to reproduce the figures:  
- Forest plots (pCR: OR; DFS: HR; OS single study)  
- Funnel plots with Egger’s regression test (+ optional Trim-and-Fill)  
- Leave-one-out sensitivity analyses  
- Subgroup summaries (CSV)  

---

## Reproduce locally  
```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis (with dummy data included here)
python3 meta_plots_polished.py --data data/extraction_dummy.csv --format pdf --dpi 600 --trimfill python

# 4. Deactivate environment when done
deactivate
