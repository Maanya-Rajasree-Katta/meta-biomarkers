# Meta-analysis: Biomarkers and Neoadjuvant Outcomes  

![Python](https://img.shields.io/badge/Python-3.13-blue)  
![Reproducibility](https://img.shields.io/badge/Reproducible-100%25-brightgreen)  
![OpenScience](https://img.shields.io/badge/Open%20Science-%F0%9F%8C%90-lightgrey)  

📊 **Reproducible meta-analysis of tumor biomarkers and neoadjuvant outcomes (pCR, DFS, OS) with forest plots, funnel plots, Egger’s regression, Trim-and-Fill, and leave-one-out sensitivity analyses. 

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
