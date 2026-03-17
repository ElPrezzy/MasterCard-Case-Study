# MasterClothe TSS Impact Analysis

Analyzes the revenue impact of competitor (The Style Spot) opening on MasterClothe 
stores using a Difference-in-Differences (DiD) method.

## What the code does
- It classifies stores as treatment (nearby) or control (no TSS)
- Calculates DiD % change per product category after a store was built near
- Plots grouped before/after revenue bars with 95% confidence intervals

## Required to run
pip install pandas matplotlib seaborn numpy scipy openpyxl

## How to use
Update TX_PATH and MASTER_PATH at the top of sales_analysis.py then run:
python sales_analysis.py