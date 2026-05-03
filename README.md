# GDP and Life Expectancy Analysis 🌍📊

**Author:** Maitham Al-Shamery  
**Course:** Data Scientist – Codecademy  
**Data Sources:** World Health Organization (WHO) & World Bank  

---

## Project Overview

This project analyzes the relationship between **GDP** and **Life Expectancy** across six countries from 2000 to 2015:

| Country | Region |
|---|---|
| 🇨🇱 Chile | South America |
| 🇨🇳 China | Asia |
| 🇩🇪 Germany | Europe |
| 🇲🇽 Mexico | North America |
| 🇺🇸 United States of America | North America |
| 🇿🇼 Zimbabwe | Africa |

---

## Project Objectives

- Analyze and visualize GDP and life expectancy data
- Identify the relationship between economic output and health outcomes
- Use **Seaborn** and **Matplotlib** to create clear, meaningful visualizations
- Present findings in a blog post format for the WHO website

---

## Files in this Repository

| File | Description |
|---|---|
| `life_expectancy_gdp.ipynb` | Main Jupyter notebook with full analysis |
| `all_data.csv` | Dataset (GDP and life expectancy, 2000–2015) |
| `README.md` | This file |

---

## Visualizations Produced

1. Life Expectancy over time (line chart)
2. GDP over time (line chart)
3. GDP vs Life Expectancy (scatter plot)
4. Country subplots — GDP and Life Expectancy together
5. Average Life Expectancy per country (bar chart)
6. Average GDP per country (bar chart)
7. Correlation heatmap
8. Distribution of Life Expectancy (violin plot)
9. FacetGrid — GDP vs Life Expectancy per country with trend line

---

## Key Findings

- ✅ **Strong positive correlation** between GDP and life expectancy
- 📈 **China** showed the fastest GDP growth (from $1.2T to $11T)
- 💪 **Zimbabwe** showed the most dramatic life expectancy improvement
- 🏆 **Germany** consistently had the highest average life expectancy
- 🇺🇸 **USA** had the highest average GDP across the period

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/maithamphysics/Ghufran_GDP.git

# Navigate to the folder
cd Ghufran_GDP

# Install required libraries
pip install pandas numpy matplotlib seaborn jupyter

# Open the notebook
jupyter notebook life_expectancy_gdp.ipynb
```

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## Conclusion

GDP is strongly associated with life expectancy, but it is not the only factor.
Zimbabwe's dramatic improvement despite a very low GDP shows that healthcare
investment and political stability also play a crucial role in population health.

---

*Data Science Portfolio Project | Codecademy Data Scientist Path*
