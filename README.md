# Student Depression Analysis 

## Project Overview
A comprehensive exploratory data analysis (EDA) and feature engineering study designed to explore the nuances of what influences depression among students. By utilizing the `Student Depression Dataset.csv` (27,901 records, post-clean), we aim to pull out definitive patterns across demographics, physical wellness markers (sleep, diet), and academic constraints. 

## Data Pipeline & Decisions

### 1. Data Cleaning
- **Handling Missing Elements:** Text columns with erroneous placeholder strings (like "na", "n/a", "none") were converted to native `NaN` to prevent inaccurate string groupings.
- **Micro-Missing Values:** 
  - `Financial Stress` natively contained exactly 3 missing values (`NaN`).
  - **The Decision:** Rather than imputing them with mean or median structures, the 3 rows were explicitly dropped. Why? Because dropping 3 out of 27,901 rows represents a structural loss of less than `0.01%`. Leaving them out entirely averts artificial noise or flattening in the financial signal without meaningful loss to statistical integrity.
- **Normalization:** Object columns were standardized by stripping extra spaces and applying uniform Title casing (i.e. 'Yes', 'Male'). 

### 2. Feature Engineering Logic
Raw textual survey data must be mathematically processed before it becomes useful for aggregate visualization. Thus, the following engine logic was applied:

1. **`Suicidal Thoughts Flag` & `Family History Flag`**: Shifted textual logic (`Yes`/`No`) into binary boolean states (`1`/`0`), streamlining direct correlation and regression tracking.
2. **`Age Group`**: Binned the age spectrum into categorical buckets (`19-22`, `23-26`, etc.). This smooths over isolated anomalies at specific integer ages, allowing us to spot dominant **generational** themes in depression triggers.
3. **`Sleep Deficit Hours`**: 
   - Sleep categories like "5-6 hours" were quantified to median scalars (`5.5`).
   - We assumed a clinical baseline of `7` healthy hours; subtracting the student's scalar outputs a distinct `Sleep Deficit` parameter. It is much more effective to model exactly how many hours of sleep a student is *missing* rather than sorting strings.
4. **`Diet Risk Score`**: Mapped `Healthy(0) / Moderate(1) / Unhealthy(2)` to create an increasing scale of physical stress. 
5. **`Total Pressure`**: We built an aggregated feature combining `Academic Pressure`, `Work Pressure`, and `Financial Stress`. 
6. **`Mental Health Risk Score`**: A definitive, weighted composite score considering Pressures (0.30 weight), Sleep Deficit (0.20), Diet Risk (0.15), and baseline predispositions (0.20 for Suicidal Thoughts, 0.15 for Family History). 

*(Note: The cleaned and engineered final dataframe has been exported directly to `final_preprocessed_dataset.csv` for independent review or modeling).*

---

## Exploratory Data Analysis (EDA) & Extracted Insights

Through a series of tailored visualizations from the notebook, we extract several major insights dictating student depression metrics!

### Insight A: Density Dynamics of Academic Success 
![CGPA Density vs Depression](notebook_files/notebook_19_0.png)

**Observation:** We plotted the continuous CGPA distributions mapped to depression outcomes. 
**Pattern:** Quite counter-intuitively, students marked under the Depression demographic (red) peak in density at slightly *higher* academic grades than the non-depressed tier (green). Higher CGPAs track visibly alongside higher depressive states, hinting that immense academic perfectionism actively degrades baseline mental health.

### Insight B: The Physical Interplay - Diet & Sleep Deficit
![Diet/Sleep Heatmap](notebook_files/notebook_20_0.png)

**Observation:** This cross-sectional heatmap proves compounding physiological pressures. 
**Pattern:** For individuals with 0 sleep deficit (healthy sleep), bad diets still push depression rates to `~61%`. However, once individuals maintain high Sleep Deficits (2.5 hours missed) combined with an "Unhealthy" (Tier 2) diet risk score, depression incidence jumps above **`90%`**. Poor physiological maintenance virtually guarantees depression onset irrespective of life stressors. 

### Insight C: Work & Study Volumes 
![Work Hours vs Satisfaction Violin Plot](notebook_files/notebook_21_0.png)

**Observation:** Segmenting average daily work/study hours alongside the student's 'Study Satisfaction' output.
**Pattern:** The inner quartiles show that severely depressed students (orange split) routinely clock excessive workloads extending far above the 8-hour density bulge, regardless of whether their Study Satisfaction is a 0 or a 5. Non-depressed students enforce much tighter bounds around a 5–7 hour work cycle. 

### Insight D: Course Type Stressors 
![Degrees vs Depression](notebook_files/notebook_22_1.png)

**Observation:** We subsetted the dataset down to the top 10 highest-volume degrees reported and graphed average depression rates. 
**Pattern:** STEM and heavily analytical coursework such as B.Arch, M.Tech, and B.Tech present structurally higher baseline depressive rates compared to standard undergraduate arrays. 

### Insight E: Generational and Gender Shifts
![Depression Prevalence by Age Group and Gender](notebook_files/notebook_8_1.png)

**Observation:** Depression prevalence broken down directly by age bounds.
**Pattern:** There is a sharp linear increase in depression likelihood as ages ascend towards the `31-35` range. Both males and females succumb roughly equally at the limits, indicating that prolonged exposure to high-pressure graduate frameworks or balancing late-stage study drives systematic exhaustion.

### Insight F: The `Mental Health Risk Score` Cliff
![Risk Score](notebook_files/notebook_10_1.png)

**Observation:** Mapping our engineered risk scalar against raw depression incidence. 
**Pattern:** The model hits a distinct logistical cliff. Entering the `2.0` Risk Score tier marks the inflection point where depression escalates rapidly over 80%. This confirms our engineered logic is a highly predictive metric. 

### Insight G: Correlational Maps
![Correlation Map](notebook_files/notebook_17_0.png)

**Observation:** We see strong systemic correlations binding our custom values. `Mental Health Risk Score` strictly maps to occurrences of clinical depression far heavier than pure CGPA or base Age do. 

---

## Conclusion
The data emphatically concludes that academic success metrics (CGPA, high study volumes) and grueling degree structures act as direct escalators for student depression when baseline physical markers (sleep structure and diet) are abandoned. Correcting sleep deficits and financial safety nets offer mathematically stronger prevention bounds than isolating standard academic factors.
