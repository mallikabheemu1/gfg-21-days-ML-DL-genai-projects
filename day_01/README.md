# 🚢 Day 01: Data Storytelling — Analyzing Survival on the Titanic

## 📘 Project Overview
This project performs a **comprehensive Exploratory Data Analysis (EDA)** on the Titanic dataset to uncover key factors that influenced passenger survival.  
It follows a complete **data science workflow** — from data loading and cleaning to visualization, profiling, and insight generation.

---

## 🎯 Objective
To perform a step-by-step EDA that uncovers insights about survival patterns on the Titanic.  
Covers **data loading, cleaning, analysis, feature engineering, and visualization** with theoretical explanations at each stage.

---

## 🧩 Dataset
- **Source:** Titanic passenger dataset  
- **Size:** 891 passengers, 12 features  
- **Key Variables:** Passenger Class, Gender, Age, Fare, Embarkation Port, Survival Status  

---

## 🔍 Analysis Workflow

### 1️⃣ Data Loading & Initial Exploration
- Loaded Titanic dataset from CSV  
- Examined structure, data types, and dimensions  
- Identified missing values and data quality issues  

### 2️⃣ Data Cleaning & Preprocessing
**Missing Value Treatment:**  
- `Age`: Filled missing values with median age  
- `Embarked`: Filled missing values with mode  
- `Cabin`: Created binary feature `Has_Cabin` due to 77% missing values  

**Feature Engineering:**  
- Extracted **titles** from passenger names  
- Created categorical variables  
- Binned continuous variables for better analysis  

### 3️⃣ Exploratory Data Analysis
- **Survival Analysis:** Overall survival rate — 38.4%  
- **Demographic Insights:** Gender, age, and class trends  
- **Correlation Analysis:** Relationships between variables  
- **Visualization:** Histograms, boxplots, countplots, heatmaps  

### 4️⃣ Data Profiling
- Generated an **interactive HTML profiling report** using `ydata-profiling`  
- Included summary statistics, correlations, and missing value patterns  

---

## 🧠 Key Findings
- **Gender Effect:** Females had higher survival rates  
- **Class Impact:** Higher classes survived more  
- **Age Patterns:** Younger passengers had higher survival probabilities  
- **Embarkation Port:** Survival rates varied by port  

---

## ⚙️ Technical Implementation

**Tools & Libraries:**  
- Python 3.x  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- ydata-profiling  

---

## 📂 Deliverables
- **Main Notebook:** Complete EDA with detailed explanations  
- **HTML Profiling Report:** Comprehensive automated data analysis  
- **Documentation:** Professional project documentation  

---

## ✅ Assignment Completion
**Status:** COMPLETED  

All assignment requirements have been successfully fulfilled:  
- Comprehensive exploratory data analysis performed  
- Data cleaning and preprocessing completed  
- ydata-profiling HTML report generated  
- Professional documentation provided

---

## 🚀 Usage Instructions

1. **Run the Analysis:**  
```bash
jupyter notebook 1_Data_Storytelling_Analysing_Survival_on_the_Titanic.ipynb
```
---

2. **View Profiling Report:**

- Open Assignment/MB.html in any web browser
- Interactive report with detailed statistics and visualizations

---

3. **Requirements:**
```bash
- pip install pandas numpy matplotlib seaborn ydata-profiling
```
---

## Results and Impact

This analysis provides valuable insights into the factors that influenced survival on the Titanic, demonstrating the power of data storytelling in uncovering historical patterns. The methodology can be       applied to similar datasets for comprehensive exploratory analysis.

---

## Future Enhancements
- Predictive modeling for survival prediction
- Advanced feature engineering techniques
- Machine learning model implementation
- Interactive dashboard development

---

**Note:** This project is part of a 21-day data science challenge focusing on practical applications of data analysis and visualization techniques.

---

