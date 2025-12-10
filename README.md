# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

---

# King County Housing Analytics 🏠

![](https://github.com/dumindagamage/Home-Value-Analysis/blob/wip/resources/images/home_analysis_banner.png?raw=true)

---
## Project Overview

This Data Analytics project analyzes housing sales data in King County, USA (May 2014 - May 2015). The goal is to provide actionable insights for two distinct user groups: **Buyers** looking for value and affordability, and **Sellers** aiming to maximize their profit. 

The project culminates in an interactive Streamlit dashboard that allows users to explore market trends, assess property value based on condition and location, and predict house prices using a Machine Learning model.

## Executive Summary
**Client:** King County Real Estate Agency.  
**Goal:** The client wants to launch a digital dashboard to assist two user groups:
1. **Buyers:** Finding real value of the properties and understanding affordability.
2. **Sellers:** Determining the optimal pricing strategy, listing time and renovation strategy.

## Dataset Content
The project uses the **King County House Sales dataset** (`kc_house_data.csv`).
* **Source:** [Kaggle - House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
* **Volume:** ~21,613 records (from May 2014 - May 2015)
* **Features:** 21 variables including Price (Target), Bedrooms, Bathrooms, Sqft Living, Floors, Waterfront, View, Condition, Grade, Zipcode, and Year Built.
* **Data Dictionary:**

| Variable | Data Type | Description |
| :--- | :--- | :--- |
| **id** | String | A unique identifier for each home sold. |
| **date** | Date | The date the home sale was completed. |
| **price** | Float | The price the home was sold for (Target Variable). |
| **bedrooms** | Integer | Number of bedrooms in the house. |
| **bathrooms** | Float | Number of bathrooms (0.5 indicates a toilet/sink but no shower). |
| **sqft_living** | Integer | Square footage of the interior living space. |
| **sqft_lot** | Integer | Square footage of the entire land lot. |
| **floors** | Float | Total number of floors (levels) in the house. |
| **waterfront** | Boolean | Whether the house has a view of the waterfront (0 = No, 1 = Yes). |
| **view** | Integer | A rating of the view quality (0 = Poor to 4 = Excellent). |
| **condition** | Integer | A rating of the overall condition of the house (1 = Poor to 5 = Excellent). |
| **grade** | Integer | An overall grade given to the housing unit based on the King County grading system (1 = Poor to 13 = Excellent). |
| **sqft_above** | Integer | The square footage of the interior housing space that is above ground level. |
| **sqft_basement**| Integer | The square footage of the interior housing space that is below ground level. |
| **yr_built** | Integer | The year the house was initially built. |
| **yr_renovated** | Integer | The year of the house’s last renovation (0 if never renovated). |
| **zipcode** | String | The zip code area where the house is located. |
| **lat** | Float | Latitude coordinate of the property. |
| **long** | Float | Longitude coordinate of the property. |
| **sqft_living15**| Integer | The average interior living space square footage of the nearest 15 neighbors. |
| **sqft_lot15** | Integer | The average lot size square footage of the nearest 15 neighbors. |

## Business Requirements
### 🏷️ User Group 1: Buyers
* **Affordability:** Identify the top 10 most affordable zip codes.
* **Value Assessment:** Quantify the premium for scenic attributes (Waterfront or High-Quality Views) for properties.
* **Feature Importance:** Determine house grade vs condition impact on price.
* **Prediction:** Estimate fair market value to make competitive offers.

### 💰 User Group 2: Sellers
* **Feature Value:** Determine which specific home attributes contribute most significantly to property valuation
* **ROI Analysis:** Determine if renovations yield a statistically significant return.
* **Timing:** Identify the best month to sell for maximum profit.
* **Listing Strategy:** Set optimal prices based on neighborhood trends.

## Project Hypothesis and Validation

The following hypotheses were posed at the project's inception and validated through the data analysis process.

### Hypothesis 1: There are distinct geographic clusters (zip codes) within King County that are significantly more affordable than the county median.
* **Validation:** Analysis of median price per zip code visualized via bar charts.
* **Outcome:** **Confirmed**. The analysis identified specific zip codes (e.g., 98002, 98168) where the median price is less than half of the county average, validating the "Affordability" user story.

### Hypothesis 2: Properties with scenic attributes (Waterfront or High-Quality Views) command a statistically significant price premium compared to standard properties.
* **Validation Tests:**
    1.  **Mann-Whitney U Test (Waterfront):** Used to compare the binary "Waterfront vs. Non-Waterfront" groups.
    2.  **Kruskal-Wallis H Test (View):** Used to compare the five distinct view categories (0–4) to ensure the price difference wasn't just random.
* **Outcome:** **Confirmed**.
    * **Waterfront:** Validated as the single most valuable binary feature.
    * **View:** The Kruskal-Wallis test (Statistic: 1,936, p < 0.05) confirmed that view quality is a significant price driver.

### Hypothesis 3: Construction Grade has a stronger impact on price than Condition.
* **Validation:** Box plot and Bar Plot for the high level analysis of the each feature. Heatmap Interaction Analysis & Correlation Comparison.Use Spearman Correlation because Grade/Condition are ordinal categories
* **Outcome:** **Confirmed** Grade (~0.65) is a far stronger correlation than Condition (~0.04).

### Hypothesis 4: Sales prices follow a seasonal trend, suggesting an optimal window for sellers.
* **Validation:** Time-series analysis of `Median Price` by `Month Sold`.Kruskal-Wallis Test (Comparing > 2 groups on non-normal data).
* **Outcome:** **Confirmed**. The data shows a visible trend where median prices and sales volume tick upward starting in April/May, supporting the advice for sellers to list during these months for maximum potential profit. It confirmed by the statistical test as the p-value (p-unc) is far less than 0.05.

### Hypothesis 5: The value of renovation is not uniform
* **Validation:**  Interaction Plot (Age x Renovation).
* **Outcome:** **Confirmed**. It confimrmed that the ROI varies significantly by era as Mid-Century (1950-1990) shows ~60% ROI. It provides a significantly higher ROI for Mid-Century homes (1950-1990) than for pre-war or modern homes.

### Hypothesis 6:  `sqft_living` (total living space) has a significantly stronger correlation with Price than room counts (`bedrooms` or `bathrooms`), indicating that buyers value total usable space more than just the number of rooms."
* **Validation Tests:**
    1.  **Pearson Correlation Matrix:** Used to identify the strength of the linear relationship between Price and key features.
    2.  **Spearman Rank Correlation:** *Secondary Test.* Used this to validate the rankings. Since Pearson can be sensitive to outliers (like mega-mansions), Spearman checks the "rank" order instead of raw values, ensuring the findings hold true even with skewed data.
* **Outcome:** **Confirmed**. Total living space (`sqft_living`) is the dominant driver of price (Pearson: **0.70**), significantly outperforming simple room counts like `bedrooms` (Pearson: **0.32**).

## Project Management
This project was managed using Agile methodologies with a **GitHub Project Board**.
* **Kanban Board:** [Link to your GitHub Project Board](https://github.com/users/dumindagamage/projects/5/views/1)
![Project Board](https://github.com/dumindagamage/House-Price-Analysis/blob/wip/resources/images/project_board.png)

## Project Plan
### 1. High-Level Steps
The analysis was executed in four phases:

* **Phase 1: Business Understanding & Data Collection**
    * **Goal:** Defined the scope to help Buyers find value and Sellers maximize profit.
    * **Data:** Acquired the King County Housing dataset (21k+ records) from Kaggle.
    * **Hypotheses:** Set up core questions regarding location, condition, and how renovations affect value.

* **Phase 2: Data Cleaning, EDA & Transformation**
    * **Cleaning:** Handled missing values, removed duplicates, and corrected data types (e.g., converting `date` to datetime).
    * **EDA:** Checked data distributions (univariate analysis) and explored relationships between variables (correlation analysis).
    * **Feature Engineering:** Created new metrics to drive better insights, such as `price_per_sqft`, `house_age`, and `age_group` (Pre-War, Mid-Century, Modern).

* **Phase 3: Hypothesis Testing (Buyer & Seller Personas)**
    * **Analysis:** Performed deep-dive analysis to test specific questions (e.g., *Does renovation impact price differently by era?*).
    * **Validation:** Used statistical tests (like Mann-Whitney U and Correlation) to prove that the findings were real, not just random chance.

* **Phase 4: Predictive Modeling & Dashboarding**
    * **Modeling:** Trained Machine Learning models (Random Forest) to estimate house prices.
    * **Dashboard:** Built an interactive Streamlit dashboard to allow users to explore the data and findings themselves.

### 2. Data Management Strategy
Data integrity was maintained through a strict separation of concerns:

* **Storage Architecture:** The project utilizes a "Raw vs. Processed" directory structure.
    * `data/raw/`: Contains the immutable original CSV file (`kc_house_data.csv`).
    * `data/processed/`: Stores the cleaned and transformed dataset (`final_house_data.csv`) used for modeling and the dashboard. This ensures the original data is never overwritten.
* **Privacy & Ethics:** The dataset contains no Personally Identifiable Information (PII). All analysis is performed at the property level, aggregated by Zip Code or structural features, ensuring ethical compliance.
* **Version Control:** All code, notebooks, and dataset metadata are managed via Git and GitHub to ensure reproducibility.

### 3. Rationale for Methodologies
The research methodologies were chosen based on the specific distribution of the data and the nature of the business questions.

**Statistical Analysis Choices:**
* **Median vs. Mean:** House prices are right-skewed with significant outliers (luxury properties). We utilized the **Median** as the primary central tendency metric to provide a realistic representation of the "typical" buyer experience.
* **Mann-Whitney U Test:** Used to validate the "Waterfront Premium." Since the Shapiro-Wilk test confirmed price data was **non-normal**, this non-parametric test was chosen over the T-Test to compare independent groups.
* **Spearman Correlation:** Used for the "Feature Importance" analysis. Features like `Grade` (1-13) and `Condition` (1-5) are **ordinal** (ranked categories). Spearman correlation is more appropriate than Pearson for detecting monotonic relationships in ranked data.

**Machine Learning Choices:**
* **Random Forest Regressor:** Selected for the Price Prediction tool. Real estate data contains non-linear relationships (e.g., the complex interaction between latitude/longitude and price). Random Forest handles these non-linearities and feature interactions better than Linear Regression, resulting in a higher R² score (0.873) and a more accurate "Fair Value" estimate.

## The rationale to map the business requirements to the Data Visualisations
| Business Requirement | Data Visualisation(s) | Rationale |
| :--- | :--- | :--- |
| **Buyers** | | |
| **1. Affordability:**<br>Identify the top 10 most affordable zip codes. | **Bar Chart:**<br>Top 10 Zip Codes by Median Price. | **Validates Hypothesis 1.**<br>A bar chart allows for a clear ranking of categorical data (zip codes). We use the **Median** rather than the Mean to prevent high-priced outliers (luxury estates) from skewing the perception of affordability, ensuring buyers see a realistic entry point. |
| **2. Scenery & View Value:**<br>Quantify the premium for "Waterfront" and "High View" properties. | **Box Plots & Statistical Tests:**<br>Mann-Whitney U (Waterfront) & Kruskal-Wallis (View). | **Validates Hypothesis 2.**<br>Since price data is non-normal, I used non-parametric tests to confirm significance. The **Mann-Whitney U** test confirmed the waterfront premium, while the **Kruskal-Wallis** test proved that the view quality (from 0 to 4) significantly increases value, with "Excellent" views adding over $750k in median value. |
| **3. Feature Importance:**<br>Determine if House Grade or Condition matters more. | **Heatmap, Spearman Correlation, Box Plot & Bar Plot,:**<br>Interaction between Grade, Condition, and Price. | **Validates Hypothesis 3.**<br>Since Grade and Condition are **ordinal** variables (ranked categories), Spearman Correlation is the appropriate statistical measure. The heatmap visualizes the interaction, confirming that higher construction grades correlate more strongly with price than cosmetic condition. |
| **4. Prediction:**<br>Estimate fair market value to make competitive offers. | **Predictive Model:**<br>Random Forest Regressor. | **Operationalizes Findings.**<br>House prices are influenced by **non-linear relationships** (e.g., the interaction between latitude/longitude and property size). A Random Forest model captures these complexities better than simple linear formulas, providing a "Fair Value" estimate to the buyers |
| **Sellers** | | |
**5. Feature Value:**<br>Identify and validate specific home features that add the most financial value. | **Heatmap & Validation Chart:**<br>Feature Correlation (Pearson vs. Spearman). | **Validates Hypothesis 6.**<br>Sellers need a reliable hierarchy of value. I used a dual-method comparison (Pearson vs. Spearman) to validate that **Living Space** and **Grade** are the top drivers, ensuring outliers didn't skew the results. The analysis visually proves that **Bathrooms** outperform **Bedrooms**, allowing sellers to prioritize the right attributes in marketing. |
| **6. Timing:**<br>Identify the best month to sell for maximum profit. | **Line Chart (Time-Series):**<br>Median Price vs. Month Sold. | **Validates Hypothesis 4.**<br>To identify seasonal trends, a time-series view is essential. This visualization exposes the cyclical nature of the market, highlighting the Spring/Summer peak (April/May) to advise sellers on the optimal window for listing. |
| **7. ROI Analysis:**<br>Determine if renovations yield a statistically significant return. | **Interaction Plot (Bar Chart):**<br>Price by Age Group grouped by Renovation Status. | **Validates Hypothesis 5.**<br>The value of renovation is **not uniform**. A simple average would hide the truth. This plot separates the data by Era (Pre-war, Mid-Century, Modern), revealing that Mid-Century homes yield a significantly higher ROI (~60%) than other eras. |
| **8. Listing Strategy:**<br>Set optimal prices based on neighborhood trends. | **Predictive Model:**<br>Random Forest. | **Operationalizes Findings.**<br>While buyers use the model to find deals, sellers use it to establish a **baseline market value**. By inputting their specific home features, they get a data-driven starting price that removes emotional bias from the listing strategy. |

## Analysis techniques used
* **Methods & Limitations:** I used standard descriptive statistics (like calculating the mean and median) and created visualizations (histograms, boxplots, scatter plots) to find patterns. A main limitation was that the data is older and only covers one year, so I couldn't look at long-term price trends.

* **Analysis Structure:** I structured my analysis to cover the "Big Picture" (looking at the whole  King County) and then drill down into specific details (like bedrooms) and finally location (zip codes). I did this to make sure I understood the both general market and specific neighborhoods.

* **Data Challenges:** The biggest challenge was "outliers" a few massive mansions that skewed the average prices. To fix this, I switched to using the median price instead of the average (mean) because it gives a more accurate picture of a "normal" home.

* **Generative AI Usage:** I used AI tools to help me brainstorm questions to ask the data and to generate ideas first. It was also used heavily for fixing bugs in the code, optimizing and formatting the code, notebooks including charts, and formatting this README documentation.

## Ethical Considerations & Data Privacy
Although this is a public dataset, ethical usage of data is paramount:
* **Privacy:** The dataset contains property features and locations (Latitude/Longitude) but **no Personally Identifiable Information (PII)** regarding previous owners.
* **Fairness:** We excluded variables that could introduce bias (e.g., zip codes were used for location value, not for any sort of demographic profiling).
* **Usage:** Data is used strictly for educational and market analysis purposes, complying with Kaggle's open license terms.

## Dashboard Design - TODO
* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other item that your dashboard library supports.
* Later, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but subsequently you used another plot type).
* How were data insights communicated to technical and non-technical audiences?
* Explain how the dashboard was designed to communicate complex data insights to different audiences. 

## Unfixed Bugs - TODO
* Please mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable to consider, paucity of time and difficulty understanding implementation are not valid reasons to leave bugs unfixed.
* Did you recognise gaps in your knowledge, and how did you address them?
* If applicable, include evidence of feedback received (from peers or instructors) and how it improved your approach or understanding.

## Development Roadmap
The project followed a 5-phase development lifecycle:

1.  **Phase 1: ETL & Cleaning (Notebook 01)**
    * Handling outliers (e.g., the 33-bedroom error).
    * Imputing missing values and fixing data types.
2.  **Phase 2: Feature Engineering (Notebook 02)**
    * Creating `log_price` for normalization.
    * Extracting `sale_month` and `year_renovated`.
3.  **Phase 3: Statistical Analysis (Notebooks 03 & 04)**
    * Conducting Hypothesis Testing (Hypothesis H1-H5).
    * Visualizing correlations and distributions.
4.  **Phase 4: Machine Learning (Notebook 05)**
    * Building and tuning the Random Forest Regressor.
5.  **Phase 5: Dashboard & Documentation**
    * Streamlit app development and README creation.

### Challenges & Strategies
* Data Skewness & Quality: Addressed significant right-skewness in the target variable (price) by applying Log Transformations to improve model performance. Mitigated data quality issues, such as the erroneous "33-bedroom" outlier, through rigorous cleaning and domain-aware filtering.
* Computational Constraints: Overcame performance bottlenecks during Random Forest Hyperparameter Tuning on local hardware (laptop) by optimizing the feature set and prioritizing RandomizedSearchCV over exhaustive grid searches to balance accuracy with training time.

### Future Learning
* Advanced Modeling: I plan to explore Gradient Boosting algorithms (XGBoost, LightGBM) to potentially surpass the performance of the Random Forest model.
* Advanced Statistical Analysis: Aim to deepen expertise in Statistical Inference, specifically mastering the application of complex Parametric vs. Non-Parametric tests


## Deployment
* The App live link is: https://YOUR_APP_NAME.streamlit.com/ 

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/dumindagamage/House-Price-Analysis.git)
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd [REPO-NAME]
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit Dashboard:**
    ```bash
    streamlit run dashboard/app.py
    ```


## Main Data Analysis Libraries
* **Language:** Python 3.12.X
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Plotly, Seaborn, Matplotlib
* **Machine Learning:** Scikit-Learn (Pipeline, RandomForestRegressor)
* **Dashboarding:** Streamlit


## Credits 
* **Dataset:** [Kaggle House Sales in King County](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction).
* **Template:** Code Institute README Template.
* **CI LMS:** Code Institute LMS.
* **AI Tools:** GitHub Co-Pilot, Google Gemini

### Content
- The image in the README was generated with Google Gemini

### Media - TODO

- The photos used on the home and sign-up page are from This Open-Source site
- The images used for the gallery page were taken from this other open-source site


## Acknowledgements - TODO
* Thank the people who provided support through this project.