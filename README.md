# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

---

# 🏠 King County Housing Analytics 🏠

![](https://github.com/dumindagamage/Home-Value-Analysis/blob/main/resources/images/home_analysis_banner.png?raw=true)

---
## 📝 Project Overview

This Data Analytics project analyzes housing sales data in King County, USA (May 2014 - May 2015). The goal is to provide actionable insights for two distinct user groups: **Buyers** looking for value and affordability, and **Sellers** aiming to optimize the selling price. 

The project culminates in an interactive Streamlit dashboard that allows users to explore market trends, assess property value based on size, grade, condition and location, and predict house prices using a Machine Learning model.

## 📑 Executive Summary
**Client:** King County Real Estate Agency.  
**Goal:** The client wants to launch a digital dashboard to assist two user groups:
1. **Buyers:** Finding real value of the properties and understanding affordability.
2. **Sellers:** Determining the optimal pricing strategy, listing time and renovation strategy.

## 💾 Dataset Content
The project uses the **King County House Sales dataset** (`kc_house_data.csv`).
* **Source:** [Kaggle - House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
* **Volume:** ~21,613 records (from May 2014 - May 2015)
* **Features:** 21 variables including Price (Target), Bedrooms, Bathrooms, Sqft Living, Floors, Waterfront, View, Condition, Grade, Zipcode, and Year Built.

## 🎯 Business Requirements
### User Group 1: Buyers
* **Affordability:** Identify the top 10 most affordable zip codes.
* **Value Assessment:** Quantify the premium for scenic attributes (Waterfront or High-Quality Views) for properties.
* **Feature Importance:** Determine house grade vs condition impact on price.
* **Prediction:** Estimate fair market value to make competitive offers.

### User Group 2: Sellers
* **Feature Value:** Determine which specific home attributes contribute most significantly to property valuation
* **ROI Analysis:** Determine if renovations yield a significant return.
* **Timing:** Identify the best month to sell for maximum profit.
* **Listing Strategy:** Set optimal prices based on neighborhood market trends.

## 🏗️ Project Management
This project was managed using Agile methodologies with a **GitHub Project Board**.
* **Kanban Board:** [Link to your GitHub Project Board](https://github.com/users/dumindagamage/projects/5/views/1)
![Project Board](https://github.com/dumindagamage/House-Price-Analysis/blob/wip/resources/images/project_board.png)

## 🗓️ Project Plan
### 1. High-Level Steps
The analysis was executed in four phases:

* **Phase 1: Business Understanding & Data Collection**
    * **Goal:** Defined the scope to help Buyers find value and Sellers maximize profit.
    * **Data:** Acquired the King County Housing dataset (21k+ records) from Kaggle.
    * **Hypotheses:** Set up core questions regarding location, condition, and how renovations affect value.

* **Phase 2: Data Cleaning, EDA & Transformation**
    * **Cleaning:** Handled missing values, removed duplicates, and corrected data types (e.g., converting `date` to datetime).
    * **EDA:** Checked data distributions and explored relationships between variables.
    * **Feature Engineering:** Created new metrics to drive better insights, such as `price_per_sqft`, `house_age`, and `age_group` (Pre-War, Mid-Century, Modern).

* **Phase 3: Hypothesis Testing (Buyer & Seller Personas)**
    * **Analysis:** Conducted a granular analysis to investigate key business questions tailored to specific user personas.
    * **Validation:** Where applicable, Used statistical tests to prove that the findings were real, not just random chance.

* **Phase 4: Predictive Modeling & Dashboarding**
    * **Modeling:** Trained Machine Learning models to estimate house prices.
    * **Dashboard:** Built an interactive Streamlit dashboard to allow users to explore the data and findings themselves.

### 2. Data Management Strategy
Data integrity was maintained through a strict separation of concerns:

* **Storage Architecture:** The project utilizes a "Raw vs. Processed" directory structure.
    * `data/raw/`: Contains the immutable original CSV file (`kc_house_data.csv`).
    * `data/processed/`: Stores the cleaned (`cleaned_house_data.csv`) and transformed (`final_house_data.csv`) dataset used for modeling and the dashboard. This ensures the original data is never overwritten.
* **Privacy & Ethics:** The dataset contains no Personally Identifiable Information (PII). All analysis is performed at the property level, aggregated by Zip Code or structural features, ensuring ethical compliance.
* **Version Control:** All code, notebooks, and dataset metadata are managed via Git and GitHub to ensure reproducibility.

## 🧪 Project Hypothesis & Validations
The following hypotheses were posed at the project's inception and validated through the data analysis process.

* **Hypothesis 1:** Geographic location (Zip Code) is the primary driver of affordability, creating distinct clusters of accessible housing.
    * **Validation:** Median Price Analysis per Zipcode (Bar Chart).
* **Hypothesis 2:** Properties with scenic attributes (Waterfront or High-Quality Views) command a statistically significant price premium compared to standard properties.
    * **Validation:** Mann-Whitney U Test (Waterfront) & Kruskal-Wallis Test (View).
* **Hypothesis 3:** Construction Grade has a stronger correlation with price than Condition, implying buyers pay more for structural quality than cosmetic state.
    * **Validation:** Spearman Rank Correlation & Heatmap visualization.
* **Hypothesis 4:** Sales prices follow a seasonal trend, suggesting an optimal window (Spring/Summer) for sellers to maximize profit.
    * **Validation:** Time-Series Trend Analysis & Kruskal-Wallis Test.
* **Hypothesis 5:** The ROI of renovation is not uniform; it provides a significantly higher return for Mid-Century homes (1950-1990) than for pre-war or modern builds.
    * **Validation:** Multivariate Interaction Plot (Price × Era × Renovation Status).
* **Hypothesis 6:** Total living space (`sqft_living`) is a stronger predictor of price than simple room counts (`bedrooms` or `bathrooms`), indicating buyers value overall volume over segmentation.
    * **Validation:** Correlation Matrix Comparison (Pearson vs. Spearman).


## 📊 The rationale to map the business requirements to the Data Visualisations
| Business Requirement | Data Visualisation(s) | Rationale & Hypothesis Outcome |
| :--- | :--- | :--- |
| **Buyers** | | |
| **1. Affordability:**<br>Identify the top 10 most affordable zip codes. | **Bar Chart:**<br>Top 10 Zip Codes by Median Price. | **Validates Hypothesis 1 (Confirmed).**<br>A bar chart allows for a clear ranking of zip codes. I used the **Median** to prevent high-priced outliers from skewing affordability perception. The analysis identified specific areas (e.g., 98002) where median prices are <50% of the county average. |
| **2. Scenery & View Value:**<br>Quantify the premium for "Waterfront" and "High View" properties. | **Box Plots & Statistical Tests:**<br>Mann-Whitney U (Compare the binary "Waterfront vs. Non-Waterfront" groups) & Kruskal-Wallis (View -  Comparing > 2 groups on non-normal data). | **Validates Hypothesis 2 (Confirmed).**<br>Since price data is non-normal, non-parametric tests were essential. **Mann-Whitney U** confirmed the waterfront premium is statistically significant, while **Kruskal-Wallis** proved that "Excellent" views (Rating 4) add substantial value over standard views. |
| **3. Feature Importance:**<br>Determine if House Grade or Condition matters more. | **Heatmap & Spearman Correlation:**<br>Interaction between Grade, Condition, and Price. | **Validates Hypothesis 3 (Confirmed).**<br>Since Grade and Condition are ordinal variables, Spearman Correlation is appropriate. The heatmap visually confirms that **Grade (~0.65)** correlates far more strongly with price than **Condition (~0.04)**. |
| **4. Prediction:**<br>Estimate fair market value to make competitive offers. | **Predictive Model:**<br>Random Forest Regressor. | **Operationalizes Findings.**<br>House prices are influenced by non-linear relationships (e.g., location coordinates x size). Random Forest captures these complexities better than linear formulas, providing a precise "Fair Value" estimate. |
| **Sellers** | | |
| **5. Feature Value:**<br>Identify specific home features that add the most financial value. | **Heatmap & Validation Chart:**<br>Feature Correlation (Pearson vs. Spearman). | **Validates Hypothesis 6 (Confirmed).**<br>A dual-method comparison (Pearson/Spearman) validated that **Living Space (0.70)** is the dominant price driver, significantly outperforming simple room counts like Bedrooms (0.32). This allows sellers to focus marketing on space rather than room number. |
| **6. Timing:**<br>Identify the best month to sell for maximum profit. | **Line Chart (Time-Series):**<br>Median Price vs. Month Sold **& Statistical Tests:** Kruskal-Wallis Test (Comparing > 2 groups on non-normal data) | **Validates Hypothesis 4 (Confirmed).**<br>Time-series analysis reveals a clear cyclical trend. Median prices and volume consistently tick upward starting in **April/May**, confirming this as the optimal listing window with statistical test as well. |
| **7. ROI Analysis:**<br>Determine if renovations yield a statistically significant return. | **Interaction Plot (Bar Chart):**<br>Price by Age Group grouped by Renovation Status. | **Validates Hypothesis 5 (Confirmed).**<br>The value of renovation is not uniform. The interaction plot reveals that **Mid-Century homes (1950-1990)** yield a significantly higher ROI (~60%) from renovation compared to Pre-War or Modern homes. |
| **8. Listing Strategy:**<br>Set optimal prices based on neighborhood and features. | **Predictive Model:**<br>Random Forest Regressor. | **Operationalizes Findings.**<br>While buyers use the model to find deals, sellers use it to establish a **baseline market value**, ensuring they list aggressively but realistically based on their specific features. |


## ⚙️ Rationale for Methodologies
The research methodologies were chosen based on the specific distribution of the data and the nature of the business questions.

**Statistical Analysis Choices:**
* **Median vs. Mean:** House prices are right-skewed with significant outliers (luxury properties). I utilized the **Median** as the primary central tendency metric to provide a realistic representation of the "typical" buyer experience.
* **Mann-Whitney U Test:** Used to validate the "Waterfront Premium." Since the Shapiro-Wilk test confirmed price data was **non-normal**, this non-parametric test was chosen over the T-Test to compare independent groups.
* **Kruskal-Wallis H Test:** Used to validate the "View Premium." Since the Shapiro-Wilk test confirmed price data was non-normal, this non-parametric test was chosen over One-Way ANOVA to compare price distributions across the five distinct view categories (0–4).
* **Spearman Correlation:** Used for the "Feature Importance" analysis. Features like `Grade` (1-13) and `Condition` (1-5) are **ordinal** (ranked categories). Spearman correlation is more appropriate than Pearson for detecting monotonic relationships in ranked data.

**Machine Learning Choices:**

* **Model Evaluation:** Conducted a comparative analysis of three algorithms to balance complexity with accuracy:
    1.  **Linear Regression:** Established a baseline for performance.
    2.  **Decision Tree Regressor:** Tested for capturing non-linear patterns.
    3.  **Random Forest Regressor:** Tested for ensemble accuracy and generalization.

* **Target Transformation:** Applied a Log Transformation (`np.log1p`) to the target variable (`price`). Since real estate data is highly right-skewed, this normalized the distribution, reduced the impact of extreme luxury outliers, and improved the model's ability to minimize relative error.

* **Pre-Pipeline Processing:** Before entering the machine learning pipeline, the dataset underwent strict manual filtering to ensure integrity:
    * **Leakage Prevention:** Dropped columns directly containing target information (`price`, `price_log`, `price_per_sqft`) to prevent data leakage.
    * **Noise Reduction:** Removed non-predictive identifiers and metadata columns (`id`, `date`, `sale_month_name`, `age_group`) that do not contribute to property valuation.

* **Pipeline Architecture:** To ensure reproducibility and prevent data leakage, all models were evaluated using a Scikit-Learn pipeline consisting of:
    * **Data encoder:**  Encode `zipcode` with OneHotEncoder from feature_engine (Important for Linear models).
    * **Feature Selection:** SmartCorrelatedSelection(threshold=0.85) from feature_engine. This automatically finds correlated features (like sqft_living vs sqft_above) and removes the redundant ones to reduce noise.
    * **Feature Scaling:** StandardScaler (Important for Linear models). `StandardScaler` normalized all numerical variables to a standard scale (mean=0, variance=1) to ensure consistent interpretation during the model evaluation phase.

* **Hyperparameter Tuning (HPT):** Performed rigorous tuning on the Random Forest model using **GridSearchCV** with 3-fold cross-validation. The analysis revealed that the **default hyperparameters** actually yielded a higher R² score and better generalization than the tuned parameters. Consequently, the default configuration was retained to prevent overfitting.

* **Final Selection (Random Forest Regressor):** The Random Forest model was selected for the final Price Prediction tool. It handled the dataset's non-linearities (e.g., the complex interaction between Latitude/Longitude and Price) significantly better than Linear Regression, achieving a robust **R² score of 0.865**.

![](https://github.com/dumindagamage/House-Price-Analysis/blob/main/resources/images/rfmodel.png?raw=true)


## 📉 Analysis techniques used
* **Methods & Limitations:** I used standard descriptive statistics (mean, median, std, percentile, min and max) and created visualizations (histograms, boxplots, scatter plots) to find patterns. A main limitation was that the data is older and only covers one year, so I couldn't look at long-term price trends.

* **Analysis Structure:** I structured my analysis to cover the "Big Picture" (looking at the whole  King County) and then drill down into specific details and finally location (zip codes). I did this to make sure I understood the both general market, price drivers, patterns and specific neighborhoods.

* **Data Challenges:** The biggest challenge was "outliers" a few massive mansions that skewed the average prices. To fix this, I switched to using the median price instead of the average (mean) because it gives a more accurate picture of a "normal" home.

* **Generative AI Usage:** I used AI tools to help me brainstorm questions to ask the data and to generate ideas first. It was also used heavily for fixing bugs in the code, optimizing and formatting the code, notebooks including charts, and optinmizing, formatting this README documentation. While AI tools facilitated workflow efficiency, all analytical reasoning, modeling strategies, and code execution were exclusively performed by the author, ensuring strict adherence to academic integrity.

## ⚖️ Ethical Considerations & Data Privacy
Although this is a public dataset, ethical usage of data is paramount:
* **Privacy:** The dataset contains property features and locations (Latitude/Longitude) but **no Personally Identifiable Information (PII)** regarding owners.
* **Fairness:** I excluded variables that could introduce bias (e.g., zip codes were used for location value, not for any sort of demographic profiling).
* **Usage:** Data is used strictly for educational and market analysis purposes, complying with Kaggle's open license terms.

## 🎨 Dashboard Design
The dashboard was designed with accessibility for non-technical users as a priority. Visualizations were carefully selected for clarity to ensure insights are immediately understandable, while Plotly was utilized to deliver a visually appealing and interactive user experience. Furthermore, the dedicated "General Recommendations" sections for both buyers and sellers synthesize key findings into plain language, allowing users to grasp critical market trends without needing to interpret complex charts.
* **Project Overview:** A high-level summary of the dataset (Total Sales, Average Price) featuring an interactive map to visualize property distribution across King County.
* **Buyer Insights:** Tools designed to help buyers find value. This section identifies the most affordable zip codes, analyzes the relationship between house size and price, and helps quantify the "Scenery Value" (View/Waterfront premiums).
* **Seller Analytics:** Tools designed to maximize profit. This section highlights the best months to sell (Seasonality), ranks the most valuable home features, and calculates the ROI of renovations based on the property's era.
* **Price Estimator:** An interactive prediction tool where users select their role (Buyer or Seller). It provides a specific "Fair Value" estimate with a calculated safety margin (taking MAE as the margin) to help negotiate deals or set listing prices. 

Dashboard: https://house-price-analysis-kcha.streamlit.app/
![](https://github.com/dumindagamage/House-Price-Analysis/blob/main/resources/images/dashboard.png?raw=true)


## 💡 Conclusion
This project demonstrates that successful real estate decisions in King County are driven by specific, quantifiable factors rather than general market movements:

* **For Buyers:** Value is best found by prioritizing **"Mid-Century" homes with lower condition ratings but high construction grades**, as these offer the best price-per-square-foot entry point with high renovation potential. Buyers should also be aware that "Scenery" (Waterfront/View) commands a statistically validated premium that acts as a distinct luxury tax on top of standard living space costs.
* **For Sellers:** Maximizing profit relies on timing and feature highlighting. Listing in **April/May** captures peak seasonal demand. Furthermore, marketing efforts should focus heavily on **Total Living Space** and **Construction Quality (Grade)** rather than room counts, as these are the strongest drivers of final sale price. For owners of homes built between 1950-1990, renovation offers a significantly higher ROI than for any other property era.

## 🐛 Unfixed Bugs 
At the time of final deployment, there are **no known unfixed bugs**. All core features, including the predictive model, dashboard filtering, and interactive charts, function as expected.
**Closing Knowledge Gaps**
During the project lifecycle, I identified gaps in my understanding of advanced ML models, hyper parametyer tuning, advanced statistical testing and the specific dependency requirements for cloud deployment. I addressed these challenges by:
* **Revisiting Core Concepts:** I reviewed specific modules in the Code Institute LMS to reinforce my understanding of hypothesis testing.
* **Documentation:** I relied heavily on the official documentation for **Streamlit** and **Scikit-Learn** to understand version compatibility (e.g., solving the `monotonic_cst` error).
* **AI Assistance:** I utilized AI tools to clarify doubts regarding complex concepts and to optimize code for performance.

## 🛣️ Development Roadmap
The project followed a 5-phase development lifecycle:

1.  **Phase 1: ETL & Cleaning (Notebook 01)**
    * Handling outliers (e.g., the 33-bedroom error).
    * Imputing missing values and fixing data types.
2.  **Phase 2: Feature Engineering (Notebook 02)**
    * Creating `log_price` for normalization.
    * Extracting `sale_month` and `year_renovated`.
3.  **Phase 3: Statistical Analysis (Notebooks 03 & 04)**
    * Conducting Hypothesis Testing (Hypothesis H1-H6).
    * Visualizing correlations and distributions.
4.  **Phase 4: Machine Learning (Notebook 05)**
    * Building and tuning the Random Forest Regressor.
5.  **Phase 5: Dashboard & Documentation**
    * Streamlit dashboard development and README creation.

## 🏔️ Challenges & Strategies
* **Data Skewness & Quality:** Addressed significant right-skewness in the target variable (price) by applying Log Transformations to improve model performance. Mitigated data quality issues, such as the erroneous "33-bedroom" outlier, through rigorous cleaning and domain-aware filtering.
* **Computational Constraints:** Overcame performance bottlenecks during Random Forest Hyperparameter Tuning on local hardware (laptop) by narrowed it down to the most impactful ranges for Random Forest to test and balanced the accuracy with training time.

## 🎓 Learning Reflection

* This project refined my ability to manage complex, real-world datasets and bridge the gap between classical analysis and modern machine learning. By architecting predictive pipelines and performing rigorous hyperparameter tuning, I gained a deeper understanding of the trade-offs between accuracy and performance in a business context.

* Developing the Streamlit dashboard significantly improved my data storytelling skills, teaching me how to translate technical metrics into actionable business insights for non-technical stakeholders. 

* Furthermore, integrating AI tools into my workflow demonstrated how to leverage generative assistance responsibly to accelerate development without compromising code quality. This experience has prepared me for continuous learning in advanced statistical modeling and domain-specific analytics.

## 📚 Future Learning
* **Advanced Modeling:** I plan to explore Boosting algorithms (XGBoost, etc..) to potentially surpass the performance of the Random Forest model and dig deeper into hyper-parameter tuning. 
* **Advanced Statistical Analysis:** Aim to deepen expertise in Statistical Inference, specifically mastering the application of complex Parametric vs. Non-Parametric tests.


## 🚀 Deployment
* The Dashboard live link: https://house-price-analysis-kcha.streamlit.app/

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
    streamlit run dashboard/dashboard.py
    ```

## 🐍 Main Data Analysis Libraries
* **Language:** Python 3.12.X
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Plotly, Seaborn, Matplotlib
* **Statistical Testing:** Pingouin , Scipy 
* **Machine Learning:** Feature-Engine, Scikit-Learn (Pipeline, RandomForestRegressor)
* **Dashboarding:** Streamlit


## 📜 Credits 
* **Dataset:** [Kaggle House Sales in King County](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction).
* **Template:** Code Institute README Template.
* **CI LMS:** Code Institute LMS.
* **AI Tools:** GitHub Co-Pilot, Google Gemini

### Content
* The custom imagery featured in both the Dashboard and the README was generated using Google Gemini

### Media 
* **Dataset:** [House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) (Kaggle).
* **Header Image:** Custom dashboard banner created for this project (`resources/images/dashboard_header.png`).
* **Icons:** Dashboard interface icons provided by Streamlit.


## 👏 Acknowledgements
* **Code Institute:** For the Data Analytics curriculum, learning materials, and assessment criteria. Special thanks to the instructors and course coordinator for their support.
* **Kaggle:** For providing the open-source dataset used in this analysis.
* **AI Tools:**
    * **GitHub Copilot:** For code auto-completion and troubleshooting syntax errors.
    * **Google Gemini:** For code optimization, debugging complex functions, and refining the technical documentation.