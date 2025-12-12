# ==============================================================================================================
# Framework: Built upon the standardized single-page Streamlit template provided by the Code Institute.
# UX/UI: Leveraged Plotly to create an engaging, interactive user experience that simplifies complex data exploration.
# Data Handling: To prevent model failure on limited user inputs, the system constructs a complete input vector by using local median values for all background features.
# ==============================================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# =============================================================================
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="King County Housing Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.image("resources/images/dashboard_header.png", use_container_width=True)

# --- CUSTOM CSS ---
# Adjust tab font size and weight (Generated with AI assistance)
st.markdown("""
<style>
    button[data-baseweb="tab"] {
        font-size: 18px !important;
        font-weight: 700 !important;
    }
    .metric-card {
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #ddd;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .highlight-buyer {
        border-left: 5px solid #2ecc71;
        background-color: #f0f9f4;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .highlight-seller {
        border-left: 5px solid #e67e22;
        background-color: #fdf2e9;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. DATA & MODEL LOADING
# -----------------------------------------------------------------------------
DATA_PATH = 'data/processed/final_house_data.csv'
MODEL_PATH = 'data/models/house_price_model.pkl'

# Model Metrics
MODEL_METRICS = {
    "R2": 0.865,
    "MAE": 67640,
    "RMSE": 132851
}

@st.cache_data # memory caching for data loading for memory optimization
def load_data():
    paths_to_try = [DATA_PATH, '../' + DATA_PATH]
    df = None
    loaded_path = ""
    
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                loaded_path = path
                break
            except:
                continue
                
    if df is None:
        st.error(f"❌ Critical Error: Could not find dataset at {DATA_PATH}.") # Check DATA_PATH
        st.stop()

    # --- CLEANING & PREP ---
    if 'Unnamed: 0' in df.columns: # Drop Index Column
        df = df.drop(columns=['Unnamed: 0'])

    if 'date' in df.columns: # Date type conversion & Create Month Name
        df['date'] = pd.to_datetime(df['date'])
        df['month_name'] = df['date'].dt.strftime('%b')
    
    # Ensure String Zipcode
    if 'zipcode' in df.columns:
        df['zipcode'] = df['zipcode'].astype(str).str.replace(r'\.0$', '', regex=True)

    # Feature: House Age
    if 'house_age' not in df.columns and 'yr_built' in df.columns:
        df['house_age'] = df['sale_year'] - df['yr_built']

    # Feature: Era Categorization with explanatory labels
    def get_era(yr):
        if yr < 1950: return 'Pre-War'
        elif yr <= 1990: return 'Mid-Century'
        else: return 'Modern'
        
    df['era'] = df['yr_built'].apply(get_era)
    
    # Feature: Price per SqFt
    df['price_per_sqft'] = df['price'] / df['sqft_living']

    return df, loaded_path

@st.cache_resource
def load_model():
    paths_to_try = [MODEL_PATH, '../' + MODEL_PATH]
    model = None
    loaded_path = ""
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                loaded_path = path
                break
            except Exception as e:
                st.error(f"Error loading model: {e}")
                continue
    return model, loaded_path

df, data_source = load_data()
model, model_source = load_model()

# =============================================================================
# 3. HELPER: MODEL INPUT PREP (FIXED)
# -----------------------------------------------------------------------------
def prepare_input_data(user_inputs, ref_df, trained_model):
    """
    Constructs a DataFrame matching the EXACT feature set of the trained model.
    """
    # 1. Filter Context (Zipcode)
    zip_subset = ref_df[ref_df['zipcode'] == user_inputs['zipcode']]
    if zip_subset.empty: zip_subset = ref_df 
    
    # Create template row from medians (Includes ALL numeric columns like lat, long, etc)
    input_df = pd.DataFrame([zip_subset.select_dtypes(include=[np.number]).median()])
    
    # 2. Overwrite with User Inputs
    input_df['bedrooms'] = user_inputs['bedrooms']
    input_df['bathrooms'] = user_inputs['bathrooms']
    input_df['sqft_living'] = user_inputs['sqft_living']
    input_df['floors'] = user_inputs['floors']
    input_df['waterfront'] = 1 if user_inputs['waterfront'] else 0
    input_df['view'] = user_inputs['view']
    input_df['condition'] = user_inputs['condition']
    input_df['grade'] = user_inputs['grade']
    input_df['yr_built'] = user_inputs['yr_built']
    input_df['zipcode'] = str(user_inputs['zipcode'])
    
    # 3. Setting values for the engineered features
    input_df['sale_year'] = 2015 # Set latest year in dataset
    input_df['house_age'] = input_df['sale_year'] - input_df['yr_built']
    if 'sale_month' not in input_df.columns:
        input_df['sale_month'] = 6 # Default to June if missing
    
    # --- CRITICAL STEP: FEATURE ALIGNMENT ---
    # Need to pass the exact columns the model expects.  
    if hasattr(trained_model, 'feature_names_in_'):
        # Best Case: Extract features from the model 
        expected_cols = trained_model.feature_names_in_
        
        # Add any missing columns (set to 0)
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 0
                
        # Select & Sort to match Model EXACTLY
        input_df = input_df[expected_cols]
        
    else:
        # Manually drop the known "Dashboard-Only" columns
        cols_to_drop = [
            'price', 'price_log', 'price_per_sqft', 'id', 'date', 
            'sale_month_name', 'age_group', 'era', 'sqft_bin', 
            'month_name', 'Unnamed: 0', 'is_renovated'
        ]
        existing_drop = [c for c in cols_to_drop if c in input_df.columns]
        input_df = input_df.drop(columns=existing_drop)

    return input_df

# =============================================================================
# 4. SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.title("🏠 KCHA Dashboard")
st.sidebar.markdown("---")

# Global Filters
st.sidebar.header("📉 Global Filters")
all_zipcodes = sorted(df['zipcode'].unique())
selected_zipcodes = st.sidebar.multiselect("Filter Charts by Zipcode", all_zipcodes, default=[])
st.sidebar.markdown("---")

# Location Price Checker
with st.sidebar.container(border=True):
    st.markdown("### 📍 Location Price Checker")
    st.info("Quickly check market average for any Zipcode.")
    check_zip = st.selectbox("Select Zipcode:", sorted(df['zipcode'].unique()), key="checker_zip")
    if check_zip:
        area_stats = df[df['zipcode'] == check_zip]['price']
        c1, c2 = st.columns(2)
        c1.metric("Average (Median) Price", f"${area_stats.median():,.0f}")

if not selected_zipcodes:
    filtered_df = df.copy()
else:
    filtered_df = df[df['zipcode'].isin(selected_zipcodes)]

# =============================================================================
# 5. MAIN TABS
# -----------------------------------------------------------------------------
#st.title("King County Housing Analytics 🏠")
st.markdown("""
<style>
    /* Style the tab labels */
    button[data-baseweb="tab"] {
        font-size: 30px !important;  /* Change font size */
        font-weight: 700 !important; /* Make text bold */
    }
    
    /* Optional: Style the active tab to look distinct */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #f0f2f6 !important;
        border-radius: 5px;
    }
    
    /* Custom Metric Card Style */
    div.metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    div.metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    div.metric-value {
        font-size: 26px;
        font-weight: bold;
        color: #2c3e50;
        margin: 5px 0;
    }
    div.metric-label {
        font-size: 14px;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div.metric-icon {
        font-size: 24px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# --- TABS ---
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Project Overview", 
    "🔍 Buyer Insights", 
    "📈 Seller Analytics", 
    "🤖 Price Estimator"
])

# =============================================================================
# --- TAB 1: SUMMARY ---
# -----------------------------------------------------------------------------
with tab1:
    # ... inside Tab 1 ...
    st.subheader("Property Overview")

    # Calculate Stats
    total_sales = filtered_df.shape[0]
    avg_price = filtered_df['price'].mean()
    median_price = filtered_df['price'].median()

    # Create 3 Columns for Cards
    k1, k2, k3 = st.columns(3)

    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🏠</div>
            <div class="metric-value">{total_sales:,}</div>
            <div class="metric-label">Total Homes Sold</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">💵</div>
            <div class="metric-value">${median_price:,.0f}</div>
            <div class="metric-label">Market Average (Median)</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📈</div>
            <div class="metric-value">${avg_price:,.0f}</div>
            <div class="metric-label">Market Average (Ceiling)</div>
        </div>
        """, unsafe_allow_html=True)
   
    st.markdown("---")

    # 1. Professional Heading & Instruction
    st.subheader("🗺️ Geographic Distribution of Property Sales")
    st.caption("Explore how property prices are distributed across the county. Zoom in to see specific neighborhoods and hover over points for details.")

    # 2. Performance Optimization: Sample data if dataset is too large
    # Rendering 21k points on a map can lag the browser, so we sample 3,000 points for smoothness.
    map_data = filtered_df.sample(min(len(filtered_df), 3000), random_state=42)

    # 3. Create Interactive Map
    fig_map = px.scatter_mapbox(
        map_data, 
        lat="lat", 
        lon="long", 
        color="price", 
        size="price",                  # Bubble size varies slightly by price
        size_max=12,                   # Cap the bubble size
        zoom=9,
        hover_data=['zipcode', 'price'], # <--- Added Zipcode to Tooltip
        mapbox_style="carto-positron", 
        color_continuous_scale="RdYlGn_r", # Red = Expensive, Green = Cheap (or inverted based on preference)
        labels={'price': 'Sale Price'}
    )

    # 4. Map Layout Adjustments
    fig_map.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0}, # Remove white margins
        height=500                        # Set a fixed height for better visibility
    )

    st.plotly_chart(fig_map, use_container_width=True)


# =============================================================================
# RECOMMENDATIONS CARD STYLE
# =============================================================================   
    recomendation_card_style = """
    <style>
    .rec-card {
        background-color: #e8f4f8; padding: 20px; border-radius: 10px;
        border-left: 5px solid #3498db; margin-bottom: 10px;
    }
    .rec-title { font-weight: bold; font-size: 18px; color: #2c3e50; margin-bottom: 5px; }
    .rec-text { font-size: 14px; color: #34495e; }
    </style>
    """
    st.markdown(recomendation_card_style, unsafe_allow_html=True)
# =============================================================================
# --- TAB 2: BUYER ANALYSIS ---
# -----------------------------------------------------------------------------
with tab2:
    st.header("Analysis for Buyers 🔍")
    
    b1, b2 = st.columns(2)
    with b1:
        # st.subheader("1. Most Affordable Zipcodes")
        st.markdown("""<div class="rec-card"><div class="rec-title">📍 Location & Affordability</div>
            <div class="rec-text">- The data identifies specific zip codes, such as 98002 and 98168, as distinct affordable clusters that trade at less than 50% of the county median price.</div></div>""", unsafe_allow_html=True)
        affordability = df.groupby('zipcode')['price'].median().sort_values().head(10).reset_index()
        overall_median = df['price'].median()
        
        fig_afford = px.bar(
            affordability, x='zipcode', y='price', text_auto='.2s',
            title="Average Price by Zipcode (vs. County Median)",
            labels={'price': 'Median Price ($)', 'zipcode': 'Zip Code'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_afford.add_hline(y=overall_median, line_dash="dot", line_color="red", 
                             annotation_text=f"County Median: ${overall_median:,.0f}")
        fig_afford.update_layout(xaxis_type='category')
        st.plotly_chart(fig_afford, use_container_width=True)

    with b2:
        # st.subheader("2. Best Value for Money (Size)")
        st.markdown("""<div class="rec-card"><div class="rec-title">📐 Best Value for Money (Size) </div>
                    <div class="rec-text">- The 2,000–2,500 sq. ft. size bracket demonstrates the lowest Price per Square Foot, representing the most efficient volume-to-cost ratio in the market.</div>
                    <div class="rec-text">- Filter by  <b>Zipcode</b> to analyze specific local market information.</div></div>""", unsafe_allow_html=True)  
        filtered_df['sqft_detailed_bin'] = pd.cut(filtered_df['sqft_living'], bins=range(0, 5000, 250))
        size_value = filtered_df.groupby('sqft_detailed_bin')['price_per_sqft'].median().reset_index()
        size_value['sqft_mid'] = size_value['sqft_detailed_bin'].apply(lambda x: x.mid)
        
        fig_value = px.line(
            size_value, x='sqft_mid', y='price_per_sqft', markers=True,
            title="Price per Sq. Ft. by House Size",
            labels={'sqft_mid': 'House Size (Sq. Ft.)', 'price_per_sqft': 'Price / Sq. Ft. ($)'},
            color_discrete_sequence=['#2ca02c']
        )
        st.plotly_chart(fig_value, use_container_width=True)

    st.markdown("---")
    st.markdown("""<div class="rec-card"><div class="rec-title">🏗️ House Condition & Grade</div>
                <div class="rec-text">- Analysis shows that Construction Grade has a stronger correlation with value than cosmetic Condition. High-grade structures in fair condition historically hold more value than lower-grade properties in perfect condition.</div>
                <div class="rec-text">- Filter by  <b>Zipcode</b> to analyze specific local market information.</div></div>""", unsafe_allow_html=True)  

    b3, b4 = st.columns(2)
    with b3:
        #st.subheader("3. Average Price vs. Grade")
        def categorize_grade(g):
            if g <= 6: return "Low (1-6)"
            elif g <= 9: return "Good (7-9)"
            else: return "Excellent (10-13)"
        
        filtered_df['grade_cat'] = filtered_df['grade'].apply(categorize_grade)
        grade_order = ["Low (1-6)", "Good (7-9)", "Excellent (10-13)"]
        grade_price = filtered_df.groupby('grade_cat')['price'].median().reindex(grade_order).reset_index()
        
        fig_grade = px.bar(grade_price, x='grade_cat', y='price', text_auto='.2s',
                           title="Average Price by Construction Grade",
                           color='grade_cat', color_discrete_map={"Low (1-6)": "#bdc3c7", "Good (7-9)": "#f39c12", "Excellent (10-13)": "#27ae60"})
        fig_grade.update_layout(showlegend=False)
        st.plotly_chart(fig_grade, use_container_width=True)

    with b4:
        #st.subheader("4. Average Price vs. Condition")
        cond_price = filtered_df.groupby('condition')['price'].median().reset_index()
        fig_cond = px.bar(cond_price, x='condition', y='price', text_auto='.2s',
                          title="Median Price by House Condition", color='price', color_continuous_scale='OrRd')
        fig_cond.update_xaxes(tickmode='linear', dtick=1)
        st.plotly_chart(fig_cond, use_container_width=True)

    st.markdown("---")
    st.markdown("""<div class="rec-card"><div class="rec-title">🌊 Scenery Attributes</div>
                    <div class="rec-text">- Waterfront and High-View properties command a significant price jump independent of other key house features. Properties without these features reflect value based primarily on functional utility (Grade and Sq. Ft.).</div>
                    <div class="rec-text">- Filter by  <b>Zipcode</b> to analyze specific local market information.</div></div>""", unsafe_allow_html=True)
    b5, b6 = st.columns(2)
    with b5:
        view_price = filtered_df.groupby('view')['price'].median().reset_index()
        fig_view_bar = px.bar(view_price, x='view', y='price', text_auto='.2s',
                              title="Average Price by View", color='price', color_continuous_scale='Teal')
        st.plotly_chart(fig_view_bar, use_container_width=True)

    with b6:
            wf_stats = filtered_df.groupby('waterfront')['price'].median().reset_index()
            wf_stats['label'] = wf_stats['waterfront'].map({0: 'Non-Waterfront', 1: 'Waterfront'})
            fig_wf_pie = px.pie(wf_stats, values='price', names='label', title="Average Price by for Waterfront Properties",
                                color='label', color_discrete_map={'Non-Waterfront': '#bdc3c7', 'Waterfront': '#3498db'}, hole=0.4)
            fig_wf_pie.update_traces(textinfo='label+value', texttemplate='%{label}<br>$%{value:,.0f}')
            st.plotly_chart(fig_wf_pie, use_container_width=True)

# =============================================================================
# --- TAB 3: SELLER ANALYSIS ---
# -----------------------------------------------------------------------------
with tab3:
    st.header("Analysis for Sellers 📈")
    
    # ROW 1: Drivers & Expensive Markets
    s1, s2 = st.columns(2)
    
    with s1:
        st.markdown("""<div class="rec-card"><div class="rec-title">✨ Relative Influence of Features </div>
            <div class="rec-text">- Analysis identifies Living Space (Sq. Ft.) and Construction Grade as the primary determinants of final sale price, outweighing specific room counts and other features.</div>
                    <div class="rec-text">- Filter by  <b>Zipcode</b> to analyze specific local market information.</div></div>""", unsafe_allow_html=True)
        corr_cols = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'floors', 'view', 'condition', 'grade']
        corr = filtered_df[[c for c in corr_cols if c in df.columns]].corr()['price'].sort_values(ascending=False).drop('price')
        corr_pos = corr[corr > 0].reset_index()
        corr_pos.columns = ['Feature', 'Correlation']
        corr_pos['Feature'] = corr_pos['Feature'].replace({'sqft_living': 'Living Space', 'sqft_above': 'Space Above Ground'})
        
        fig_corr = px.pie(corr_pos, values='Correlation', names='Feature', title="What drives the price most?",
                          hole=0.4, color_discrete_sequence=px.colors.sequential.Teal)
        fig_corr.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with s2:
        st.markdown("""<div class="rec-card"><div class="rec-title">📍 Highest Average Values and Areas</div>
            <div class="rec-text">- The identified premium zip codes represent the highest valuation tiers in the region, establishing the upper benchmark for market pricing.</div></div>""", unsafe_allow_html=True)
        exp_zip = df.groupby('zipcode')['price'].median().sort_values(ascending=False).head(15).reset_index()
        fig_exp = px.bar(exp_zip, x='zipcode', y='price', text_auto='.2s', title="Highest Median Price by Zipcode",
                         labels={'zipcode': 'Zip Code', 'price': 'Median Price ($)'}, color='price', color_continuous_scale='Magma')
        fig_exp.update_layout(xaxis_type='category')
        st.plotly_chart(fig_exp, use_container_width=True)
        
    st.markdown("---")

    # ROW 2: Seasonality & Renovation
    s3, s4 = st.columns(2)
    
    with s3:
        st.markdown("""<div class="rec-card"><div class="rec-title">🗓️ Best Time to Sell</div>
                    <div class="rec-text">- Historical sales data reveals a consistent cyclical peak, with median transaction prices and market activity reaching their highest levels during April and May.</div>
                    <div class="rec-text">- Filter by  <b>Zipcode</b> to analyze specific local market information.</div></div>""", unsafe_allow_html=True)
        # Define the correct order
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        # Convert column to Categorical type
        filtered_df['month_name'] = pd.Categorical(
            filtered_df['month_name'], 
            categories=month_order, 
            ordered=True
        )
        # Groupby (Pandas will now respect the logical order)
        seasonal = filtered_df.groupby('month_name')['price'].median().reset_index()
        # Explicit sort to be safe
        seasonal = seasonal.sort_values('month_name')
        fig_season = px.line(seasonal, x='month_name', y='price', markers=True, title="Median Price Trend by Month",
                             labels={'month_name': 'Month of the Year', 'price': 'Median Price ($)'}, color_discrete_sequence=['#ff7f0e'])
        fig_season.update_xaxes(tickmode='linear', dtick=1)
        st.plotly_chart(fig_season, use_container_width=True)

    with s4:
        st.markdown("""<div class="rec-card"><div class="rec-title">🔨 Renovate Strategy</div>
                    <div class="rec-text">- Comparative analysis by era demonstrates that Mid-Century homes (1950-1990) exhibit the largest value appreciation following renovation, outperforming both pre-war and modern builds.</div>
                    <div class="rec-text">- Filter by  <b>Zipcode</b> to analyze specific local market information.</div></div>""", unsafe_allow_html=True)
        if 'is_renovated' in filtered_df.columns and 'era' in filtered_df.columns:
            reno_era = filtered_df.groupby(['era', 'is_renovated'])['price'].median().reset_index()
            reno_era['Status'] = reno_era['is_renovated'].map({0: 'Not Renovated', 1: 'Renovated'})
            era_order = ['Pre-1950', '1950-1990', 'Post-1990']
            
            fig_reno_era = px.bar(reno_era, x='era', y='price', color='Status', barmode='group',
                                  category_orders={'era': era_order}, text_auto='.2s', title="Renovated vs. Unrenovated Price | Renovation ROI by Era",
                                  labels={'era': 'Construction Era - Pre 1950 (war) | 1950 - 1990 (Mid) | Post 1990 (Modern)', 'price': 'Median Price ($)'},
                                  color_discrete_map={'Not Renovated': '#95a5a6', 'Renovated': '#27ae60'})
            st.plotly_chart(fig_reno_era, use_container_width=True)
        else:
            st.error("Data for Renovation Analysis not available.")
    
# =============================================================================
# --- TAB 4: PREDICTOR ---
# -----------------------------------------------------------------------------
with tab4:
    st.header("Price Estimator 🤖")
    
    if model is None:
        st.error("Model not loaded.")
    else:
        # --- 1. Role Selection ---
        st.markdown("#### Select Your Role for a taylored experience:")
        role = st.radio("I am a:", ["Buyer", "Seller"], horizontal=True, label_visibility="collapsed")
        if role == "Buyer":
            st.info("As a Buyer, focus on properties that offer the best value for your budget. Set the property features below to see how different features impact price estimation and make an informed offering decision.")
        else:
            st.info("As a Seller, understand how various features of your property can influence its market value. Set the property features below to see how the changes in features can affect the value and optimize your listing price.")    

        # --- 2. User Inputs ---
        if role == "Buyer":
            st.markdown("#### Specify the Desired Property Features for an accurate Offer Eatimation:")
        else:
            st.markdown("#### Specify the Features of Your Property for a more precise Valuation:")
        
        # --- CRITICAL FIX: ZIPCODE OUTSIDE THE FORM ---
        # This allows the app to rerun immediately when Zipcode changes
        p_zip = st.selectbox("Select Zipcode Area (The values below are optimized based on market data for this Zipcode):", sorted(df['zipcode'].unique()))
        
        # Filter context to set min/max/avg defaults based on the selection
        zip_df = df[df['zipcode'] == p_zip]
        if zip_df.empty: zip_df = df
        
        # Define ranges based on the selected Zipcode
        # Bedrooms
        min_bed = int(zip_df['bedrooms'].min())
        max_bed = int(zip_df['bedrooms'].max())
        avg_bed = int(zip_df['bedrooms'].median())
        
        # Bathrooms
        min_bath = float(zip_df['bathrooms'].min())
        max_bath = float(zip_df['bathrooms'].max())
        avg_bath = float(zip_df['bathrooms'].median())
        
        # SqFt
        min_sq = int(zip_df['sqft_living'].min())
        max_sq = int(zip_df['sqft_living'].max())
        avg_sq = int(zip_df['sqft_living'].median())
        
        # Year
        min_yr = int(zip_df['yr_built'].min())
        max_yr = int(zip_df['yr_built'].max())
        avg_yr = int(zip_df['yr_built'].median())

        # --- FORM START ---
        with st.form("pred_form"):
            # --- ROW 1 ---
            c1, c2, c3 = st.columns(3)
            with c1:
                # Key includes p_zip to force reset when zip changes
                p_bed = st.slider("Bedrooms", min_bed, max(max_bed, 6), avg_bed, key=f"bed_{p_zip}")
                p_bath = st.slider("Bathrooms", min_bath, max(max_bath, 5.0), avg_bath, step=0.25, key=f"bath_{p_zip}")
                
            with c2:
                st.write("Living Space (Sq. Ft.)")
                # Slider logic: default to average, range starts from min
                sq_range = st.slider("Select Range", min_sq, max(max_sq, 3000), (avg_sq, avg_sq + 500), key=f"sq_{p_zip}")
                p_sqft = sum(sq_range) / 2
                
                p_floor = st.selectbox("Floors", [1.0, 1.5, 2.0, 2.5, 3.0], key=f"floor_{p_zip}")
                
            with c3:
                st.write("Year Built")
                yr_range = st.slider("Select Era", min_yr, max_yr, (avg_yr-10, avg_yr+10), key=f"yr_{p_zip}")
                p_year = int(sum(yr_range)/2)
                
                p_water = st.checkbox("Waterfront Property?", key=f"water_{p_zip}")

            st.markdown("---")
            
            # --- ROW 2 (Qualitative) ---
            q1, q2, q3 = st.columns(3)
            with q1:
                grade_map = {
                    "Standard (1-6)": 6, 
                    "Good (7-8)": 7.5, 
                    "Better (9-10)": 9.5, 
                    "Luxury (11-13)": 12
                }
                grade_label = st.selectbox("Construction Grade", list(grade_map.keys()), index=1, key=f"grade_{p_zip}")
                p_grade = grade_map[grade_label]
                
            with q2:
                cond_map = {
                    "1 - Worn Out": 1, "2 - Fair": 2, "3 - Average": 3, 
                    "4 - Good": 4, "5 - Excellent": 5
                }
                cond_label = st.selectbox("Condition", list(cond_map.keys()), index=2, key=f"cond_{p_zip}")
                p_cond = cond_map[cond_label]
                
            with q3:
                view_map = {
                    "0 - No View": 0, "1 - Fair": 1, "2 - Average": 2, 
                    "3 - Good": 3, "4 - Stunning": 4
                }
                view_label = st.selectbox("View Quality", list(view_map.keys()), index=0, key=f"view_{p_zip}")
                p_view = view_map[view_label]
                
            submit = st.form_submit_button(" Estimate Value", type="primary", use_container_width=True)            
        
        if submit:
            u_input = {
                'zipcode': p_zip, 'bedrooms': p_bed, 'bathrooms': p_bath,
                'sqft_living': p_sqft, 'floors': p_floor, 'yr_built': p_year,
                'grade': p_grade, 'condition': p_cond, 'view': p_view, 'waterfront': p_water
            }
            try:
                # Predict
                in_df = prepare_input_data(u_input, df, model)
                log_p = model.predict(in_df)[0]
                real_p = np.expm1(log_p)
                
                # Metrics
                mae = MODEL_METRICS['MAE']
                min_p = real_p - mae
                max_p = real_p + mae
                
                st.markdown("### 📊 Valuation Result")
                st.metric("Estimated Fair Market Value", f"${real_p:,.0f}")
                
                # Tailored Message
                if role == "Buyer":
                    st.markdown(f"""
                        <div class="highlight-buyer">
                            <h4 style="margin-top:0;">🛡️ Buyer Strategy</h4>
                            <p> The upper safety limit for this home is <b>${max_p:,.0f}</b>.</p>
                            <p> Able to negotiate closer to <b>${min_p:,.0f}</b> depending on the condition and necessary repairs.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="highlight-seller">
                            <h4 style="margin-top:0;">🚀 Seller Strategy</h4>
                            <p><b>Listing Potential:</b> You could aggressively list near <b>${max_p:,.0f}</b> if staged well.</p>
                            <p>A competitive quick-sale floor price would be around <b>${min_p:,.0f}</b> taking the condition and necessary repairs into account.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Prediction Failed: {e}")
                # Debug Info (Visible only if error happens)
                st.info("Debugging Hint: Check if the model has 'feature_names_in_' attribute.")

# =============================================================================
# --- FOOTER ---
# -----------------------------------------------------------------------------
st.markdown("---")
f1, f2 = st.columns(2)
with f1: st.caption("Developed for Code Institute Data Analytics Project.")
with f2: st.caption(f"Data: {data_source} | Model: {model_source}")