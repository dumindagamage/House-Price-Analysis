import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
st.image("https://github.com/dumindagamage/House-Price-Analysis/blob/wip/resources/images/dashboard_header.png?raw=true", use_container_width=True)

# --- CUSTOM CSS ---
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
    "R2": 0.872,
    "MAE": 68047,
    "RMSE": 129513
}

@st.cache_data
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
        st.error(f"❌ Critical Error: Could not find dataset at {DATA_PATH}.")
        st.stop()

    # --- CLEANING & PREP ---
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['yr_sold'] = df['date'].dt.year
        df['month_sold'] = df['date'].dt.month
        df['month_name'] = df['date'].dt.strftime('%b')
    
    # Ensure String Zipcode
    if 'zipcode' in df.columns:
        df['zipcode'] = df['zipcode'].astype(str).str.replace(r'\.0$', '', regex=True)

    # Feature: House Age
    if 'house_age' not in df.columns and 'yr_built' in df.columns:
        df['house_age'] = df['yr_sold'] - df['yr_built']

    # Feature: Era
    def get_era(yr):
        if yr < 1950: return 'Pre-1950'
        elif yr <= 1990: return '1950-1990'
        else: return 'Post-1990'
        
    df['era'] = df['yr_built'].apply(get_era)

    # Feature: SqFt Bin
    sqft_bins = [0, 1000, 1500, 2000, 2500, 3000, 4000, 10000]
    sqft_labels = ['<1k', '1k-1.5k', '1.5k-2k', '2k-2.5k', '2.5k-3k', '3k-4k', '4k+']
    df['sqft_bin'] = pd.cut(df['sqft_living'], bins=sqft_bins, labels=sqft_labels)
    
    # Feature: Price per SqFt
    df['price_per_sqft'] = df['price'] / df['sqft_living']

    # Feature: Is Renovated
    if 'yr_renovated' in df.columns:
        df['is_renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)

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
    
    # 3. Feature Engineering (Match Notebook Logic)
    input_df['yr_sold'] = 2015 
    input_df['house_age'] = input_df['yr_sold'] - input_df['yr_built']
    if 'month_sold' not in input_df.columns:
        input_df['month_sold'] = 6 # Default to June if missing
    
    # --- CRITICAL FIX: FEATURE ALIGNMENT ---
    # We must only pass the exact columns the model expects.
    
    if hasattr(trained_model, 'feature_names_in_'):
        # Best Case: The model knows its own features
        expected_cols = trained_model.feature_names_in_
        
        # Add any missing columns (set to 0)
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 0
                
        # Select & Sort to match Model EXACTLY
        input_df = input_df[expected_cols]
        
    else:
        # Fallback for older sklearn models
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
    st.info("Quickly check market stats for any Zipcode.")
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
</style>
""", unsafe_allow_html=True)

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
    st.header("Project Overview 📝")
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Sales", f"{filtered_df.shape[0]:,}")
    k2.metric("Avg Price", f"${filtered_df['price'].mean():,.0f}")
    k3.metric("Median Price", f"${filtered_df['price'].median():,.0f}")
    
    st.markdown("### 🗺️ Sales Map")
    map_data = filtered_df.sample(min(len(filtered_df), 3000))
    fig_map = px.scatter_mapbox(
        map_data, lat="lat", lon="long", color="price", size_max=15, zoom=9,
        mapbox_style="carto-positron", title="Sold Properties (Color = Price)",
        color_continuous_scale="RdYlGn_r"
    )
    st.plotly_chart(fig_map, use_container_width=True)

# =============================================================================
# --- TAB 2: BUYER ANALYSIS ---
# -----------------------------------------------------------------------------
with tab2:
    st.header("Analysis for Buyers 🔍")
    
    b1, b2 = st.columns(2)
    with b1:
        st.subheader("1. Most Affordable Zipcodes")
        affordability = df.groupby('zipcode')['price'].median().sort_values().head(10).reset_index()
        overall_median = df['price'].median()
        
        fig_afford = px.bar(
            affordability, x='zipcode', y='price', text_auto='.2s',
            title="Median Price by Zipcode (vs. County Median)",
            labels={'price': 'Median Price ($)', 'zipcode': 'Zip Code'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_afford.add_hline(y=overall_median, line_dash="dot", line_color="red", 
                             annotation_text=f"County Median: ${overall_median:,.0f}")
        fig_afford.update_layout(xaxis_type='category')
        st.plotly_chart(fig_afford, use_container_width=True)

    with b2:
        st.subheader("2. Best Value for Money (Size)")
        df['sqft_detailed_bin'] = pd.cut(df['sqft_living'], bins=range(0, 5000, 250))
        size_value = df.groupby('sqft_detailed_bin')['price_per_sqft'].median().reset_index()
        size_value['sqft_mid'] = size_value['sqft_detailed_bin'].apply(lambda x: x.mid)
        
        fig_value = px.line(
            size_value, x='sqft_mid', y='price_per_sqft', markers=True,
            title="Price per Sq. Ft. by House Size",
            labels={'sqft_mid': 'House Size (Sq. Ft.)', 'price_per_sqft': 'Price / Sq. Ft. ($)'},
            color_discrete_sequence=['#2ca02c']
        )
        st.plotly_chart(fig_value, use_container_width=True)

    st.markdown("---")
    
    b3, b4 = st.columns(2)
    with b3:
        st.subheader("3. Average Price vs. Grade")
        def categorize_grade(g):
            if g <= 6: return "Low (1-6)"
            elif g <= 9: return "Good (7-9)"
            else: return "Excellent (10-13)"
        
        filtered_df['grade_cat'] = filtered_df['grade'].apply(categorize_grade)
        grade_order = ["Low (1-6)", "Good (7-9)", "Excellent (10-13)"]
        grade_price = filtered_df.groupby('grade_cat')['price'].median().reindex(grade_order).reset_index()
        
        fig_grade = px.bar(grade_price, x='grade_cat', y='price', text_auto='.2s',
                           title="Median Price by Construction Grade",
                           color='grade_cat', color_discrete_map={"Low (1-6)": "#bdc3c7", "Good (7-9)": "#f39c12", "Excellent (10-13)": "#27ae60"})
        fig_grade.update_layout(showlegend=False)
        st.plotly_chart(fig_grade, use_container_width=True)

    with b4:
        st.subheader("4. Average Price vs. Condition")
        cond_price = filtered_df.groupby('condition')['price'].median().reset_index()
        fig_cond = px.bar(cond_price, x='condition', y='price', text_auto='.2s',
                          title="Median Price by House Condition", color='price', color_continuous_scale='OrRd')
        fig_cond.update_xaxes(tickmode='linear', dtick=1)
        st.plotly_chart(fig_cond, use_container_width=True)

    st.markdown("---")

    b5, b6 = st.columns(2)
    with b5:
        st.subheader("5. Quantify the Premium for View")
        view_price = filtered_df.groupby('view')['price'].median().reset_index()
        fig_view_bar = px.bar(view_price, x='view', y='price', text_auto='.2s',
                              title="Median Price by View Rating", color='price', color_continuous_scale='Teal')
        st.plotly_chart(fig_view_bar, use_container_width=True)

    with b6:
            st.subheader("6. Quantify the Premium for Waterfront")
            wf_stats = filtered_df.groupby('waterfront')['price'].median().reset_index()
            wf_stats['label'] = wf_stats['waterfront'].map({0: 'Non-Waterfront', 1: 'Waterfront'})
            fig_wf_pie = px.pie(wf_stats, values='price', names='label', title="Median Price Comparison",
                                color='label', color_discrete_map={'Non-Waterfront': '#bdc3c7', 'Waterfront': '#3498db'}, hole=0.4)
            fig_wf_pie.update_traces(textinfo='label+value', texttemplate='%{label}<br>$%{value:,.0f}')
            st.plotly_chart(fig_wf_pie, use_container_width=True)

    # BUYER RECOMMENDATIONS
    st.markdown("### 💡 General Recommendations for Buyers")
    
    card_style = """
    <style>
    .rec-card {
        background-color: #f0f2f6; padding: 20px; border-radius: 10px;
        border-left: 5px solid #ff4b4b; margin-bottom: 10px;
    }
    .rec-title { font-weight: bold; font-size: 18px; color: #31333F; margin-bottom: 5px; }
    .rec-text { font-size: 14px; color: #555; }
    </style>
    """
    st.markdown(card_style, unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("""<div class="rec-card"><div class="rec-title">🔨 Renovate Strategy</div>
                    <div class="rec-text">It's advisable to target <b>1950-1990 builds</b> for maximum ROI. These homes often have good "bones" but outdated finishes, offering potential for value creation.</div></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="rec-card"><div class="rec-title">🏗️ Features & Grade</div>
                    <div class="rec-text">It's advisable to prioritize <b>Construction Grade</b> over immediate condition. A high-grade house in fair condition is generally a better investment than a low-grade 'mint condition' property.</div></div>""", unsafe_allow_html=True)

    with rc2:
        st.markdown("""<div class="rec-card"><div class="rec-title">🌊 Scenery Attributes</div>
                    <div class="rec-text">It's advisable to focus on <b>Space (SqFt) and Grade</b> for functional value. Waterfront properties often carry a significant "Scenery Tax" that drives up cost without adding daily utility.</div></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="rec-card"><div class="rec-title">📐 Space Optimization</div>
                    <div class="rec-text">It's advisable to look within the <b>2,000–2,500 sq. ft. bracket</b>. This range often offers the most efficient Price per Sq. Ft. for buyers seeking value.</div></div>""", unsafe_allow_html=True)

# =============================================================================
# --- TAB 3: SELLER ANALYSIS ---
# -----------------------------------------------------------------------------
with tab3:
    st.header("Analysis for Sellers 📈")
    
    # ROW 1: Drivers & Expensive Markets
    s1, s2 = st.columns(2)
    
    with s1:
        st.subheader("1. Relative Influence of Features")
        corr_cols = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'floors', 'view', 'condition', 'grade']
        corr = df[[c for c in corr_cols if c in df.columns]].corr()['price'].sort_values(ascending=False).drop('price')
        corr_pos = corr[corr > 0].reset_index()
        corr_pos.columns = ['Feature', 'Correlation']
        corr_pos['Feature'] = corr_pos['Feature'].replace({'sqft_living': 'Living Space', 'sqft_above': 'Space Above Ground'})
        
        fig_corr = px.pie(corr_pos, values='Correlation', names='Feature', title="What drives the price most?",
                          hole=0.4, color_discrete_sequence=px.colors.sequential.Teal)
        fig_corr.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with s2:
        st.subheader("2. Top 15 Expensive Markets")
        exp_zip = df.groupby('zipcode')['price'].median().sort_values(ascending=False).head(15).reset_index()
        fig_exp = px.bar(exp_zip, x='zipcode', y='price', text_auto='.2s', title="Highest Median Price by Zipcode",
                         labels={'zipcode': 'Zip Code', 'price': 'Median Price ($)'}, color='price', color_continuous_scale='Magma')
        fig_exp.update_layout(xaxis_type='category')
        st.plotly_chart(fig_exp, use_container_width=True)

    st.markdown("---")

    # ROW 2: Seasonality & Renovation
    s3, s4 = st.columns(2)
    
    with s3:
        st.subheader("3. Best Time to Sell")
        seasonal = filtered_df.groupby('month_sold')['price'].median().reset_index()
        fig_season = px.line(seasonal, x='month_sold', y='price', markers=True, title="Median Price Trend by Month",
                             labels={'month_sold': 'Month (1-12)', 'price': 'Median Price ($)'}, color_discrete_sequence=['#ff7f0e'])
        fig_season.update_xaxes(tickmode='linear', dtick=1)
        st.plotly_chart(fig_season, use_container_width=True)

    with s4:
        st.subheader("4. Renovation ROI by Era")
        if 'is_renovated' in filtered_df.columns and 'era' in filtered_df.columns:
            reno_era = filtered_df.groupby(['era', 'is_renovated'])['price'].median().reset_index()
            reno_era['Status'] = reno_era['is_renovated'].map({0: 'Not Renovated', 1: 'Renovated'})
            era_order = ['Pre-1950', '1950-1990', 'Post-1990']
            
            fig_reno_era = px.bar(reno_era, x='era', y='price', color='Status', barmode='group',
                                  category_orders={'era': era_order}, text_auto='.2s', title="Renovated vs. Unrenovated Price by Era",
                                  labels={'era': 'Construction Era', 'price': 'Median Price ($)'},
                                  color_discrete_map={'Not Renovated': '#95a5a6', 'Renovated': '#27ae60'})
            st.plotly_chart(fig_reno_era, use_container_width=True)
        else:
            st.error("Data for Renovation Analysis not available.")

    # SELLER RECOMMENDATIONS
    st.markdown("### 💡 General Recommendations for Sellers")
    
    seller_card_style = """
    <style>
    .seller-card {
        background-color: #e8f4f8; padding: 20px; border-radius: 10px;
        border-left: 5px solid #3498db; margin-bottom: 10px;
    }
    .seller-title { font-weight: bold; font-size: 18px; color: #2c3e50; margin-bottom: 5px; }
    .seller-text { font-size: 14px; color: #34495e; }
    </style>
    """
    st.markdown(seller_card_style, unsafe_allow_html=True)
    
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown("""<div class="seller-card"><div class="seller-title">🔨 Renovate Strategy</div>
                    <div class="seller-text">It's advisable to target <b>1950-1990 builds</b> for renovations. Data shows these homes yield the highest price jump compared to unrenovated peers.</div></div>""", unsafe_allow_html=True)
    with sc2:
        st.markdown("""<div class="seller-card"><div class="seller-title">🗓️ Timing</div>
                    <div class="seller-text">It's advisable to list your property in <b>April or May</b>. The market consistently shows a peak in median sales prices during these spring months.</div></div>""", unsafe_allow_html=True)
    with sc3:
        st.markdown("""<div class="seller-card"><div class="seller-title">✨ Features to Highlight</div>
                    <div class="seller-text">It's advisable to highlight <b>Living Space (Sq. Ft.)</b> and <b>Construction Grade</b> in your descriptions, as these are the top drivers of property value.</div></div>""", unsafe_allow_html=True)

# =============================================================================
# --- TAB 4: PREDICTOR ---
# -----------------------------------------------------------------------------
with tab4:
    st.header("Price Estimator 🤖")
    
    if model is None:
        st.error("Model not loaded.")
    else:
        # --- 1. Role Selection ---
        st.markdown("#### 1. Select Your Role")
        role = st.radio("I am a:", ["Buyer", "Seller"], horizontal=True, label_visibility="collapsed")
        
        # --- 2. Input Form ---
        st.markdown("#### 2. Enter Property Details")
        st.caption("Ranges below update automatically based on the selected Zipcode.")
        
        with st.form("pred_form"):
            p_zip = st.selectbox("Select Zipcode Area", sorted(df['zipcode'].unique()))
            
            # Filter context to set min/max/avg defaults
            zip_df = df[df['zipcode'] == p_zip]
            if zip_df.empty: zip_df = df
            
            # --- ROW 1 ---
            c1, c2, c3 = st.columns(3)
            with c1:
                # Bedrooms
                min_bed = int(zip_df['bedrooms'].min())
                max_bed = int(zip_df['bedrooms'].max())
                avg_bed = int(zip_df['bedrooms'].median())
                p_bed = st.slider("Bedrooms", min_bed, max(max_bed, 6), avg_bed)
                
                # Bathrooms
                min_bath = float(zip_df['bathrooms'].min())
                max_bath = float(zip_df['bathrooms'].max())
                avg_bath = float(zip_df['bathrooms'].median())
                p_bath = st.slider("Bathrooms", min_bath, max(max_bath, 5.0), avg_bath, step=0.25)
                
            with c2:
                # Living Space
                min_sq = int(zip_df['sqft_living'].min())
                max_sq = int(zip_df['sqft_living'].max())
                avg_sq = int(zip_df['sqft_living'].median())
                st.write("Living Space (Sq. Ft.)")
                sq_range = st.slider("Select Range", min_sq, max(max_sq, 3000), (avg_sq, avg_sq + 500))
                p_sqft = sum(sq_range) / 2
                
                # Floors
                p_floor = st.selectbox("Floors", [1.0, 1.5, 2.0, 2.5, 3.0])
                
            with c3:
                # Year Built
                min_yr = int(zip_df['yr_built'].min())
                max_yr = int(zip_df['yr_built'].max())
                avg_yr = int(zip_df['yr_built'].median())
                st.write("Year Built")
                yr_range = st.slider("Select Era", min_yr, max_yr, (avg_yr-10, avg_yr+10))
                p_year = int(sum(yr_range)/2)
                
                p_water = st.checkbox("Waterfront Property?")

            st.markdown("---")
            
            # --- ROW 2 (Qualitative) ---
            q1, q2, q3 = st.columns(3)
            with q1:
                # Grade - Categorized
                grade_map = {
                    "Standard (1-6)": 6, 
                    "Good (7-8)": 7.5, 
                    "Better (9-10)": 9.5, 
                    "Luxury (11-13)": 12
                }
                grade_label = st.selectbox("Construction Grade", list(grade_map.keys()), index=1)
                p_grade = grade_map[grade_label]
                
            with q2:
                # Condition
                cond_map = {
                    "1 - Worn Out": 1, "2 - Fair": 2, "3 - Average": 3, 
                    "4 - Good": 4, "5 - Excellent": 5
                }
                cond_label = st.selectbox("Condition", list(cond_map.keys()), index=2)
                p_cond = cond_map[cond_label]
                
            with q3:
                # View
                view_map = {
                    "0 - No View": 0, "1 - Fair": 1, "2 - Average": 2, 
                    "3 - Good": 3, "4 - Stunning": 4
                }
                view_label = st.selectbox("View Quality", list(view_map.keys()), index=0)
                p_view = view_map[view_label]
                
            submit = st.form_submit_button("Estimate Value", type="primary", use_container_width=True)
            
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
                            <p>A competitive quick-sale price would be around <b>${min_p:,.0f}</b> taking the condition and necessary repairs into account.</p>
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