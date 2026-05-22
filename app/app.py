# ============================================================
# CONSUMER BEHAVIOUR INTELLIGENCE PLATFORM
# Chennai — What People Watch vs What They Buy
# ============================================================
# Built with: Streamlit, Plotly, XGBoost, VADER, Gemini API
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title = "Chennai Consumer Intelligence",
    page_icon  = "🛍️",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1565C0;
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #757575;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #1565C0;
    }
    .insight-box {
        background: #E8F5E9;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #1B5E20;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: #F3E5F5;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #4A148C;
        margin: 1rem 0;
    }
    .warning-box {
        background: #FFF3E0;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #E65100;
    }
</style>
""", unsafe_allow_html=True)

# ── Load All Data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    """
    Cache data so it only loads once per session.
    Without caching the app reloads all CSVs on every
    user interaction making it very slow.
    """
    data = {}
    base = "data/cleaned"
    files = {
        "instagram"  : "instagram_clean.csv",
        "trends"     : "trends_clean.csv",
        "retail"     : "retail_clean.csv",
        "consumers"  : "chennai_consumers.csv",
        "posts"      : "chennai_posts_with_sentiment.csv",
        "stores"     : "chennai_stores.csv",
        "monthly"    : "monthly_trends.csv",
        "master"     : "master_merged.csv",
        "locality"   : "locality_summary.csv",
        "sent_summary": "sentiment_summary.csv",
    }
    for key, filename in files.items():
        path = os.path.join(base, filename)
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
        else:
            data[key] = pd.DataFrame()
    return data

@st.cache_resource
def load_model():
    """
    Cache model so it only loads once.
    Model file is 3.8MB — loading every interaction
    would make the predictor very slow.
    """
    try:
        model    = joblib.load("app/models/xgboost_purchase_predictor.pkl")
        encoders = joblib.load("app/models/label_encoders.pkl")
        cats     = joblib.load("app/models/purchase_categories.pkl")
        return model, encoders, cats
    except Exception as e:
        return None, None, None

# Load everything
data         = load_data()
model, encoders, categories = load_model()

# ── Sidebar Navigation ────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/fluency/96/shop.png",
    width=60
)
st.sidebar.title("Chennai Consumer\nIntelligence Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home",
     "📊 EDA Dashboard",
     "🧠 Sentiment Analysis",
     "🛍️ Purchase Predictor",
     "💬 Ask The Data"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Summary**")
st.sidebar.metric("Instagram Posts",  "29,999")
st.sidebar.metric("Consumer Profiles","5,000")
st.sidebar.metric("Store Records",    "200")
st.sidebar.metric("Social Posts",     "3,000")

# ══════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":

    st.markdown('<div class="main-header">🛍️ Chennai Consumer Intelligence Platform</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">What People Watch vs What They Buy — AI-Powered Consumer Insights</div>',
                unsafe_allow_html=True)

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Data Points", "48,125", "rows analysed")
    with col2:
        st.metric("Screen Time Correlation", "0.815", "with impulse buying")
    with col3:
        st.metric("Top Platform", "Instagram", "highest spend")
    with col4:
        st.metric("Top Genre", "Education", "most positive sentiment")
    with col5:
        st.metric("Model Accuracy", "45.1%", "215% above random")

    st.markdown("---")

    # Master insight table
    st.subheader("🔍 Core Project Insight — The Master Table")
    st.markdown("*What each age group watches, where, and what they buy*")

    if not data["master"].empty:
        master_display = data["master"][[
            "age_group", "top_platform", "top_content",
            "top_purchase_category", "avg_spend",
            "avg_screen_time", "top_impulse_freq"
        ]].copy()
        master_display.columns = [
            "Age Group", "Top Platform", "Top Content",
            "Buys Most", "Avg Spend (₹)", "Screen Time (hrs)",
            "Impulse Buying"
        ]
        master_display["Avg Spend (₹)"] = master_display[
            "Avg Spend (₹)"
        ].apply(lambda x: f"₹{float(x):,.0f}")
        master_display["Screen Time (hrs)"] = master_display[
            "Screen Time (hrs)"
        ].apply(lambda x: f"{float(x):.1f} hrs")

        st.dataframe(
            master_display,
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # Five key findings
    st.subheader("📌 Five Key Findings")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="insight-box">
        <b>Finding 1 — Screen Time Predicts Impulse Buying</b><br>
        Correlation of 0.815 between daily screen time and
        impulse buying frequency. Users spending 4.8hrs/day
        impulse buy 2x more than users spending 2.3hrs/day.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
        <b>Finding 2 — Instagram Users Spend Most</b><br>
        Instagram users average ₹18,373/month vs TikTok
        users at ₹16,278/month. Platform choice is a
        strong spending predictor.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
        <b>Finding 3 — Content Genre = #1 Purchase Predictor</b><br>
        XGBoost feature importance: content genre scored 0.551,
        more than all other features combined.
        What you watch predicts what you buy.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="insight-box">
        <b>Finding 4 — Negative Content Gets More Views</b><br>
        Negative sentiment posts average 51,672 views vs
        positive posts at 49,763. Negativity bias drives
        higher attention in Chennai social media.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
        <b>Finding 5 — Travel is Chennai's Most Negative Genre</b><br>
        Travel content scored -0.131 average sentiment.
        Complaints about delayed flights, tourist traps and
        overbooked hotels dominate travel posts.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
        <b>Finding 6 — T Nagar Leads in Content Volume</b><br>
        T Nagar generated 6,017 posts — highest of all
        localities. Porur leads in reach per post at 6,451
        average reach despite fewer posts.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — EDA DASHBOARD
# ══════════════════════════════════════════════════════════════
elif page == "📊 EDA Dashboard":

    st.title("📊 Exploratory Data Analysis Dashboard")
    st.markdown("*Interactive exploration of Chennai consumer behaviour data*")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        age_filter = st.multiselect(
            "Filter by Age Group",
            ["13-17","18-24","25-34","35-50","50+"],
            default=["13-17","18-24","25-34","35-50","50+"]
        )
    with col2:
        platform_filter = st.multiselect(
            "Filter by Platform",
            ["instagram","tiktok","youtube","twitter"],
            default=["instagram","tiktok","youtube","twitter"]
        )
    with col3:
        locality_filter = st.multiselect(
            "Filter by Locality",
            ["T Nagar","Anna Nagar","Velachery","Adyar",
             "OMR","Tambaram","Porur","Nungambakkam",
             "Chromepet","Perambur"],
            default=["T Nagar","Anna Nagar","Velachery",
                     "Adyar","OMR"]
        )

    # Filter data
    if not data["consumers"].empty:
        filtered = data["consumers"][
            (data["consumers"]["age_group"].isin(age_filter)) &
            (data["consumers"]["primary_platform"].isin(platform_filter)) &
            (data["consumers"]["locality"].isin(locality_filter))
        ]

        if len(filtered) == 0:
            st.warning("No data matches your filters. Please adjust.")
            st.stop()

        st.markdown(f"*Showing **{len(filtered):,}** consumers*")
        st.markdown("---")

        # Row 1 charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Platform Distribution")
            plat_counts = filtered[
                "primary_platform"
            ].value_counts().reset_index()
            plat_counts.columns = ["Platform", "Count"]
            fig = px.pie(
                plat_counts,
                values="Count",
                names="Platform",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.35
            )
            fig.update_layout(height=350, margin=dict(t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Content Genre by Age Group")
            genre_age = filtered.groupby(
                ["age_group","top_content_genre"]
            ).size().reset_index(name="count")
            fig = px.bar(
                genre_age,
                x="age_group",
                y="count",
                color="top_content_genre",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                barmode="stack"
            )
            fig.update_layout(height=350, margin=dict(t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)

        # Row 2 charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Screen Time vs Monthly Spend")
            fig = px.scatter(
                filtered,
                x="daily_screen_time_hrs",
                y="monthly_spend_inr",
                color="age_group",
                opacity=0.5,
                size_max=8,
                color_discrete_sequence=px.colors.qualitative.Set1,
                labels={
                    "daily_screen_time_hrs": "Daily Screen Time (hrs)",
                    "monthly_spend_inr":     "Monthly Spend (INR)"
                }
            )
            fig.update_layout(height=350, margin=dict(t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Purchase Category Distribution")
            purchase_counts = filtered[
                "top_purchase_category"
            ].value_counts().reset_index()
            purchase_counts.columns = ["Category","Count"]
            fig = px.bar(
                purchase_counts,
                x="Category",
                y="Count",
                color="Category",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig.update_layout(
                height=350,
                margin=dict(t=20,b=20),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # Row 3 — Monthly trends
        st.subheader("📈 Monthly Content Trends 2024")
        if not data["monthly"].empty:
            social_m = data["monthly"][
                data["monthly"]["record_type"] == "social_media"
            ]
            genre_select = st.multiselect(
                "Select Genres to Plot",
                social_m["category"].unique().tolist(),
                default=["fashion","food","technology","education"]
            )
            filtered_monthly = social_m[
                social_m["category"].isin(genre_select)
            ]
            fig = px.line(
                filtered_monthly,
                x="month_date",
                y="total_posts",
                color="category",
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Set1,
                labels={
                    "month_date":  "Month",
                    "total_posts": "Total Posts",
                    "category":    "Genre"
                }
            )
            fig.update_layout(height=380, margin=dict(t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "🧠 Sentiment Analysis":

    st.title("🧠 VADER Sentiment Analysis — AI Layer 1")
    st.markdown("*Emotional tone analysis of 3,000 Chennai social media posts*")

    if not data["posts"].empty and not data["sent_summary"].empty:

        # Top metrics
        total     = len(data["posts"])
        pos_count = (data["posts"]["sent_label"] == "positive").sum()
        neg_count = (data["posts"]["sent_label"] == "negative").sum()
        neu_count = (data["posts"]["sent_label"] == "neutral").sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Posts Analysed",  f"{total:,}")
        col2.metric("Positive Posts",
                    f"{pos_count:,}",
                    f"{pos_count/total*100:.1f}%")
        col3.metric("Negative Posts",
                    f"{neg_count:,}",
                    f"-{neg_count/total*100:.1f}%")
        col4.metric("Neutral Posts",
                    f"{neu_count:,}",
                    f"{neu_count/total*100:.1f}%")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sentiment by Content Genre")
            genre_sent = data["sent_summary"][[
                "content_genre","avg_compound",
                "pct_positive","pct_negative"
            ]].sort_values("avg_compound", ascending=True)

            fig = px.bar(
                genre_sent,
                x="avg_compound",
                y="content_genre",
                orientation="h",
                color="avg_compound",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                labels={
                    "avg_compound":    "Avg Compound Score",
                    "content_genre":   "Genre"
                }
            )
            fig.update_layout(height=400, margin=dict(t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Sentiment by Platform")
            platform_sent = data["posts"].groupby(
                ["platform","sent_label"]
            ).size().reset_index(name="count")
            fig = px.bar(
                platform_sent,
                x="platform",
                y="count",
                color="sent_label",
                barmode="group",
                color_discrete_map={
                    "positive": "#43A047",
                    "neutral":  "#FB8C00",
                    "negative": "#E53935"
                }
            )
            fig.update_layout(height=400, margin=dict(t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)

        # Locality sentiment
        st.subheader("📍 Sentiment by Chennai Locality")
        locality_sent = data["posts"].groupby("locality")[
            "sent_compound"
        ].mean().reset_index()
        locality_sent.columns = ["Locality","Avg Sentiment"]
        locality_sent = locality_sent.sort_values(
            "Avg Sentiment", ascending=False
        )
        fig = px.bar(
            locality_sent,
            x="Locality",
            y="Avg Sentiment",
            color="Avg Sentiment",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0
        )
        fig.update_layout(height=380, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Sample posts
        st.subheader("🔍 Explore Sample Posts")
        genre_pick = st.selectbox(
            "Select Genre",
            data["posts"]["content_genre"].unique()
        )
        sentiment_pick = st.selectbox(
            "Select Sentiment",
            ["positive","negative","neutral"]
        )
        samples = data["posts"][
            (data["posts"]["content_genre"] == genre_pick) &
            (data["posts"]["sent_label"]    == sentiment_pick)
        ][["caption","sent_compound","platform","locality"]].head(5)

        if len(samples) > 0:
            for _, row in samples.iterrows():
                color = ("#E8F5E9" if row["sent_compound"] > 0.05
                         else "#FFEBEE" if row["sent_compound"] < -0.05
                         else "#FFF3E0")
                st.markdown(
                    f'<div style="background:{color};padding:10px;'
                    f'border-radius:8px;margin:5px 0">'
                    f'<b>{row["platform"].title()}</b> · '
                    f'{row["locality"]} · '
                    f'Score: {row["sent_compound"]:.3f}<br>'
                    f'{row["caption"]}</div>',
                    unsafe_allow_html=True
                )

# ══════════════════════════════════════════════════════════════
# PAGE 4 — PURCHASE PREDICTOR
# ══════════════════════════════════════════════════════════════
elif page == "🛍️ Purchase Predictor":

    st.title("🛍️ AI Purchase Predictor — AI Layer 2")
    st.markdown("*XGBoost model predicting purchase category from consumer profile*")

    if model is None:
        st.error("Model not loaded. Please check app/models/ folder.")
        st.stop()

    st.markdown("### Enter Consumer Profile")
    st.markdown("*Fill in the profile below and click Predict*")

    col1, col2, col3 = st.columns(3)

    with col1:
        age_group = st.selectbox(
            "Age Group",
            ["13-17","18-24","25-34","35-50","50+"],
            index=1
        )
        platform = st.selectbox(
            "Primary Platform",
            ["instagram","tiktok","youtube","twitter"]
        )
        genre = st.selectbox(
            "Top Content Genre",
            ["fashion","food","technology","fitness",
             "education","entertainment","travel","finance"]
        )

    with col2:
        screen_time = st.slider(
            "Daily Screen Time (hours)",
            min_value=0.5,
            max_value=8.0,
            value=3.0,
            step=0.5
        )
        income = st.selectbox(
            "Income Bracket",
            ["low","lower-middle","middle",
             "upper-middle","high"],
            index=2
        )
        online_pref = st.selectbox(
            "Online Shopping Preference",
            ["low","medium","high"],
            index=1
        )

    with col3:
        locality = st.selectbox(
            "Locality",
            ["T Nagar","Anna Nagar","Velachery","Adyar",
             "OMR","Tambaram","Porur","Nungambakkam",
             "Chromepet","Perambur"]
        )
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button(
            "🔮 Predict Purchase Category",
            type="primary",
            use_container_width=True
        )

    if predict_btn:
        person = {
            "age_group":             age_group,
            "primary_platform":      platform,
            "top_content_genre":     genre,
            "daily_screen_time_hrs": screen_time,
            "income_bracket":        income,
            "online_shopping_pref":  online_pref,
            "locality":              locality
        }

        # Encode and predict
        encoded = {}
        feature_cols = encoders.get("feature_cols", list(person.keys()))
        if isinstance(feature_cols, list):
            feat_list = feature_cols
        else:
            feat_list = list(person.keys())

        for col in feat_list:
            if col in encoders and col not in ["target","feature_cols"]:
                try:
                    encoded[col] = encoders[col].transform(
                        [str(person[col])]
                    )[0]
                except Exception:
                    encoded[col] = 0
            else:
                encoded[col] = person.get(col, 0)

        import pandas as pd_pred
        input_df    = pd_pred.DataFrame([encoded])
        pred_enc    = model.predict(input_df)[0]
        pred_proba  = model.predict_proba(input_df)[0]

        if hasattr(encoders.get("target"), "classes_"):
            classes    = encoders["target"].classes_
            pred_label = encoders["target"].inverse_transform(
                [pred_enc]
            )[0]
        else:
            classes    = categories
            pred_label = categories[pred_enc]

        confidence = float(pred_proba.max()) * 100
        proba_dict = {
            classes[i]: round(float(p)*100, 1)
            for i, p in enumerate(pred_proba)
        }
        proba_sorted = dict(sorted(
            proba_dict.items(),
            key=lambda x: x[1], reverse=True
        ))

        # Display result
        st.markdown("---")
        col1, col2 = st.columns([1, 1])

        with col1:
            status_color = (
                "#E8F5E9" if confidence >= 50 else
                "#FFF3E0" if confidence >= 35 else
                "#FFEBEE"
            )
            st.markdown(
                f'<div style="background:{status_color};'
                f'padding:1.5rem;border-radius:12px;'
                f'border-left:4px solid #4A148C">'
                f'<h3 style="color:#4A148C;margin:0">'
                f'🎯 Predicted Purchase</h3>'
                f'<h1 style="color:#1565C0;margin:0.5rem 0">'
                f'{pred_label.upper()}</h1>'
                f'<p style="color:#757575;margin:0">'
                f'Confidence: {confidence:.1f}%</p>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f'<div class="insight-box">'
                f'<b>💡 Insight</b><br>'
                f'A {age_group} year old spending {screen_time}h daily '
                f'on {platform} watching {genre} content in {locality} '
                f'is most likely to purchase '
                f'<b>{pred_label}</b>.'
                f'</div>',
                unsafe_allow_html=True
            )

        with col2:
            st.markdown("**All Category Probabilities**")
            fig = go.Figure(go.Bar(
                x=list(proba_sorted.values()),
                y=list(proba_sorted.keys()),
                orientation="h",
                marker_color=[
                    "#1565C0" if k == pred_label else "#BBDEFB"
                    for k in proba_sorted.keys()
                ]
            ))
            fig.update_layout(
                height=320,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis_title="Probability (%)",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 5 — ASK THE DATA
# ══════════════════════════════════════════════════════════════
elif page == "💬 Ask The Data":

    st.title("💬 Ask The Data — AI Layer 3")
    st.markdown("*Powered by Gemini API — ask any question about Chennai consumer behaviour*")

    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.markdown("""
        <div class="warning-box">
        <b>⚠️ Gemini API Key Not Found</b><br>
        Create a .env file in your project root with:<br>
        <code>GEMINI_API_KEY=your_key_here</code><br>
        Get your free key at aistudio.google.com
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Build context from master data
    if not data["master"].empty:
        context = f"""
You are a data analyst for the Chennai Consumer Behaviour
Intelligence Platform. Here is the project data summary:

MASTER INSIGHTS TABLE:
{data["master"].to_string()}

KEY FINDINGS:
- Screen time vs impulse buying correlation: 0.815 (strong)
- Instagram users spend most: Rs 18,373/month average
- TikTok users spend least: Rs 16,278/month average
- Technology content watchers spend most: Rs 18,151/month
- Education content has highest positive sentiment: 0.156
- Travel content has most negative sentiment: -0.131
- T Nagar has highest post volume: 6,017 posts
- Porur has highest reach per post: 6,451 average
- Negative content gets more views: 51,672 vs 49,763
- XGBoost model: 45.1% accuracy, 215% above random baseline
- Top predictive feature: content genre (55.1% importance)

DATASET SIZES:
- Instagram Analytics: 29,999 posts
- Consumer Profiles: 5,000 Chennai consumers
- Store Data: 200 Chennai stores
- Social Posts with Sentiment: 3,000 posts
- Monthly Trends: 12 months of 2024 data

Answer questions clearly and concisely using this data.
If asked something not in the data say so honestly.
Always relate answers to Chennai context.
        """
    else:
        context = "Chennai Consumer Behaviour Platform data analyst."

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Suggested questions
    if len(st.session_state.messages) == 0:
        st.markdown("**💡 Try asking:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Which age group spends the most?"):
                st.session_state.suggested = \
                    "Which age group spends the most?"
            if st.button("What does T Nagar buy most?"):
                st.session_state.suggested = \
                    "What does T Nagar buy most?"
        with col2:
            if st.button("Which platform has most positive content?"):
                st.session_state.suggested = \
                    "Which platform has most positive content?"
            if st.button("How does screen time affect buying?"):
                st.session_state.suggested = \
                    "How does screen time affect buying?"

    # Chat input
    user_input = st.chat_input("Ask anything about Chennai consumer behaviour...")

    # Handle suggested question clicks
    if hasattr(st.session_state, "suggested"):
        user_input = st.session_state.suggested
        del st.session_state.suggested

    if user_input:
        # Add user message
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.write(user_input)

        # Get Gemini response
        with st.chat_message("assistant"):
            with st.spinner("Analysing your data..."):
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    gemini = genai.GenerativeModel(
                        "gemini-1.5-flash"
                    )
                    prompt   = f"{context}\n\nQuestion: {user_input}"
                    response = gemini.generate_content(prompt)
                    answer   = response.text
                except Exception as e:
                    answer = (
                        f"I encountered an issue connecting to "
                        f"Gemini API: {str(e)}. "
                        f"Please check your API key in the .env file."
                    )

                st.write(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )