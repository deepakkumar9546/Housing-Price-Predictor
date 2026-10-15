import os

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

CSV_PATH = os.path.join(os.path.dirname(__file__), "USA_Housing 2.csv")

FEATURES = [
    "Avg. Area Income",
    "Avg. Area House Age",
    "Avg. Area Number of Rooms",
    "Avg. Area Number of Bedrooms",
    "Area Population",
]
TARGET = "Price"

st.set_page_config(page_title="USA Housing Price Predictor", page_icon="🏠", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)


@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    return model, r2, mae


df = load_data()
model, r2, mae = train_model(df)

st.title("🏠 USA Housing Price Predictor")
st.caption("Linear regression model trained on `USA_Housing 2.csv`")

tab_predict, tab_explore, tab_data = st.tabs(["🔮 Predict", "📊 Explore", "🗂️ Data"])

with tab_predict:
    col_form, col_metrics = st.columns([2, 1])

    with col_form:
        st.subheader("Enter house / area details")
        c1, c2 = st.columns(2)
        with c1:
            income = st.number_input(
                "Avg. Area Income ($)",
                min_value=0.0,
                value=float(df["Avg. Area Income"].mean()),
                step=1000.0,
            )
            house_age = st.number_input(
                "Avg. Area House Age (years)",
                min_value=0.0,
                value=float(df["Avg. Area House Age"].mean()),
                step=0.1,
            )
            rooms = st.number_input(
                "Avg. Area Number of Rooms",
                min_value=0.0,
                value=float(df["Avg. Area Number of Rooms"].mean()),
                step=0.1,
            )
        with c2:
            bedrooms = st.number_input(
                "Avg. Area Number of Bedrooms",
                min_value=0.0,
                value=float(df["Avg. Area Number of Bedrooms"].mean()),
                step=0.1,
            )
            population = st.number_input(
                "Area Population",
                min_value=0.0,
                value=float(df["Area Population"].mean()),
                step=100.0,
            )

        if st.button("Predict Price", type="primary"):
            input_df = pd.DataFrame(
                [[income, house_age, rooms, bedrooms, population]], columns=FEATURES
            )
            prediction = model.predict(input_df)[0]
            st.success(f"Estimated Price: **${prediction:,.2f}**")

    with col_metrics:
        st.subheader("Model performance")
        st.metric("R² score", f"{r2:.3f}")
        st.metric("Mean Absolute Error", f"${mae:,.0f}")
        st.caption(f"Trained on {len(df):,} records")

with tab_explore:
    st.subheader("Feature vs Price")
    feature_choice = st.selectbox("Choose a feature", FEATURES)
    fig = px.scatter(
        df, x=feature_choice, y=TARGET, trendline="ols", opacity=0.5,
        title=f"{feature_choice} vs {TARGET}",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price distribution")
    st.plotly_chart(px.histogram(df, x=TARGET, nbins=50), use_container_width=True)

    st.subheader("Correlation heatmap")
    corr = df[FEATURES + [TARGET]].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r"), use_container_width=True)

with tab_data:
    st.subheader("Raw dataset")
    st.dataframe(df, use_container_width=True)
    st.subheader("Summary statistics")
    st.dataframe(df.describe(), use_container_width=True)
