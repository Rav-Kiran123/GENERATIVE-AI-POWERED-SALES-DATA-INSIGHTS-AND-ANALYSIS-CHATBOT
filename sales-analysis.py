import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
import numpy as np
import os
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Setting Streamlit page configuration
st.set_page_config(
    page_title="Sales Data Analysis Chatbot",
    page_icon=":bar_chart:",
)

# Cache the data to prevent reloading on each run
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\xampp\htdocs\Langchain\GenAI-LLM-LC-Code\GenAI-LLM-LC-Code\Sales Data.csv")
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')  # Dropping unwanted column
    return df

def check_data_quality(df):
    """Perform data quality checks on the dataset."""
    st.write("### Data Quality Checks")

    # Number of rows and columns
    st.write(f"**Number of rows:** {df.shape[0]}")
    st.write(f"**Number of columns:** {df.shape[1]}")

    # Missing values
    missing_values = df.isnull().sum()
    missing_columns = missing_values[missing_values > 0]
    if not missing_columns.empty:
        st.write("**Columns with Missing Values:**")
        st.write(missing_columns)
    else:
        st.write("No missing values in the dataset.")

    # Data types
    st.write("**Data Types:**")
    data_types_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
    st.write(data_types_df)

    # Anomaly Detection
    st.write("**Anomaly Detection**")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if numeric_columns.any():
        st.write("Detecting anomalies in numeric columns...")
        # Initializing Isolation Forest
        iso_forest = IsolationForest(contamination=0.01)  # Adjusting contamination as needed
        df_numeric = df[numeric_columns].fillna(df[numeric_columns].mean())  # Filling missing values for anomaly detection
        anomalies = iso_forest.fit_predict(df_numeric.values)  # Using NumPy array
        anomaly_count = (anomalies == -1).sum()
        st.write(f"**Number of detected anomalies:** {anomaly_count}")

def generate_visualizations(df):
    """Generate various visualizations and return necessary data for explanations."""
    st.write("### Generating Visualizations")

    # Histogram for each numeric column
    st.write("**Histograms of Numeric Columns**")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    histograms = {}
    for column in numeric_columns:
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)
        st.pyplot(fig)
        histograms[column] = df[column].describe()
    
    # Scatter Plot 
    scatter_plot = None
    if len(numeric_columns) >= 2:
        x_col, y_col = numeric_columns[:2]  # Using first two numeric columns for scatter plot
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
        st.pyplot(fig)
        scatter_plot = (x_col, y_col, df[[x_col, y_col]].corr().iloc[0, 1])

    # Pie Chart
    pie_chart = None
    if 'City' in df.columns:
        city_sales = df.groupby('City')['Sales'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots()
        ax.pie(city_sales, labels=city_sales.index, autopct='%1.1f%%', startangle=140)
        ax.axis('equal')  # Pie drawn as circle
        st.pyplot(fig)
        pie_chart = city_sales

    # Bar Plot example
    bar_plot = None
    if 'Product' in df.columns:
        product_sales = df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots()
        product_sales.plot(kind='barh', ax=ax)
        st.pyplot(fig)
        bar_plot = product_sales

    # Correlation Heatmap
    correlation_heatmap = None
    if numeric_columns.any():
        fig, ax = plt.subplots()
        corr_matrix = df[numeric_columns].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        correlation_heatmap = corr_matrix

    return histograms, scatter_plot, pie_chart, bar_plot, correlation_heatmap

def provide_analysis_explanations(df, histograms, scatter_plot, pie_chart, bar_plot, correlation_heatmap):
    """Provide detailed explanations of different types of analyses based on visualizations."""
    st.write("### Analysis Types and Recommendations")

    # Descriptive Analysis
    st.write("**Descriptive Analysis**")
    if histograms:
        st.write("**Histograms** reveal the distribution of numeric columns, showing data spread, patterns, and potential skewness. Here are the statistics:")
        for column, stats in histograms.items():
            st.write(f"**{column}**: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, Min={stats['min']:.2f}, Max={stats['max']:.2f}")

    if scatter_plot:
        x_col, y_col, correlation = scatter_plot
        st.write(f"**Scatter Plot** between {x_col} and {y_col} shows a correlation of {correlation:.2f}. A correlation close to 1 or -1 indicates a strong linear relationship.")

    if correlation_heatmap is not None:
        st.write("**Correlation Heatmap** shows relationships between numeric variables. Strong correlations (near ±1) suggest significant relationships. For example, a high correlation between 'Quantity Ordered' and 'Sales' indicates that more orders generally lead to higher sales.")

    # Prescriptive Analysis
    st.write("**Prescriptive Analysis**")
    if pie_chart is not None and not pie_chart.empty:
        st.write("**Pie Chart of Sales by City** shows the sales distribution among cities. For example, if 'New York' contributes 40% of total sales, it indicates a major market. Focus marketing and expansion efforts on high-performing cities like:")
        st.write(pie_chart)

    if bar_plot is not None and not bar_plot.empty:
        st.write("**Bar Plot of Sales by Product** identifies top-performing products. For example, if 'Macbook Pro Laptop' has the highest sales, prioritize this product in marketing strategies and inventory management. The top products are:")
        st.write(bar_plot)

    # Predictive Analysis
    st.write("**Predictive Analysis**")
    st.write("Based on historical data, machine learning models can forecast future sales. Use time series forecasting or other predictive models to project future sales trends and adjust strategies accordingly.")

    # Product with Most Sales
    if 'Product' in df.columns:
        top_product = df.groupby('Product')['Sales'].sum().idxmax()
        st.write(f"**Product with Most Sales**: The product with the highest sales is '{top_product}'. This is a top performer and should be prioritized in marketing and inventory strategies.")
    
    # Explain Insights
    st.write("**Insights and Recommendations**")
    if scatter_plot:
        x_col, y_col, correlation = scatter_plot
        st.write(f"The scatter plot between {x_col} and {y_col} shows a correlation of {correlation:.2f}. If correlation is high, focus on improving or leveraging this relationship to enhance performance.")

    if correlation_heatmap is not None:
        st.write("The correlation heatmap indicates which variables are strongly related. Use this information to focus on improving the most impactful relationships.")

    if pie_chart is not None and not pie_chart.empty:
        st.write("The pie chart highlights high-performing cities or categories. Concentrate marketing and expansion efforts in these areas to maximize returns.")

    if bar_plot is not None and not bar_plot.empty:
        st.write("The bar plot reveals top-selling products. Invest in promoting these products and consider enhancing strategies for those with lower sales.")

def main():
    load_dotenv()
    df = load_data()

    # Integrating Gemini
    agent = create_csv_agent(
        ChatGoogleGenerativeAI(
            model="gemini-pro",  # Ensure model identifier is correct
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0
        ),
        r"C:\xampp\htdocs\Langchain\GenAI-LLM-LC-Code\GenAI-LLM-LC-Code\Sales Data.csv",
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True
    )

    st.title("Sales Data Analysis Chatbot")
    st.subheader("Discover Insights from Your Sales Data!")
    st.markdown(
        """
        This chatbot helps you analyze your sales data interactively. 
        The chatbot will generate various visualizations, provide explanations of different types of analyses, and offer insights based on the data.
        """
    )
    st.write(df.head())

    # Precomputing and displaying data quality checks, visualizations, and explanations
    check_data_quality(df)
    histograms, scatter_plot, pie_chart, bar_plot, correlation_heatmap = generate_visualizations(df)
    provide_analysis_explanations(df, histograms, scatter_plot, pie_chart, bar_plot, correlation_heatmap)

if __name__ == "__main__":
    main()









