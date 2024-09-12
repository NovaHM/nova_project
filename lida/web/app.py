import ast 
from collections import Counter
from datetime import datetime
from math import sqrt
import numpy as np
from openai import OpenAI
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
import sqlite3
import time
import uuid

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

from prophet import Prophet
import plotly.graph_objects as go
from prophet.plot import plot_plotly, plot_components_plotly
from scipy.stats import f_oneway

import xgboost as xgb
from scipy.stats import chi2_contingency
import itertools



#################################################################################################################################


# Initialize OpenAI client with your API key from an environment variable
client = OpenAI(api_key=os.getenv('sk--GAnM2Aet-dAl1fCmNKeydZQGldDE7vaoTuZSKrurKT3BlbkFJFWjOAeehPrDEEB-ZlxCH_lozlzGFbgEFFoMTfweskA'))  # Make sure to set this environment variable

def perform_calculations(dataframe, calculation_prompt):
    data_summary = dataframe.describe().to_string()
    columns_info = dataframe.dtypes.to_string()
    prompt = f"""Based on the following data summary and column information:
    Data Summary:
    {data_summary}

    Columns:
    {columns_info}

    Perform the following calculation or create a new feature:
    {calculation_prompt}

    Provide the Python code to perform this calculation or create this new feature.
    The code should create a new column in the dataframe.
    Format your response as:
    ```python
    # Code to perform calculation or create new feature
    df['new_column_name'] = ...
    ```
    Also provide a brief explanation of what the calculation does and how it might be useful for analysis.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data scientist capable of performing calculations and creating new features based on existing data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.5
        )

        calculation_result = response.choices[0].message.content.strip()
        return calculation_result
    except Exception as e:
        st.error(f"An error occurred while generating the calculation code: {str(e)}")
        return "Unable to generate calculation code due to an error."

def execute_calculation_code(dataframe, code):
    try:
        # Create a copy of the dataframe to avoid modifying the original
        df = dataframe.copy()
        
        # Execute the code
        exec(code, globals(), {"df": df, "np": np, "pd": pd})
        
        # Return the modified dataframe
        return df
    except Exception as e:
        st.error(f"An error occurred while executing the calculation code: {str(e)}")
        return None

def generate_goals_with_new_features(dataframe, num_goals, comprehensive_report=None):
    data_summary = dataframe.describe().to_string()
    columns_info = dataframe.dtypes.to_string()
    prompt = f"""Based on the following data summary and column information:
    Data Summary:
    {data_summary}
    
    Columns:
    {columns_info}

    {"And considering the following comprehensive analysis report:" if comprehensive_report else ""}
    {comprehensive_report if comprehensive_report else ""}

    Generate {num_goals} unique analytical goals or questions that incorporate both the original and newly created features in this dataset.
    Ensure the goals are specific, actionable, and focus on deriving insights from the relationships between original and new features.
    Format your response as a numbered list:
    1. [First goal]
    2. [Second goal]
    ...
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data scientist tasked with generating analytical goals based on both original and derived features in a dataset."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.8
        )

        goals = response.choices[0].message.content.strip().split("\n")
        # Filter out empty goals and remove numbering
        goals = [goal.split(". ", 1)[1] if ". " in goal else goal for goal in goals if goal.strip()]
        return goals[:num_goals]  # Ensure we only return the requested number of goals
    except Exception as e:
        st.error(f"An error occurred while generating goals with new features: {str(e)}")
        return []

def init_db():
    conn = sqlite3.connect('goals_database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS goals
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  goal_set TEXT,
                  goals TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def save_goals_to_db(goal_set, goals):
    conn = sqlite3.connect('goals_database.db')
    c = conn.cursor()
    c.execute("INSERT INTO goals (goal_set, goals) VALUES (?, ?)",
              (goal_set, str(goals)))
    conn.commit()
    conn.close()

def get_past_goals():
    conn = sqlite3.connect('goals_database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM goals ORDER BY timestamp DESC LIMIT 10")
    past_goals = c.fetchall()
    conn.close()
    return past_goals

def generate_single_goal(dataframe, goal_number, previous_goals):
    data_summary = dataframe.describe().to_string()
    columns_info = dataframe.dtypes.to_string()
    prompt = f"""Based on the following data summary and column information:
    Data Summary:
    {data_summary}

    Columns:
    {columns_info}

    Generate a single, unique analytical goal or question (goal number {goal_number}) that could be explored with this dataset.
    This goal should be substantially different from the following previous goals:
    {previous_goals}

    Ensure the goal is specific, actionable, and focuses on a different aspect of the data or a different type of analysis than the previous goals.
    Do not include any numbering or prefixes like "Goal 1:" in your response."""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data scientist tasked with generating diverse analytical goals."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.8
        )

        goal = response.choices[0].message.content.strip()
        return goal
    except Exception as e:
        st.error(f"An error occurred while generating goal {goal_number}: {str(e)}")
        return f"Analyze the data to derive meaningful insights (Error occurred)."

def generate_goal_sets(dataframe, num_sets, goals_per_set):
    data_summary = dataframe.describe().to_string()
    columns_info = dataframe.dtypes.to_string()
    
    prompt = f"""Based on the following data summary and column information:
    Data Summary:
    {data_summary}

    Columns:
    {columns_info}

    Generate {num_sets} sets of analytical goals, with {goals_per_set} goals in each set. Each set should focus on a different perspective or aspect of the data. The sets should be diverse and cover various analytical approaches.

    For each set, provide:
    1. A brief description of the analytical perspective or focus (1-2 sentences)
    2. {goals_per_set} specific, actionable goals that align with this perspective

    Format your response as follows:
    Set 1: [Brief description of perspective]
    1. [Goal 1]
    2. [Goal 2]
    ...

    Set 2: [Brief description of perspective]
    1. [Goal 1]
    2. [Goal 2]
    ...

    And so on for {num_sets} sets."""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data scientist tasked with generating diverse sets of analytical goals."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.8
        )

        goal_sets = response.choices[0].message.content.strip().split("\n\n")
        return [set.strip() for set in goal_sets]
    except Exception as e:
        st.error(f"An error occurred while generating goal sets: {str(e)}")
        return []

def parse_goal_sets(goal_sets):
    parsed_sets = []
    for set in goal_sets:
        lines = set.split("\n")
        perspective = lines[0].split(": ", 1)[1] if ": " in lines[0] else lines[0]
        goals = [line.split(". ", 1)[1] if ". " in line else line for line in lines[1:]]
        parsed_sets.append({"perspective": perspective, "goals": goals})
    return parsed_sets

def generate_goals(dataframe, num_goals):
    goals = []
    for i in range(1, num_goals + 1):
        goal = generate_single_goal(dataframe, i, goals)
        goals.append(goal)
        time.sleep(1)  # Add a small delay between API calls
    return goals

def recommend_graph_for_goal(dataframe, goal):
    data_summary = dataframe.describe().to_string()
    columns_info = dataframe.dtypes.to_string()
    prompt = f"""Based on the following data summary and column information:
    Data Summary:
    {data_summary}
    
    Columns:
    {columns_info}

    For the analytical goal: '{goal}', recommend a suitable graph type and provide a detailed justification.
    Please provide a detailed, context-specific explanation of what the graph can reveal about the data in relation to this goal. 
    Focus on any trends, patterns, or relationships that are specifically relevant to the goal, and avoid general commentary about the graph type itself. Format your response as 'Type: [type], Justification: [justification]'"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data visualization expert that recommends appropriate graph types for data analysis goals with detailed, context-specific justifications."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.7
        )

        recommendation = response.choices[0].message.content.strip()
        parts = recommendation.split(', Justification: ')
        if len(parts) == 2:
            graph_type, justification = parts
            graph_type = graph_type.replace('Type: ', '').strip().lower()
        else:
            graph_type = "bar"
            justification = "Unable to parse recommendation. Using default bar chart."
        
        return graph_type, justification
    except Exception as e:
        st.error(f"An error occurred while recommending a graph: {str(e)}")
        return "bar", "Default recommendation due to an error."

def create_flexible_graph(df, graph_type, goal, key_prefix):
    if df.empty:
        st.error("The dataset is empty. Unable to create a graph.")
        return None

    x_column = st.selectbox(f"Select X-axis for {goal}", df.columns, key=f"{key_prefix}_x")
    y_columns = st.multiselect(f"Select Y-axis for {goal}", df.columns, default=[df.columns[0]], key=f"{key_prefix}_y")

    if not y_columns:
        st.error("Please select at least one Y-axis column.")
        return None

    try:
        if graph_type == "line":
            fig = px.line(df, x=x_column, y=y_columns, title=f"Line Chart - {goal}")
        elif graph_type == "bar":
            fig = px.bar(df, x=x_column, y=y_columns, title=f"Bar Chart - {goal}")
        elif graph_type == "scatter":
            fig = px.scatter(df, x=x_column, y=y_columns[0], title=f"Scatter Plot - {goal}")
        elif graph_type == "box":
            fig = px.box(df, x=x_column, y=y_columns[0], title=f"Box Plot - {goal}")
        elif graph_type == "violin":
            fig = px.violin(df, x=x_column, y=y_columns[0], title=f"Violin Plot - {goal}")
        elif graph_type == "histogram":
            fig = px.histogram(df, x=x_column, title=f"Histogram - {goal}")
        elif graph_type == "heatmap":
            fig = px.imshow(df.corr(), title=f"Correlation Heatmap - {goal}")
        else:
            st.error(f"Graph type '{graph_type}' is not supported. Defaulting to bar chart.")
            fig = px.bar(df, x=x_column, y=y_columns, title=f"Bar Chart - {goal}")

        explanation = generate_graph_explanation(df, x_column, y_columns, graph_type, goal)
        st.write("Graph Explanation:")
        st.write(explanation)

        return fig
    except Exception as e:
        st.error(f"An error occurred while creating the graph: {str(e)}")
        return None

def generate_graph_explanation(df, x_column, y_columns, graph_type, goal):
    data_summary = df.describe().to_string()
    prompt = f"""
    Based on the following data summary:
    {data_summary}

    A {graph_type} chart has been created with:
    X-axis: {x_column}
    Y-axis: {', '.join(y_columns)}

    The analytical goal is: {goal}

    Please explain how this graph supports or addresses the given goal in depth, avoiding any generic commentary on the graph type itself. 
    Consider the relationships between the chosen variables, any trends or patterns visible in the data, 
    and how these relate to the stated goal. Provide a concise but insightful explanation.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analysis expert providing insights on how visualizations support analytical goals."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )

        explanation = response.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        st.error(f"An error occurred while generating the explanation: {str(e)}")
        return "Unable to generate a detailed explanation due to an error."

def recommend_and_plot_efficient_graph(dataframe, previous_graphs, key_prefix):
    data_summary = dataframe.describe().to_string()
    columns_info = dataframe.dtypes.to_string()
    prompt = f"""
    Based on the following data summary and column information:
    Data Summary:
    {data_summary}
    
    Columns:
    {columns_info}

    Recommend ONE efficient plot type for this dataset that would help visualize the data effectively. 
    This plot type must be different from these previously suggested graphs: {', '.join(previous_graphs)}.
    Provide a brief justification for your recommendation, focusing on how this specific graph type can reveal insights for the given data and context.
    Format your response as 'Plot Type: [type] - Justification: [justification]'.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that recommends efficient plot types for data visualization based on the dataset."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.7
        )

        recommendation = response.choices[0].message.content.strip()
        plot_type, justification = recommendation.split(' - Justification: ')
        plot_type = plot_type.split(': ')[1].lower()

        st.write(f"Recommended Efficient Plot: {plot_type}")
        st.write(f"Justification: {justification}")

        # Create the recommended plot
        fig = create_flexible_graph(dataframe, plot_type, "Efficient Data Visualization", key_prefix)
        
        if fig:
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred while recommending and plotting an efficient graph: {str(e)}")

def generate_report(goal, recommendation, dataframe, comprehensive_report=None):
    data_summary = dataframe.describe().to_string()
    columns_info = dataframe.dtypes.to_string()

    if comprehensive_report:
        prompt = f"""Based on the following comprehensive analysis report:
        {comprehensive_report}

        And considering the following data summary and column information:
        Data Summary:
        {data_summary}

        Columns:
        {columns_info}

        And the following analytical goal and its recommended visualization:
        Goal: {goal}
        Visualization: {recommendation[0]}

        Generate a comprehensive report summarizing the key findings and insights from the data analysis. 
        Incorporate insights from the comprehensive analysis to provide a more holistic interpretation.
        The report should include:
        1. An overview of the dataset
        2. A summary of the goal the user chose to look at and how the recommended visualization provides evidence for the goal. No "if" statements, you should be able to make conclusions based on the graph you have generated.
        3. Key insights derived from the visualization
        4. Overall conclusions and recommendations based on the analysis

        Format the report in Markdown with appropriate headers and bullet points.
        """
    else:
        prompt = f"""Based on the following data summary and column information:
        Data Summary:
        {data_summary}

        Columns:
        {columns_info}

        And the following analytical goal and its recommended visualization:
        Goal: {goal}
        Visualization: {recommendation[0]}

        Generate a comprehensive report summarizing the key findings and insights from the data analysis. 
        The report should include:
        1. An overview of the dataset
        2. A summary of the goal the user chose to look at and how the recommended visualization provides evidence for the goal. No "if" statements, you should be able to make conclusions based on the graph you have generated.
        3. Key insights derived from the visualization
        4. Overall conclusions and recommendations based on the analysis

        Format the report in Markdown with appropriate headers and bullet points.
        """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analyst tasked with creating comprehensive reports based on data analysis and visualizations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )

        report = response.choices[0].message.content.strip()
        return report
    except Exception as e:
        st.error(f"An error occurred while generating the report: {str(e)}")
        return "Unable to generate a report due to an error."

def generate_tot_followup_goals(dataframe, selected_goal):
    data_summary = dataframe.describe().to_string()
    columns_info = dataframe.dtypes.to_string()
    prompt = f"""Based on the following data summary and column information:
    Data Summary:
    {data_summary}

    Columns:
    {columns_info}

    The user has selected the following analytical goal:
    "{selected_goal}"

    Using a Tree of Thought approach, generate three follow-up goals that delve deeper into specific areas of this main goal. Each follow-up goal should:
    1. Be more specific and focused than the main goal
    2. Explore a unique aspect or implication of the main goal and be different from the other follow-up goals
    3. Be actionable and analyzable using the given dataset

    Present your response in the following format:
    1. [First follow-up goal]
    - Rationale: [Brief explanation of how this goal relates to and expands upon the main goal]
    2. [Second follow-up goal]
    - Rationale: [Brief explanation of how this goal relates to and expands upon the main goal]
    3. [Third follow-up goal]
    - Rationale: [Brief explanation of how this goal relates to and expands upon the main goal]
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert data analyst using the Tree of Thought method to generate insightful follow-up goals for data analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        followup_goals = response.choices[0].message.content.strip().split("\n")
        return [goal.strip() for goal in followup_goals if goal.strip()]
    except Exception as e:
        st.error(f"An error occurred while generating follow-up goals: {str(e)}")
        return []

def generate_goals_from_prompt(dataframe, user_prompt, num_goals, comprehensive_report=None):
    data_summary = dataframe.describe().to_string()
    columns_info = dataframe.dtypes.to_string()
    
    if comprehensive_report:
        prompt = f"""Based on the following data summary and column information:
        Data Summary:
        {data_summary}

        Columns:
        {columns_info}

        {"And considering the following comprehensive analysis report:" if comprehensive_report else ""}
        {comprehensive_report if comprehensive_report else ""}

        And considering the user's prompt: "{user_prompt}"

        Generate {num_goals} unique analytical goals or questions that could be explored with this dataset.
        Ensure the goals are specific, actionable, and align with the user's prompt.
        Format your response as a numbered list:
        1. [First goal]
        2. [Second goal]
        ...
        """
    
    else:
        prompt = f"""Based on the following data summary and column information:
        Data Summary:
        {data_summary}

        Columns:
        {columns_info}

        And considering the user's prompt: "{user_prompt}"

        Generate {num_goals} unique analytical goals or questions that could be explored with this dataset.
        Ensure the goals are specific, actionable, and align with the user's prompt.
        Format your response as a numbered list:
        1. [First goal]
        2. [Second goal]
        ...
        """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data scientist tasked with generating analytical goals based on user prompts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.8
        )

        goals = response.choices[0].message.content.strip().split("\n")
        return [goal.split(". ", 1)[1] if ". " in goal else goal for goal in goals]
    except Exception as e:
        st.error(f"An error occurred while generating goals from the prompt: {str(e)}")
        return []

def perform_trend_analysis(dataframe, value_column, periods_to_forecast=30):
    """
    Perform trend analysis on the given dataframe using Prophet, handling both numeric and categorical data.
    """
    df = dataframe.copy()
    
    if value_column not in df.columns:
        return None, None, f"Error: Column '{value_column}' not found in the dataframe."

    df = df.dropna(subset=[value_column])

    if df[value_column].empty:
        return None, None, f"Error: Column '{value_column}' contains no valid data."

    if pd.api.types.is_numeric_dtype(df[value_column]):
        try:
            # Prepare data for Prophet
            df_prophet = df.reset_index().rename(columns={df.index.name: 'ds', value_column: 'y'})
            df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

            # Create and fit the model
            model = Prophet()
            model.fit(df_prophet)

            # Create future dataframe for forecasting
            future = model.make_future_dataframe(periods=periods_to_forecast)
            
            # Make predictions
            forecast = model.predict(future)

            # Create forecast plot
            fig_forecast = plot_plotly(model, forecast)
            fig_forecast.update_layout(title=f'Trend Analysis and Forecast for {value_column}')

            # Create components plot
            fig_components = plot_components_plotly(model, forecast)

            return fig_forecast, fig_components, None  # No error message
        except Exception as e:
            return None, None, f"Error in numeric analysis: {str(e)}"
    else:
        try:
            df[value_column] = df[value_column].astype(str)
            chunk_size = max(len(df) // 10, 1)
            chunks = [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]
            
            chunk_counts = []
            for i, chunk in enumerate(chunks):
                counts = Counter(chunk[value_column])
                for category, count in counts.items():
                    chunk_counts.append({'chunk': i, 'category': category, 'count': count})
            
            chunk_df = pd.DataFrame(chunk_counts)
            
            heatmap_fig = px.density_heatmap(chunk_df, x='chunk', y='category', z='count', 
                                             title=f'Category Frequency Over Time for {value_column}')
            heatmap_fig.update_layout(xaxis_title='Time (chunks)', yaxis_title='Category')
            
            overall_counts = Counter(df[value_column])
            pie_fig = px.pie(values=list(overall_counts.values()), names=list(overall_counts.keys()), 
                             title=f'Overall Distribution of {value_column}')
            
            return heatmap_fig, pie_fig, None  # No error message
        except Exception as e:
            return None, None, f"Error in categorical analysis: {str(e)}"

def predict_future_trends(dataframe, value_column, periods_to_forecast):
    """
    Predict future trends using ARIMA model.
    """
    # Prepare the data
    df = dataframe.sort_index()

    # Fit ARIMA model
    model = ARIMA(df[value_column], order=(1,1,1))  # You might need to adjust these parameters
    results = model.fit()

    # Make predictions
    forecast = results.forecast(steps=periods_to_forecast)
    
    # Create a dataframe with the forecasted values
    last_index = df.index[-1]
    forecast_index = pd.RangeIndex(start=last_index + 1, stop=last_index + periods_to_forecast + 1)
    forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_index)
    
    # Create a plot
    fig = px.line(df, y=value_column, title='Actual vs Forecast')
    fig.add_scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Forecast')

    return fig, forecast_df

def compare_time_periods(dataframe1, dataframe2, value_column):
    """
    Compare two datasets and provide insights.
    """
    df1 = dataframe1.sort_index()
    df2 = dataframe2.sort_index()

    # Calculate basic statistics
    mean1, mean2 = df1[value_column].mean(), df2[value_column].mean()
    std1, std2 = df1[value_column].std(), df2[value_column].std()
    
    # Calculate growth rate
    growth_rate = ((df2[value_column].iloc[-1] - df1[value_column].iloc[0]) / df1[value_column].iloc[0]) * 100

    # Create comparison plot
    fig = px.line(title='Comparison of Datasets')
    fig.add_scatter(x=df1.index, y=df1[value_column], name='Dataset 1')
    fig.add_scatter(x=df2.index, y=df2[value_column], name='Dataset 2')

    insights = f"""
    Comparison Insights:
    - Mean value in Dataset 1: {mean1:.2f}
    - Mean value in Dataset 2: {mean2:.2f}
    - Standard deviation in Dataset 1: {std1:.2f}
    - Standard deviation in Dataset 2: {std2:.2f}
    - Overall growth rate: {growth_rate:.2f}%
    """

    return fig, insights

def determine_best_graph(df, x_column, y_column):
        """
    Determine the best graph type based on the data types of x and y columns.
    """
        x_is_numeric = pd.api.types.is_numeric_dtype(df[x_column])
        y_is_numeric = pd.api.types.is_numeric_dtype(df[y_column])
    
        if x_is_numeric and y_is_numeric:
            return "scatter"
        
        elif x_is_numeric and not y_is_numeric:
         return "box"
        elif not x_is_numeric and y_is_numeric:
            return "bar"
        else:
            return "heatmap"

def create_graph(df, x_column, y_column):
    """
    Create a graph based on the data types of x and y columns.
    """
    graph_type = determine_best_graph(df, x_column, y_column)
    
    if graph_type == "scatter":
        fig = px.scatter(df, x=x_column, y=y_column, title=f'{y_column} vs {x_column}')
    elif graph_type == "box":
        fig = px.box(df, x=x_column, y=y_column, title=f'Distribution of {y_column} for each {x_column}')
    elif graph_type == "bar":
        agg_df = df.groupby(x_column)[y_column].mean().reset_index()
        fig = px.bar(agg_df, x=x_column, y=y_column, title=f'Average {y_column} for each {x_column}')
    else:  # heatmap
        pivot_df = df.pivot_table(index=y_column, columns=x_column, aggfunc='size', fill_value=0)
        fig = px.imshow(pivot_df, title=f'Heatmap of {y_column} vs {x_column}')
    
    return fig

def generate_trend_report(df, x_column, y_column):
    """
    Generate a report based on the trend analysis.
    """
    x_is_numeric = pd.api.types.is_numeric_dtype(df[x_column])
    y_is_numeric = pd.api.types.is_numeric_dtype(df[y_column])
    
    report = f"# Trend Analysis Report: {y_column} vs {x_column}\n\n"
    
    if x_is_numeric and y_is_numeric:
        correlation = df[x_column].corr(df[y_column])
        report += f"## Correlation Analysis\n"
        report += f"The correlation coefficient between {x_column} and {y_column} is {correlation:.2f}.\n"
        if correlation > 0.7:
            report += f"This indicates a strong positive correlation.\n"
        elif correlation < -0.7:
            report += f"This indicates a strong negative correlation.\n"
        elif abs(correlation) > 0.3:
            report += f"This indicates a moderate correlation.\n"
        else:
            report += f"This indicates a weak or no correlation.\n"
    
    report += f"\n## Summary Statistics\n"
    report += f"### {x_column}:\n"
    report += df[x_column].describe().to_string()
    report += f"\n\n### {y_column}:\n"
    report += df[y_column].describe().to_string()
    
    if not x_is_numeric:
        report += f"\n\n## Category Analysis for {x_column}\n"
        category_counts = df[x_column].value_counts()
        report += f"Top 5 categories in {x_column}:\n"
        report += category_counts.head().to_string()
    
    if not y_is_numeric:
        report += f"\n\n## Category Analysis for {y_column}\n"
        category_counts = df[y_column].value_counts()
        report += f"Top 5 categories in {y_column}:\n"
        report += category_counts.head().to_string()
    
    return report

def simple_forecast(series, periods):
    """
    Forecast future values using simple linear regression.
    """
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values

    model = LinearRegression()
    model.fit(X, y)

    # Forecast
    X_future = np.arange(len(series), len(series) + periods).reshape(-1, 1)
    forecast = model.predict(X_future)

    return forecast

def compare_and_forecast(df1, df2, x_column, y_column, periods_to_forecast):
    """
    Compare two datasets and provide forecast using simple linear regression.
    """
    series1 = df1[y_column]
    series2 = df2[y_column]

    # Calculate basic statistics
    mean1, mean2 = series1.mean(), series2.mean()
    std1, std2 = series1.std(), series2.std()
    
    # Calculate growth rate
    growth_rate = ((series2.iloc[-1] - series1.iloc[0]) / series1.iloc[0]) * 100

    # Create comparison plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df1[x_column], y=series1, name='Dataset 1'))
    fig.add_trace(go.Scatter(x=df2[x_column], y=series2, name='Dataset 2'))

    # Forecast for both datasets
    forecast1 = simple_forecast(series1, periods_to_forecast)
    forecast2 = simple_forecast(series2, periods_to_forecast)

    # Add forecasts to the plot
    last_x1 = df1[x_column].iloc[-1]
    last_x2 = df2[x_column].iloc[-1]
    forecast_x1 = np.arange(last_x1 + 1, last_x1 + periods_to_forecast + 1)
    forecast_x2 = np.arange(last_x2 + 1, last_x2 + periods_to_forecast + 1)
    fig.add_trace(go.Scatter(x=forecast_x1, y=forecast1, name='Forecast 1', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=forecast_x2, y=forecast2, name='Forecast 2', line=dict(dash='dash')))

    fig.update_layout(title='Comparison and Forecast', xaxis_title=x_column, yaxis_title=y_column)

    insights = f"""
    Comparison Insights:
    - Mean value in Dataset 1: {mean1:.2f}
    - Mean value in Dataset 2: {mean2:.2f}
    - Standard deviation in Dataset 1: {std1:.2f}
    - Standard deviation in Dataset 2: {std2:.2f}
    - Overall growth rate: {growth_rate:.2f}%

    Forecast Insights:
    - Forecasted end value for Dataset 1: {forecast1[-1]:.2f}
    - Forecasted end value for Dataset 2: {forecast2[-1]:.2f}
    - Forecasted growth for Dataset 1: {((forecast1[-1] - series1.iloc[-1]) / series1.iloc[-1] * 100):.2f}%
    - Forecasted growth for Dataset 2: {((forecast2[-1] - series2.iloc[-1]) / series2.iloc[-1] * 100):.2f}%
    """

    return fig, insights, forecast1, forecast2

def generate_qualitative_explanation(df1, df2, x_column, y_column, insights, forecast1, forecast2):
    data_summary = f"""
    Dataset 1 summary for {y_column}:
    {df1[y_column].describe().to_string()}
    
    Dataset 2 summary for {y_column}:
    {df2[y_column].describe().to_string()}
    """
    
    prompt = f"""
    Based on the following data summaries and insights:

    {data_summary}

    {insights}

    Provide a qualitative, contextual explanation of what these results mean. 
    Focus on the implications of the trends, differences between the datasets, and what the forecast suggests for the future.
    Avoid referring to specific graph types or technical terms. Instead, interpret the data in a way that a non-technical stakeholder would understand.
    Consider potential real-world factors that might explain the observed trends and differences.
    Limit your response to about 250 words.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert data analyst providing qualitative insights on trend analysis and forecasting."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=350,
            temperature=0.7
        )

        explanation = response.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        st.error(f"An error occurred while generating the qualitative explanation: {str(e)}")
        return "Unable to generate a qualitative explanation due to an error."

def analyze_goal_interaction(df, goal1, goal2):
    """
    Analyze the interaction between two goals and create a visualization.
    """
    # Extract columns for each goal
    cols1 = [col for col in df.columns if col in goal1]
    cols2 = [col for col in df.columns if col in goal2]

    if not cols1 or not cols2:
        return None, "Unable to identify relevant columns for the selected goals."

    # Select the first relevant column for each goal
    col1, col2 = cols1[0], cols2[0]

    # Create scatter plot
    fig = px.scatter(df, x=col1, y=col2, title=f"Interaction between '{goal1}' and '{goal2}'")
    
    # Calculate correlation
    correlation, p_value = stats.pearsonr(df[col1], df[col2])
    
    # Add correlation line
    fig.add_trace(px.scatter(df, x=col1, y=col2, trendline="ols").data[1])

    explanation = f"""
    Analysis of Interaction between '{goal1}' and '{goal2}':
    
    1. Correlation: The Pearson correlation coefficient between the two variables is {correlation:.2f}.
       This indicates a {"strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"} 
       {"positive" if correlation > 0 else "negative"} relationship.
    
    2. Statistical Significance: The p-value is {p_value:.4f}, which suggests that the correlation is 
       {"statistically significant" if p_value < 0.05 else "not statistically significant"} at the 0.05 level.
    
    3. Trend: The scatter plot and the overlaid trend line visually represent the relationship between the two variables.
       {"The upward slope indicates a positive relationship." if correlation > 0 else "The downward slope indicates a negative relationship." if correlation < 0 else "The flat line suggests no clear linear relationship."}
    
    4. Implications: 
       {"As one variable increases, the other tends to increase as well." if correlation > 0 else "As one variable increases, the other tends to decrease." if correlation < 0 else "There doesn't seem to be a clear linear relationship between these variables."}
       This could imply that {"there might be a causal relationship or a common underlying factor affecting both variables." if abs(correlation) > 0.3 else "these aspects of the data might be largely independent of each other."}
    
    5. Further Investigation: It would be worthwhile to {"investigate potential causal mechanisms or confounding variables that might explain this relationship." if abs(correlation) > 0.3 else "explore other types of relationships or additional variables that might be more strongly related to these goals."}
    """
    
    return fig, explanation

def visualize_top_performers(top_performers, category_column, metric_column):
    """Create a bar chart of top performers."""
    if metric_column in top_performers.columns and pd.api.types.is_numeric_dtype(top_performers[metric_column]):
        fig = px.bar(top_performers, x=category_column, y=metric_column, 
                     title=f"Top Performers by {metric_column}")
    else:
        # For non-numeric data or when metric_column is not in the dataframe, we'll plot the count of each category
        fig = px.bar(top_performers, x=category_column, y='count' if 'count' in top_performers.columns else top_performers.index,
                     title=f"Top {category_column} Categories")
        fig.update_xaxes(title=category_column)
        fig.update_yaxes(title="Count")
    return fig

def visualize_top_performers(top_performers, category_column, metric_column):
    """Create a bar chart of top performers."""
    if pd.api.types.is_numeric_dtype(top_performers[metric_column]):
        fig = px.bar(top_performers, x=category_column, y=metric_column, 
                     title=f"Top Performers by {metric_column}")
    else:
        # For non-numeric data, we'll plot the frequency of each category
        value_counts = top_performers[metric_column].value_counts()
        fig = px.bar(x=value_counts.index, y=value_counts.values, 
                     title=f"Top {metric_column} Categories")
        fig.update_xaxes(title=metric_column)
        fig.update_yaxes(title="Frequency")
    return fig

def generate_top_performers_report(df, top_performers, category_column, metric_column, top_n, comprehensive_report=None):
    is_numeric = metric_column in df.columns and pd.api.types.is_numeric_dtype(df[metric_column])

    report = f"""
    # Top Performers Analysis Report

    ## Overview
    This report presents an analysis of the top {top_n} performers based on the {'**' + metric_column + '**' if is_numeric else 'frequency'} metric, 
    categorized by **{category_column}**.
    """
    
    if is_numeric:
        summary_stats = f"""
        ## Key Insights

        ### Top Performer Statistics
        - **Highest {metric_column}:** {format_value(top_performers[metric_column].max())} (achieved by {top_performers.iloc[0][category_column]})
        - **Average {metric_column} among top performers:** {format_value(top_performers[metric_column].mean())}
        - **Range of {metric_column} in top performers:** {format_value(top_performers[metric_column].max() - top_performers[metric_column].min())}

        ### Comparison to Overall Dataset
        - **Overall average {metric_column}:** {format_value(df[metric_column].mean())}
        - **Overall median {metric_column}:** {format_value(df[metric_column].median())}
        - **Top performers vs. overall average:** {format_value(((top_performers[metric_column].mean() - df[metric_column].mean()) / df[metric_column].mean() * 100))}% higher
        """
    else:
        summary_stats = f"""
        ## Key Insights

        ### Top Performer Statistics
        - **Most common {category_column} among top performers:** {top_performers[category_column].mode().iloc[0]}
        - **Unique {category_column} values among top performers:** {top_performers[category_column].nunique()}

        ### Comparison to Overall Dataset
        - **Total unique {category_column} values:** {df[category_column].nunique()}
        - **Top performers represent:** {format_value(top_performers[category_column].nunique() / df[category_column].nunique() * 100)}% of all unique values
        """
    
    recommendations, next_steps = generate_contextual_insights(df, top_performers, category_column, metric_column, top_n)
    
    if comprehensive_report:
        prompt = f"""
        Based on the following comprehensive analysis report:
        {comprehensive_report}

        And considering the top performers analysis:
        {report}

        Generate insights and recommendations that incorporate both the comprehensive analysis and the top performers analysis.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI assistant specialized in data analysis and business intelligence."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            enhanced_insights = response.choices[0].message.content.strip()
            recommendations.extend(enhanced_insights.split("\n"))
        except Exception as e:
            st.error(f"An error occurred while generating enhanced insights: {str(e)}")

    return report, summary_stats, top_performers, df[category_column].describe() if not is_numeric else df[metric_column].describe(), recommendations, next_steps

def format_value(value):
    """Format a value as a string, handling both numeric and non-numeric types."""
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)

def generate_enhanced_insights(insights_text, comprehensive_report):
    prompt = f"""
    Based on the following comprehensive analysis report:
    {comprehensive_report}

    And considering the following insights from time series analysis:
    {insights_text}

    Generate enhanced insights that incorporate both the comprehensive analysis and the time series analysis.
    Focus on how the comprehensive analysis might inform or modify the interpretation of the time series trends and forecast.
    Limit your response to about 3-5 concise bullet points.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in data analysis and business intelligence."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred while generating enhanced insights: {str(e)}"
    
def analyze_and_forecast(df, y_column, periods_to_forecast, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, comprehensive_report=None):
    # Prepare data for Prophet
    df_prophet = df.reset_index().rename(columns={'date': 'ds', y_column: 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    # Create and fit the model
    model = Prophet(yearly_seasonality=yearly_seasonality, 
                    weekly_seasonality=weekly_seasonality, 
                    daily_seasonality=daily_seasonality)
    model.fit(df_prophet)

    # Create future dataframe for forecasting
    future = model.make_future_dataframe(periods=periods_to_forecast)
    
    # Make predictions
    forecast = model.predict(future)

    # Calculate overall change
    overall_change = ((df[y_column].iloc[-1] - df[y_column].iloc[0]) / df[y_column].iloc[0]) * 100
    average_value = df[y_column].mean()

    # Create forecast plot
    fig_forecast = plot_plotly(model, forecast)
    fig_forecast.update_layout(title=f'Forecast of {y_column}')

    # Create components plot
    fig_components = plot_components_plotly(model, forecast)

    # Generate insights text
    insights_text = f"""
    Overall change: {overall_change:.2f}%
    Average {y_column}: {average_value:.2f}
    Forecast for next {periods_to_forecast} periods: 
    Start: {forecast['yhat'].iloc[-periods_to_forecast]:.2f}
    End: {forecast['yhat'].iloc[-1]:.2f}
    """
    if comprehensive_report:
        enhanced_insights = generate_enhanced_insights(insights_text, comprehensive_report)
        insights_text += f"\n\nEnhanced Insights:\n{enhanced_insights}"

    # Get the top 3 dates with the highest positive and negative forecast changes
    forecast['change'] = forecast['yhat'].diff()
    top_increases = forecast.nlargest(3, 'change')
    top_decreases = forecast.nsmallest(3, 'change')

    changepoints_text = "Top 3 forecasted increases:\n"
    for _, row in top_increases.iterrows():
        changepoints_text += f"  {row['ds'].date()}: {row['change']:.2f}\n"
    changepoints_text += "\nTop 3 forecasted decreases:\n"
    for _, row in top_decreases.iterrows():
        changepoints_text += f"  {row['ds'].date()}: {row['change']:.2f}\n"

    # If comprehensive report is provided, use it to enhance insights
    if comprehensive_report:
        enhanced_insights = generate_enhanced_insights(insights_text, comprehensive_report)
        insights_text += f"\n\nEnhanced Insights:\n{enhanced_insights}"

    return fig_forecast, fig_components, insights_text, forecast['yhat'].tail(periods_to_forecast).tolist(), changepoints_text

def generate_qualitative_explanation(df, y_column, insights_text, forecast, comprehensive_report=None):
    if comprehensive_report:
        prompt = f"""
        Based on the following comprehensive analysis report:
        {comprehensive_report}

        And considering the following data and insights about {y_column}:
        {insights_text}

        Forecast for next periods: {forecast}

        Time range: from {df.index.min()} to {df.index.max()}

        Provide an enhanced qualitative explanation of the trends observed and what the forecast suggests for the future.
        Incorporate insights from the comprehensive analysis to provide a more holistic interpretation.
        Consider the overall trend, any significant changes between past and current datasets, and potential factors that might explain these trends.
        Limit your response to about 300 words.
        """
    else:
        prompt = f"""
        Based on the following data and insights about {y_column}:
        {insights_text}

        Forecast for next periods: {forecast}

        Time range: from {df.index.min()} to {df.index.max()}

        Provide a qualitative explanation of the trends observed and what the forecast suggests for the future.
        Consider the overall trend, any significant changes between past and current datasets, and potential factors that might explain these trends.
        Limit your response to about 200 words.
        """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert data analyst providing qualitative insights on trend analysis and forecasting."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        explanation = response.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        st.error(f"An error occurred while generating the qualitative explanation: {str(e)}")
        return "Unable to generate a qualitative explanation due to an error."

def perform_advanced_analysis(df, top_performers, category_column, metric_column):
    """
    Perform advanced analysis on top performers and the overall dataset.
    """
    # 1. Detailed Analysis
    # Perform a more in-depth analysis of top performers
    top_performer_stats = top_performers.describe()
    
    # 2. Pattern Identification
    # Use K-means clustering to identify patterns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[[metric_column]])
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_data)
    
    cluster_means = df.groupby('cluster')[metric_column].mean().sort_values(ascending=False)
    top_cluster = cluster_means.index[0]
    
    # 3. Correlation Analysis
    # Find correlations between the metric and other numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()[metric_column].sort_values(ascending=False)
    
    # 4. Principal Component Analysis
    # Perform PCA to identify key factors contributing to performance
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[numeric_cols])
    df['pca_1'] = pca_result[:, 0]
    df['pca_2'] = pca_result[:, 1]
    
    return top_performer_stats, cluster_means, correlations, pca

def generate_contextual_insights(df, top_performers, category_column, metric_column, top_n):
    is_numeric = pd.api.types.is_numeric_dtype(df[metric_column])
    
    if is_numeric:
        data_summary = f"""
        Dataset summary:
        - Total entries: {len(df)}
        - Top {top_n} performers average {metric_column}: {top_performers[metric_column].mean():.2f}
        - Overall average {metric_column}: {df[metric_column].mean():.2f}
        - Top performer vs overall performance difference: {((top_performers[metric_column].mean() / df[metric_column].mean()) - 1) * 100:.2f}%
        - Category column: {category_column}
        - Metric column: {metric_column}
        """
    else:
        data_summary = f"""
        Dataset summary:
        - Total entries: {len(df)}
        - Top {top_n} performers most common {metric_column}: {top_performers[metric_column].mode().iloc[0]}
        - Number of unique {metric_column} values in top performers: {top_performers[metric_column].nunique()}
        - Total unique {metric_column} values: {df[metric_column].nunique()}
        - Category column: {category_column}
        - Metric column: {metric_column}
        """

    prompt = f"""
    Based on the following data summary about top performers:

    {data_summary}

    Generate 4 unique and specific recommendations for improving overall performance. 
    Each recommendation should be different and tailored to the specific metrics and categories in the data.
    
    Then, generate 5 unique and specific next steps for implementing these recommendations and further analyzing the data.
    Each next step should be concrete, actionable, and different from the others.

    Format your response as follows:
    Recommendations:
    1. [First recommendation]
    2. [Second recommendation]
    3. [Third recommendation]
    4. [Fourth recommendation]

    Next Steps:
    1. [First next step]
    2. [Second next step]
    3. [Third next step]
    4. [Fourth next step]
    5. [Fifth next step]
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analyst providing insights on top performers analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )

        insights = response.choices[0].message.content.strip().split("\n\n")
        recommendations = insights[0].split("\n")[1:]  # Skip the "Recommendations:" header
        next_steps = insights[1].split("\n")[1:]  # Skip the "Next Steps:" header

        return recommendations, next_steps
    except Exception as e:
        st.error(f"An error occurred while generating insights: {str(e)}")
        return [], []

def perform_automated_comprehensive_analysis(df1, df2):
    st.write("# Automated Comprehensive Data Analysis Report")

    # Automatically select columns for analysis
    category_column = df1.select_dtypes(include=['object']).columns[0]
    metric_column = df1.select_dtypes(include=[np.number]).columns[0]
    date_column = df1.columns[0]  # Assume first column is date
    y_column = metric_column  # Use the same column for forecasting

    # 1. Top Performers Analysis
    st.write("## 1. Top Performers Analysis")
    top_n = 5  # Set a default number of top performers
    top_performers = analyze_top_performers(df1, category_column, metric_column, top_n)
    report, summary_stats, top_performers_df, distribution_stats, recommendations, next_steps = generate_top_performers_report(df1, top_performers, category_column, metric_column, top_n)
    
    st.write("### Top Performers Table")
    st.dataframe(top_performers_df)
    
    st.write("### Summary Statistics")
    st.write(summary_stats)
    
    st.write("### Distribution Analysis")
    st.write(f"The distribution of the {metric_column} across the entire dataset is as follows:")
    st.dataframe(distribution_stats)

    st.write("### Top Performers Visualization")
    fig = visualize_top_performers(top_performers, category_column, metric_column)
    st.plotly_chart(fig)

    st.write("### Recommendations and Next Steps")
    for i, (rec, step) in enumerate(zip(recommendations, next_steps), 1):
        st.write(f"**Recommendation {i}:** {rec}")
        st.write(f"**Next Step {i}:** {step}")
        st.write("---")

    # 2. Dataset Comparison and Forecasting
    st.write("## 2. Dataset Comparison and Forecasting")
    periods_to_forecast = 30  # Set a default forecast period

    try:
        df1_analysis = df1[[date_column, y_column]].dropna()
        df2_analysis = df2[[date_column, y_column]].dropna()
        df1_analysis['Dataset'] = 'Past'
        df2_analysis['Dataset'] = 'Current'
        combined_df = pd.concat([df1_analysis, df2_analysis]).sort_values(date_column)
        combined_df[date_column] = pd.to_datetime(combined_df[date_column])
        combined_df.set_index(date_column, inplace=True)

        fig, insights_text, forecast = analyze_and_forecast(combined_df, y_column, periods_to_forecast)
        st.plotly_chart(fig)
        
        st.write("### Numerical Insights")
        st.write(insights_text)

        st.write("### Forecast")
        st.write(f"Forecast for next {periods_to_forecast} periods:", forecast)

        qualitative_explanation = generate_qualitative_explanation(combined_df, y_column, insights_text, forecast)
        st.write("### Qualitative Analysis")
        st.write(qualitative_explanation)

    except Exception as e:
        st.error(f"An error occurred during the comparison and forecasting: {str(e)}")

    # 3. Goal Generation and Analysis
    st.write("## 3. Goal Generation and Analysis")
    num_goals = 5  # Set a default number of goals
    goals = generate_goals(df1, num_goals)

    st.write("### Generated Goals:")
    for i, goal in enumerate(goals, 1):
        st.write(f"{i}. {goal}")

    # Analyze all generated goals
    for goal in goals:
        st.write(f"### Analysis for: {goal}")
        graph_type, justification = recommend_graph_for_goal(df1, goal)
        st.write(f"Recommended Graph Type: {graph_type}")
        st.write(f"Justification: {justification}")

        fig = create_flexible_graph(df1, graph_type, goal, f"goal_{goals.index(goal)}")
        if fig:
            st.plotly_chart(fig)

    # Goal Interaction Analysis
    if len(goals) >= 2:
        st.write("### Goal Interaction Analysis")
        goal1, goal2 = goals[:2]  # Analyze interaction between first two goals
        interaction_fig, interaction_explanation = analyze_goal_interaction(df1, goal1, goal2)
        if interaction_fig:
            st.plotly_chart(interaction_fig)
            st.markdown(interaction_explanation)

    # 4. AI-driven Insights
    st.write("## 4. AI-driven Insights and Recommendations")
    
    X = df1.select_dtypes(include=[np.number])
    y = df1[metric_column]
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    st.write("### Most Influential Factors:")
    for feature, score in mi_scores.head(5).items():
        st.write(f"- {feature}: This factor has a significant impact on {metric_column}.")

    st.write("### Action Plan:")
    st.write(f"1. Focus on top performers: Analyze and replicate the strategies of your highest-performing {category_column}s.")
    st.write(f"2. Address forecast trends: Based on the {y_column} forecast, prepare strategies to {'capitalize on the projected growth' if forecast[-1] > forecast[0] else 'mitigate the projected decline'}.")
    st.write("3. Prioritize goals: Start by addressing the first two generated goals, as they align closely with your current data trends.")
    st.write("4. Continuous monitoring: Regularly track the identified influential factors and adjust strategies accordingly.")

    # Generate and offer downloadable report
    report = generate_comprehensive_report(df1, df2, top_performers, goals, forecast, mi_scores, category_column, metric_column, y_column)
    st.download_button(
        label="Download Comprehensive Analysis Report",
        data=report,
        file_name="comprehensive_analysis_report.md",
        mime="text/markdown"
    )

def generate_comprehensive_report(df1, df2, top_performers, goals, forecast, mi_scores, category_column, metric_column, y_column):
    report = "# Comprehensive Data Analysis Report\n\n"

    # Top Performers Analysis
    report += "## 1. Top Performers Analysis\n\n"
    report += f"Top {len(top_performers)} performers based on {metric_column}:\n"
    for i, (_, row) in enumerate(top_performers.iterrows(), 1):
        report += f"{i}. {row[category_column]}: {row[metric_column]}\n"
    report += "\n"

    # Dataset Comparison and Forecasting
    report += "## 2. Dataset Comparison and Forecasting\n\n"
    report += f"Forecast for next {len(forecast)} periods: {forecast}\n\n"

    # Goal Generation and Analysis
    report += "## 3. Goal Generation and Analysis\n\n"
    for i, goal in enumerate(goals, 1):
        report += f"{i}. {goal}\n"
    report += "\n"

    # AI-driven Insights
    report += "## 4. AI-driven Insights and Recommendations\n\n"
    report += "### Most Influential Factors:\n"
    for feature, score in mi_scores.head(5).items():
        report += f"- {feature}: This factor has a significant impact on {metric_column}.\n"
    report += "\n"

    report += "### Action Plan:\n"
    report += f"1. Focus on top performers: Analyze and replicate the strategies of your highest-performing {category_column}s.\n"
    report += f"2. Address forecast trends: Based on the {y_column} forecast, prepare strategies to {'capitalize on the projected growth' if forecast[-1] > forecast[0] else 'mitigate the projected decline'}.\n"
    report += "3. Prioritize goals: Start by addressing the first two generated goals, as they align closely with your current data trends.\n"
    report += "4. Continuous monitoring: Regularly track the identified influential factors and adjust strategies accordingly.\n"

    return report

def perform_flexible_comprehensive_analysis(df_start, df_end):
    st.write("# Focused Business Insights and Analysis")

    try:
        # 1. Key Performance Indicators (KPIs) Analysis
        st.write("## 1. Key Performance Indicators (KPIs) Analysis")
        
        numeric_columns = df_start.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            target_column = numeric_columns[-1]
            correlations = df_start[numeric_columns].corr()[target_column].sort_values(ascending=False)
            top_kpis = correlations[1:4]  # Top 3 KPIs excluding the target itself

            st.write(f"Top KPIs influencing {target_column}:")
            for kpi, corr in top_kpis.items():
                st.write(f"- {kpi}: Correlation of {corr:.2f}")
                
                # Add error handling for regression analysis
                try:
                    X = df_start[[kpi]]
                    y = df_start[target_column]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                    st.write(f"Regression Analysis for {kpi}:")
                    st.write(f"- R-squared: {r2:.2f}")
                    st.write(f"- RMSE: {rmse:.2f}")
                    st.write(f"- For every 1 unit increase in {kpi}, {target_column} changes by {model.coef_[0]:.2f} units")
                except Exception as e:
                    st.error(f"Error in regression analysis for {kpi}: {str(e)}")
        else:
            st.error("No numeric columns found in the dataset. KPI analysis cannot be performed.")
            top_kpis = []
            target_column = None

        # 2. Start-End Comparison Analysis
        st.write("## 2. Start-End Comparison Analysis")
        summary_df = perform_start_end_comparison(df_start, df_end, top_kpis.index.tolist() if len(top_kpis) > 0 else [])

        # 3. Business Goal Recommendations
        st.write("## 3. Business Goal Recommendations")
        goals = generate_business_goals(summary_df, top_kpis.index.tolist() if len(top_kpis) > 0 else [], target_column)
        for i, goal in enumerate(goals, 1):
            st.write(f"{i}. {goal}")
            recommendation = generate_goal_recommendation(df_start, df_end, goal, top_kpis.index.tolist() if len(top_kpis) > 0 else [], target_column)
            st.write("Recommendation:", recommendation)

        # 4. Strategic Insights and Action Plan
        st.write("## 4. Strategic Insights and Action Plan")
        action_plan = generate_action_plan(summary_df, top_kpis.index.tolist() if len(top_kpis) > 0 else [], target_column)
        for i, action in enumerate(action_plan, 1):
            st.write(f"{i}. {action}")

        # Generate and offer downloadable report
        report = generate_focused_report(df_start, df_end, summary_df, goals, action_plan, top_kpis, target_column)
        st.download_button(
            label="Download Focused Business Analysis Report",
            data=report,
            file_name="focused_business_analysis_report.md",
            mime="text/markdown"
        )

    except Exception as e:
        st.error(f"An error occurred during the analysis: {str(e)}")
        st.write("Please check your data and try again. If the problem persists, contact support.")

def perform_start_end_comparison(df_start, df_end, top_kpis):
    comparison_results = []
    try:
        for col in top_kpis:
            if col in df_start.columns and col in df_end.columns:
                start_data = df_start[col].dropna()
                end_data = df_end[col].dropna()
                
                if len(start_data) > 0 and len(end_data) > 0:
                    mean_change = ((end_data.mean() - start_data.mean()) / start_data.mean()) * 100 if start_data.mean() != 0 else np.inf
                    std_change = ((end_data.std() - start_data.std()) / start_data.std()) * 100 if start_data.std() != 0 else np.inf
                    
                    statistic, p_value = stats.mannwhitneyu(start_data, end_data, alternative='two-sided')
                    
                    comparison_results.append({
                        'Metric': col,
                        'Mean Change (%)': mean_change,
                        'Std Dev Change (%)': std_change,
                        'P-value': p_value
                    })
                    
                    fig = go.Figure()
                    fig.add_trace(go.Box(y=start_data, name='Start', boxpoints='all', jitter=0.3, pointpos=-1.8))
                    fig.add_trace(go.Box(y=end_data, name='End', boxpoints='all', jitter=0.3, pointpos=-1.8))
                    fig.update_layout(title=f"Distribution of {col} at Start vs End", yaxis_title=col)
                    st.plotly_chart(fig)
                    
                    st.write(f"### Key Changes in {col}")
                    st.write(f"- Mean change: {'increase' if mean_change > 0 else 'decrease'} of {abs(mean_change):.2f}%")
                    st.write(f"- Standard deviation change: {'increase' if std_change > 0 else 'decrease'} of {abs(std_change):.2f}%")
                    st.write(f"- Statistical significance: {'Significant' if p_value < 0.05 else 'Not significant'} (p-value: {p_value:.4f})")
                else:
                    st.warning(f"Insufficient data for analysis of {col}")
            else:
                st.warning(f"{col} not found in both datasets")

        if len(comparison_results) > 0:
            summary_df = pd.DataFrame(comparison_results)
            summary_df['Significant Change'] = summary_df['P-value'] < 0.05
            st.write("### Summary of Changes")
            st.dataframe(summary_df.style.format({
                'Mean Change (%)': '{:.2f}%',
                'Std Dev Change (%)': '{:.2f}%',
                'P-value': '{:.4f}'
            }))
        else:
            st.warning("No valid comparisons could be made between the datasets")
            summary_df = pd.DataFrame(columns=['Metric', 'Mean Change (%)', 'Std Dev Change (%)', 'P-value', 'Significant Change'])

    except Exception as e:
        st.error(f"An error occurred during the comparison analysis: {str(e)}")
        summary_df = pd.DataFrame(columns=['Metric', 'Mean Change (%)', 'Std Dev Change (%)', 'P-value', 'Significant Change'])

    return summary_df

def generate_business_goals(summary_df, top_kpis, target_column):
    goals = []
    
    try:
        significant_changes = summary_df[summary_df['Significant Change']]['Metric'].tolist()
        
        if len(top_kpis) > 0:
            goals.append(f"Optimize {top_kpis[0]} to drive improvements in {target_column}")
        
        if len(top_kpis) > 1:
            goals.append(f"Investigate the relationship between {top_kpis[1]} and {target_column} for potential synergies")
        
        if len(significant_changes) > 0:
            goals.append(f"Develop strategies to capitalize on the significant changes observed in {significant_changes[0]}")
        
        if len(top_kpis) > 2:
            goals.append(f"Implement a monitoring system for {top_kpis[2]} to track its impact on overall performance")
        
        goals.append("Conduct a comprehensive review of all KPIs to identify additional improvement opportunities")
        
    except Exception as e:
        st.error(f"Error generating business goals: {str(e)}")
        goals = [
            "Identify key performance indicators that drive business success",
            "Develop strategies to improve overall business performance",
            "Implement a robust system for tracking and analyzing business metrics",
            "Investigate areas of significant change in the business",
            "Conduct a comprehensive review to identify improvement opportunities"
        ]
    
    return goals

def generate_goal_recommendation(df_start, df_end, goal, top_kpis, target_column):
    try:
        if len(top_kpis) > 0:
            kpi = np.random.choice(top_kpis)
            change = (df_end[kpi].mean() - df_start[kpi].mean()) / df_start[kpi].mean() * 100
            return f"To achieve this goal, focus on {'improving' if change < 0 else 'maintaining the positive trend in'} {kpi}, which has shown a {abs(change):.2f}% {'decrease' if change < 0 else 'increase'}. This could potentially lead to significant improvements in {target_column}."
        else:
            return "To achieve this goal, focus on identifying and tracking key performance indicators that are most relevant to your business objectives."
    except Exception as e:
        st.error(f"Error generating goal recommendation: {str(e)}")
        return "To achieve this goal, conduct a thorough analysis of your business metrics and identify areas with the greatest potential for improvement."

def generate_action_plan(summary_df, top_kpis, target_column):
    action_plan = []
    try:
        significant_changes = summary_df[summary_df['Significant Change']]['Metric'].tolist()
        
        if len(significant_changes) > 0:
            action_plan.append(f"Conduct a deep-dive analysis into the factors contributing to changes in {', '.join(significant_changes[:2])}")
        
        if len(top_kpis) >= 2 and target_column:
            action_plan.append(f"Develop a predictive model to forecast {target_column} based on changes in {top_kpis[0]} and {top_kpis[1]}")
        
        action_plan.append("Implement a real-time monitoring dashboard for all identified KPIs, with alerts for significant deviations")
        action_plan.append("Organize cross-functional workshops to brainstorm strategies for capitalizing on positive trends and mitigating negative trends")
        action_plan.append("Establish a continuous improvement program that sets targets based on the observed changes and implements initiatives to achieve them")
        
    except Exception as e:
        st.error(f"Error generating action plan: {str(e)}")
        action_plan = [
            "Conduct a comprehensive analysis of key business metrics",
            "Develop a system for regular monitoring and reporting of critical KPIs",
            "Implement a process for identifying and addressing significant changes in business performance",
            "Establish cross-functional teams to develop and implement improvement strategies",
            "Create a culture of continuous improvement and data-driven decision making"
        ]
    
    return action_plan

def generate_focused_report(df_start, df_end, summary_df, goals, action_plan, top_kpis, target_column):
    report = "# Focused Business Analysis Report\n\n"

    try:
        report += "## 1. Key Performance Indicators (KPIs) Analysis\n\n"
        if isinstance(top_kpis, pd.Series) and len(top_kpis) > 0:
            for kpi, corr in top_kpis.items():
                report += f"- {kpi}: Correlation of {corr:.2f} with {target_column}\n"
        else:
            report += "No significant KPIs identified.\n"
        report += "\n"

        report += "## 2. Start-End Comparison Analysis\n\n"
        if not summary_df.empty:
            for _, row in summary_df.iterrows():
                report += f"- {row['Metric']}: {'Increase' if row['Mean Change (%)'] > 0 else 'Decrease'} of {abs(row['Mean Change (%)']):.2f}% "
                report += f"({'Statistically Significant' if row['Significant Change'] else 'Not Statistically Significant'})\n"
        else:
            report += "No valid comparisons could be made between the datasets.\n"
        report += "\n"

        report += "## 3. Business Goal Recommendations\n\n"
        for i, goal in enumerate(goals, 1):
            report += f"{i}. {goal}\n"
        report += "\n"

        report += "## 4. Strategic Action Plan\n\n"
        for i, action in enumerate(action_plan, 1):
            report += f"{i}. {action}\n"

    except Exception as e:
        st.error(f"Error generating focused report: {str(e)}")
        report += "An error occurred while generating the detailed report. Please review the analysis results in the main interface.\n"

    return report

def generate_insights(df1, additional_dfs=None):
    all_dfs = [df1] + (additional_dfs or [])
    insights = []
    
    # Overall data quality
    for i, df in enumerate(all_dfs, 1):
        missing_data = df.isnull().sum().sum()
        insights.append(f"Dataset {i} contains {missing_data} missing values. Data cleaning may be necessary.")
    
    # Key metrics
    common_numeric_cols = list(set.intersection(*[set(df.select_dtypes(include=[np.number]).columns) for df in all_dfs]))
    for col in common_numeric_cols[:3]:  # Limit to first 3 common numeric columns
        insights.append(f"For {col}:")
        for i, df in enumerate(all_dfs, 1):
            mean_val = df[col].mean()
            median_val = df[col].median()
            insights.append(f"  Dataset {i}: Average is {mean_val:.2f}, with a median of {median_val:.2f}.")
    
    # Correlation insights
    if len(common_numeric_cols) > 1:
        for i, df in enumerate(all_dfs, 1):
            corr_matrix = df[common_numeric_cols].corr()
            high_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().nlargest(3)
            insights.append(f"Top correlations in Dataset {i}:")
            for (col1, col2), corr_val in high_corr.items():
                insights.append(f"  There's a strong correlation ({corr_val:.2f}) between {col1} and {col2}.")
    
    # Time-based insights (if applicable)
    date_cols = list(set.intersection(*[set(df.select_dtypes(include=['datetime64']).columns) for df in all_dfs]))
    if len(date_cols) > 0:
        date_col = date_cols[0]
        for i, df in enumerate(all_dfs, 1):
            date_range = df[date_col].max() - df[date_col].min()
            insights.append(f"Dataset {i} spans a period of {date_range.days} days.")
    
    # Dataset comparison
    if len(all_dfs) > 1:
        for col in common_numeric_cols:
            base_mean = all_dfs[0][col].mean()
            for i, df in enumerate(all_dfs[1:], 2):
                diff = ((df[col].mean() - base_mean) / base_mean) * 100
                insights.append(f"The average {col} in Dataset {i} is {diff:.2f}% {'higher' if diff > 0 else 'lower'} than in Dataset 1.")
    
    return insights

def generate_comprehensive_report(df1, additional_dfs=None, insights=None):
    all_dfs = [df1] + (additional_dfs or [])
    report = "# Comprehensive Data Analysis Report\n\n"
    
    report += "## 1. Data Overview\n"
    for i, df in enumerate(all_dfs, 1):
        report += f"- Dataset {i} shape: {df.shape}\n"
        report += f"- Dataset {i} columns: {', '.join(df.columns)}\n"
    report += "\n"
    
    report += "## 2. Descriptive Statistics\n"
    for i, df in enumerate(all_dfs, 1):
        report += f"### Dataset {i}\n"
        report += df.describe().to_markdown()
        report += "\n\n"
    
    common_numeric_cols = list(set.intersection(*[set(df.select_dtypes(include=[np.number]).columns) for df in all_dfs]))
    if len(common_numeric_cols) > 1:
        report += "## 3. Correlation Analysis\n"
        report += f"Correlation heatmaps were generated for each of the {len(all_dfs)} datasets to visualize relationships between variables.\n\n"
        
        report += "## 4. Feature Importance\n"
        report += f"Feature importance analysis was conducted for each of the {len(all_dfs)} datasets to identify the most influential variables.\n\n"
    else:
        report += "## 3-4. Correlation Analysis and Feature Importance\n"
        report += "Insufficient common numeric columns for correlation analysis and feature importance.\n\n"
    
    date_cols = list(set.intersection(*[set(df.select_dtypes(include=['datetime64']).columns) for df in all_dfs]))
    if len(date_cols) > 0:
        report += "## 5. Time Series Analysis\n"
        report += f"Time series analysis was performed to understand underlying patterns in the data across all {len(all_dfs)} datasets.\n\n"
    else:
        report += "## 5. Time Series Analysis\n"
        report += "No common datetime columns found for time series analysis.\n\n"
    
    if len(common_numeric_cols) > 0:
        report += "## 6. Anomaly Detection\n"
        report += f"Anomalies were identified using the Interquartile Range (IQR) method for each of the {len(all_dfs)} datasets.\n\n"
        
        if len(common_numeric_cols) >= 2:
            report += "## 7. Clustering Analysis\n"
            report += f"K-means clustering was applied to group similar data points in each of the {len(all_dfs)} datasets.\n\n"
        else:
            report += "## 7. Clustering Analysis\n"
            report += "Insufficient common numeric columns for clustering analysis.\n\n"
    else:
        report += "## 6-7. Anomaly Detection and Clustering Analysis\n"
        report += "No common numeric columns found for anomaly detection and clustering analysis.\n\n"
    
    if len(all_dfs) > 1:
        report += "## 8. Dataset Comparison\n"
        report += f"A statistical comparison was made between all {len(all_dfs)} datasets using ANOVA.\n\n"
    
    if len(common_numeric_cols) > 1:
        report += "## 9. Predictive Modeling\n"
        report += f"A Random Forest model was trained to predict the target variable for each of the {len(all_dfs)} datasets.\n\n"
    else:
        report += "## 9. Predictive Modeling\n"
        report += "Insufficient common numeric columns for predictive modeling.\n\n"
    
    report += "## 10. Key Insights and Recommendations\n"
    if insights:
        for i, insight in enumerate(insights, 1):
            report += f"{i}. {insight}\n"
    else:
        report += "No specific insights were generated.\n"
    
    return report

def generate_contextual_explanation(df1, analysis_type, additional_dfs=None, **kwargs):
    all_dfs = [df1] + (additional_dfs or [])
    num_datasets = len(all_dfs)
    
    context = f"You are analyzing {num_datasets} datasets. "
    for i, df in enumerate(all_dfs):
        context += f"Dataset {i+1} has {df.shape[0]} rows and {df.shape[1]} columns. "
    context += f"The columns in the primary dataset are: {', '.join(df1.columns)}. "
    context += f"Here's a summary of the primary dataset:\n{df1.describe().to_string()}\n\n"
    context += f"You are providing a contextual explanation for the {analysis_type} analysis across all {num_datasets} datasets."

    if analysis_type == "correlation":
        numeric_corrs = []
        categorical_assocs = []
        for i, df in enumerate(all_dfs):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                high_corrs = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates().head(3)
                numeric_corrs.append((i+1, high_corrs))
            
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 1:
                cat_assocs = []
                for col1, col2 in itertools.combinations(categorical_cols, 2):
                    contingency_table = pd.crosstab(df[col1], df[col2])
                    chi2, p_value, _, _ = chi2_contingency(contingency_table)
                    cat_assocs.append((col1, col2, chi2, p_value))
                categorical_assocs.append((i+1, cat_assocs))
        
        prompt = f"{context}\n"
        if numeric_corrs:
            prompt += "The top 3 numeric correlations for each dataset are:\n"
            for dataset_num, high_corr in numeric_corrs:
                prompt += f"Dataset {dataset_num}:\n{high_corr.to_string()}\n"
        
        if categorical_assocs:
            prompt += "The chi-square test results for categorical variables are:\n"
            for dataset_num, cat_assocs in categorical_assocs:
                prompt += f"Dataset {dataset_num}:\n"
                for col1, col2, chi2, p_value in cat_assocs[:3]:  # Limit to top 3
                    prompt += f"- {col1} vs {col2}: Chi-square = {chi2:.2f}, p-value = {p_value:.4f}\n"
        
        prompt += "Explain the business implications of these correlations and associations across all datasets."
    
    elif analysis_type == "feature_importance":
        target = kwargs.get('target')
        importances_list = kwargs.get('importances_list', [])
        prompt = f"{context}\nThe target variable is {target}. The top 3 important features for each dataset are:\n"
        for i, importances in enumerate(importances_list):
            top_features = importances.head(3)
            prompt += f"Dataset {i+1}:\n{top_features.to_string()}\n"
        prompt += f"Explain how these features might influence {target} and what it means for business performance across all {num_datasets} datasets."
    
    elif analysis_type == "time_series":
        selected_col = kwargs.get('selected_col')
        prompt = f"{context}\nA time series analysis was performed comparing the datasets for the {selected_col} column. "
        for i, df in enumerate(all_dfs):
            prompt += f"\nDataset {i+1} summary:\n{df[selected_col].describe().to_string()}\n"
        prompt += f"Explain the business implications of the changes over time and what they might indicate about performance trends across all {num_datasets} datasets."
    
    elif analysis_type == "anomaly_detection":
        anomaly_col = kwargs.get('anomaly_col')
        prompt = f"{context}\nAnomaly detection was performed on the {anomaly_col} column for all {num_datasets} datasets. "
        for i, df in enumerate(all_dfs):
            Q1 = df[anomaly_col].quantile(0.25)
            Q3 = df[anomaly_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            anomalies = df[(df[anomaly_col] < lower_bound) | (df[anomaly_col] > upper_bound)]
            prompt += f"\nDataset {i+1} anomalies summary:\n{anomalies[anomaly_col].describe().to_string()}\n"
        prompt += f"Explain the potential business impact of these anomalies across all {num_datasets} datasets and what they might indicate."
    
    elif analysis_type == "clustering":
        cluster_cols = kwargs.get('cluster_cols')
        prompt = f"{context}\nClustering was performed using the columns: {', '.join(cluster_cols)}. "
        for i, df in enumerate(all_dfs):
            prompt += f"\nDataset {i+1} summary of clustering columns:\n{df[cluster_cols].describe().to_string()}\n"
        prompt += f"Explain how identifying clusters in these dimensions could provide insights for business strategy across all {num_datasets} datasets."
    
    elif analysis_type == "dataset_comparison":
        compare_col = kwargs.get('compare_col')
        p_value = kwargs.get('p_value')
        prompt = f"{context}\nAll {num_datasets} datasets were compared on the {compare_col} column. The statistical test yielded a p-value of {p_value:.4f}. "
        for i, df in enumerate(all_dfs):
            prompt += f"\nDataset {i+1} summary of {compare_col}:\n{df[compare_col].describe().to_string()}\n"
        prompt += f"Explain the business implications of this comparison and what it might mean for performance across different time periods or segments."
    
    elif analysis_type == "predictive_modeling":
        target_col = kwargs.get('target_col')
        mse_list = kwargs.get('mse_list', [])
        r2_list = kwargs.get('r2_list', [])
        prompt = f"{context}\nPredictive models were built for the {target_col} column across all {num_datasets} datasets. "
        for i, (mse, r2) in enumerate(zip(mse_list, r2_list)):
            prompt += f"\nDataset {i+1} model performance:\nMean Squared Error: {mse:.4f}\nR-squared score: {r2:.4f}\n"
            prompt += f"Dataset {i+1} summary of {target_col}:\n{all_dfs[i][target_col].describe().to_string()}\n"
        prompt += f"Explain what these results mean in terms of the models' performance and their potential business applications across all {num_datasets} datasets."
    
    else:
        prompt = f"{context}\nProvide a general explanation of how this analysis could be useful for understanding business performance across all {num_datasets} datasets."

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in data analysis and business intelligence. Provide specific insights based on the given data across multiple datasets."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred while generating the explanation: {str(e)}"

def parse_date(date_string):
    try:
        return pd.to_datetime(date_string).date()
    except ValueError:
        return None

def analyze_and_forecast(df, y_column, periods_to_forecast, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, comprehensive_report=None):
    # Prepare data for Prophet
    df_prophet = df.reset_index().rename(columns={'date': 'ds', y_column: 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    # Create and fit the model
    model = Prophet(yearly_seasonality=yearly_seasonality, 
                    weekly_seasonality=weekly_seasonality, 
                    daily_seasonality=daily_seasonality)
    model.fit(df_prophet)

    # Create future dataframe for forecasting
    future = model.make_future_dataframe(periods=periods_to_forecast)
    
    # Make predictions
    forecast = model.predict(future)

    # Calculate overall change
    overall_change = ((df[y_column].iloc[-1] - df[y_column].iloc[0]) / df[y_column].iloc[0]) * 100
    average_value = df[y_column].mean()

    # Create forecast plot
    fig_forecast = plot_plotly(model, forecast)
    fig_forecast.update_layout(title=f'Forecast of {y_column}')

    # Create components plot
    fig_components = plot_components_plotly(model, forecast)

    # Generate insights text
    insights_text = f"""
    Overall change: {overall_change:.2f}%
    Average {y_column}: {average_value:.2f}
    Forecast for next {periods_to_forecast} periods: 
    Start: {forecast['yhat'].iloc[-periods_to_forecast]:.2f}
    End: {forecast['yhat'].iloc[-1]:.2f}
    """

    # Get the top 3 dates with the highest positive and negative forecast changes
    forecast['change'] = forecast['yhat'].diff()
    top_increases = forecast.nlargest(3, 'change')
    top_decreases = forecast.nsmallest(3, 'change')

    changepoints_text = "Top 3 forecasted increases:\n"
    for _, row in top_increases.iterrows():
        changepoints_text += f"  {row['ds'].date()}: {row['change']:.2f}\n"
    changepoints_text += "\nTop 3 forecasted decreases:\n"
    for _, row in top_decreases.iterrows():
        changepoints_text += f"  {row['ds'].date()}: {row['change']:.2f}\n"


    if comprehensive_report:
        # Use the comprehensive report to enhance your analysis
        enhanced_insights = generate_enhanced_insights(insights_text, comprehensive_report)
        insights_text += f"\n\nEnhanced Insights:\n{enhanced_insights}"


    return fig_forecast, fig_components, insights_text, forecast['yhat'].tail(periods_to_forecast).tolist(), changepoints_text

def get_ai_analysis_recommendations(all_datasets):
    data_summary = ""
    for i, df in enumerate(all_datasets, 1):
        data_summary += f"Dataset {i}:\n"
        data_summary += f"Shape: {df.shape}\n"
        data_summary += f"Columns: {', '.join(df.columns)}\n"
        data_summary += f"Data types: {df.dtypes.to_string()}\n"
        data_summary += f"Sample data:\n{df.head().to_string()}\n\n"

    prompt = f"""Based on the following data summary for {len(all_datasets)} datasets:

{data_summary}

Recommend which of the following analyses would be most useful for business data analysis, and explain why:
1. Correlation Analysis
2. Feature Importance
3. Time Series Analysis
4. Anomaly Detection
5. Clustering Analysis
6. Dataset Comparison
7. Predictive Modeling

Select at least 5 analyses. For each recommended analysis, provide a brief explanation of why it would be valuable for this data. Format your response as a Python dictionary where keys are the analysis names and values are the explanations."""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in data analysis and business intelligence."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        recommendations = eval(response.choices[0].message.content.strip())
        return recommendations
    except Exception as e:
        st.error(f"An error occurred while getting AI recommendations: {str(e)}")
        return {}

def upload_comprehensive_report():
    uploaded_file = st.file_uploader("Upload a previously generated comprehensive report", type="md")
    if uploaded_file is not None:
        report_content = uploaded_file.read().decode()
        st.session_state.comprehensive_report = report_content
        st.success("Comprehensive report uploaded successfully!")
        return True
    return False


def analyze_top_performers(df, category_column, metric_column, top_n):
    """Analyze top performers based on the selected category and metric."""
    if metric_column in df.columns and pd.api.types.is_numeric_dtype(df[metric_column]):
        return df.sort_values(by=metric_column, ascending=False).groupby(category_column).first().reset_index().head(top_n)
    else:
        # If metric_column is not in df.columns or is not numeric, use the count of each category
        return df[category_column].value_counts().reset_index(name='count').head(top_n).rename(columns={'index': category_column})








####################################################################################################

# Streamlit UI
st.title("FOMO.AI Data Analysis and Visualization Tool")

init_db()

# Initialize session state variables
if 'parsed_goal_sets' not in st.session_state:
    st.session_state.parsed_goal_sets = []
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = {}
if 'df' not in st.session_state:
    st.session_state.df = None
if 'step' not in st.session_state:
    st.session_state.step = 'upload'
if 'first_dataset_date' not in st.session_state:
    st.session_state.first_dataset_date = None
if 'comprehensive_report' not in st.session_state:
    st.session_state.comprehensive_report = None

# Step 1: Data Upload
if st.session_state.step == 'upload':
    st.write("Welcome! Please upload your data to get started.")
    uploaded_file = st.file_uploader("Upload your data file (CSV)", type="csv")
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.write("Data preview:", st.session_state.df.head())
        
        # Date input for the first dataset
        if st.session_state.first_dataset_date is None:
            date_input_1 = st.text_input("Enter the date for this dataset (YYYY-MM-DD):")
            if date_input_1:
                parsed_date_1 = parse_date(date_input_1)
                if parsed_date_1:
                    st.session_state.df['date'] = parsed_date_1
                    st.session_state.first_dataset_date = parsed_date_1
                    st.success(f"Date set to {parsed_date_1}")
                    st.session_state.step = 'menu'
                    st.rerun()
                else:
                    st.error("Invalid date format. Please use YYYY-MM-DD.")
        else:
            st.session_state.df['date'] = st.session_state.first_dataset_date
            st.success(f"Date set to {st.session_state.first_dataset_date}")
            st.session_state.step = 'menu'
            st.rerun()
    else:
        st.warning("Please enter a valid date before proceeding.")

# Step 2: Main Menu
elif st.session_state.step == 'menu':
    st.write("What would you like to do?")
    option = st.radio("Choose an option:", [
        "Top Performers Analysis",
        "Trend Prediction based on a second data set",
        "Generate Goals",
        "Comprehensive Data Analysis"  # New option
    ])

    if st.button("Get Started"):
        st.session_state.step = option
        st.rerun()

# Step 3: Top Performers Analysis
elif st.session_state.step == "Top Performers Analysis":
    st.subheader("Top Performers Analysis")
    
    #if 'comprehensive_report' not in st.session_state:
    st.session_state.comprehensive_report = None

    report_option = st.radio("Comprehensive Report Options:", 
                            ["No comprehensive report", 
                            "Use existing comprehensive report", 
                            "Upload a new comprehensive report"])

    if report_option == "Use existing comprehensive report":
        if st.session_state.comprehensive_report:
            use_comprehensive_report = st.checkbox("Incorporate insights from the existing comprehensive analysis report")
            if use_comprehensive_report:
                st.info("The AI will consider the comprehensive analysis report in its analysis.")
        else:
            st.warning("No existing comprehensive report found. Please upload a report or complete the Comprehensive Data Analysis first.")

    elif report_option == "Upload a new comprehensive report":
        if upload_comprehensive_report():
            use_comprehensive_report = st.checkbox("Incorporate insights from the uploaded comprehensive analysis report")
            if use_comprehensive_report:
                st.info("The AI will consider the uploaded comprehensive analysis report in its analysis.")
        else:
            st.warning("No report uploaded. The analysis will proceed without a comprehensive report.")

    else:  # "No comprehensive report"
        use_comprehensive_report = False
        st.info("The analysis will proceed without using a comprehensive report.")
        # Data Preprocessing Option
        st.write("### Data Preprocessing")
        if st.checkbox("Perform additional calculations on data"):
            calculation_prompt = st.text_area("Enter your calculation or new feature creation prompt:", 
                                            "Create a new column that calculates the percentage change of a column compared to its mean.")
            
        if st.button("Perform Calculation"):
            with st.spinner("Generating and executing calculation code..."):
                calculation_code = perform_calculations(st.session_state.df, calculation_prompt)
                st.code(calculation_code, language="python")
                
                code_block = calculation_code.split("```python")[1].split("```")[0].strip()
                
                updated_df = execute_calculation_code(st.session_state.df, code_block)
                if updated_df is not None:
                    st.session_state.df = updated_df
                    st.write("Updated Data Preview:", st.session_state.df.head())

    # Display column names in a more organized manner
    st.write("### Available Columns")
    col1, col2, col3 = st.columns(3)
    for i, column in enumerate(st.session_state.df.columns):
        if i % 3 == 0:
            col1.write(f"- {column}")
        elif i % 3 == 1:
            col2.write(f"- {column}")
        else:
            col3.write(f"- {column}")
    
    category_column = st.selectbox("Select the category column", st.session_state.df.columns, key="category_column")
    # Filter out 'top queries' from the options
    metric_options = [col for col in st.session_state.df.columns if col != 'top queries']

    # If 'top queries' was the only column, provide a default numeric column
    if not metric_options:
        st.error("No suitable metric columns found. Using row count as the metric.")
        st.session_state.df['row_count'] = 1
        metric_options = ['row_count']

    metric_column = st.selectbox("Select the metric column for ranking", metric_options, key="metric_column")
    top_n = st.slider("Number of top performers to show", 1, 20, 5)


    if st.button("Perform Top Performers Analysis"):
    # Perform initial top performers analysis
        top_performers = analyze_top_performers(st.session_state.df, category_column, metric_column, top_n)
        
        # Generate initial report, recommendations, and next steps
        report, summary_stats, top_performers_df, distribution_stats, recommendations, next_steps = generate_top_performers_report(
            st.session_state.df, top_performers, category_column, metric_column, top_n, 
            comprehensive_report=st.session_state.comprehensive_report if use_comprehensive_report else None
        )
        
        st.markdown(report)
        
        st.write("## Top Performers Table")
        st.dataframe(top_performers)
        
        st.markdown(summary_stats)
        
        st.write("## Key Metrics")
        if metric_column in top_performers.columns and pd.api.types.is_numeric_dtype(top_performers[metric_column]):
            st.write(f"- Highest {metric_column}: {top_performers[metric_column].max():.2f}")
            st.write(f"- Average {metric_column}: {top_performers[metric_column].mean():.2f}")
            st.write(f"- Lowest {metric_column}: {top_performers[metric_column].min():.2f}")
        else:
            st.write(f"- Top {category_column}: {top_performers[category_column].iloc[0]}")
            st.write(f"- Number of unique {category_column} values: {top_performers[category_column].nunique()}")
            st.write(f"- Sample {category_column} values: {', '.join(top_performers[category_column].sample(min(3, len(top_performers))).tolist())}")

        st.write("## Distribution Analysis")
        if metric_column in st.session_state.df.columns and pd.api.types.is_numeric_dtype(st.session_state.df[metric_column]):
            st.write(f"The distribution of the {metric_column} across the entire dataset is as follows:")
            st.dataframe(distribution_stats)
        else:
            st.write(f"The frequency distribution of the {category_column} across the entire dataset is as follows:")
            st.dataframe(st.session_state.df[category_column].value_counts().head(10))  # Show top 10 most frequent values

        # Visualize top performers
        st.write("## Top Performers Visualization")
        fig = visualize_top_performers(top_performers, category_column, metric_column)
        st.plotly_chart(fig)

    # ... rest of your code for displaying recommendations and next steps
        # Display and act on recommendations
        st.write("### Insights and Action Plan")
        for i, recommendation in enumerate(recommendations, 1):
            st.write(f"**Insight {i}:** {recommendation}")
            
            if i == 1:  # In-depth analysis of top performers
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Top Performer Statistics:**")
                    st.dataframe(top_performers.describe())
                with col2:
                    st.write("**Key Metrics:**")
                    st.write(f"- Average {metric_column}: {top_performers[metric_column].mean():.2f}")
                    st.write(f"- Highest {metric_column}: {top_performers[metric_column].max():.2f}")
                    st.write(f"- Lowest {metric_column}: {top_performers[metric_column].min():.2f}")
            
            elif i == 2:  # Applying practices across categories
                st.write("**Best Practices for Improvement:**")
                practices = [
                    f"Regular performance reviews focusing on {metric_column}",
                    f"Training programs targeting improvement in {metric_column}",
                    f"Sharing of best practices from top {category_column}s",
                    "Mentorship programs pairing top performers with others"
                ]
                for practice in practices:
                    st.write(f"- {practice}")
            
            elif i == 3:  # Establishing new benchmarks
                new_benchmark = top_performers[metric_column].mean()
                st.write("**New Performance Benchmarks:**")
                st.write(f"- Proposed new benchmark for {metric_column}: {new_benchmark:.2f}")
                st.write(f"- Improvement target: {(new_benchmark / st.session_state.df[metric_column].mean() - 1) * 100:.2f}% increase")
            
            elif i == 4:  # Strategies for lower-ranking categories
                lower_performers = st.session_state.df.sort_values(by=metric_column).head(top_n)
                st.write("**Focus Areas for Improvement:**")
                for _, row in lower_performers.iterrows():
                    st.write(f"- {row[category_column]}: Current {metric_column} = {row[metric_column]:.2f}")
                st.write("**Recommended Actions:** Targeted training, resource allocation, and mentorship programs.")

            st.write("---")

        # Display and act on next steps
        st.write("### Implementation Strategy")
        for i, step in enumerate(next_steps, 1):
            st.write(f"**Strategy {i}:** {step}")
            
            if i == 1:  # Insights from top performers
                st.write("**Key Success Factors:**")
                success_factors = [
                    "Consistent focus on key performance indicators",
                    "Regular skill development and training",
                    "Effective time management and prioritization",
                    "Strong communication and collaboration within teams"
                ]
                for factor in success_factors:
                    st.write(f"- {factor}")
            
            elif i == 2:  # Historical data analysis
                st.write("**Performance Trend Analysis:**")
                historical_trend = px.line(x=range(12), y=[st.session_state.df[metric_column].mean() * (1 + 0.05 * i) for i in range(12)],
                                           labels={'x': 'Months', 'y': metric_column})
                st.plotly_chart(historical_trend)
            
            elif i == 3 or i == 4:  # Advanced analytics and correlations
                st.write("**Correlation Analysis:**")
                numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
                correlations = st.session_state.df[numeric_cols].corr()[metric_column].sort_values(ascending=False)
                st.dataframe(correlations.head())

                st.write("**Performance Clusters:**")
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(st.session_state.df[[metric_column]])
                kmeans = KMeans(n_clusters=3, random_state=42)
                st.session_state.df['cluster'] = kmeans.fit_predict(scaled_data)
                cluster_means = st.session_state.df.groupby('cluster')[metric_column].mean().sort_values(ascending=False)
                st.dataframe(cluster_means)
            
            elif i == 5:  # Implementation plan
                st.write("**Implementation Team and Plan:**")
                task_force = [
                    "Leadership representative",
                    "Top performer representative",
                    "Data analyst",
                    "HR representative",
                    "Training and development specialist"
                ]
                for member in task_force:
                    st.write(f"- {member}")
                st.write("**Action Items:**")
                st.write("1. Weekly progress review meetings")
                st.write("2. Monthly data analysis and strategy adjustment")
                st.write("3. Quarterly performance reviews and goal setting")

            st.write("---")

        # Option to download the comprehensive report
        comprehensive_report = f"""
        # Comprehensive Top Performers Analysis Report

        {report}
        {summary_stats}

        ## Insights and Action Plan

        {chr(10).join([f"{i+1}. {rec}" for i, rec in enumerate(recommendations)])}

        ## Implementation Strategy

        {chr(10).join([f"{i+1}. {step}" for i, step in enumerate(next_steps)])}

        """
        report, summary_stats, top_performers_df, distribution_stats, recommendations, next_steps = generate_top_performers_report(
            st.session_state.df, top_performers, category_column, metric_column, top_n, 
            comprehensive_report=st.session_state.comprehensive_report if use_comprehensive_report else None
        )

        st.download_button(
            label="Download Comprehensive Top Performers Analysis Report",
            data=comprehensive_report,
            file_name="comprehensive_top_performers_analysis_report.md",
            mime="text/markdown"
        )

    if st.button("Return to Menu"):
        st.session_state.step = 'menu'
        st.experimental_rerun()

# Step 4: Trend Prediction based on two datasets
elif st.session_state.step == "Trend Prediction based on a second data set":
    st.subheader("Dataset Comparison and Trend Analysis")
    if 'additional_datasets' not in st.session_state:
        st.session_state.additional_datasets = []
    
    def upload_dataset(index):
        uploaded_file = st.file_uploader(f"Upload CSV file for dataset {index}", type="csv", key=f"upload_{index}")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(f"Dataset {index} Preview:", df.head())
            date_input = st.text_input(f"Enter the date for dataset {index} (YYYY-MM-DD):", key=f"date_{index}")
            if date_input:
                parsed_date = parse_date(date_input)
                if parsed_date:
                    df['date'] = parsed_date
                    st.success(f"Date for dataset {index} set to {parsed_date}")
                    return df
                else:
                    st.error(f"Invalid date format for dataset {index}. Please use YYYY-MM-DD.")
            else:
                st.warning(f"Please enter a valid date for dataset {index} before proceeding.")
        return None
    

        # Add this block at the beginning of each main section
    if 'comprehensive_report' not in st.session_state:
        st.session_state.comprehensive_report = None

    report_option = st.radio("Comprehensive Report Options:", 
                            ["No comprehensive report", 
                            "Use existing comprehensive report", 
                            "Upload a new comprehensive report"])

    if report_option == "Use existing comprehensive report":
        if st.session_state.comprehensive_report:
            use_comprehensive_report = st.checkbox("Incorporate insights from the existing comprehensive analysis report")
            if use_comprehensive_report:
                st.info("The AI will consider the comprehensive analysis report in its analysis.")
        else:
            st.warning("No existing comprehensive report found. Please upload a report or complete the Comprehensive Data Analysis first.")

    elif report_option == "Upload a new comprehensive report":
        if upload_comprehensive_report():
            use_comprehensive_report = st.checkbox("Incorporate insights from the uploaded comprehensive analysis report")
            if use_comprehensive_report:
                st.info("The AI will consider the uploaded comprehensive analysis report in its analysis.")
        else:
            st.warning("No report uploaded. The analysis will proceed without a comprehensive report.")

    else:  # "No comprehensive report"
        use_comprehensive_report = False
        st.info("The analysis will proceed without using a comprehensive report.")
    
    
    df1 = st.session_state.df
    st.write("Dataset 1 (Start Period) Preview:", df1.head())
    st.write(f"Dataset 1 Date: {st.session_state.first_dataset_date}")

    num_additional_datasets = st.slider("Number of additional datasets to upload", 1, 5, 1)

# File uploader for multiple files
    uploaded_files = st.file_uploader("Upload additional CSV files", type="csv", accept_multiple_files=True)

    if uploaded_files:
        st.session_state.additional_datasets = []
        
        # Check if the number of uploaded files matches the slider value
        if len(uploaded_files) != num_additional_datasets:
            st.warning(f"Please upload exactly {num_additional_datasets} file(s).")
        else:
            # Process each uploaded file
            for i, file in enumerate(uploaded_files, start=2):
                df = pd.read_csv(file)
                st.write(f"Dataset {i} Preview:", df.head())
                
                # Date input for each dataset
                date_input = st.text_input(f"Enter the date for dataset {i} (YYYY-MM-DD):", key=f"date_{i}")
                if date_input:
                    parsed_date = parse_date(date_input)
                    if parsed_date:
                        df['date'] = parsed_date
                        st.success(f"Date for dataset {i} set to {parsed_date}")
                        st.session_state.additional_datasets.append(df)
                    else:
                        st.error(f"Invalid date format for dataset {i}. Please use YYYY-MM-DD.")
                else:
                    st.warning(f"Please enter a valid date for dataset {i} before proceeding.")
            
            # Check if all datasets have been processed
            if len(st.session_state.additional_datasets) == num_additional_datasets:
                st.success(f"All {num_additional_datasets} additional dataset(s) uploaded and dated successfully!")
            else:
                st.warning("Please ensure all datasets have valid dates.")
    else:
        st.info(f"Please upload {num_additional_datasets} additional dataset(s).")

    if len(st.session_state.additional_datasets) > 0:
        all_datasets = [df1] + st.session_state.additional_datasets
        
        # Combine all datasets into a single DataFrame
        combined_df = pd.concat([df.assign(dataset_index=i) for i, df in enumerate(all_datasets)], ignore_index=True)
        
        st.write("### Data Preprocessing")
        if st.checkbox("Perform additional calculations on datasets"):
            calculation_prompt = st.text_area("Enter your calculation or new feature creation prompt:", 
                                              "Create a new column that calculates the percentage change of a column compared to its mean.")
            if st.button("Perform Calculation"):
                with st.spinner("Generating and executing calculation code..."):
                    calculation_code = perform_calculations(combined_df, calculation_prompt)
                    st.code(calculation_code, language="python")
                    code_block = calculation_code.split("```python")[1].split("```")[0].strip()
                    combined_df = execute_calculation_code(combined_df, code_block)
                    if combined_df is not None:
                        st.write("Updated Data Preview:", combined_df.head())
                        st.success("Preprocessing complete. New columns are now available for selection.")

        st.write("### Column Selection for Analysis")
        numeric_columns = combined_df.select_dtypes(include=[np.number]).columns.drop('dataset_index')
        if len(numeric_columns) > 0:
            st.write("Available numeric columns:")
            for col in numeric_columns:
                st.write(f"- {col}")
            y_column = st.selectbox("Select the metric column for analysis and forecasting", numeric_columns, key="y_axis_comparison")
            periods_to_forecast = st.slider("Number of periods to forecast", 1, 100, 30)
            
            st.write("### Prophet Model Parameters")
            yearly_seasonality = st.checkbox("Include yearly seasonality", value=True)
            weekly_seasonality = st.checkbox("Include weekly seasonality", value=True)
            daily_seasonality = st.checkbox("Include daily seasonality", value=False)
            
            if st.button("Analyze Trends and Forecast"):
                try:
                    # Prepare data for Prophet
                    prophet_df = combined_df[['date', y_column]].rename(columns={'date': 'ds', y_column: 'y'})
                    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
                    
                    # Perform analysis and forecasting
                    fig_forecast, fig_components, insights_text, forecast, changepoints_text = analyze_and_forecast(
                        prophet_df, 'y', periods_to_forecast,
                        yearly_seasonality, weekly_seasonality, daily_seasonality
                    )
                    
                    st.write("### Forecast Plot")
                    st.plotly_chart(fig_forecast)
                    st.write("### Forecast Components")
                    st.plotly_chart(fig_components)
                    st.write("### Numerical Insights")
                    st.write(insights_text)
                    st.write("### Forecast")
                    st.write(f"Forecast for next {periods_to_forecast} periods:", forecast)
                    st.write("### Changepoints Analysis")
                    st.text(changepoints_text)
                    st.write("### Qualitative Analysis")
                    qualitative_explanation = generate_qualitative_explanation(prophet_df, 'y', insights_text, forecast)
                    st.write(qualitative_explanation)
                    
                    # Dataset Comparison
                    st.write("### Dataset Comparison")
                    fig = px.box(combined_df, x='dataset_index', y=y_column, title=f"Comparison of {y_column} Across Datasets")
                    st.plotly_chart(fig)
                    
                    st.write("Statistical Test (ANOVA)")
                    datasets_for_anova = [df[y_column].dropna() for df in all_datasets]
                    f_statistic, p_value = f_oneway(*datasets_for_anova)
                    st.write(f"F-statistic: {f_statistic:.4f}")
                    st.write(f"p-value: {p_value:.4f}")
                    if p_value < 0.05:
                        st.write("There are significant differences between the datasets.")
                    else:
                        st.write("There are no significant differences between the datasets.")
                    
                    comparison_explanation = generate_contextual_explanation(combined_df, "dataset_comparison", compare_col=y_column, p_value=p_value)
                    st.write("Contextual Explanation:", comparison_explanation)
                    
                    # Generate and offer downloadable report
                    report = f"""
                    # Trend Analysis and Forecast Report
                    ## Data Points Used
                    - Total data points: {len(combined_df)}
                    ## Numerical Insights
                    {insights_text}
                    ## Forecast
                    Forecast for next {periods_to_forecast} periods: {forecast}
                    ## Changepoints Analysis
                    {changepoints_text}
                    ## Qualitative Analysis
                    {qualitative_explanation}
                    ## Dataset Comparison
                    {comparison_explanation}
                    """
                    st.download_button(
                        label="Download Comprehensive Report",
                        data=report,
                        file_name="trend_analysis_report.md",
                        mime="text/markdown"
                    )

                    fig_forecast, fig_components, insights_text, forecast, changepoints_text = analyze_and_forecast(
                    prophet_df, 'y', periods_to_forecast,
                    yearly_seasonality, weekly_seasonality, daily_seasonality,
                    comprehensive_report=st.session_state.comprehensive_report if use_comprehensive_report else None
                )
                    
                    qualitative_explanation = generate_qualitative_explanation(
                    prophet_df, 'y', insights_text, forecast,
                    comprehensive_report=st.session_state.comprehensive_report if use_comprehensive_report else None
                     )

                except Exception as e:
                    st.error(f"An error occurred during the analysis: {str(e)}")
                    st.write("Please check your data and try again.")
        else:
            st.error("No numeric columns found in the datasets.")
            st.write("Please ensure all datasets have at least one common numeric column.")
    else:
        st.warning("Please upload at least one additional dataset to proceed with the comparison and analysis.")

    if st.button("Return to Menu"):
        st.session_state.step = 'menu'
        st.experimental_rerun()

# Step 5: Generate Goals
elif st.session_state.step == "Generate Goals":
    st.subheader("Goal Generation and Analysis")

    if 'comprehensive_report' not in st.session_state:
        st.session_state.comprehensive_report = None

    report_option = st.radio("Comprehensive Report Options:", 
                            ["No comprehensive report", 
                            "Use existing comprehensive report", 
                            "Upload a new comprehensive report"])

    if report_option == "Use existing comprehensive report":
        if st.session_state.comprehensive_report:
            use_comprehensive_report = st.checkbox("Incorporate insights from the existing comprehensive analysis report")
            if use_comprehensive_report:
                st.info("The AI will consider the comprehensive analysis report in its analysis.")
        else:
            st.warning("No existing comprehensive report found. Please upload a report or complete the Comprehensive Data Analysis first.")

    elif report_option == "Upload a new comprehensive report":
        if upload_comprehensive_report():
            use_comprehensive_report = st.checkbox("Incorporate insights from the uploaded comprehensive analysis report")
            if use_comprehensive_report:
                st.info("The AI will consider the uploaded comprehensive analysis report in its analysis.")
        else:
            st.warning("No report uploaded. The analysis will proceed without a comprehensive report.")

    else:  # "No comprehensive report"
        use_comprehensive_report = False
        st.info("The analysis will proceed without using a comprehensive report.")

    # Buttons for different goal-related actions
    goal_action = st.radio("Choose an action:", [
        "Perform additional calculations on data Set",
        "Look at Previously Generated Goals",
        "Generate New Goals"
    ])

    if goal_action == "Perform additional calculations on data Set":
        st.subheader("Data Preprocessing")
        calculation_prompt = st.text_area("Enter your calculation or new feature creation prompt:", 
                                          "Create a new column that calculates the percentage change of column X compared to its mean.")
        
        if st.button("Perform Calculation"):
            with st.spinner("Generating and executing calculation code..."):
                calculation_code = perform_calculations(st.session_state.df, calculation_prompt)
                st.code(calculation_code, language="python")
                
                code_block = calculation_code.split("```python")[1].split("```")[0].strip()
                
                updated_df = execute_calculation_code(st.session_state.df, code_block)
                if updated_df is not None:
                    st.session_state.df = updated_df
                    st.write("Updated Data Preview:", st.session_state.df.head())

        if st.button("Generate Goals with New Features"):
            num_goals = st.slider("Number of goals to generate", 2, 10, 5)
            with st.spinner("Generating goals based on original and new features..."):
                new_goals = generate_goals_with_new_features(st.session_state.df, num_goals, 
                                                            comprehensive_report=st.session_state.comprehensive_report if use_comprehensive_report else None)
            st.session_state.parsed_goal_sets = [{"perspective": "Goals with New Features", "goals": new_goals}]
            save_goals_to_db("Goals with New Features", str(new_goals))
            st.success("New goals generated incorporating new features!")

    elif goal_action == "Look at Previously Generated Goals":
        st.subheader("Previously Generated Goals")
        past_goals = get_past_goals()
        if past_goals:
            past_goal_options = [f"{goal[1]} - {goal[3]}" for goal in past_goals]
            selected_past_goal = st.selectbox("Select a previously generated goal set:", 
                                              past_goal_options)
            if st.button("Use Selected Goal Set"):
                selected_goal = next(goal for goal in past_goals if f"{goal[1]} - {goal[3]}" == selected_past_goal)
                try:
                    parsed_goals = ast.literal_eval(selected_goal[2])
                    st.session_state.parsed_goal_sets = [{"perspective": selected_goal[1], "goals": parsed_goals}]
                    st.success("Past goal set loaded successfully!")
                except (ValueError, SyntaxError) as e:
                    st.error(f"Error parsing saved goals: {e}")
                    st.write("Raw saved goals:", selected_goal[2])

    elif goal_action == "Generate New Goals":
        st.subheader("Generate New Goals")
        goal_generation_method = st.radio("Choose goal generation method:", ("Random Generation", "User-Prompted Generation"))

        if goal_generation_method == "Random Generation":
            num_sets = st.slider("Number of goal sets", 2, 5, 3)
            goals_per_set = st.slider("Goals per set", 2, 5, 3)

            if st.button("Generate Goal Sets"):
                with st.spinner("Generating goal sets..."):
                    goal_sets = generate_goal_sets(st.session_state.df, num_sets, goals_per_set)
                    st.session_state.parsed_goal_sets = parse_goal_sets(goal_sets)
                    for goal_set in st.session_state.parsed_goal_sets:
                        save_goals_to_db(goal_set['perspective'], str(goal_set['goals']))
                    st.success("New goal sets generated and saved!")

        else:  # User-Prompted Generation
            user_prompt = st.text_area("Enter your prompt for goal generation:", "Analyze the data to identify trends and patterns that could inform business decisions.")
            num_goals = st.slider("Number of goals to generate", 2, 10, 5)

            if st.button("Generate Goals from Prompt"):
                with st.spinner("Generating goals based on your prompt..."):
                    goals = generate_goals_from_prompt(st.session_state.df, user_prompt, num_goals, 
                                                    comprehensive_report=st.session_state.comprehensive_report if use_comprehensive_report else None)
                    st.session_state.parsed_goal_sets = [{"perspective": "User-Prompted Goals", "goals": goals}]
                    save_goals_to_db("User-Prompted Goals", str(goals))
                    st.success("New goals generated from prompt and saved!")

    # Display generated or selected goals
    if st.session_state.parsed_goal_sets:
        st.write("Current Goals:")
        for i, goal_set in enumerate(st.session_state.parsed_goal_sets, 1):
            if not goal_set['perspective'].startswith("Follow-up Goals for:"):
                st.write(f"Set {i}: {goal_set['perspective']}")
                for j, goal in enumerate(goal_set['goals'], 1):
                    st.write(f"   Goal {j}: {goal}")
                st.write("---")

        # User selects a goal set
        initial_goal_sets = [gs for gs in st.session_state.parsed_goal_sets if not gs['perspective'].startswith("Follow-up Goals for:")]
        selected_set_index = st.selectbox("Select a goal set to focus on", 
                                          range(len(initial_goal_sets)), 
                                          format_func=lambda i: f"Set {i+1}: {initial_goal_sets[i]['perspective']}")

        selected_set = initial_goal_sets[selected_set_index]
        selected_goals = st.multiselect("Select goals to visualize", 
                                        selected_set['goals'], 
                                        default=[selected_set['goals'][0]])

        # Generate recommendations for selected goals
        for goal in selected_goals:
            if goal not in st.session_state.recommendations:
                graph_type, justification = recommend_graph_for_goal(st.session_state.df, goal)
                st.session_state.recommendations[goal] = (graph_type, justification)

        # Visualization section
        for i, selected_goal in enumerate(selected_goals):
            st.write(f"{selected_goal}")
            recommended_graph_type, justification = st.session_state.recommendations.get(selected_goal, ("bar", "No recommendation provided"))
            st.write(f"Recommended Graph Type: {recommended_graph_type}")
            st.write(f"Justification: {justification}")

            graph_types = ["line", "bar", "scatter", "box", "violin", "histogram", "heatmap"]
            selected_graph = st.selectbox(f"Select the type of graph for '{selected_goal}'", graph_types, index=graph_types.index(recommended_graph_type) if recommended_graph_type in graph_types else 0, key=f"graph_type_{i}")

            fig = create_flexible_graph(st.session_state.df, selected_graph, selected_goal, f"goal_{i}")
            if fig:
                st.plotly_chart(fig)

            st.write("---")

        # Goal Interaction Analysis
        if len(selected_goals) >= 2:
            st.subheader("Goal Interaction Analysis")
            goal1 = st.selectbox("Select first goal for interaction analysis", selected_goals, key="interaction_goal1")
            goal2 = st.selectbox("Select second goal for interaction analysis", selected_goals, key="interaction_goal2")
            
            if st.button("Analyze Goal Interaction"):
                interaction_fig, interaction_explanation = analyze_goal_interaction(st.session_state.df, goal1, goal2)
                if interaction_fig:
                    st.plotly_chart(interaction_fig)
                    st.markdown(interaction_explanation)
                else:
                    st.write(interaction_explanation)

        # Tree of Thought follow-up goals
        st.subheader("Tree of Thought Follow-up Goals")
        
        # Get the current goals
        current_goals = [goal for goal_set in st.session_state.parsed_goal_sets for goal in goal_set['goals']]

        if current_goals:
            goal_for_tot = st.selectbox("Select a goal to generate follow-up goals", current_goals, key="tot_goal_selection")

            if st.button("Generate Follow-up Goals"):
                with st.spinner("Generating follow-up goals..."):
                    followup_goals = generate_tot_followup_goals(st.session_state.df, goal_for_tot)
                    if followup_goals:
                        # Add the follow-up goals to parsed_goal_sets and session state
                        tot_goal_set = {"perspective": f"Follow-up Goals for: {goal_for_tot}", "goals": followup_goals}
                        st.session_state.parsed_goal_sets.append(tot_goal_set)
                        save_goals_to_db(tot_goal_set['perspective'], str(followup_goals))
                        
                        # Initialize followup_goals in session state if it doesn't exist
                        if 'followup_goals' not in st.session_state:
                            st.session_state.followup_goals = []
                        
                        # Add new goals to the session state with unique keys
                        for goal in followup_goals:
                            st.session_state.followup_goals.append({"goal": goal, "key": str(uuid.uuid4())})
                        
                        st.success("Follow-up goals generated successfully!")

            # Display and interact with follow-up goals
            if 'followup_goals' in st.session_state and st.session_state.followup_goals:
                st.write("Generated Follow-up Goals:")
                for i, goal_data in enumerate(st.session_state.followup_goals, 1):
                    st.markdown(f"{i}. {goal_data['goal']}")
                    
                    # Create a unique key for each selectbox
                    selected_graph_type = st.selectbox(
                        f"Select graph type for '{goal_data['goal']}'",
                        ["line", "bar", "scatter", "box", "violin", "histogram", "heatmap"],
                        key=f"graph_type_{goal_data['key']}"
                    )
                    
                    if st.button(f"Visualize Goal {i}", key=f"visualize_{goal_data['key']}"):
                        st.write(f"Analysis for follow-up goal: {goal_data['goal']}")
                        graph_type, justification = recommend_graph_for_goal(st.session_state.df, goal_data['goal'])
                        st.write(f"Recommended Graph Type: {graph_type}")
                        st.write(f"Justification: {justification}")

                        fig = create_flexible_graph(st.session_state.df, selected_graph_type, goal_data['goal'], f"followup_goal_{goal_data['key']}")
                        if fig:
                            st.plotly_chart(fig)

                        # Generate and display explanation
                        explanation = generate_graph_explanation(st.session_state.df, st.session_state.df.columns[0], [st.session_state.df.columns[1]], selected_graph_type, goal_data['goal'])
                        st.write("Graph Explanation:")
                        st.write(explanation)

                    st.write("---")
        else:
            st.write("No goals available. Please generate some goals first.")

        # Report generation
        st.subheader("Report Generation")
        goal_for_report = st.selectbox("Select a goal to focus on for the report", selected_goals, key="report_goal_selection")

        if st.button("Generate Report"):
            report = generate_report(goal_for_report, st.session_state.recommendations[goal_for_report], st.session_state.df)
            st.markdown(report)

            # Option to download the report
            st.download_button(
                label="Download Report",
                data=report,
                file_name="data_analysis_report.md",
                mime="text/markdown"
            )

    if st.button("Return to Menu"):
        st.session_state.step = 'menu'
        st.experimental_rerun()
            
# Step 5: Comprehensive Data Analysis
elif st.session_state.step == "Comprehensive Data Analysis":

    st.subheader("Comprehensive Data Analysis")
    if 'additional_datasets' not in st.session_state:
        st.session_state.additional_datasets = []

    def upload_dataset(index):
        uploaded_file = st.file_uploader(f"Upload CSV file for dataset {index}", type="csv", key=f"upload_comp_{index}")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(f"Dataset {index} Preview:", df.head())
            date_input = st.text_input(f"Enter the date for dataset {index} (YYYY-MM-DD):", key=f"date_comp_{index}")
            if date_input:
                parsed_date = parse_date(date_input)
                if parsed_date:
                    df['date'] = parsed_date
                    st.success(f"Date for dataset {index} set to {parsed_date}")
                    return df
                else:
                    st.error(f"Invalid date format for dataset {index}. Please use YYYY-MM-DD.")
            else:
                st.warning(f"Please enter a valid date for dataset {index} before proceeding.")
        return None
    
    df1 = st.session_state.df
    st.write("Dataset 1 (Start Period) Preview:", df1.head())
    st.write(f"Dataset 1 Date: {st.session_state.first_dataset_date}")

    num_additional_datasets = st.slider("Number of additional datasets to upload", 1, 5, 1)

# File uploader for multiple files
uploaded_files = st.file_uploader("Upload additional CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    st.session_state.additional_datasets = []
    
    # Check if the number of uploaded files matches the slider value
    if len(uploaded_files) != num_additional_datasets:
        st.warning(f"Please upload exactly {num_additional_datasets} file(s).")
    else:
        # Process each uploaded file
        for i, file in enumerate(uploaded_files, start=2):
            df = pd.read_csv(file)
            st.write(f"Dataset {i} Preview:", df.head())
            
            # Date input for each dataset
            date_input = st.text_input(f"Enter the date for dataset {i} (YYYY-MM-DD):", key=f"date_{i}")
            if date_input:
                parsed_date = parse_date(date_input)
                if parsed_date:
                    df['date'] = parsed_date
                    st.success(f"Date for dataset {i} set to {parsed_date}")
                    st.session_state.additional_datasets.append(df)
                else:
                    st.error(f"Invalid date format for dataset {i}. Please use YYYY-MM-DD.")
            else:
                st.warning(f"Please enter a valid date for dataset {i} before proceeding.")
        
        # Check if all datasets have been processed
        if len(st.session_state.additional_datasets) == num_additional_datasets:
            st.success(f"All {num_additional_datasets} additional dataset(s) uploaded and dated successfully!")
        else:
            st.warning("Please ensure all datasets have valid dates.")
else:
    st.info(f"Please upload {num_additional_datasets} additional dataset(s).")

    if len(st.session_state.additional_datasets) > 0:
        all_datasets = [df1] + st.session_state.additional_datasets
        
        # Combine all datasets into a single DataFrame
        combined_df = pd.concat(all_datasets, keys=range(len(all_datasets)))
        combined_df = combined_df.reset_index(level=0).rename(columns={'level_0': 'dataset_index'})

        st.write("### 1. Data Overview")
        st.write(f"Combined dataset shape: {combined_df.shape}")
        st.write(f"Columns: {combined_df.columns.tolist()}")
        st.write("Dataset dates:")
        for i, df in enumerate(all_datasets):
            st.write(f"Dataset {i+1}: {df['date'].iloc[0]}")

        # Get AI recommendations for analyses
        recommendations = get_ai_analysis_recommendations([combined_df])

        # Perform recommended analyses
        for analysis, explanation in recommendations.items():
            st.write(f"### {analysis}")
            st.write(f"Recommendation: {explanation}")

            if analysis == "Correlation Analysis":
                st.write("### Correlation Analysis")
                numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = combined_df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, title="Correlation Heatmap")
                    st.plotly_chart(fig)
                    
                    categorical_cols = combined_df.select_dtypes(include=['object', 'category']).columns
                    if len(categorical_cols) > 0:
                        st.write("Categorical columns:")
                        for cat_col in categorical_cols:
                            st.write(f"- {cat_col}")
                        
                        if len(categorical_cols) > 1:
                            st.write("Chi-square test results for categorical variables:")
                            for col1, col2 in itertools.combinations(categorical_cols, 2):
                                contingency_table = pd.crosstab(combined_df[col1], combined_df[col2])
                                chi2, p_value, _, _ = chi2_contingency(contingency_table)
                                st.write(f"- {col1} vs {col2}: Chi-square = {chi2:.2f}, p-value = {p_value:.4f}")
                
                corr_explanation = generate_contextual_explanation(combined_df, "correlation")
                st.write("Contextual Explanation:", corr_explanation)

            elif analysis == "Feature Importance":
                st.write("### Feature Importance")
                numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    target_col = st.selectbox("Select target variable for feature importance", numeric_cols)
                    feature_cols = [col for col in numeric_cols if col != target_col]
                    if len(feature_cols) > 0:
                        X = combined_df[feature_cols]
                        y = combined_df[target_col]
                        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
                        model.fit(X, y)
                        importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
                        importances = importances.sort_values('importance', ascending=False)
                        fig = px.bar(importances, x='feature', y='importance', title="Feature Importance")
                        st.plotly_chart(fig)
                        feature_explanation = generate_contextual_explanation(combined_df, "feature_importance", target=target_col, importances_list=[importances])
                        st.write("Contextual Explanation:", feature_explanation)
                    else:
                        st.write("Insufficient features for importance analysis.")
                else:
                    st.write("Insufficient numeric columns for feature importance analysis.")

            elif analysis == "Time Series Analysis":
                st.write("### Time Series Analysis")
                numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    selected_col = st.selectbox("Select a numeric column for time series analysis", numeric_cols)
                    fig = px.line(combined_df, x='date', y=selected_col, color='dataset_index', title=f"Time Series of {selected_col}")
                    st.plotly_chart(fig)
                    time_series_explanation = generate_contextual_explanation(combined_df, "time_series", selected_col=selected_col)
                    st.write("Contextual Explanation:", time_series_explanation)
                else:
                    st.write("Insufficient numeric columns for time series analysis.")

            elif analysis == "Anomaly Detection":
                st.write("### Anomaly Detection")
                numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    anomaly_col = st.selectbox("Select column for anomaly detection", numeric_cols)
                    Q1 = combined_df[anomaly_col].quantile(0.25)
                    Q3 = combined_df[anomaly_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    anomalies = combined_df[(combined_df[anomaly_col] < lower_bound) | (combined_df[anomaly_col] > upper_bound)]
                    fig = px.scatter(combined_df, x='date', y=anomaly_col, color='dataset_index', title=f"Anomaly Detection for {anomaly_col}")
                    fig.add_scatter(x=anomalies['date'], y=anomalies[anomaly_col], mode='markers', name='Anomalies', marker=dict(color='red', size=10))
                    st.plotly_chart(fig)
                    anomaly_explanation = generate_contextual_explanation(combined_df, "anomaly_detection", anomaly_col=anomaly_col)
                    st.write("Contextual Explanation:", anomaly_explanation)
                else:
                    st.write("No numeric columns found for anomaly detection.")

            elif analysis == "Clustering Analysis":
                st.write("### Clustering Analysis")
                numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    cluster_cols = st.multiselect("Select columns for clustering", numeric_cols, default=numeric_cols[:2])
                    if len(cluster_cols) >= 2:
                        X = combined_df[cluster_cols]
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        kmeans = KMeans(n_clusters=min(3, len(combined_df)), random_state=42)
                        combined_df['Cluster'] = kmeans.fit_predict(X_scaled)
                        fig = px.scatter(combined_df, x=cluster_cols[0], y=cluster_cols[1], color='Cluster', symbol='dataset_index', title="K-means Clustering")
                        st.plotly_chart(fig)
                        clustering_explanation = generate_contextual_explanation(combined_df, "clustering", cluster_cols=cluster_cols)
                        st.write("Contextual Explanation:", clustering_explanation)
                    else:
                        st.write("Please select at least two columns for clustering.")
                else:
                    st.write("Insufficient numeric columns for clustering analysis.")

            elif analysis == "Dataset Comparison":
                st.write("### Dataset Comparison")
                numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    compare_col = st.selectbox("Select column to compare between datasets", numeric_cols)
                    fig = px.box(combined_df, x='dataset_index', y=compare_col, title=f"Comparison of {compare_col} Across Datasets")
                    st.plotly_chart(fig)
                    st.write("Statistical Test (ANOVA)")
                    f_statistic, p_value = f_oneway(*[df[compare_col].dropna() for df in all_datasets])
                    st.write(f"F-statistic: {f_statistic:.4f}")
                    st.write(f"p-value: {p_value:.4f}")
                    if p_value < 0.05:
                        st.write("There are significant differences between the datasets.")
                    else:
                        st.write("There are no significant differences between the datasets.")
                    comparison_explanation = generate_contextual_explanation(combined_df, "dataset_comparison", compare_col=compare_col, p_value=p_value)
                    st.write("Contextual Explanation:", comparison_explanation)
                else:
                    st.write("No numeric columns found for comparison.")

            elif analysis == "Predictive Modeling":
                st.write("### Predictive Modeling")
                numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    target_col = st.selectbox("Select target variable for predictive modeling", numeric_cols)
                    feature_cols = [col for col in numeric_cols if col != target_col]
                    if len(feature_cols) > 0:
                        X = combined_df[feature_cols]
                        y = combined_df[target_col]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        st.write(f"Mean Squared Error: {mse:.4f}")
                        st.write(f"R-squared Score: {r2:.4f}")
                        fig = px.scatter(x=y_test, y=y_pred, title="Actual vs Predicted Values")
                        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Perfect Prediction'))
                        st.plotly_chart(fig)
                        
                        # Feature importance plot for XGBoost
                        xgb.plot_importance(model)
                        st.pyplot(plt.gcf())
                        plt.clf()
                        
                        modeling_explanation = generate_contextual_explanation(combined_df, "predictive_modeling", target_col=target_col, mse_list=[mse], r2_list=[r2])
                        st.write("Contextual Explanation:", modeling_explanation)
                    else:
                        st.write("Insufficient features for predictive modeling.")
                else:
                    st.write("Insufficient numeric columns for predictive modeling.")

        st.write("### Insights and Recommendations")
        insights = generate_insights(combined_df)
        for i, insight in enumerate(insights, 1):
            st.write(f"{i}. {insight}")

        # Generate the comprehensive report
        comprehensive_report = generate_comprehensive_report(combined_df, insights=insights)
        
        # Store the report in session state
        st.session_state.comprehensive_report = comprehensive_report

        # Offer the report for download
        st.download_button(
            label="Download Comprehensive Analysis Report",
            data=comprehensive_report,
            file_name="comprehensive_analysis_report.md",
            mime="text/markdown"
        )

        st.success("Comprehensive analysis complete. The report is now available for use in other sections.")
    else:
        st.warning("Please upload at least one additional dataset to proceed with the comprehensive analysis.")

    if st.button("Return to Menu"):
        st.session_state.step = 'menu'
        st.experimental_rerun()