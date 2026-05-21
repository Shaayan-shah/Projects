import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go


class ExpenseTracker:
    FILE_NAME = "Expense.csv"

    def __init__(self):
        self.expenses = self.load_expenses()

    # =========================
    # FILE HANDLING
    # =========================
    def load_expenses(self):
        if not Path(self.FILE_NAME).exists():
            return []

        try:
            df = pd.read_csv(
                self.FILE_NAME, header=None, names=["amount", "category", "date"]
            )
            return df.to_dict("records")
        except Exception as e:
            st.error(f"Error loading expenses: {e}")
            return []

    def save_expense(self, expense):
        df = pd.DataFrame([expense])
        df.to_csv(self.FILE_NAME, mode="a", header=False, index=False)

    def get_dataframe(self):
        return pd.DataFrame(self.expenses)

    # =========================
    # EXPENSE OPERATIONS
    # =========================
    def add_expense(self, amount, category, date):
        expense = {"amount": amount, "category": category, "date": date}
        self.expenses.append(expense)
        self.save_expense(expense)
        return True

    def get_total_expenses(self):
        return sum(expense["amount"] for expense in self.expenses)

    def get_category_totals(self):
        df = self.get_dataframe()
        if df.empty:
            return pd.Series()
        return df.groupby("category")["amount"].sum()

    def get_daily_expenses(self):
        df = self.get_dataframe()
        if df.empty:
            return pd.Series()
        df["date"] = pd.to_datetime(df["date"])
        return df.groupby("date")["amount"].sum()

    def predict_next_expense(self):
        daily_expenses = self.get_daily_expenses()
        if len(daily_expenses) < 2:
            return None

        X = np.array(range(len(daily_expenses))).reshape(-1, 1)
        y = daily_expenses.values
        model = LinearRegression()
        model.fit(X, y)
        next_day = np.array([[len(daily_expenses)]])
        prediction = model.predict(next_day)
        return prediction[0]


# =========================
# STREAMLIT UI
# =========================
def main():
    st.set_page_config(page_title="Expense Tracker", page_icon="💰", layout="wide")

    # Initialize session state
    if "tracker" not in st.session_state:
        st.session_state.tracker = ExpenseTracker()

    tracker = st.session_state.tracker

    # Sidebar navigation
    st.sidebar.title("💰 Expense Tracker")
    st.sidebar.markdown("---")

    menu = st.sidebar.selectbox(
        "Navigation",
        [
            "Add Expense",
            "View Expenses",
            "Statistics",
            "Analysis",
            "Visualization",
            "Predictions",
            "Data Management",
        ],
    )

    # Header
    st.title("💰 Personal Expense Tracker")
    st.markdown("Track, analyze, and predict your expenses with ease!")

    # =========================
    # ADD EXPENSE
    # =========================
    if menu == "Add Expense":
        st.header("➕ Add New Expense")

        col1, col2, col3 = st.columns(3)

        with col1:
            amount = st.number_input(
                "Amount ($)", min_value=0.01, step=0.01, format="%.2f"
            )

        with col2:
            categories = [
                "Food",
                "Transport",
                "Shopping",
                "Entertainment",
                "Bills",
                "Healthcare",
                "Education",
                "Other",
            ]
            category = st.selectbox("Category", categories)

        with col3:
            date = st.date_input("Date", datetime.now())

        if st.button("Add Expense", type="primary"):
            if amount > 0:
                if tracker.add_expense(amount, category, str(date)):
                    st.success("✅ Expense added successfully!")
                    st.balloons()
            else:
                st.error("Please enter a valid amount!")

    # =========================
    # VIEW EXPENSES
    # =========================
    elif menu == "View Expenses":
        st.header("📋 Expense List")

        if not tracker.expenses:
            st.info("No expenses found. Add some expenses to get started!")
        else:
            df = tracker.get_dataframe()

            # Filters
            col1, col2 = st.columns(2)
            with col1:
                categories = ["All"] + sorted(df["category"].unique().tolist())
                filter_category = st.selectbox("Filter by Category", categories)

            with col2:
                df["date"] = pd.to_datetime(df["date"])
                min_date = df["date"].min()
                max_date = df["date"].max()
                date_range = st.date_input("Filter by Date Range", [min_date, max_date])

            # Apply filters
            filtered_df = df.copy()
            if filter_category != "All":
                filtered_df = filtered_df[filtered_df["category"] == filter_category]

            if len(date_range) == 2:
                filtered_df = filtered_df[
                    (filtered_df["date"] >= pd.to_datetime(date_range[0]))
                    & (filtered_df["date"] <= pd.to_datetime(date_range[1]))
                ]

            # Display table
            display_df = filtered_df.copy()
            display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime(
                "%Y-%m-%d"
            )
            display_df = display_df.sort_values("date", ascending=False)

            st.dataframe(
                display_df,
                column_config={
                    "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                    "category": "Category",
                    "date": "Date",
                },
                use_container_width=True,
                hide_index=True,
            )

            st.metric("Total Expenses", f"${filtered_df['amount'].sum():.2f}")

            # Export option
            if st.button("Export to CSV"):
                filtered_df.to_csv("exported_expenses.csv", index=False)
                st.success("Exported to exported_expenses.csv")

    # =========================
    # STATISTICS
    # =========================
    elif menu == "Statistics":
        st.header("📊 Statistics Overview")

        if not tracker.expenses:
            st.info("No expenses found. Add some expenses to see statistics!")
        else:
            df = tracker.get_dataframe()
            df["date"] = pd.to_datetime(df["date"])

            col1, col2, col3, col4 = st.columns(4)

            total = tracker.get_total_expenses()
            avg = df["amount"].mean()
            max_expense = df["amount"].max()
            min_expense = df["amount"].min()

            with col1:
                st.metric("Total Expenses", f"${total:.2f}")
            with col2:
                st.metric("Average Expense", f"${avg:.2f}")
            with col3:
                st.metric("Highest Expense", f"${max_expense:.2f}")
            with col4:
                st.metric("Lowest Expense", f"${min_expense:.2f}")

            st.markdown("---")

            # Monthly statistics
            st.subheader("Monthly Breakdown")
            df["month"] = df["date"].dt.strftime("%Y-%m")
            monthly = (
                df.groupby("month")["amount"].agg(["sum", "mean", "count"]).round(2)
            )
            monthly.columns = ["Total", "Average", "Number of Expenses"]

            st.dataframe(monthly, use_container_width=True)

            # Category statistics
            st.subheader("Category Breakdown")
            category_stats = (
                df.groupby("category")["amount"].agg(["sum", "mean", "count"]).round(2)
            )
            category_stats.columns = ["Total", "Average", "Number of Expenses"]
            category_stats = category_stats.sort_values("Total", ascending=False)

            st.dataframe(category_stats, use_container_width=True)

    # =========================
    # ANALYSIS
    # =========================
    elif menu == "Analysis":
        st.header("🔍 Deep Analysis")

        if not tracker.expenses:
            st.info("No expenses found. Add some expenses to analyze!")
        else:
            df = tracker.get_dataframe()
            df["date"] = pd.to_datetime(df["date"])

            tab1, tab2, tab3 = st.tabs(
                ["Category Analysis", "Time Analysis", "Statistical Insights"]
            )

            with tab1:
                st.subheader("Category Distribution")
                category_totals = tracker.get_category_totals()

                col1, col2 = st.columns(2)

                with col1:
                    fig = px.pie(
                        values=category_totals.values,
                        names=category_totals.index,
                        title="Expense Distribution by Category",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = px.bar(
                        x=category_totals.index,
                        y=category_totals.values,
                        title="Expenses by Category",
                        labels={"x": "Category", "y": "Amount ($)"},
                        color=category_totals.index,
                        color_discrete_sequence=px.colors.qualitative.Set3,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("Expense Trends Over Time")

                daily = tracker.get_daily_expenses()
                daily_df = pd.DataFrame({"Date": daily.index, "Amount": daily.values})

                # Weekly average
                daily_df["Week"] = pd.to_datetime(daily_df["Date"]).dt.strftime(
                    "%Y-W%W"
                )
                weekly_avg = daily_df.groupby("Week")["Amount"].mean()

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=daily_df["Date"],
                        y=daily_df["Amount"],
                        mode="lines+markers",
                        name="Daily Expenses",
                        line=dict(color="blue", width=2),
                        marker=dict(size=6),
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=weekly_avg.index,
                        y=weekly_avg.values,
                        mode="lines+markers",
                        name="Weekly Average",
                        line=dict(color="red", width=2, dash="dash"),
                    )
                )

                fig.update_layout(
                    title="Daily Expense Trend",
                    xaxis_title="Date",
                    yaxis_title="Amount ($)",
                    hovermode="x unified",
                )

                st.plotly_chart(fig, use_container_width=True)

            with tab3:
                st.subheader("Statistical Insights")

                col1, col2 = st.columns(2)

                with col1:
                    # Histogram
                    fig = px.histogram(
                        df,
                        x="amount",
                        nbins=20,
                        title="Expense Amount Distribution",
                        labels={"amount": "Amount ($)"},
                        color_discrete_sequence=["green"],
                    )
                    fig.update_layout(bargap=0.1)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Box plot by category
                    fig = px.box(
                        df,
                        x="category",
                        y="amount",
                        title="Expense Distribution by Category",
                        labels={"category": "Category", "amount": "Amount ($)"},
                        color="category",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Correlation analysis (if there's date-based patterns)
                df["day_of_week"] = df["date"].dt.dayofweek
                df["day_name"] = df["date"].dt.day_name()

                weekday_avg = (
                    df.groupby("day_name")["amount"]
                    .mean()
                    .reindex(
                        [
                            "Monday",
                            "Tuesday",
                            "Wednesday",
                            "Thursday",
                            "Friday",
                            "Saturday",
                            "Sunday",
                        ]
                    )
                )

                fig = px.bar(
                    x=weekday_avg.index,
                    y=weekday_avg.values,
                    title="Average Expense by Day of Week",
                    labels={"x": "Day of Week", "y": "Average Amount ($)"},
                    color=weekday_avg.index,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                st.plotly_chart(fig, use_container_width=True)

    # =========================
    # VISUALIZATION
    # =========================
    elif menu == "Visualization":
        st.header("📈 Interactive Visualizations")

        if not tracker.expenses:
            st.info("No expenses found. Add some expenses to visualize!")
        else:
            df = tracker.get_dataframe()
            df["date"] = pd.to_datetime(df["date"])

            viz_type = st.selectbox(
                "Select Visualization Type",
                [
                    "Bar Chart",
                    "Pie Chart",
                    "Line Chart",
                    "Area Chart",
                    "Scatter Plot",
                    "Heatmap",
                ],
            )

            if viz_type == "Bar Chart":
                category_totals = tracker.get_category_totals()
                fig = px.bar(
                    x=category_totals.index,
                    y=category_totals.values,
                    title="Expenses by Category",
                    labels={"x": "Category", "y": "Total Amount ($)"},
                    text=category_totals.values,
                    color=category_totals.index,
                )
                fig.update_traces(texttemplate="$%{text:.2f}", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Pie Chart":
                category_totals = tracker.get_category_totals()
                fig = px.pie(
                    values=category_totals.values,
                    names=category_totals.index,
                    title="Expense Distribution",
                    hole=0.3,
                )
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Line Chart":
                daily = tracker.get_daily_expenses()
                fig = px.line(
                    x=daily.index,
                    y=daily.values,
                    title="Daily Expense Trend",
                    labels={"x": "Date", "y": "Amount ($)"},
                    markers=True,
                )
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Area Chart":
                df["month"] = df["date"].dt.strftime("%Y-%m")
                monthly = (
                    df.groupby(["month", "category"])["amount"].sum().reset_index()
                )
                fig = px.area(
                    monthly,
                    x="month",
                    y="amount",
                    color="category",
                    title="Monthly Expense Trend by Category",
                    labels={"month": "Month", "amount": "Amount ($)"},
                )
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Scatter Plot":
                df["day_of_month"] = df["date"].dt.day
                fig = px.scatter(
                    df,
                    x="day_of_month",
                    y="amount",
                    color="category",
                    title="Expense Distribution by Day of Month",
                    labels={"day_of_month": "Day of Month", "amount": "Amount ($)"},
                    size="amount",
                    hover_data=["date"],
                )
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Heatmap":
                # Create pivot table for heatmap
                df["month"] = df["date"].dt.month
                df["day"] = df["date"].dt.day
                pivot = df.pivot_table(
                    values="amount",
                    index="day",
                    columns="month",
                    aggfunc="sum",
                    fill_value=0,
                )

                fig = px.imshow(
                    pivot,
                    title="Expense Heatmap (Day vs Month)",
                    labels={"x": "Month", "y": "Day", "color": "Amount ($)"},
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig, use_container_width=True)

    # =========================
    # PREDICTIONS
    # =========================
    elif menu == "Predictions":
        st.header("🔮 Expense Predictions")

        if not tracker.expenses:
            st.info("Not enough data for predictions. Add at least 2 expenses!")
        else:
            df = tracker.get_dataframe()
            df["date"] = pd.to_datetime(df["date"])

            daily_expenses = tracker.get_daily_expenses()

            if len(daily_expenses) < 2:
                st.warning("Need at least 2 days of data for prediction!")
            else:
                prediction = tracker.predict_next_expense()

                if prediction:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Predicted Next Day Expense",
                            f"${prediction:.2f}",
                            delta=f"{prediction - daily_expenses.iloc[-1]:.2f}",
                        )

                    with col2:
                        st.metric(
                            "Average Daily Expense", f"${daily_expenses.mean():.2f}"
                        )

                    # Visualization with prediction
                    st.subheader("Expense Trend with Prediction")

                    # Prepare data
                    last_7_days = daily_expenses.tail(7)
                    X_actual = np.array(range(len(daily_expenses))).reshape(-1, 1)
                    y_actual = daily_expenses.values

                    model = LinearRegression()
                    model.fit(X_actual, y_actual)

                    # Generate future predictions
                    future_days = 7
                    future_X = np.array(
                        range(len(daily_expenses), len(daily_expenses) + future_days)
                    ).reshape(-1, 1)
                    future_predictions = model.predict(future_X)

                    # Create plot
                    fig = go.Figure()

                    # Historical data
                    fig.add_trace(
                        go.Scatter(
                            x=daily_expenses.index,
                            y=daily_expenses.values,
                            mode="lines+markers",
                            name="Historical",
                            line=dict(color="blue", width=2),
                        )
                    )

                    # Future predictions
                    future_dates = pd.date_range(
                        start=daily_expenses.index[-1] + pd.Timedelta(days=1),
                        periods=future_days,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=future_predictions,
                            mode="lines+markers",
                            name="Predicted",
                            line=dict(color="red", width=2, dash="dash"),
                        )
                    )

                    fig.update_layout(
                        title="Expense Prediction (Linear Regression)",
                        xaxis_title="Date",
                        yaxis_title="Amount ($)",
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Model metrics
                    st.subheader("Model Performance")
                    r2_score = model.score(X_actual, y_actual)
                    st.info(f"R² Score: {r2_score:.3f}")

                    if r2_score < 0.3:
                        st.warning(
                            "⚠️ The prediction model has low accuracy. Add more data for better predictions!"
                        )
                    elif r2_score > 0.7:
                        st.success("✅ Model has good predictive power!")

    # =========================
    # DATA MANAGEMENT
    # =========================
    elif menu == "Data Management":
        st.header("⚙️ Data Management")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Current Data Summary")
            if tracker.expenses:
                st.write(f"Total Records: {len(tracker.expenses)}")
                st.write(
                    f"Date Range: {tracker.get_dataframe()['date'].min()} to {tracker.get_dataframe()['date'].max()}"
                )
                st.write(
                    f"Categories: {', '.join(tracker.get_dataframe()['category'].unique())}"
                )
            else:
                st.write("No data available")

        with col2:
            st.subheader("Data Operations")

            if st.button("Refresh Data", type="primary"):
                st.session_state.tracker = ExpenseTracker()
                st.success("Data refreshed!")
                st.rerun()

            if st.button("Clear All Data", type="secondary"):
                if Path(ExpenseTracker.FILE_NAME).exists():
                    Path(ExpenseTracker.FILE_NAME).unlink()
                    st.session_state.tracker = ExpenseTracker()
                    st.warning("All data cleared!")
                    st.rerun()
                else:
                    st.info("No data to clear")

        st.markdown("---")
        st.subheader("Import/Export Data")

        # Export all data
        if tracker.expenses:
            df = tracker.get_dataframe()
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Data (CSV)",
                data=csv,
                file_name="expense_data.csv",
                mime="text/csv",
            )

        # Import data
        uploaded_file = st.file_uploader("Import Data (CSV)", type=["csv"])
        if uploaded_file is not None:
            try:
                imported_df = pd.read_csv(uploaded_file)
                required_cols = ["amount", "category", "date"]
                if all(col in imported_df.columns for col in required_cols):
                    for _, row in imported_df.iterrows():
                        tracker.add_expense(row["amount"], row["category"], row["date"])
                    st.success(f"Imported {len(imported_df)} records!")
                    st.rerun()
                else:
                    st.error("CSV must have columns: amount, category, date")
            except Exception as e:
                st.error(f"Error importing file: {e}")


if __name__ == "__main__":
    main()
