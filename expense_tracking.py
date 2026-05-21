import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from pathlib import Path


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
            print(f"Error loading expenses: {e}")
            return []

    def save_expense(self, expense):

        df = pd.DataFrame([expense])

        df.to_csv(self.FILE_NAME, mode="a", header=False, index=False)

    # =========================
    # MENU
    # =========================

    def menu(self):

        print("\n===== Expense Tracker =====")
        print("1. Add Expense")
        print("2. View Expenses")
        print("3. Total Expenses")
        print("4. Category Analysis")
        print("5. Pandas Analysis")
        print("6. Visualize Data")
        print("7. Predict Expenses")
        print("8. Exit")

    # =========================
    # ADD EXPENSE
    # =========================

    def add_expense(self):

        amount = self.get_valid_amount()
        category = self.get_valid_category()
        date = self.get_valid_date()

        expense = {"amount": amount, "category": category, "date": date}

        self.expenses.append(expense)

        self.save_expense(expense)

        print("✅ Expense added successfully!")

    # =========================
    # VALIDATION METHODS
    # =========================

    def get_valid_amount(self):

        while True:

            try:

                amount = float(input("Enter amount: "))

                if amount <= 0:
                    print("Amount must be positive.")
                    continue

                return amount

            except ValueError:
                print("Invalid number.")

    def get_valid_category(self):

        while True:

            category = input("Enter category: ").strip()

            if category:
                return category

            print("Category cannot be empty.")

    def get_valid_date(self):

        while True:

            date = input("Enter date (YYYY-MM-DD): ")

            try:
                pd.to_datetime(date)
                return date

            except:
                print("Invalid date format.")

    # =========================
    # VIEW EXPENSES
    # =========================

    def show_expenses(self):

        if not self.expenses:
            print("No expenses found.")
            return

        print("\n===== Expenses =====")

        for expense in self.expenses:

            print(
                f"Amount: {expense['amount']} | "
                f"Category: {expense['category']} | "
                f"Date: {expense['date']}"
            )

    # =========================
    # TOTAL EXPENSES
    # =========================

    def total_expenses(self):

        total = sum(expense["amount"] for expense in self.expenses)

        print(f"\nTotal Expenses: {total:.2f}")

    # =========================
    # CATEGORY ANALYSIS
    # =========================

    def category_analysis(self):

        if not self.expenses:
            print("No expense data available.")
            return

        df = self.get_dataframe()

        category_totals = df.groupby("category")["amount"].sum()

        print("\n===== Category Analysis =====")
        print(category_totals)

    # =========================
    # DATAFRAME
    # =========================

    def get_dataframe(self):

        return pd.DataFrame(self.expenses)

    # =========================
    # PANDAS ANALYSIS
    # =========================

    def pandas_analysis(self):

        df = self.get_dataframe()

        if df.empty:
            print("No data available.")
            return

        print("\n===== Data Analysis =====")

        print(df)

        print("\nTotal Expense:")
        print(df["amount"].sum())

        print("\nAverage Expense:")
        print(df["amount"].mean())

        print("\nHighest Expense:")
        print(df["amount"].max())

        print("\nCategory Totals:")
        print(df.groupby("category")["amount"].sum())

    # =========================
    # VISUALIZATION
    # =========================

    def visualize_data(self):

        df = self.get_dataframe()

        if df.empty:
            print("No data available.")
            return

        while True:

            print("\n===== Visualization Menu =====")
            print("1. Bar Chart")
            print("2. Pie Chart")
            print("3. Line Chart")
            print("4. Histogram")
            print("5. Back")

            choice = input("Enter choice: ")

            if choice == "1":
                self.bar_chart(df)

            elif choice == "2":
                self.pie_chart(df)

            elif choice == "3":
                self.line_chart(df)

            elif choice == "4":
                self.histogram_chart(df)

            elif choice == "5":
                break

            else:
                print("Invalid choice.")

    def bar_chart(self, df):

        plt.figure(figsize=(8, 5))

        (df.groupby("category")["amount"].sum().plot(kind="bar"))

        plt.title("Expenses by Category")
        plt.xlabel("Category")
        plt.ylabel("Amount")
        plt.grid(True)

        plt.show()

    def pie_chart(self, df):

        plt.figure(figsize=(8, 5))

        (df.groupby("category")["amount"].sum().plot(kind="pie", autopct="%1.1f%%"))

        plt.title("Expense Distribution")
        plt.ylabel("")

        plt.show()

    def line_chart(self, df):

        plt.figure(figsize=(8, 5))

        df["date"] = pd.to_datetime(df["date"])

        daily_expenses = df.groupby("date")["amount"].sum()

        daily_expenses.plot(kind="line")

        plt.title("Daily Spending Trend")
        plt.xlabel("Date")
        plt.ylabel("Amount")
        plt.grid(True)

        plt.show()

    def histogram_chart(self, df):

        plt.figure(figsize=(8, 5))

        df["amount"].plot(kind="hist")

        plt.title("Expense Distribution")
        plt.xlabel("Amount")
        plt.grid(True)

        plt.show()

    # =========================
    # MACHINE LEARNING
    # =========================

    def predict_expenses(self):

        df = self.get_dataframe()

        if len(df) < 2:
            print("Not enough data for prediction.")
            return

        daily_expenses = df.groupby("date")["amount"].sum()

        X = np.array(range(len(daily_expenses))).reshape(-1, 1)

        y = daily_expenses.values

        model = LinearRegression()

        model.fit(X, y)

        next_day = np.array([[len(daily_expenses)]])

        prediction = model.predict(next_day)

        print(f"\nPredicted next day expense: " f"{prediction[0]:.2f}")

    # =========================
    # APPLICATION RUNNER
    # =========================

    def run(self):

        while True:

            self.menu()

            choice = input("Enter choice: ")

            if choice == "1":
                self.add_expense()

            elif choice == "2":
                self.show_expenses()

            elif choice == "3":
                self.total_expenses()

            elif choice == "4":
                self.category_analysis()

            elif choice == "5":
                self.pandas_analysis()

            elif choice == "6":
                self.visualize_data()

            elif choice == "7":
                self.predict_expenses()

            elif choice == "8":

                print("Exiting application...")
                break

            else:
                print("Invalid choice.")


if __name__ == "__main__":

    tracker = ExpenseTracker()

    tracker.run()
