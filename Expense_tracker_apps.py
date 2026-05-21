# We will build such a machine learning model which predict the actual expenses

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


def load_expenses():

    expenses = []

    file = open("Expense.csv", "r")

    for line in file:

        line = line.strip()

        if not line:
            continue

        amount, category, date = line.split(",")

        expense = {"amount": float(amount), "category": category, "date": date}

        expenses.append(expense)

    file.close()

    return expenses


expenses = load_expenses()


def menu():

    print("\n1. Add Expense ")
    print("2. View Expense")
    print("3. Total Expenses")
    print("4. Category Analysis")
    print("5. Pandas Analysis")
    print("6. Visualize Data")
    print("7. Predict Expenses")
    print("8. Exit")


def add_expense(expenses):

    while True:

        try:

            amount = float(input("Enter amount: "))

            if amount <= 0:
                print("Amount must be positive.")
                continue

            break

        except ValueError:
            print("Please enter a valid number.")
    while True:

        category = input("Enter category: ")

        if category.strip() == "":
            print("Category cannot be empty.")

        else:
            break
    date = input("Enter date (YYYY-MM-DD): ")

    if len(date) != 10:
        print("Invalid date format.")

    expense = {"amount": amount, "category": category, "date": date}

    expenses.append(expense)
    save_to_file(expense)
    print("Expenes Added!")


def show_expense(expenses):

    if not expenses:
        print("No expenses yet!")
    else:
        for expense in expenses:
            print("-" * 20)
            print(
                f"Amount: {expense['amount']} | "
                f"Category: {expense['category']} | "
                f"Date: {expense['date']}"
            )
        print("-" * 20)


def total_expenses(expenses):

    total = 0
    if not expenses:
        print("No expenses to calculate !!")
    else:
        for expense in expenses:
            total += expense["amount"]

        print(f"Total Expenses: {total}")


def category_analysis(expenses):

    if not expenses:
        print("No expenses available.")
        return

    category_total = {}

    for expense in expenses:

        category = expense["category"]
        amount = expense["amount"]

        if category in category_total:
            category_total[category] += amount
        else:
            category_total[category] = amount
    print("\n Category Analysis")

    for category, amount in category_total.items():
        print(f"{category}: {amount}")


def save_to_file(expense):

    with open("Expense.csv", "a") as f:
        f.write(f"{expense['amount']} , {expense['category']}, {expense['date']}\n")


def pandas_analysis():
    import pandas as pd

    df = pd.read_csv("Expense.csv", header=None, names=["amount", "category", "date"])

    print(df)
    print(f"All Amounts = {df["amount"]}")
    print(f"The total expenses = {df["amount"].sum()}")
    print(f"Category Expenses \n {df.groupby('category')['amount'].sum()}")
    print(f"Highest Expense = {df['amount'].max()}")


def visualize_data():

    df = pd.read_csv("Expense.csv", header=None, names=["amount", "category", "date"])

    def bar_chart(df):

        category_totals = df.groupby("category")["amount"].sum()

        category_totals.plot(kind="bar")

        plt.title("Expenses by Category")
        plt.xlabel("Category")
        plt.ylabel("Amount")

        plt.show()

    def pie_chart(df):

        category_totals = df.groupby("category")["amount"].sum()

        category_totals.plot(kind="pie", autopct="%1.1f%%")

        plt.title("Expense Distribution")

        plt.ylabel("")

        plt.show()

    def line_chart(df):

        df["date"] = pd.to_datetime(df["date"])

        daily_expenses = df.groupby("date")["amount"].sum()

        daily_expenses.plot(kind="line")

        plt.title("Daily Spending Trend")
        plt.xlabel("Date")
        plt.ylabel("Amount")

        plt.show()

    def histogram_chart(df):

        df["amount"].plot(kind="hist")

        plt.title("Expense Amount Distribution")
        plt.xlabel("Amount")

        plt.show()

    while True:

        print("\n--- Visualization Menu ---")
        print("1. Bar Chart")
        print("2. Pie Chart")
        print("3. Daily Spending Trend")
        print("4. Expense Distribution Histogram")
        print("5. Back")

        choice = input("Enter choice: ")

        if choice == "1":
            bar_chart(df)

        elif choice == "2":
            pie_chart(df)

        elif choice == "3":
            line_chart(df)

        elif choice == "4":
            histogram_chart(df)

        elif choice == "5":
            break

        else:
            print("Invalid choice!")


def predict_expenses():

    df = pd.read_csv("Expense.csv", header=None, names=["amount", "category", "date"])

    daily_expenses = df.groupby("date")["amount"].sum()

    days = np.array(range(len(daily_expenses))).reshape(-1, 1)

    amounts = daily_expenses.values

    model = LinearRegression()

    model.fit(days, amounts)

    next_day = np.array([[len(daily_expenses)]])

    prediction = model.predict(next_day)

    print(f"Predicted next day expense: {prediction[0]:.2f}")


while True:

    menu()
    choice = input("Enter Choice: ")
    if choice == "1":
        add_expense(expenses)
    elif choice == "2":
        show_expense(expenses)
    elif choice == "3":
        total_expenses(expenses)
    elif choice == "4":
        category_analysis(expenses)
    elif choice == "5":
        pandas_analysis()
    elif choice == "6":
        visualize_data()
    elif choice == "7":
        predict_expenses()
    elif choice == "8":

        print("Exiting....")
        break
    else:
        print("Invalid Choice.!!")
