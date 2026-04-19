# Big Data Homework — Written Answers

## Question 1: Steps for Data Preprocessing in Python

1. **Load the data** — import the dataset using `pandas` (`pd.read_csv()` or a library like `seaborn`).
2. **Inspect the data** — check shape, data types, and summary statistics (`df.info()`, `df.describe()`).
3. **Handle missing values** — fill with mean/median/mode (`df.fillna()`) or drop rows/columns (`df.dropna()`).
4. **Remove duplicates** — identify and drop duplicate rows (`df.drop_duplicates()`).
5. **Drop irrelevant columns** — remove features that do not contribute to the analysis.
6. **Encode categorical variables** — convert text labels to numbers using label encoding or one-hot encoding (`pd.get_dummies()`).
7. **Scale numerical features** — normalize or standardize values using `MinMaxScaler` or `StandardScaler` from `sklearn.preprocessing`.

---

## Question 2: Visualization Methods for Analysis

- **Histogram / KDE plot** — shows the distribution and density of a continuous variable (e.g., age).
- **Bar chart** — compares counts or frequencies across categorical groups (e.g., survival by sex).
- **Box plot** — displays the spread, median, and outliers of a numerical variable across groups (e.g., fare by survival).
- **Heatmap** — visualizes correlation between multiple numerical features using a color-coded matrix.
- **Grouped bar chart** — compares a metric (e.g., survival rate) across multiple category combinations (e.g., class × survival).
- **Scatter plot** — reveals relationships or clusters between two continuous variables.
- **Pie chart** — shows proportional composition of a categorical variable.
