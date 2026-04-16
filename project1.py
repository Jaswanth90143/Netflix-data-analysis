# ==========================================
# NETFLIX DATA ANALYSIS PROJECT (FULL CODE)
# ==========================================

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Dataset
df = pd.read_csv(r"C:\Users\JASWANTH PULI\Downloads\netflix_titles.csv\netflix_titles.csv")

# Step 3: Basic Info
print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA INFO =====")
print(df.info())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Step 4: Data Cleaning (FIXED - no inplace)
df['director'] = df['director'].fillna('Unknown')
df['cast'] = df['cast'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')
df['rating'] = df['rating'].fillna('Not Rated')

# Convert date
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['year_added'] = df['date_added'].dt.year

# ==========================================
# VISUALIZATIONS
# ==========================================

# 1. Movies vs TV Shows
plt.figure()
sns.countplot(x='type', data=df)
plt.title("Movies vs TV Shows")
plt.show()

# 2. Top Genres
genres = df['listed_in'].str.split(', ', expand=True).stack()
plt.figure()
genres.value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Genres")
plt.show()

# 3. Content Added Over Years
plt.figure()
df['year_added'].value_counts().sort_index().plot(kind='line')
plt.title("Content Added Over Years")
plt.show()

# 4. Top Countries
plt.figure()
df['country'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Countries")
plt.show()

# 5. Ratings Distribution
plt.figure()
sns.countplot(y='rating', data=df, order=df['rating'].value_counts().index)
plt.title("Ratings Distribution")
plt.show()

# ==========================================
# MOVIE-SPECIFIC ANALYSIS
# ==========================================

df_movies = df[df['type'] == 'Movie'].copy()

# Convert duration
df_movies['duration'] = df_movies['duration'].str.replace(' min', '', regex=False)
df_movies['duration'] = pd.to_numeric(df_movies['duration'], errors='coerce')

# 6. Histogram
plt.figure()
sns.histplot(df_movies['duration'].dropna(), bins=30)
plt.title("Movie Duration Distribution")
plt.show()

# 7. Box Plot
plt.figure()
sns.boxplot(x=df_movies['duration'])
plt.title("Box Plot of Movie Duration")
plt.show()

# ==========================================
# OUTLIER DETECTION (IQR)
# ==========================================

Q1 = df_movies['duration'].quantile(0.25)
Q3 = df_movies['duration'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_movies[(df_movies['duration'] < lower_bound) | 
                     (df_movies['duration'] > upper_bound)]

print("\n===== OUTLIERS =====")
print("Number of outliers:", outliers.shape[0])
print(outliers[['title', 'duration']].head())

# ==========================================
# PAIR PLOT
# ==========================================

# Create new numeric features
df_movies['year_added'] = df_movies['year_added']
df_movies['release_year'] = df_movies['release_year']

# Pairplot with multiple columns
sns.pairplot(df_movies[['duration', 'release_year', 'year_added']].dropna())
plt.show()

# ==========================================
# PIE CHARTS
# ==========================================

# 1. Movies vs TV Shows Pie Chart
type_counts = df['type'].value_counts()

plt.figure()
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%')
plt.title("Movies vs TV Shows Distribution")
plt.show()

# 2. Top 5 Genres Pie Chart
top_genres = genres.value_counts().head(5)

plt.figure()
plt.pie(top_genres, labels=top_genres.index, autopct='%1.1f%%')
plt.title("Top 5 Genres")
plt.show()


# ==========================================
# INSIGHTS (FOR VIVA)
# ==========================================

print("\n===== KEY INSIGHTS =====")
print("Most common genre:", genres.value_counts().idxmax())
print("Top country:", df['country'].value_counts().idxmax())
print("Most common rating:", df['rating'].value_counts().idxmax())

# ==========================================
# END OF PROJECT
# ==========================================
