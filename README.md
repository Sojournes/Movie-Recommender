# Movie Recommendation

This project demonstrates a movie recommendation system using collaborative filtering with the MovieLens dataset.

## Dataset

The MovieLens dataset contains 100,000 ratings from 943 users on 1682 movies.

## Getting Started

### Prerequisites

- Python 3.x
- Pandas
- Numpy
- Matplotlib
- Seaborn

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Sojournes/Movie-Recommendation.git
    ```

2. Install the required packages:
    ```bash
    pip install pandas numpy matplotlib seaborn
    ```

3. Download the MovieLens dataset from [here](https://grouplens.org/datasets/movielens/100k/) and place it in the project directory.

## Code Walkthrough

### Importing Libraries and Loading Data

```python
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Load dataset
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=column_names)
```

### Merging Movie Titles

```python
# Load movie titles
movies_title = pd.read_csv("ml-100k/u.item", sep="|", header=None, encoding='ISO-8859-1')
movies_title = movies_title[[0, 1]]
movies_title.columns = ['item_id', 'title']

# Merge datasets
df = pd.merge(df, movies_title, on="item_id")
```

### Exploratory Data Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')

# Calculate mean rating for each movie
ratings = pd.DataFrame(df.groupby('title').mean()['rating'])

# Add number of ratings column
ratings['num of ratings'] = pd.DataFrame(df.groupby('title').count()['rating'])

# Plot number of ratings distribution
plt.figure(figsize=(10, 6))
plt.hist(ratings['num of ratings'], bins=70)
plt.show()

# Plot ratings distribution
plt.figure(figsize=(10, 6))
plt.hist(ratings['rating'], bins=70)
plt.show()

# Jointplot of ratings and number of ratings
sns.jointplot(x='rating', y='num of ratings', data=ratings, alpha=0.5)
plt.show()
```

### Creating Movie Recommendation System

```python
# Create user-movie matrix
moviemat = df.pivot_table(index='user_id', columns='title', values='rating')

# Function to predict movies
def predict_movies(movie_name):
    movie_user_ratings = moviemat[movie_name]
    similar_to_movie = moviemat.corrwith(movie_user_ratings)
    corr_movie = pd.DataFrame(similar_to_movie, columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(ratings['num of ratings'])
    predictions = corr_movie[corr_movie['num of ratings'] > 100].sort_values('Correlation', ascending=False)
    return predictions

# Example prediction
predictions = predict_movies('Titanic (1997)')
print(predictions.head())
```

## Results

The recommendation system can provide movie recommendations based on the user's preferences by calculating the correlation between the ratings of different movies.
