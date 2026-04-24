from src.data_loader import load_all

movies, ratings = load_all()

print(movies.head())
print(ratings.head())
