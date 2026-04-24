import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(ROOT_DIR)

from src.data_loader import load_all

movies, ratings = load_all()

print(movies.head())
print(ratings.head())
