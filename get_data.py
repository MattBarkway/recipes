"""
Scrapes BBC food for recipe data
"""
import asyncio
import os

import pandas as pd

from web_scraping import get_recipes, get_recipe_links

if __name__ == "__main__":
    base_data_output_path = os.path.join('data', 'bbc', 'recipe_pages.csv')
    if not os.path.exists(base_data_output_path):
        asyncio.run(get_recipe_links(base_data_output_path))
    details_output_path = os.path.join('data', 'bbc', 'recipe_details.csv')
    if not os.path.exists(details_output_path):
        df = pd.read_csv(base_data_output_path, index_col=0)
        asyncio.run(get_recipes(df, details_output_path))
    else:
        print('Data already exists')
