"""
Scrapes BBC food for recipe data
"""
import asyncio
import os

import pandas as pd

from web_scraping import get_recipes, get_recipe_links


def scrape():
    if not os.path.isdir('data'):
        os.mkdir('data')
    base_data_output_path = os.path.join('data', 'recipe_pages.csv')
    if not os.path.exists(base_data_output_path):
        asyncio.run(get_recipe_links(base_data_output_path))
    details_output_path = os.path.join('data', 'recipe_details.csv')
    if not os.path.exists(details_output_path):
        df = pd.read_csv(base_data_output_path, index_col=0)
        asyncio.run(get_recipes(df, details_output_path))
        print(f'Data saved to file {details_output_path}')
    else:
        print('Data already exists')


if __name__ == "__main__":
    scrape()
    # TODO avoid loading all data in memory at any point
    # Save each line as it is processed
    # TODO scrape all details in one run, dont use intermediate file
    # import csv
    # fields = ['first', 'second', 'third']
    # with open(r'name', 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(fields)
