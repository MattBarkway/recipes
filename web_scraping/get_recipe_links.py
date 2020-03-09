"""
Scrapes the BBC food recipe index for links to every recipe
"""

import asyncio
import csv
import os
import string

import requests
from aiohttp import ClientSession
from bs4 import BeautifulSoup

from utils.core import limited_as_completed, get_content


def get_recipe_pages(session):
    recipes_home = 'https://www.bbc.co.uk/food/recipes/a-z/'  # a/1 , a/2 , b/1 ...
    for letter in string.ascii_lowercase:
        recipe_page_letter = recipes_home + letter
        num_pages = get_num_of_pages(recipe_page_letter)
        print(f'found {num_pages} pages for {letter}')
        for idx in range(num_pages):
            recipe_page = f'{recipe_page_letter}/{idx + 1}'
            try:
                data = get_recipes_on_page(recipe_page, session)
            except Exception as e:
                print(e)
                continue
            yield data


def get_num_of_pages(url):
    html = requests.get(url).content
    soup = BeautifulSoup(html, 'html.parser')
    pages = soup.findAll('a', {'class': 'pagination__link'})
    number = 1
    for page in pages:
        try:
            candidate_number = int(page.text)
            if candidate_number > number:
                number = candidate_number
        except ValueError:
            continue
    return number


async def get_recipes_on_page(recipe_page, session):
    html = await get_content(recipe_page, session)
    print(f'got data for {recipe_page}')
    soup = BeautifulSoup(html, 'html.parser')
    recipe_cells = soup.findAll('a', {'class': 'promo'})
    links = []
    for recipe_cell in recipe_cells:
        link = get_recipe_from_cell(recipe_cell)
        if link:
            links.append(link)
    return links


def get_recipe_from_cell(recipe_cell):
    try:
        recipe_dict = {
            'name': recipe_cell.find('h3', {'class': 'promo__title'}).text,
            'category': recipe_cell.find('span', {'class': 'promo__type'}).text,
            'link': recipe_cell['href']
        }
        print(f'Added recipe \'{recipe_dict["name"]}\'')
    except Exception as e:
        print(e)
        recipe_dict = {}
    return recipe_dict


async def run(output_path, k=100):
    fieldnames = ['name', 'category', 'link']
    async with ClientSession() as session:
        with open(output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for result in limited_as_completed(get_recipe_pages(session), k):
                try:
                    recipe_page = await result
                except Exception as e:
                    print(e)
                    raise
                    continue
                if recipe_page:
                    for recipe in recipe_page:
                        writer.writerow([recipe.get(field, '') for field in fieldnames])


if __name__ == "__main__":
    asyncio.run(run(os.path.join('data', 'recipe_pages.csv')))
