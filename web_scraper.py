import os
import string

import asyncio
import pandas as pd
import requests
from aiohttp import ClientSession

from recipes.core import (
    fetch, limited_as_completed
)
from bs4 import BeautifulSoup

recipes_home = 'https://www.bbc.co.uk/food/recipes/a-z/'  # a/1 , a/2 , b/1 ...


def get_recipe_pages(session):
    letters = string.ascii_lowercase
    for letter in letters:
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
    try:
        print(f'fetching {recipe_page}')
        html = await fetch(recipe_page, session)
    except Exception as e:
        print(e)
        print(f'page not found {recipe_page}')
        return []
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
        name = recipe_cell.text
    except Exception as e:
        print(e)
        return {}
    recipe_dict = {
        'name': name,
        'category': recipe_cell.find('span', {'class': 'promo__type'}).text,
        'link': recipe_cell['href']
    }
    print(f'Added recipe {recipe_dict["name"]}')
    return recipe_dict


async def run():
    recipe_pages = []
    async with ClientSession() as session:
        for result in limited_as_completed(get_recipe_pages(session), 1000):
            try:
                recipe_page = await result
            except Exception as e:
                print(e)
                continue
            if recipe_page:
                recipe_pages.extend(recipe_page)
        pd.DataFrame(recipe_pages).to_csv(os.path.join('data', 'bbc', 'recipe_pages.csv'))


if __name__ == "__main__":
    asyncio.run(run())
