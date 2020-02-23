from datetime import datetime

from recipes.core import (
    fetch, limited_as_completed
)
import os
import pandas as pd
from aiohttp import ClientSession
import asyncio
from bs4 import BeautifulSoup


def save_recipes(df, session):
    for idx, row in df.iterrows():
        print(f'row: {idx + 1}/{len(df)}')
        try:
            yield get_recipe(row['link'], row['name'], session)
        except Exception as e:
            print(e)
            continue


async def get_recipe(link, name, session):
    recipe_dict = {}
    try:
        full_link = f'https://www.bbc.co.uk{link}'
        html = await fetch(full_link, session)
    except Exception as e:
        print(f'request failed, message: {e}')
        return {}
    start = datetime.now()
    soup = BeautifulSoup(html, 'html.parser')
    recipe_dict['name'] = getattr(soup.find('h1', {'class': 'gel-trafalgar content-title__text'}), 'text', name)
    try:
        recipe_dict['cook_time'] = soup.findAll("p", {"class": 'recipe-metadata__cook-time'})[-1].text
    except IndexError:
        print('No cooking time info found')
    try:
        recipe_dict['prep_time'] = soup.findAll("p", {"class": 'recipe-metadata__prep-time'})[-1].text
    except IndexError:
        print('No prep time info found')
    try:
        recipe_dict['serves'] = soup.findAll("p", {"class": 'recipe-metadata__serving'})[-1].text
    except IndexError:
        print('No serving info found')

    ingredients_html = soup.findAll("li", {"class": 'recipe-ingredients__list-item'})
    ingredients = []
    for ingredient_cell in ingredients_html:
        ingredients.append(ingredient_cell.text)
    recipe_dict['ingredients'] = '::'.join(ingredients)
    instructions_html = soup.findAll("p", {"class": 'recipe-method__list-item-text'})
    instructions = []
    for instruction_cell in instructions_html:
        instructions.append(instruction_cell.text)
    recipe_dict['instructions'] = '::'.join(instructions)
    end = datetime.now()
    print(f'{(end - start).total_seconds()} seconds:')
    print(f'Processed details for {recipe_dict["name"]}')
    return recipe_dict


async def run(df):
    recipes = []
    async with ClientSession() as session:
        for result in limited_as_completed(save_recipes(df, session), limit=1000):
            try:
                recipe = await result
            except Exception as e:
                print(e)
                continue
            if recipe:
                recipes.append(recipe)
    pd.DataFrame(recipes).to_csv(os.path.join('data', 'bbc',  'recipe_details.csv'))


if __name__ == "__main__":
    _df = pd.read_csv(os.path.join('data', 'bbc', 'recipe_pages.csv'), index_col=0)
    _df = _df.drop_duplicates()
    asyncio.run(run(_df))
