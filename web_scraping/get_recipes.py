import asyncio
import csv
import os

import pandas as pd
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from utils.core import limited_as_completed, get_content


def save_recipes(df, session):
    for idx, row in df.iterrows():
        print(f'row: {idx + 1}/{len(df)}')
        try:
            yield get_recipe(row['link'], row['name'], session)
        except Exception as e:
            print(e)
            continue


def process_recipe_html(html, name):
    recipe_dict = {}
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
    recipe_dict['ingredients'] = '::'.join(get_ingredients(soup))
    recipe_dict['instructions'] = '::'.join(get_instructions(soup))
    print(f'Processed details for {recipe_dict["name"]}')
    return recipe_dict


async def get_recipe(link, name, session):
    html = await get_content(f'https://www.bbc.co.uk{link}', session)
    return process_recipe_html(html, name)


def get_ingredients(soup):
    ingredients_html = soup.findAll("li", {"class": 'recipe-ingredients__list-item'})
    ingredients = []
    for ingredient_cell in ingredients_html:
        ingredients.append(ingredient_cell.text)
    return ingredients


def get_instructions(soup):
    instructions_html = soup.findAll("p", {"class": 'recipe-method__list-item-text'})
    instructions = []
    for instruction_cell in instructions_html:
        instructions.append(instruction_cell.text)
    return instructions


async def run(df, output_path, k=100):
    fieldnames = ['name', 'cook_time', 'prep_time', 'serves', 'ingredients', 'instructions']
    async with ClientSession() as session:
        with open(output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for result in limited_as_completed(save_recipes(df, session), limit=k):
                try:
                    recipe = await result
                except Exception as e:
                    print(e)
                    continue
                if recipe:
                    writer.writerow([recipe.get(field, '') for field in fieldnames])


if __name__ == "__main__":
    _output_path = os.path.join('data', 'recipe_details.csv')
    _df = pd.read_csv(os.path.join('data', 'recipe_pages.csv'), index_col=0)
    _df = _df.drop_duplicates()
    asyncio.run(run(_df, _output_path))
