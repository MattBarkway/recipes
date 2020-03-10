import asyncio
from itertools import islice


async def fetch(url, session):
    async with session.get(url) as response:
        return await response.read()


def limited_as_completed(coroutines, limit=100):
    """
    taken from:
    https://www.artificialworlds.net/blog/2017/05/31/python-3-large-numbers-of-tasks-with-limited-concurrency/
    """
    futures = [
        asyncio.create_task(c)
        for c in islice(coroutines, 0, limit)
    ]

    async def first_to_finish():
        while True:
            await asyncio.sleep(0)
            for f in futures:
                if f.done():
                    futures.remove(f)
                    try:
                        new_f = next(coroutines)
                        futures.append(asyncio.create_task(new_f))
                    except StopIteration:
                        pass
                    return f.result()

    while len(futures) > 0:
        yield first_to_finish()


async def get_content(link, session, retries=5):
    attempts = 1
    while attempts <= retries:
        try:
            return await fetch(link, session)
        except Exception as e:
            print(f'request failed, message: \'{e}\' \nsleeping for 5 seconds...')
            await asyncio.sleep(5)
            print(f'retry: {attempts}')
            attempts += 1
    return None
