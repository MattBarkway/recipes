from itertools import islice

import aiohttp
import asyncio


async def fetch(url, session):
    async with session.get(url) as response:
        return await response.read()


async def bound_fetch(url, session, sem):
    async with sem:
        await fetch(url, session)


def limited_as_completed(coros, limit=100):
    futures = [
        asyncio.create_task(c)
        for c in islice(coros, 0, limit)
    ]

    async def first_to_finish():
        while True:
            await asyncio.sleep(0)
            for f in futures:
                if f.done():
                    futures.remove(f)
                    try:
                        new_f = next(coros)
                        futures.append(
                            asyncio.create_task(new_f))
                    except StopIteration as e:
                        pass
                    return f.result()
    while len(futures) > 0:
        yield first_to_finish()
