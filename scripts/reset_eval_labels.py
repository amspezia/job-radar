import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from job_radar.config import settings


async def main() -> None:
    engine = create_async_engine(settings.database_url)
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE eval_labels;"))
    print("All eval labels cleared.")


asyncio.run(main())
