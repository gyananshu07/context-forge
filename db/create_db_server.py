import asyncio
import asyncpg

async def main():
    try:
        # Connect to the default 'postgres' database
        conn = await asyncpg.connect(user='db_user', password='db_user_2026', host='localhost', port=5432, database='postgres')
        # We need to run CREATE DATABASE outside of a transaction block
        await conn.execute('CREATE DATABASE ragdb')
        print("Database 'ragdb' created successfully!")
        await conn.close()
    except asyncpg.exceptions.DuplicateDatabaseError:
        print("Database 'ragdb' already exists.")
    except Exception as e:
        print(f"Failed to create database: {e}")

if __name__ == '__main__':
    asyncio.run(main())
