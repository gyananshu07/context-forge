import asyncio
from db.session import AsyncSessionLocal
from db.models import User
import secrets
import string


async def seed_user():
    async with AsyncSessionLocal() as session:
        # Generate a random email and password
        alphabet = string.ascii_letters + string.digits
        random_str = "".join(secrets.choice(alphabet) for i in range(8))
        email = "admin@contextforge.com"
        # Just putting a dummy hashed password for now
        hashed_password = f"dummy_hash_{random_str}"

        new_user = User(email=email, hashed_password=hashed_password)
        session.add(new_user)
        await session.commit()
        print(f"Created random admin user with email: {email}")


if __name__ == "__main__":
    asyncio.run(seed_user())
