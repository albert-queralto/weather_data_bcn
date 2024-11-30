from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class HashPassword:
    def create(self, password: str) -> str:
        return pwd_context.hash(password)

    def verify(self, password: str, hashed_password: str) -> bool:
        return pwd_context.verify(password, hashed_password)