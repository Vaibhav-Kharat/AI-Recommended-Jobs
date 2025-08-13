from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

SQLALCHEMY_DATABASE_URL = "postgresql://ryan_ai_aviation_user:riBVsmNCckogNtl9GAGKMe3dNXuKrizD@dpg-d2522j49c44c73b4f3e0-a.oregon-postgres.render.com/ryan_ai_aviation?connection_limit=5&pool_timeout=20&sslmode=require"

engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_table():
    Base.metadata.create_all(bind=engine)