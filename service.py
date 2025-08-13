from models import Job
from sqlalchemy.orm import Session
# from schemas import jobCreate

def get_job(db: Session):
    return db.query(Job).all() 