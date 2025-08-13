from db import Base 
from sqlalchemy import Column, Integer, String, Float

class Job(Base):
    __tablename__ = "Job"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String, index=True)
    location = Column(String, index=True)
    salaryMin = Column(Float, index=True)
    salaryMax = Column(Float, index=True)
    type = Column(String, index=True)
    vacancies = Column(Integer, index=True)
