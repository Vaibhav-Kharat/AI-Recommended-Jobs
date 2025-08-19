from db import Base 
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
import datetime

class Job(Base):
    __tablename__ = "Job"

    id = Column(String, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String, index=True)
    location = Column(String, index=True)
    salaryMin = Column(Float, index=True)
    salaryMax = Column(Float, index=True)
    type = Column(String, index=True)
    vacancies = Column(Integer, index=True)

class User(Base):
    __tablename__ = "User"   

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    fullName = Column("full_name", String)
    image = Column(String)
    role = Column(String, default="USER")
    createdAt = Column(DateTime, default=datetime.datetime.utcnow)
    updatedAt = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
