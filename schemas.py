from pydantic import BaseModel

class jobBase(BaseModel):
    title: str
    description: str
    location: str
    salaryMin: float
    salaryMax: float

class Config:
    orm_mode = True

class job(jobBase):
    id: str

    class Config:
        from_attribute = True