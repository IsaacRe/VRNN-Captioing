from sqlalchemy import Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine

Base = declarative_base()

class Input(Base):
    __tablename__ = 'input'

    id = Column(Integer, primary_key = True)
    name = Column(String(250), nullable = False)
    ground_true = Column(String(250), nullable = False)
    translation = Column(String(250), nullable = True)

engine = create_engine('sqlite:///course.db')
 

Base.metadata.create_all(engine)
