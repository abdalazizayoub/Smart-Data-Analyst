# sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship 
from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime

Base = declarative_base() 
engine = create_engine('sqlite:///smart_da.db')
Session = sessionmaker(bind=engine) 
session = Session()  

class Datasets(Base):
    __tablename__ = "datasets"  
    
    id = Column(Integer, primary_key=True, autoincrement=True) 
    dataset_path = Column(String, unique=True)
    dataset_name = Column(String)
    description = Column(String, default=None)
    pushed_at = Column(DateTime, default=datetime.utcnow) 
    
    plots = relationship("Plot", back_populates="dataset") 

    def __repr__(self):
        return f"<Dataset(dataset='{self.dataset_name}', description='{self.description}')>"

class Results(Base): 
    __tablename__ = "results"  
    
    id = Column(Integer, primary_key=True)
    descrpitive_stat = Column(String) 
    feature_importance = Column(String)
    correlation = Column(String)
    audio_path = Column(String)
    dataset_id = Column(Integer, ForeignKey("datasets.id")) 
    
    dataset = relationship("Datasets", back_populates="plots")

Base.metadata.create_all(engine)