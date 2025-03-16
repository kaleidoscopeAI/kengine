from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Analysis(Base):
    """Analysis results table"""
    __tablename__ = 'analyses'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    file_path = Column(String)
    file_type = Column(String)
    results = Column(JSON)
    status = Column(String)

class QuantumState(Base):
    """Quantum simulation states"""
    __tablename__ = 'quantum_states'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    n_qubits = Column(Integer)
    state_vector = Column(JSON)
    measurement = Column(JSON)
