"""
Profile Manager Module for the Reconciliation App.
Handles SQLite database operations for matching profiles.
"""
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

Base = declarative_base()


class Profile(Base):
    """SQLAlchemy model for matching profiles."""
    __tablename__ = 'profiles'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False)
    sales_headers = Column(Text)  # JSON array of column names
    settlement_headers = Column(Text)  # JSON array of column names
    match_rules_json = Column(Text)  # JSON array of match rules
    populate_rules_json = Column(Text)  # JSON array of populate rules
    output_columns = Column(Text)  # JSON array of selected output columns
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_last_used = Column(Boolean, default=False)


class AppSettings(Base):
    """SQLAlchemy model for app settings."""
    __tablename__ = 'app_settings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(255), unique=True, nullable=False)
    value = Column(Text)


@dataclass
class MatchRule:
    """Data class for match rules."""
    id: str
    sales_column: str
    settlement_column: str
    match_type: str  # 'exact', 'fuzzy', 'numeric_range', 'date_range'
    tolerance: Optional[float] = None  # For numeric/fuzzy matching
    fuzzy_threshold: Optional[int] = None  # For fuzzy matching (0-100)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MatchRule':
        return cls(**data)


@dataclass
class PopulateRule:
    """Data class for population rules."""
    id: str
    target_sheet: str  # 'Sales' or 'Settlement'
    target_column_letter: str  # Excel column letter (A, B, C, ...)
    source_sheet: str  # 'Sales', 'Settlement', or 'Credit Note'
    source_column: str  # Column name from source sheet
    target_column_name: Optional[str] = ''  # Custom name for the target column
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PopulateRule':
        # Handle legacy format
        if 'target_column' in data and 'target_column_letter' not in data:
            data['target_column_letter'] = data.pop('target_column', 'A')
        if 'source_file' in data and 'source_sheet' not in data:
            source_file = data.pop('source_file', 'settlement')
            data['source_sheet'] = 'Settlement' if source_file == 'settlement' else 'Sales'
        if 'target_sheet' not in data:
            data['target_sheet'] = 'Sales'
        if 'condition' in data:
            data.pop('condition')  # Remove legacy field
        if 'target_column_name' not in data:
            data['target_column_name'] = ''  # Default empty for legacy profiles
        return cls(**data)


@dataclass
class ProfileData:
    """Complete profile data structure."""
    id: Optional[int]
    name: str
    sales_headers: List[str]
    settlement_headers: List[str]
    match_rules: List[MatchRule]
    populate_rules: List[PopulateRule]
    output_columns: List[str]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ProfileManager:
    """
    Manages SQLite database operations for matching profiles.
    Uses SQLAlchemy for ORM operations.
    """
    
    def __init__(self, db_path: str = 'data/profiles.db'):
        """
        Initialize ProfileManager with database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"ProfileManager initialized with database: {db_path}")
    
    def create_profile(
        self,
        name: str,
        sales_headers: List[str],
        settlement_headers: List[str],
        match_rules: List[MatchRule],
        populate_rules: List[PopulateRule],
        output_columns: List[str]
    ) -> Optional[int]:
        """
        Create a new profile.
        
        Args:
            name: Profile name (unique)
            sales_headers: Column names from sales file
            settlement_headers: Column names from settlement file
            match_rules: List of matching rules
            populate_rules: List of population rules
            output_columns: Selected output columns
            
        Returns:
            Profile ID on success, None on failure
        """
        session = self.Session()
        try:
            profile = Profile(
                name=name,
                sales_headers=json.dumps(sales_headers),
                settlement_headers=json.dumps(settlement_headers),
                match_rules_json=json.dumps([r.to_dict() for r in match_rules]),
                populate_rules_json=json.dumps([r.to_dict() for r in populate_rules]),
                output_columns=json.dumps(output_columns)
            )
            session.add(profile)
            session.commit()
            logger.info(f"Created profile: {name} (ID: {profile.id})")
            return profile.id
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error creating profile: {str(e)}")
            return None
        finally:
            session.close()
    
    def get_all_profiles(self) -> List[Dict[str, Any]]:
        """
        Get all saved profiles.
        
        Returns:
            List of profile dictionaries with id and name
        """
        session = self.Session()
        try:
            profiles = session.query(Profile).order_by(Profile.updated_at.desc()).all()
            return [
                {
                    'id': p.id,
                    'name': p.name,
                    'created_at': p.created_at,
                    'updated_at': p.updated_at,
                    'is_last_used': p.is_last_used
                }
                for p in profiles
            ]
        finally:
            session.close()
    
    def get_profile(self, profile_id: int) -> Optional[ProfileData]:
        """
        Get a profile by ID.
        
        Args:
            profile_id: Profile ID to fetch
            
        Returns:
            ProfileData object or None if not found
        """
        session = self.Session()
        try:
            profile = session.query(Profile).filter(Profile.id == profile_id).first()
            if not profile:
                return None
            
            return ProfileData(
                id=profile.id,
                name=profile.name,
                sales_headers=json.loads(profile.sales_headers or '[]'),
                settlement_headers=json.loads(profile.settlement_headers or '[]'),
                match_rules=[
                    MatchRule.from_dict(r) 
                    for r in json.loads(profile.match_rules_json or '[]')
                ],
                populate_rules=[
                    PopulateRule.from_dict(r) 
                    for r in json.loads(profile.populate_rules_json or '[]')
                ],
                output_columns=json.loads(profile.output_columns or '[]'),
                created_at=profile.created_at,
                updated_at=profile.updated_at
            )
        finally:
            session.close()
    
    def get_profile_by_name(self, name: str) -> Optional[ProfileData]:
        """
        Get a profile by name.
        
        Args:
            name: Profile name to fetch
            
        Returns:
            ProfileData object or None if not found
        """
        session = self.Session()
        try:
            profile = session.query(Profile).filter(Profile.name == name).first()
            if not profile:
                return None
            return self.get_profile(profile.id)
        finally:
            session.close()
    
    def update_profile(
        self,
        profile_id: int,
        name: Optional[str] = None,
        sales_headers: Optional[List[str]] = None,
        settlement_headers: Optional[List[str]] = None,
        match_rules: Optional[List[MatchRule]] = None,
        populate_rules: Optional[List[PopulateRule]] = None,
        output_columns: Optional[List[str]] = None
    ) -> bool:
        """
        Update an existing profile.
        
        Args:
            profile_id: ID of profile to update
            name: New name (optional)
            sales_headers: New sales headers (optional)
            settlement_headers: New settlement headers (optional)
            match_rules: New match rules (optional)
            populate_rules: New populate rules (optional)
            output_columns: New output columns (optional)
            
        Returns:
            True on success, False on failure
        """
        session = self.Session()
        try:
            profile = session.query(Profile).filter(Profile.id == profile_id).first()
            if not profile:
                return False
            
            if name is not None:
                profile.name = name
            if sales_headers is not None:
                profile.sales_headers = json.dumps(sales_headers)
            if settlement_headers is not None:
                profile.settlement_headers = json.dumps(settlement_headers)
            if match_rules is not None:
                profile.match_rules_json = json.dumps([r.to_dict() for r in match_rules])
            if populate_rules is not None:
                profile.populate_rules_json = json.dumps([r.to_dict() for r in populate_rules])
            if output_columns is not None:
                profile.output_columns = json.dumps(output_columns)
            
            profile.updated_at = datetime.utcnow()
            session.commit()
            logger.info(f"Updated profile ID: {profile_id}")
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error updating profile: {str(e)}")
            return False
        finally:
            session.close()
    
    def delete_profile(self, profile_id: int) -> bool:
        """
        Delete a profile by ID.
        
        Args:
            profile_id: ID of profile to delete
            
        Returns:
            True on success, False on failure
        """
        session = self.Session()
        try:
            profile = session.query(Profile).filter(Profile.id == profile_id).first()
            if not profile:
                return False
            
            session.delete(profile)
            session.commit()
            logger.info(f"Deleted profile ID: {profile_id}")
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error deleting profile: {str(e)}")
            return False
        finally:
            session.close()
    
    def set_last_used(self, profile_id: int) -> bool:
        """
        Mark a profile as last used.
        
        Args:
            profile_id: ID of profile to mark
            
        Returns:
            True on success, False on failure
        """
        session = self.Session()
        try:
            # Clear previous last used
            session.query(Profile).update({Profile.is_last_used: False})
            
            # Set new last used
            profile = session.query(Profile).filter(Profile.id == profile_id).first()
            if profile:
                profile.is_last_used = True
                session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error setting last used: {str(e)}")
            return False
        finally:
            session.close()
    
    def get_last_used_profile(self) -> Optional[ProfileData]:
        """
        Get the last used profile.
        
        Returns:
            ProfileData object or None if no last used profile
        """
        session = self.Session()
        try:
            profile = session.query(Profile).filter(Profile.is_last_used == True).first()
            if not profile:
                return None
            return self.get_profile(profile.id)
        finally:
            session.close()
    
    def profile_exists(self, name: str) -> bool:
        """
        Check if a profile with the given name exists.
        
        Args:
            name: Profile name to check
            
        Returns:
            True if exists, False otherwise
        """
        session = self.Session()
        try:
            return session.query(Profile).filter(Profile.name == name).first() is not None
        finally:
            session.close()
