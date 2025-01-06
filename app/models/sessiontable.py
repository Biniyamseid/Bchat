from pynamodb.models import Model
from pynamodb.attributes import UnicodeAttribute, UTCDateTimeAttribute, MapAttribute
from datetime import datetime
import os

from app.models.dynamodb import SessionTable





# Ensure the table is created
def initialize_table():
    if not SessionTable.exists():
        SessionTable.create_table(
            read_capacity_units=5,
            write_capacity_units=5,
            wait=True
        )
        print("SessionTable created successfully.")
        
        
    


# Create a new session
def create_session(session_id: str, user_id: str, metadata: dict = None, data: dict = None):
    try:
        session = SessionTable(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            data=data
        )
        session.save()
        print(f"Session {session_id} created successfully.")
        return session
    except Exception as e:
        print(f"Error creating session: {e}")
        raise


# Get a session
def get_session(session_id: str):
    try:
        session = SessionTable.get(session_id)
        print(f"Session {session_id} retrieved successfully.")
        return session
    except SessionTable.DoesNotExist:
        print(f"Session {session_id} does not exist.")
        return None
    except Exception as e:
        print(f"Error retrieving session: {e}")
        raise
