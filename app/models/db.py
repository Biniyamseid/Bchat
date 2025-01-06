from pynamodb.models import Model
from pynamodb.attributes import UnicodeAttribute, MapAttribute, ListAttribute
from pynamodb.exceptions import TableError, PynamoDBException
from pynamodb.connection import Connection
from dotenv import load_dotenv
import os

# Kelechi wrote here
# Load environment variables from .env file
load_dotenv()

# PynamoDB Model for the DynamoDB Table
class DataMap(MapAttribute):
    id = UnicodeAttribute(null=True)
    name = UnicodeAttribute(null=True)
    email = UnicodeAttribute(null=True)
    phone_number = UnicodeAttribute(null=True)
    messages = ListAttribute(default=list)
    content = UnicodeAttribute(null=True)
    uploaded_at = UnicodeAttribute(null=True)
    
class SMSTable(Model):
    class Meta:
        table_name = "smstable"
        region = os.getenv("AWS_REGION")
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    id = UnicodeAttribute(hash_key=True)  # Partition key
    data = DataMap()

def health_check():
    try:
        # Get all tables in DynamoDB
        conn = Connection(
            region=os.getenv("AWS_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        tables = conn.list_tables().get("TableNames", [])
        if SMSTable.Meta.table_name not in tables:
            # Create table if it doesn't exist
            SMSTable.create_table(
                read_capacity_units=5,
                write_capacity_units=5,
                wait=True,
            )
            return {
                "status": "created",
                "message": f"Table '{SMSTable.Meta.table_name}' was created successfully",
            }
        return {
            "status": "success",
            "message": f"Connection to DynamoDB is successful. Existing tables: {tables}",
        }
    except TableError as e:
        return {"status": "error", "message": f"Table error: {str(e)}"}
    except PynamoDBException as e:
        return {"status": "error", "message": str(e)}

def add_item(item: dict):
    try:
        # Expect `id` in item payload
        item_id = item.get("id")
        if not item_id:
            raise ValueError("Missing 'id' in item")

        # Retrieve existing item if it exists
        try:
            record = SMSTable.get(hash_key=item_id)
            existing_messages = record.data.messages
        except SMSTable.DoesNotExist:
            existing_messages = []

        # Append new messages to existing messages
        new_messages = item.get("data", {}).get("messages", [])
        updated_messages = existing_messages + new_messages

        # Save the item to DynamoDB
        record = SMSTable(
            id=item_id,
            data=DataMap(
                id=item_id,
                name=item.get("data", {}).get("name"),
                email=item.get("data", {}).get("email"),
                phone_number = item.get("data", {}).get("phone_number"),
                messages=updated_messages,
                content=item.get("data", {}).get("content"),  # Add content to the item
                uploaded_at=item.get("data", {}).get("uploaded_at")  # 
            )
        )
        record.save()
        return {"status": "Item added", "item": item}
    except PynamoDBException as e:
        raise Exception(str(e))

def get_item(item_id: str):
    try:
        # Retrieve item by `id`
        record = SMSTable.get(hash_key=item_id)
        return {"id": record.id, "data": record.data.as_dict()}
    except SMSTable.DoesNotExist:
        raise ValueError("Item not found")
    except PynamoDBException as e:
        raise Exception(str(e))

def get_or_create_item(item_id: str):
    try:
        # Retrieve item by `id`
        record = SMSTable.get(hash_key=item_id)
        return {"id": record.id, "data": record.data.as_dict()}
    except SMSTable.DoesNotExist:
        # Create a new item if it does not exist
        new_item = SMSTable(
            id=item_id,
            data=DataMap(
                id=item_id,
                name="",
                email="",
                phone_number="",
                messages=[]
            )
        )
        new_item.save()
        return {"id": new_item.id, "data": new_item.data.as_dict()}
    except PynamoDBException as e:
        raise Exception(str(e))

def update_item(item_id: str, new_data: dict):
    try:
        # Retrieve existing item
        record = SMSTable.get(hash_key=item_id)
        
        # Update fields
        if "name" in new_data:
            record.data.name = new_data["name"]
        if "email" in new_data:
            record.data.email = new_data["email"]
        if "phone_number" in new_data:
            record.data.phone_number = new_data["phone_number"]
        if "phone" in new_data:
            record.data.phone = new_data["phone"]
        
        # Append new messages to existing messages
        new_messages = new_data.get("messages", [])
        record.data.messages.extend(new_messages)
        
        # Save the updated item to DynamoDB
        record.save()
        return {"status": "Item updated", "item": {"id": record.id, "data": record.data.as_dict()}}
    except SMSTable.DoesNotExist:
        raise ValueError("Item not found")
    except PynamoDBException as e:
        raise Exception(str(e))



    










