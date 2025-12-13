import os

import pymongo
from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi


class Document:
    def __init__(self, _id: str, document_number: str):
        self._id = _id
        self.document_number = document_number

    def to_dict(self):
        return {"_id": self._id, "document_number": self.document_number}


class Concept:
    def __init__(self, name: str, documents: list[Document], synonym: list[str] = None, description=None, _id=None):
        self.id = _id or ObjectId()
        self.name = name
        self.documents = documents
        self.synonym = synonym or []
        self.description = description

    def to_dict(self):
        return {
            "_id": self.id,
            "name": self.name,
            "documents": [doc.to_dict() if isinstance(doc, Document) else doc for doc in self.documents],
            "synonym": self.synonym,
            "description": self.description
        }


class Relation:
    def __init__(self, name: str, documents: list[Document], synonym: list[str] = None, description=None, _id=None):
        self.id = _id or ObjectId()
        self.name = name
        self.documents = documents
        self.synonym = synonym or []
        self.description = description

    def to_dict(self):
        return {
            "_id": self.id,
            "name": self.name,
            "documents": [doc.to_dict() if isinstance(doc, Document) else doc for doc in self.documents],
            "synonym": self.synonym,
            "description": self.description
        }


class Triplet:
    def __init__(self, subject_id, relation_id, object_id, subject_name=None, relation_name=None, object_name=None, _id=None):
        self.id = _id or ObjectId()
        self.subject_id = subject_id
        self.relation_id = relation_id
        self.object_id = object_id
        self.subject_name = subject_name
        self.relation_name = relation_name
        self.object_name = object_name

    def to_dict(self):
        return {
            "_id": self.id,
            "subject_id": self.subject_id,
            "relation_id": self.relation_id,
            "object_id": self.object_id,
            "subject_name": self.subject_name,
            "relation_name": self.relation_name,
            "object_name": self.object_name
        }



def init_mongo(uri=None):
    """Initialize MongoDB connection"""
    if uri is None:
        uri = os.getenv("KG_MONGO_URI")

    client = MongoClient(uri, server_api=ServerApi('1'))
    try:
        client.admin.command('ping')
        print("You successfully connected to MongoDB!")
        return client
    except Exception as e:
        print(e)
        return None


def get_or_create_concept(concepts_collection, name, document_id, document_number, synonym_dict=None):
    """Get existing concept or create new one, adding document reference
    Also checks synonyms to find existing concepts"""

    # First try to find by exact name
    concept = concepts_collection.find_one({"name": name})

    # If not found, try to find by synonym
    if not concept:
        concept = concepts_collection.find_one({"synonym": name})

    doc_ref = {"document_id": document_id, "document_number": document_number}

    if concept:
        # Update existing concept: add document if not already present
        concepts_collection.update_one(
            {"_id": concept["_id"]},
            {
                "$addToSet": {"documents": doc_ref}
            }
        )
        return concept["_id"]
    else:
        # Get synonyms list from synonym_dict if available
        synonyms = []
        if synonym_dict and 'synonyms' in synonym_dict and name in synonym_dict['synonyms']:
            synonyms = synonym_dict['synonyms'][name]

        # Create new concept
        new_concept = {
            "_id": ObjectId(),
            "name": name,
            "documents": [doc_ref],
            "synonym": synonyms,
            "description": None
        }
        concepts_collection.insert_one(new_concept)
        return new_concept["_id"]


def get_or_create_relation(relations_collection, name, document_id, document_number, synonym_dict=None):
    """Get existing relation or create new one, adding document reference
    Also checks synonyms to find existing relations"""

    # First try to find by exact name
    relation = relations_collection.find_one({"name": name})

    # If not found, try to find by synonym
    if not relation:
        relation = relations_collection.find_one({"synonym": name})

    doc_ref = {"document_id": document_id, "document_number": document_number}

    if relation:
        # Update existing relation: add document if not already present
        relations_collection.update_one(
            {"_id": relation["_id"]},
            {
                "$addToSet": {"documents": doc_ref}
            }
        )
        return relation["_id"]
    else:
        # Get synonyms list from synonym_dict if available
        synonyms = []
        if synonym_dict and 'synonyms' in synonym_dict and name in synonym_dict['synonyms']:
            synonyms = synonym_dict['synonyms'][name]

        # Create new relation
        new_relation = {
            "_id": ObjectId(),
            "name": name,
            "documents": [doc_ref],
            "synonym": synonyms,
            "description": None
        }
        relations_collection.insert_one(new_relation)
        return new_relation["_id"]


def insert_triplet_batch_mongo(db, triplets_list, metadata, synonym_dict=None):
    """Insert batch of triplets into MongoDB with concept/relation names"""
    concepts_collection = db["concepts"]
    relations_collection = db["relations"]
    triplets_collection = db["triplets"]

    document_id = metadata['document_id']
    document_number = metadata['document_number']

    triplets_to_insert = []

    for triplet in triplets_list:
        c1_name = triplet.get('c1')
        r_name = triplet.get('r')
        c2_name = triplet.get('c2')

        if not c1_name or not r_name or not c2_name:
            continue

        # Get or create concept and relation IDs
        subject_id = get_or_create_concept(concepts_collection, c1_name, document_id, document_number, synonym_dict)
        relation_id = get_or_create_relation(relations_collection, r_name, document_id, document_number, synonym_dict)
        object_id = get_or_create_concept(concepts_collection, c2_name, document_id, document_number, synonym_dict)

        # Create triplet with names
        triplet_doc = {
            "_id": ObjectId(),
            "subject_id": subject_id,
            "relation_id": relation_id,
            "object_id": object_id,
            "subject_name": c1_name,
            "relation_name": r_name,
            "object_name": c2_name,
            "document_id": document_id,
            "document_number": document_number
        }
        triplets_to_insert.append(triplet_doc)

    if triplets_to_insert:
        triplets_collection.insert_many(triplets_to_insert)

    return len(triplets_to_insert)


def delete_all_mongo(db):
    """Delete all documents from all collections"""
    db["concepts"].delete_many({})
    db["relations"].delete_many({})
    db["triplets"].delete_many({})


def create_indexes(db):
    """Create indexes for better query performance"""
    db["concepts"].create_index("name")
    db["concepts"].create_index("synonym")
    db["relations"].create_index("name")
    db["relations"].create_index("synonym")

def update_existing_triplets_with_names(db):
    concepts_collection = db["concepts"]
    relations_collection = db["relations"]
    triplets_collection = db["triplets"]

    triplets = triplets_collection.find({})

    bulk_updates = []

    for triplet in triplets:
        subject_id = triplet.get("subject_id")
        object_id = triplet.get("object_id")
        relation_id = triplet.get("relation_id")

        # Fetch names from concepts/relations
        subject_doc = concepts_collection.find_one({"_id": subject_id})
        object_doc = concepts_collection.find_one({"_id": object_id})
        relation_doc = relations_collection.find_one({"_id": relation_id})

        update_fields = {}
        if subject_doc:
            update_fields["subject_name"] = subject_doc["name"]
        if object_doc:
            update_fields["object_name"] = object_doc["name"]
        if relation_doc:
            update_fields["relation_name"] = relation_doc["name"]

        if update_fields:
            bulk_updates.append(
                pymongo.UpdateOne({"_id": triplet["_id"]}, {"$set": update_fields})
            )

    if bulk_updates:
        result = triplets_collection.bulk_write(bulk_updates)
        print(f"Updated {result.modified_count} triplets with names.")
    else:
        print("No triplets needed updating.")
