from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import Optional, List
import pickle
import numpy as np

from fastapi.middleware.cors import CORSMiddleware
from src.utils import *



 
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ingrs_model = SentenceTransformer('./models/ingrsBERT')
tags_model = SentenceTransformer('all-mpnet-base-v2')
client = init_weaviate_client()
class_name = 'Recipe'
weaviate_limit = 10000
class recipeQuery(BaseModel):
  ingredients: List[str]
  tags: List[str]
  ingredient_exclusions: Optional[List[str]] = None
  tag_exclusions: Optional[List[str]] = None
  limit: Optional[int] = None

@app.get("/")
async def main():
    return {"message": "Welcome to SRRE! Go to /docs for the API documentation"}

@app.post('/search')
async def searchNeighbors(query: recipeQuery):
  embedding = encode_query(query=query.dict(), ingrs_model=ingrs_model, tags_model=tags_model)
  q_dict = {"vector": embedding.tolist()}
  schema = client.schema.get(class_name=class_name)
  properties = [d['name'] for d in schema['properties']] + ["_additional {certainty}"]
  
  res = client.query.get(
    class_name=class_name, 
    properties=properties
    ).with_near_vector(q_dict).with_limit(weaviate_limit).do()
  
  k = query.dict()['limit']
  k = min(10, (10 if not k else k))
  validated = validate_results(query=query.dict(), res=res['data']['Get'][class_name], k=k)
  
  return {
    'neighbors': validated
  }


class IrisSpecies(BaseModel):
  sepal_length: float 
  sepal_width: float 
  petal_length: float 
  petal_width: float
 
@app.post('/predict')
async def predict_species(iris: IrisSpecies):
  data = iris.dict()
  loaded_model = pickle.load(open('/app/src/LRClassifier.pkl', 'rb'))
  data_in = [[data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]]
  prediction = loaded_model.predict(data_in)
  probability = loaded_model.predict_proba(data_in).max()
    
  return {
      'prediction': prediction[0],
      'probability': probability
  }
