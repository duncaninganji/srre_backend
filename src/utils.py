import numpy as np
import weaviate
from sentence_transformers import SentenceTransformer
from typing import Tuple

def encode_query(
  query: dict, 
  ingrs_model: SentenceTransformer, 
  tags_model: SentenceTransformer) -> np.ndarray:
  
  ingrs = ' '.join(list(set(query['ingredients']))) + '\n' 
  tags = ' '.join(list(set(query['tags']))) + '\n'
  ingrs_exclusions = ' '.join(list(set(query['ingredient_exclusions'])))
  tags_exclusions = ' '.join(list(set(query['tag_exclusions']))) 
  ingrs_emb = ingrs_model.encode(ingrs)
  tags_emb = tags_model.encode(tags)
  
  if ingrs_exclusions:
    ingrs_emb -= ingrs_model.encode(ingrs_exclusions) 

  if tags_exclusions:
    tags_emb -= tags_model.encode(tags_exclusions)
  
  return np.hstack((tags_emb, ingrs_emb))
    
def init_weaviate_client() -> weaviate.client.Client:
  weaviate_container_url = 'http://20.120.216.46:8080/'
  weaviate_local_url = '0.0.0.0:8080/'
  my_credentials = weaviate.auth.AuthClientPassword('', '')
  
  return weaviate.Client(weaviate_container_url, auth_client_secret=my_credentials)

def validate_results(query: dict, res: list, k: int) -> Tuple[list, int]:
  validated = []
  for _dict in res:
    tags = _dict['tags']
    ingrs = _dict['model_ingredients']
    intersection_ingrs = list(set(ingrs) & set(query['ingredients']))
    intersection_tags = list(set(tags) & set(query['tags']))
    intersection = (0.667 * len(intersection_ingrs)) + (0.333 * len(intersection_tags))
    if intersection > 0:      
      validated.append((_dict, intersection))

  validated.sort(key=lambda tup: tup[1], reverse=True)
  
  return validated[:k]