from typing import Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy

app = FastAPI()

try:
    nlp = spacy.load('es_core_news_sm')
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo de Spacy: {e}")

class Chat(BaseModel):
    text: str


@app.post("/chat-message")
def analyze_chat(item: Chat):
    try:
        if not item.text:
            raise HTTPException(status_code=400, detail="El texto no puede estar vacío")
        text = item.text.strip()
      
        doc = nlp(text)

        nouns = [chunk.text for chunk in doc.noun_chunks]
        verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]

        noun_groups = {}
        for noun in nouns:
            noun_groups[noun] = noun_groups.get(noun, 0) + 1

        nouns_output = [{"noun": noun, "count": count} for noun, count in noun_groups.items()]
        noun_order = tuple(sorted(nouns_output, key=lambda x: x["count"], reverse=True))

        verb_order = tuple(sorted(verbs))

        entities = [{"text": entity.text, "entity": entity.label_} for entity in doc.ents]
        entity_groups = {}

        for entity in entities:
            group = entity_groups.setdefault(entity["entity"], {'count': 0, 'texts': []})
            group['count'] += 1
            group['texts'].append(entity["text"])
            group['texts'].sort()

        entities_output = [group for group in entity_groups.values()]
        entity_order = tuple(sorted(entities_output, key=lambda x: x["count"], reverse=True))

        data={"nouns": noun_order, "verbs": verb_order, "entities": entity_order}
        return HTTPException(status_code=200,detail={'error':0,'message':'Obtenido correctamente','data':data})
    except Exception as e:
        data={'error':1,'message':str(e),'data':[]}
        raise HTTPException(status_code=500, detail=data)
        
