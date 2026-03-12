from sentence_transformers import SentenceTransformer

_model=None
def get_embedding(text):
    global _model
    if _model is None: # model is loaded only once (lazy initialization) / also loaing large model several times is a hideous task
        _model=SentenceTransformer("all-MiniLM-L6-v2")
    return _model.encode(text)
