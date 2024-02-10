import os

os.environ['OPENAI_API_KEY'] = '<YOUR_OPENAI_API_KEY>'
os.environ['NEWS_DATA_API_KEY'] = '<YOUR_NEWS_DATA_API_KEY>'
os.environ['QDRANT_API_KEY']='<YOUR_QDRANT_API_KEY>'
os.environ['QDRANT_API_URL']='<YOUR_QDRANT_CLUSTER_URL>'

def set_environment():
    variable_dict = globals().items()
    for key, value in variable_dict:
        if 'API' in key or 'ID' in key:
            os.environ[key] = value