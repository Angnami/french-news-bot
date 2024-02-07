from datetime import datetime
from src.newsdata_api import download_historical_data 

# Extraire les données historiques de 10 jours (21-01-2024 au 31-01-2024) pour la France à partir de l'API NEWSDATA. 

download_historical_data(
    from_date=datetime(2024,1,21),
    to_date=datetime(2024,1,31)
    )
