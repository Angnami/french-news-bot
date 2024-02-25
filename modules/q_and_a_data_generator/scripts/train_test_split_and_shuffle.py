import numpy as np
from numpy import random
import json
import argparse
from pathlib import Path

DATASET_DIR = ''
def parseargs()->argparse.Namespace:
    """
    Analyse les arguments de la ligne de commande pour la répartition et le mélange des données.
    Returns:
        argparse.Namespace:un objet contenant les arguments analysés.
    """
    parser = argparse.ArgumentParser(description='Répartition et mélange des données') 
    
    parser.add_argument(
        '--test-samples-per-category',
        type=int,
        default=2,
        help="Choisir le nombre d'observations de test par catégorie"
    )

    
    return parser.parse_args()

# Récupérer les valeurs des arguments
args = parseargs()
test_samples_per_category = args.test_samples_per_category
# Load the data to split
with open('./data/training_data.json', 'r') as f:
    data = json.load(f)
    

# Convert the data to numpy array
data_to_np_arr = np.array(data)
# Split the data into training and testing
# Randomly choose too samples for any category
test_ids = []
start = 0
for i in range(10):
    test_ids.extend(random.choice(range(start,start+10),size=test_samples_per_category,replace=False).tolist())
    start += 10 
# Choose test and train data
test_data = data_to_np_arr[test_ids]
train_data = data_to_np_arr[np.isin(data_to_np_arr,test_ids, invert=True)]
# Shuffle the data and convert them back to list
random.shuffle(train_data)
random.shuffle(test_data)
train_data = train_data.tolist()
test_data = test_data.tolist()

# Save the training and testing data
with open(Path(DATASET_DIR)/'training_data.json','w') as f:
    json.dump(train_data, f)
    
with open(Path(DATASET_DIR)/'testing_data.json','w') as f:
    json.dump(test_data, f)
