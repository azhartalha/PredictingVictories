import json
import pickle
from os import walk

tabular_data_path = "C:/Users/syeda/PycharmProjects/PredictingVictories/raw data/tt_logs_flattened/"

dataSet = {}

with open('preprocessed_data.pkl', 'rb') as f:
    dataSet = pickle.load(f)

fileNames = []
for (dirpath, dirnames, fNames) in walk(tabular_data_path):
    fileNames.extend(fNames)
    break

for fileName in fileNames:
    game_id = fileName.partition(".")[0]
    with open(tabular_data_path + fileName) as f:
        data = json.load(f)
        game_state = data['State']
        for obj in game_state:
            if obj['id'] in dataSet[game_id][1]:
                dataSet[game_id][1][obj['id']].append(
                    round(obj['State']['hp']['current'] / obj['State']['hp']['max'], 2))
            elif obj['id'] in dataSet[game_id][2]:
                dataSet[game_id][2][obj['id']].append(
                    round(obj['State']['hp']['current'] / obj['State']['hp']['max'], 2))

with open("tabular_preprocessed_data.pkl", 'wb') as f:
    pickle.dump(dataSet, f)
