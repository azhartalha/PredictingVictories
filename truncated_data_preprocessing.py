import json
import pickle
from os import walk

truncated_data_path = "C:/Users/syeda/PycharmProjects/PredictingVictories/raw data/tt_logs_short/"

fileNames = []
for (dirpath, dirnames, fNames) in walk(truncated_data_path):
    fileNames.extend(fNames)
    break

dataSet = {}

for fileName in fileNames:
    game_id = fileName.partition(".")[0]
    # Label - who won the game
    # Player 0's data
    # Player 1's data
    dataSet[game_id] = [-1, {}, {}]
    with open(truncated_data_path + fileName) as f:
        data = json.load(f)
        objects = data['RegisteredObjects']
        for obj in objects:
            if obj['classId'] is not None and obj['classId'].startswith("unt"):
                if obj["ownerId"] == 0:
                    dataSet[game_id][1][obj["id"]] = [1]
                else:
                    dataSet[game_id][2][obj["id"]] = [1]

        for state in data["States"]:
            if state['t'] == 0.0:
                continue

            if "hp" in state:
                if state['id'] in dataSet[game_id][1]:
                    for playerId in dataSet[game_id][1]:
                        if state['id'] == playerId:
                            dataSet[game_id][1][playerId].append(round(state['hp']['current']/state['hp']['max'], 2))
                        else:
                            dataSet[game_id][1][playerId].append(dataSet[game_id][1][playerId][-1])
                    for playerId in dataSet[game_id][2]:
                        dataSet[game_id][2][playerId].append(dataSet[game_id][2][playerId][-1])

                elif state['id'] in dataSet[game_id][2]:
                    for playerId in dataSet[game_id][2]:
                        if state['id'] == playerId:
                            dataSet[game_id][2][playerId].append(round(state['hp']['current']/state['hp']['max'], 2))
                        else:
                            dataSet[game_id][2][playerId].append(dataSet[game_id][2][playerId][-1])
                    for playerId in dataSet[game_id][1]:
                        dataSet[game_id][1][playerId].append(dataSet[game_id][1][playerId][-1])


with open("preprocessed_data.pkl", 'wb') as f:
    pickle.dump(dataSet, f)

