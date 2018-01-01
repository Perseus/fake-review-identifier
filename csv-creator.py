import json
import csv
data = []
count = 0

with open('Dataset/dataset/review.json',encoding='utf-8') as file:
    csvFile = open('Dataset/dataset/review.csv', 'w', encoding='utf-8')
    output = csv.writer(csvFile, delimiter = '|')
    for line in file:
        count += 1
        data = json.loads(line)
        if count == 1:
            output.writerow(data.keys())
        
        output.writerow(data.values())

        if count == 10000:
            break
        
        