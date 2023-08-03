import csv

EID = 0
STOCK = 1
VEHICLE = 2
STATION = 3
TIME = 4

fields = []
rows= []

with open("./data/time_capture_data.csv", 'r') as file:
	reader = csv.reader(file)
	fields = next(reader)

	for row in reader:
		formatted_row = []
		formatted_row.append(int(row[0]))
		formatted_row.append(int(row[1]))
		formatted_row.append(row[2])
		formatted_row.append(row[3])
		formatted_row.append(int(row[4]))
		rows.append(formatted_row)


new_rows = []
i = 0
while i<len(rows):
	pre = rows[i]
	time = pre[TIME]
	i += 1
	while True:
		if i == len(rows):
			pre[TIME] = time
			new_rows.append(pre)
			break
		next = rows[i]
		if pre[EID] == next[EID]:
			if pre[STOCK] == next[STOCK]:
				if pre[STATION] == next[STATION]:
					time += next[TIME]
					i += 1
					continue
		pre[TIME] = time
		new_rows.append(pre)
		break


with open('./data/production_line_data_final.csv', 'w', newline='') as file:
	writer = csv.writer(file)
	writer.writerow(fields)
	for row in new_rows:
		writer.writerow(row)

print("Write Complete")









