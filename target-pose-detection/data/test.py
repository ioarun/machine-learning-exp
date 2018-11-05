import csv

file = open('0.csv', 'rb')
reader = csv.reader(file)
reader.next()
file_ = open('0_new.csv', 'w')
writer = csv.writer(file_)
writer.writerow(['new_y'])

list_ = list(reader)

for row in list_:
	val = row[10]
	if float(val) < 0.0:
		val = float(val) + 0.4565

	writer.writerow([val])



