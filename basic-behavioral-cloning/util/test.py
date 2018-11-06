import csv

source_file = open('0.csv', 'rb')
reader = csv.reader(source_file)

dest_file = open('1.csv', 'w')
writer = csv.writer(dest_file)
writer.writerow(['right_j0_next','right_j1_next','right_j2_next','right_j3_next','right_j4_next','right_j5_next','right_j6_next'])

reader.next()
reader.next()
reader = list(reader)

for i in range(len(reader)-1):
	if float(reader[i][9]) == float(reader[i+1][9]):
		writer.writerow([reader[i+1][2], reader[i+1][3], reader[i+1][4], reader[i+1][5], reader[i+1][6], reader[i+1][7], reader[i+1][8]])
	else:
		writer.writerow([reader[i][2], reader[i][3], reader[i][4], reader[i][5], reader[i][6], reader[i][7], reader[i][8]])
