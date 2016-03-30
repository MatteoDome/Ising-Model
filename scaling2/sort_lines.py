filenames = ['32_susceptibility.dat', '32_cv.dat', '32_magnetization.dat']

for filename in filenames:
	with open(filename, 'r') as unsorted_file:
		values = []
		for line in unsorted_file:
			values.append((float(line.split()[0]), float(line.split()[1])))

	values = sorted(values)


	with open(filename, 'w') as unsorted_file:
		for value in values:
			unsorted_file.write(str(value[0]) + ' ' + str(value[1]) + '\n')
