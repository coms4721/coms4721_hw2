

z = 57

def transform(theCount):
	count = 0
	for i in xrange(theCount):
		for j in xrange(theCount):
			if i == j:
				pass
			elif j <= i:
				pass
			else:
				count += 1
	return count

print transform(z) + 57 * 2

