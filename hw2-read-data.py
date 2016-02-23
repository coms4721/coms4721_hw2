from scipy.io import loadmat
data = loadmat('spam_fixed.mat')

labels = data['labels']
data   = data['data']

for i in data:
	print i
	#print i['labels']

for i in labels:
	print i
#print data