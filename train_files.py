import os

f = open('./train_files.txt','w')
for i in range(1,999):
	f.write('chess/seq-01 {}\n'.format(int(i)))

f.close()

f = open('./val_files.txt','w')
for i in range(1,999):
	f.write('chess/seq-03 {}\n'.format(int(i)))

f.close()
