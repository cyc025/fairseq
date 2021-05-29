
import sys

corpus = open(sys.argv[1],'r').readlines()

sent_tup = []
for line in corpus:
    pp_line = line.replace('\n','')
    if pp_line=='': continue
    index = int(pp_line.split('\t')[0].split('-')[1])
    sent = pp_line.split('\t')[2]
    sent_tup.append( (index,sent) )

sent_tup.sort(key=lambda tup: tup[0])
for line in sent_tup:
    print(line[1])
