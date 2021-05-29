

import sys


# read files
test_en_file = open(sys.argv[1],'r').readlines()
en_file = list(open(sys.argv[2],'r').readlines())
enib_file = list(open(sys.argv[3],'r').readlines())
ib_file = list(open(sys.argv[4],'r').readlines())
xx_file = list(open(sys.argv[5],'r').readlines())

# write files
en_writefile = open('{}.new'.format(sys.argv[2]),'w')
enib_writefile = open('{}.new'.format(sys.argv[3]),'w')
ib_writefile = open('{}.new'.format(sys.argv[4]),'w')
xx_writefile = open('{}.new'.format(sys.argv[5]),'w')


delete_ids = []

# get line numbers
for i,line in enumerate(en_file):
    if line in test_en_file:
        delete_ids.append(i)

# re-write lines
en_file = [ line for i,line in enumerate(en_file) if i not in delete_ids ]
enib_file = [ line for i,line in enumerate(enib_file) if i not in delete_ids ]
ib_file = [ line for i,line in enumerate(ib_file) if i not in delete_ids ]
xx_file = [ line for i,line in enumerate(xx_file) if i not in delete_ids ]

for line in en_file:
    en_writefile.write(line)
for line in ib_file:
    ib_writefile.write(line)
for line in enib_file:
    enib_writefile.write(line)
for line in xx_file:
    xx_writefile.write(line)

ib_writefile.close()
en_writefile.close()
enib_writefile.close()
xx_writefile.close()
