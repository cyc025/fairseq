

import sys

xx = sys.argv[6]

# id file
target_id_file = open(sys.argv[1],'r').readlines()
# target id file
id_file = open('{0}/{1}'.format(xx,sys.argv[2]),'r').readlines()
# infobox
ib_file = open('{0}/{1}'.format(xx,sys.argv[3]),'r').readlines()
# source
en_file = open('{0}/{1}'.format(xx,sys.argv[4]),'r').readlines()
# xx
xx_file = open('{0}/{1}'.format(xx,sys.argv[5]),'r').readlines()

# write files
# infobox
ib_writefile = open('testset/test.ib','w')
# source
en_writefile = open('testset/test.en','w')
# xx
xx_writefile = open('testset/test.{}'.format(xx),'w')

for target_id in target_id_file:
    for _id_,ib,en,xx in zip(id_file,ib_file,en_file,xx_file):
        if target_id==_id_:
            ib_writefile.write(ib)
            en_writefile.write(en)
            xx_writefile.write(xx)

ib_writefile.close()
en_writefile.close()
xx_writefile.close()
