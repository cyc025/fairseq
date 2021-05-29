

import sys

from lexicalrichness import LexicalRichness

word_count_sum = 0.
ttr_sum,rttr_sum = 0.,0.
for line in open(sys.argv[1],'r').readlines():
    lex = LexicalRichness(line)
    ttr_sum += lex.ttr
    rttr_sum += lex.rttr
    word_count_sum += lex.words

length = len(open(sys.argv[1],'r').readlines())
print('Ave. BPE Word Count {0}'.format(word_count_sum/length))
print('Ave. TTR {0}'.format(ttr_sum/length))
print('Ave. RTTR {0}'.format(rttr_sum/length))
