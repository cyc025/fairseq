CLASSPATH=~/stanford-corenlp-4.2.1-models-english.jar
DATAPATH=~/fairseq_cnn_data/cnn_cln

# Tokenize hypothesis and target files.
cat $DATAPATH/test.hypo | java -cp "*" edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $DATAPATH/test.hypo.tokenized
cat $DATAPATH/test.target | java -cp "*" edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $DATAPATH/test.hypo.target
# files2rouge $DATAPATH/test.hypo.tokenized $DATAPATH/test.hypo.target
# Expected output: (ROUGE-2 Average_F: 0.21238)
rouge -f $DATAPATH/test.hypo.tokenized $DATAPATH/test.hypo.target
