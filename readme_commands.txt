RNNLM TRAINING to generate source sentence representations: (Here $trainfname and $devfname contains only source sentence in form, source i.e. <s> ... </s>)

For training forward RNNLM:
./build_gpu/src/sentrnnlm --dynet_mem 11000 -t $trainfname -d $devfname --parameters $sfwdfname --lstm --layers 1 -h 512 \
-e 10 --lr_epochs 4 --lr_eta 0.1 --lr_eta_decay 2 --minibatch_size 256 --treport 15000 --dreport 60000

For training backward RNNLM:
./build_gpu/src/sentrnnlm --dynet_mem 11000 -t $trainfname -d $devfname --initialise $sfwdfname --parameters $srnnfname \
--lstm --r2l_target --layers 1 -h 512 -e 10 --lr_epochs 4 --lr_eta 0.1 --lr_eta_decay 2 --minibatch_size 256 --treport 15000 --dreport 60000

After training the bidirectional RNNLM, get the source sentence representations for training, dev and test and save them in file: (data should have document ids with it i.e. docid ||| source)
./build_gpu/src/sentrnnlm --dynet_mem 11000 -t $srctrainfname --get_rep --initialise $srnnfname --representations $trainrepfname --r2l_target --lstm --layers 1 -h 512

TRAINING THE DOCUMENT-LEVEL MODEL:

For Pre-training the document-level Memory-to-Context model: (Here the data files should be in format source ||| target)
./build_gpu/src/docmt-memnn --dynet_mem 11000 --train_sent $strainfname --devel_sent $sdevfname --parameters $sentfname --mem_to_context \
--bidirectional --gru --bi_srnn --slayers 1 --tlayers 2 -h 512 -a 256 -e 10 --lr_epochs 4 --lr_eta 0.1 --lr_eta_decay 2 --minibatch_size 256 --treport 15000 --dreport 60000
(Note: use --shared_embeddings for having joint vocabulary and shared embeddings, --use_joint_vocab for joint vocabulary but separate embeddings)

For training the document-level Memory-to-Context model with source memory: (Here the data files should be in format docid ||| source ||| target)
./build_gpu/src/docmt-memnn --dynet_mem 11000 --train_sent $strainfname --train_doc $dtrainfname --devel_doc $ddevfname --doc_src_mem --mem_to_context \
--initialise $sentfname --parameters $modelfname --srnnipt_initialise $trainrepfname --srnnipd_initialise $devrepfname --bidirectional --gru --bi_srnn \
--slayers 1 --tlayers 2 --dropout_enc 0.2 --dropout_dec 0.2 -h 512 -a 256 -e 15 --lr_epochs 1 --lr_eta 0.08 --lr_eta_decay 0.9 --dtreport 800 --ddreport 4000 --docminibatch_size 512

Decoding the document-level Memory-to-Context model with source memory:
./build_gpu/src/docmt-memnn --dynet_mem 11000 --train_sent $strainfname --train_doc $dtrainfname --test_doc $dtestfname --doc_src_mem --mem_to_context \
--initialise $modelfname --srnnipT_initialise $testrepfname --bidirectional --gru --bi_srnn --slayers 1 --tlayers 2 -h 512 -a 256
