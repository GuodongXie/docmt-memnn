#include "docmt-memnn.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace dynet;
using namespace boost::program_options;

unsigned SLAYERS = 1; // 2
unsigned TLAYERS = 1; // 2
unsigned HIDDEN_DIM = 64;  // 1024
unsigned ALIGN_DIM = 32;   // 128
unsigned SRC_VOCAB_SIZE = 0;
unsigned TGT_VOCAB_SIZE = 0;
unsigned MINIBATCH_SIZE = 1;
unsigned DOCMINIBATCH_SIZE = 1;

bool DEBUGGING_FLAG = false;

unsigned TREPORT = 5000;
unsigned DREPORT = 20000;
unsigned DTREPORT = 500;
unsigned DDREPORT = 2000;

dynet::Dict sd;
dynet::Dict td;

typedef vector<int> Sentence;
typedef pair<Sentence, Sentence> SentencePair;
typedef tuple<Sentence, Sentence, int> SentencePairID;
typedef vector<SentencePairID> Corpus;	//sentence-level corpus

typedef vector<SentencePair> Document;
typedef vector<Document> DocCorpus;	//document-level corpus

typedef pair<Sentence, int> SentenceID;
typedef vector<SentenceID> SourceCorpus;  //sentence-level corpus for decoding

typedef vector<Sentence> SourceDoc;
typedef vector<SourceDoc> SourceDocCorpus;  //document-level corpus for decoding

#define WTF(expression) \
	std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;
#define KTHXBYE(expression) \
	std::cout << *cg.get_value(expression.i) << std::endl;

#define LOLCAT(expression) \
	WTF(expression) \
	KTHXBYE(expression) 

void Initialise(Model &model, const string &filename);
void LoadRepresentations(vector<vector<vector<dynet::real>>> &sent_rep, const string &rep_file);

inline size_t Calc_Size(const Sentence & src, const Sentence & trg);
inline size_t Create_MiniBatches(const Corpus& cor, size_t max_size
	, std::vector<std::vector<Sentence> > & train_src_minibatch
	, std::vector<std::vector<Sentence> > & train_trg_minibatch
	, std::vector<size_t> & train_ids_minibatch);
inline void Create_DocMiniBatch(const Document& doc, size_t max_size,
                                std::vector<std::vector<Sentence>>& train_src_minidoc, std::vector<std::vector<Sentence>>& train_trg_minidoc,
                                std::vector<std::vector<unsigned int>>& train_mini_sids);

template <class DMT_t>
void TrainSentMTModel(Model &model, DMT_t &dmt, Corpus &training, Corpus &devel,
                      Trainer &sgd, string out_file, bool curriculum, int max_epochs, int lr_epochs);
template <class DMT_t>
void TrainDocMTSrcModel(Model &model, DMT_t &dmt, vector<vector<vector<dynet::real>>> &tsent_repi, vector<vector<vector<dynet::real>>> &dsent_repi,
                     DocCorpus &training_doc, DocCorpus &devel_doc, Trainer &sgd, string out_file, int max_epochs, int lr_epochs);
template <class DMT_t>
void TrainDocMTTrgModel(Model &model, DMT_t &dmt, DocCorpus &training_doc, DocCorpus &devel_doc, Trainer &sgd,
                        string out_file, int max_epochs, int lr_epochs, bool use_gold);
template <class DMT_t>
void TrainDocMTSrcTrgModel(Model &model, DMT_t &dmt, vector<vector<vector<dynet::real>>> &tsent_repi,
                           vector<vector<vector<dynet::real>>> &dsent_repi, DocCorpus &training_doc, DocCorpus &devel_doc,
                           Trainer &sgd, string out_file, int max_epochs, int lr_epochs, bool use_gold);

template <class DMT_t>
void TrainSentMTModel_Batch(Model &model, DMT_t &dmt, Corpus &training, Corpus &devel,
                            Trainer &sgd, string out_file, bool curriculum, int max_epochs, int lr_epochs);
template <class DMT_t>
void TrainDocMTSrcModel_Batch(Model &model, DMT_t &dmt, vector<vector<vector<dynet::real>>> &tsent_repi, vector<vector<vector<dynet::real>>> &dsent_repi,
                           DocCorpus &training_doc, DocCorpus &devel_doc, Trainer &sgd, string out_file, int max_epochs, int lr_epochs);
template <class DMT_t>
void TrainDocMTTrgModel_Batch(Model &model, DMT_t &dmt, DocCorpus &training_doc, DocCorpus &devel_doc, Trainer &sgd,
                              string out_file, int max_epochs, int lr_epochs, bool use_gold);
template <class DMT_t>
void TrainDocMTSrcTrgModel_Batch(Model &model, DMT_t &dmt, vector<vector<vector<dynet::real>>> &tsent_repi, vector<vector<vector<dynet::real>>> &dsent_repi,
                                 DocCorpus &training_doc, DocCorpus &devel_doc, Trainer &sgd, string out_file, int max_epochs, int lr_epochs, bool use_gold);

template <class DMT_t> vector<vector<vector<dynet::real>>> ComputeDocTrgRep(Model &model, DMT_t &dmt, DocCorpus &training_doc);
template <class DMT_t> vector<vector<vector<dynet::real>>> ComputeDocTrueTrgRep(Model &model, DMT_t &dmt, DocCorpus &training_doc);
template <class DMT_t> vector<vector<vector<dynet::real>>> ComputeDocTrg_SrcRep(Model &model, DMT_t &dmt, vector<vector<vector<dynet::real>>> &sent_repi, DocCorpus &training_doc);
template <class DMT_t> vector<vector<vector<dynet::real>>> ComputeDocTrueTrg_SrcRep(Model &model, DMT_t &dmt, vector<vector<vector<dynet::real>>> &sent_repi, DocCorpus &training_doc);

template <class DMT_t> void Test_Rescore(Model &model, DMT_t &dmt, Corpus &testing);
template <class DMT_t> void Test_Decode(Model &model, DMT_t &dmt, std::string test_file, bool use_joint_vocab);

template <class DMT_t> void TestDocSrc_Rescore(Model &model, DMT_t &dmt, vector<vector<vector<dynet::real>>> &Tsent_repi, DocCorpus &testing_doc);
template <class DMT_t> void TestDocSrc_Decode(Model &model, DMT_t &dmt, vector<vector<vector<dynet::real>>> &Tsent_repi, string test_file, bool use_joint_vocab);
template <class DMT_t> void TestDocTrg_Rescore(Model &model, DMT_t &dmt, DocCorpus &testing_doc);
template <class DMT_t> void TestDocTrg_Decode(Model &model, DMT_t &dmt, string test_file, bool iter_decode, bool use_joint_vocab);
template <class DMT_t> void TestDocSrcTrg_Rescore(Model &model, DMT_t &dmt, vector<vector<vector<dynet::real>>> &Tsent_repi, DocCorpus &testing_doc);
template <class DMT_t> void TestDocSrcTrg_Decode(Model &model, DMT_t &dmt, vector<vector<vector<dynet::real>>> &Tsent_repi, string test_file, bool iter_decode, bool use_joint_vocab);

const Sentence* GetContext(const Corpus &corpus, unsigned i);

//for sentence-level corpus
Corpus Read_Corpus(const string &filename, bool use_joint_vocab);
SourceDoc Read_TestCorpus(const string &filename);
//for document-level corpus
Corpus Read_DocCorpus(const string &filename, bool use_joint_vocab);
DocCorpus Read_DocCorpus(Corpus &corpus);
SourceCorpus Read_TestDocCorpus(const string &filename, bool use_joint_vocab);
SourceDocCorpus Read_TestDocCorpus(SourceCorpus &scorpus);
void Read_Numbered_Sentence(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int> &ids);
void Read_Numbered_Sentence_Pair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, std::vector<int> &ids);

template <class rnn_t>
int main_body(variables_map vm);

int main(int argc, char** argv) {
	dynet::initialize(argc, argv);

	// command line processing
	variables_map vm;
	options_description opts("Allowed options");
	opts.add_options()
		("help", "print help message")
		//-----------------------------------------
        ("train_sent", value<string>(), "file containing training sentences, with each line consisting of source ||| target.")
        ("devel_sent", value<string>(), "file containing development sentences for sentence-level corpus.")
        ("test_sent", value<string>(), "file containing testing sentences for sentence-level corpus")
        ("train_doc", value<string>(), "file containing training sentences, with each line consisting of docid ||| source ||| target.")
        ("devel_doc", value<string>(), "file containing development sentences for document-level corpus.")
        ("test_doc", value<string>(), "file containing testing sentences for document-level corpus.")
        ("slen_limit", value<unsigned>()->default_value(0), "limit the sentence length (either source or target); no by default")
        //-----------------------------------------
        ("use_joint_vocab", "whether or not to use a joint vocabulary for source and target (w/BPE)")
        ("shared_embeddings", "whether or not to share the embeddings (if true then use joint vocabulary)")
        //-----------------------------------------
		("rescore", "rescore (source, target) sentence/document pairs in testing, default: translate source only")
		("iterative_decode", "use iterative decoding when using target memory component")
		//("beam,b", value<unsigned>()->default_value(0), "size of beam in decoding; 0=greedy")
		//-----------------------------------------
		("minibatch_size", value<unsigned>()->default_value(1), "impose the minibatch size for training (support both GPU and CPU); no by default")
        ("docminibatch_size", value<unsigned>()->default_value(1), "impose the minibatch size for training Document-level model (support both GPU and CPU); no by default")
        //-----------------------------------------
		("sgd_trainer", value<unsigned>()->default_value(0), "use specific SGD trainer (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam)")
		("sparse_updates", value<bool>()->default_value(true), "enable/disable sparse update(s) for lookup parameter(s); true by default")
        ("grad-clip-threshold", value<float>()->default_value(5.f), "use specific gradient clipping threshold (https://arxiv.org/pdf/1211.5063.pdf); 5 by default")
        //-----------------------------------------
		("doc_src_mem", "perform doc-level MT algorithm with source document context")
        ("doc_trg_mem", "perform doc-level MT algorithm with target document context")
        ("local_mem", "consider only local context of source/target sentences during training instead of MemNN: default not")
        ("use_gold", "use gold output to compute target representations")
		//-----------------------------------------
        ("mem_to_context", "add memory component to the decoder")
        ("mem_to_output", "add memory component to the softmax")
        // -----------------------------------------
        ("initialise", value<string>(), "load initial parameters for sentence/document attentional model from file")
		("parameters", value<string>(), "save best parameters for sentence/document attentional model to this file")
        //-----------------------------------------
        ("srnnipt_initialise", value<string>(), "load representations for training set for source memory sentence-level RNN component from file")
        ("srnnipd_initialise", value<string>(), "load representations for dev set for source memory sentence-level RNN component from file")
        ("srnnipT_initialise", value<string>(), "load representations for test set for source memory sentence-level RNN component from file")
        //-----------------------------------------
		("slayers", value<unsigned>()->default_value(SLAYERS), "use <num> layers for source RNN components")
		("tlayers", value<unsigned>()->default_value(TLAYERS), "use <num> layers for target RNN components")
        ("align,a", value<unsigned>()->default_value(ALIGN_DIM), "use <num> dimensions for alignment projection")
		("hidden,h", value<unsigned>()->default_value(HIDDEN_DIM), "use <num> dimensions for recurrent hidden states")
		//-----------------------------------------
		("dropout_enc", value<float>()->default_value(0.f), "use dropout (Gal et al., 2016) for RNN encoder(s), e.g., 0.5 (input=0.5;hidden=0.5;cell=0.5) for LSTM; none by default")
		("dropout_dec", value<float>()->default_value(0.f), "use dropout (Gal et al., 2016) for RNN decoder, e.g., 0.5 (input=0.5;hidden=0.5;cell=0.5) for LSTM; none by default")
        ("dropout_df", value<float>()->default_value(0.f), "apply dropout technique (Gal et al., 2015) for forward document RNN")
        ("dropout_db", value<float>()->default_value(0.f), "apply dropout technique (Gal et al., 2015) for backward document RNN")
        //-----------------------------------------
		("epochs,e", value<int>()->default_value(20), "maximum number of training epochs")
		//-----------------------------------------
		("lr_eta", value<float>()->default_value(0.01f), "SGD learning rate value (e.g., 0.1 for simple SGD trainer or smaller 0.001 for ADAM trainer)")
		("lr_eta_decay", value<float>()->default_value(1.0f), "SGD learning rate decay value")
		//-----------------------------------------
		("lr_epochs", value<int>()->default_value(1), "no. of epochs for starting learning rate annealing (e.g., halving)")
		//-----------------------------------------
		("gru", "use Gated Recurrent Unit (GRU) for recurrent structure; default RNN")
		("lstm", "use Long Short Term Memory (LSTM) for recurrent structure; default RNN")
		("vlstm", "use Vanilla Long Short Term Memory (VLSTM) for recurrent structure; default RNN")
		("dglstm", "use Depth-Gated Long Short Term Memory (DGLSTM) (Kaisheng et al., 2015; https://arxiv.org/abs/1508.03790) for recurrent structure; default RNN") // FIXME: add this to dynet?
		//-----------------------------------------
		("bidirectional", "use bidirectional recurrent hidden states as source embeddings, rather than word embeddings")
        ("bi_srnn", "use bidirectional RNN for generating sentence representations, rather than a simple forward RNN as sentence representation")
        //-----------------------------------------
		("curriculum", "use 'curriculum' style learning, focusing on easy problems (e.g., shorter sentences) in earlier epochs")
		//-----------------------------------------
		("treport", value<unsigned>()->default_value(5000), "no. of training instances for reporting current model status on training data")
		("dreport", value<unsigned>()->default_value(20000), "no. of training instances for reporting current model status on development data (dreport = N * treport)")
		("dtreport", value<unsigned>()->default_value(500), "no. of training documents for reporting current model status on training data")
		("ddreport", value<unsigned>()->default_value(2000), "no. of training documents for reporting current model status on development data (ddreport = N * dtreport)")
		//-----------------------------------------
        ("debug", "enable/disable simpler debugging by immediate computing mode or checking validity (refers to http://dynet.readthedocs.io/en/latest/debugging.html)")// for CPU only
	;
	store(parse_command_line(argc, argv, opts), vm);
	notify(vm);

	cerr << "PID=" << ::getpid() << endl;

    if (!vm.count("doc_src_mem") && !vm.count("doc_trg_mem")){
        if (vm.count("help") || vm.count("train_sent") != 1 || (vm.count("devel_sent") != 1 && vm.count("test_sent") != 1)) {
            cout << opts << "\n";
            return 1;
        }
    }
    else if (vm.count("doc_src_mem") || vm.count("doc_trg_mem")){
        if (vm.count("help") || vm.count("train_sent") != 1 || vm.count("train_doc") != 1 ||
                (vm.count("devel_doc") != 1 && vm.count("test_doc") != 1)) {
            cout << opts << "\n";
            return 1;
        }
    }

	if (vm.count("lstm"))
		return main_body<LSTMBuilder>(vm);
	else if (vm.count("vlstm"))
		return main_body<VanillaLSTMBuilder>(vm);
	//else if (vm.count("dglstm"))
		//return main_body<DGLSTMBuilder>(vm);
	else if (vm.count("gru"))
		return main_body<GRUBuilder>(vm);
	else
		return main_body<SimpleRNNBuilder>(vm);
}

template <class rnn_t>
int main_body(variables_map vm)
{
	DEBUGGING_FLAG = vm.count("debug");

	kSRC_SOS = sd.convert("<s>");
	kSRC_EOS = sd.convert("</s>");
	kTGT_SOS = td.convert("<s>");
	kTGT_EOS = td.convert("</s>");

	SLAYERS = vm["slayers"].as<unsigned>();
	TLAYERS = vm["tlayers"].as<unsigned>();
    ALIGN_DIM = vm["align"].as<unsigned>();
	HIDDEN_DIM = vm["hidden"].as<unsigned>(); 

	TREPORT = vm["treport"].as<unsigned>(); 
	DREPORT = vm["dreport"].as<unsigned>(); 
	if (DREPORT % TREPORT != 0) assert("dreport must be divisible by treport.");// to ensure the reporting on development data
	DTREPORT = vm["dtreport"].as<unsigned>();
	DDREPORT = vm["ddreport"].as<unsigned>();
	if (DDREPORT % DTREPORT != 0) assert("ddreport must be divisible by dtreport.");// to ensure the reporting on development data

	MINIBATCH_SIZE = vm["minibatch_size"].as<unsigned>();
    DOCMINIBATCH_SIZE = vm["docminibatch_size"].as<unsigned>();

	bool bidir = vm.count("bidirectional");
    bool use_joint_vocab = vm.count("use_joint_vocab") || vm.count("shared_embeddings");
    bool shared_embeddings = vm.count("shared_embeddings");

    bool bi_srnn = vm.count("bi_srnn");
    bool src_mem = vm.count("doc_src_mem");
    bool trg_mem = vm.count("doc_trg_mem");
    bool loco_mem = vm.count("local_mem");
    bool mem_to_ctx = vm.count("mem_to_context");
    bool mem_to_op = vm.count("mem_to_output");
    bool use_gold = vm.count("use_gold");
	bool iter_decode = vm.count("iterative_decode");

	string flavour = "RNN";
	if (vm.count("lstm"))
	    flavour = "LSTM";
	else if (vm.count("gru"))
	    flavour = "GRU";

    Corpus training, devel, testing, training_sent, devel_sent, testing_sent;
    DocCorpus training_doc, devel_doc, testing_doc;
    cerr << "Reading sentence training data from " << vm["train_sent"].as<string>() << "...\n";
    training_sent = Read_Corpus(vm["train_sent"].as<string>(), use_joint_vocab);//contains sentence-level bigger corpus
    kSRC_UNK = sd.convert("<unk>");// add <unk> if does not exist!
    kTGT_UNK = td.convert("<unk>");
    sd.freeze(); // no new word types allowed
    td.freeze(); // no new word types allowed

    SRC_VOCAB_SIZE = sd.size();
    TGT_VOCAB_SIZE = td.size();

    if (!src_mem && !trg_mem) {
        if (vm.count("devel_sent")) {
            cerr << "Reading sentence dev data from " << vm["devel_sent"].as<string>() << "...\n";
            devel_sent = Read_Corpus(vm["devel_sent"].as<string>(), use_joint_vocab);
        }

        if (vm.count("test_sent") && vm.count("rescore")) {
            // otherwise "test" file is assumed just to contain source language strings
            cerr << "Reading sentence test examples from " << vm["test_sent"].as<string>() << endl;
            testing_sent = Read_Corpus(vm["test_sent"].as<string>(), use_joint_vocab);
        }
    }
    else {
        cerr << "Reading document training data from " << vm["train_doc"].as<string>() << "...\n";
        training = Read_DocCorpus(vm["train_doc"].as<string>(), use_joint_vocab);//contains sentence-level parallel corpus
        training_doc = Read_DocCorpus(training);//contains document-level parallel corpus

        if (vm.count("devel_doc")) {
            cerr << "Reading document dev data from " << vm["devel_doc"].as<string>() << "...\n";
            devel = Read_DocCorpus(vm["devel_doc"].as<string>(), use_joint_vocab);
            devel_doc = Read_DocCorpus(devel);
        }

        if (vm.count("test_doc") && vm.count("rescore")) {
            // otherwise "test" file is assumed just to contain source language strings
            cerr << "Reading document test examples from " << vm["test_doc"].as<string>() << endl;
            testing = Read_DocCorpus(vm["test_doc"].as<string>(), use_joint_vocab);
            testing_doc = Read_DocCorpus(testing);
        }
    }

    string fname;
	if (vm.count("parameters"))
		fname = vm["parameters"].as<string>();
	else {
			ostringstream os;
			os << "dmt"
			   << '_' << SLAYERS
			   << '_' << TLAYERS
			   << '_' << HIDDEN_DIM
			   << '_' << ALIGN_DIM
			   << '_' << flavour
			   << "_b" << bidir
			   << "-pid" << getpid() << ".params";
			fname = os.str();
		}

    // training phase
    if (!vm.count("test_sent") || !vm.count("test_doc")){
        if (!src_mem && !trg_mem)
            cerr << "Sentence-level parameters will be written to: " << fname << endl;
        else
            cerr << "Document-level parameters will be written to: " << fname << endl;
    }

	Model model;
	Trainer* sgd = nullptr;
	unsigned sgd_type = vm["sgd_trainer"].as<unsigned>();
	if (sgd_type == 1)
		sgd = new MomentumSGDTrainer(model, vm["lr_eta"].as<float>());
	else if (sgd_type == 2)
		sgd = new AdagradTrainer(model, vm["lr_eta"].as<float>());
	else if (sgd_type == 3)
		sgd = new AdadeltaTrainer(model);
	else if (sgd_type == 4)
		sgd = new AdamTrainer(model, vm["lr_eta"].as<float>());
	else if (sgd_type == 0)//Vanilla SGD trainer
		sgd = new SimpleSGDTrainer(model, vm["lr_eta"].as<float>());
	else
	   assert("Unknown SGD trainer type! (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam)");
    sgd->clip_threshold = vm["grad-clip-threshold"].as<float>();
	sgd->eta_decay = vm["lr_eta_decay"].as<float>();
	sgd->sparse_updates_enabled = vm["sparse_updates"].as<bool>();
	if (!sgd->sparse_updates_enabled)
		cerr << "Sparse updates for lookup parameter(s) to be disabled!" << endl;

	cerr << "%% Using " << flavour << " recurrent units" << endl;
	DocMTMemNNModel<rnn_t> dmt(&model, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE
		, SLAYERS, TLAYERS, HIDDEN_DIM, ALIGN_DIM, shared_embeddings
		, bidir, bi_srnn, src_mem, trg_mem, loco_mem, mem_to_ctx, mem_to_op);

	dmt.Set_Dropout(vm["dropout_enc"].as<float>(), vm["dropout_dec"].as<float>());
    dmt.Set_Dropout_DocRNN(vm["dropout_df"].as<float>(), vm["dropout_db"].as<float>());

    //test sentence-level or test/train document-level attentional model
	if (vm.count("initialise"))
		Initialise(model, vm["initialise"].as<string>());

    cerr << "Count of model parameters: " << model.parameter_count() << endl << endl;
	
    vector<vector<vector<dynet::real>>> tsent_repi, dsent_repi, Tsent_repi;

    //load the sentence representations if using document source context
    if (src_mem){
        if (!vm.count("test_doc")){
            if (vm.count("srnnipt_initialise")){
                cerr << "*Loading sentence representations of training set for input memory..." << endl;
                LoadRepresentations(tsent_repi, vm["srnnipt_initialise"].as<string>());
            }
            else{
                cerr << "No sentence representations of training set provided!" << endl;
                return 1;
            }
            if (vm.count("srnnipd_initialise")){
                cerr << "*Loading sentence representations of dev set for input memory..." << endl;
                LoadRepresentations(dsent_repi, vm["srnnipd_initialise"].as<string>());
            }
            else{
                cerr << "No sentence representations of dev set provided!" << endl;
                return 1;
            }
        }
        else {
			if (vm.count("srnnipT_initialise")) {
				cerr << "*Loading sentence representations of test set for input memory..." << endl;
				LoadRepresentations(Tsent_repi, vm["srnnipT_initialise"].as<string>());
			}
            else{
                cerr << "No sentence representations of test set provided!" << endl;
                return 1;
            }
        }
    }

    if (!vm.count("test_sent") && !vm.count("train_doc")) {
        TrainSentMTModel_Batch(model, dmt, training_sent, devel_sent, *sgd, fname, vm.count("curriculum"),
                               vm["epochs"].as<int>(), vm["lr_epochs"].as<int>());
    }
    else if (vm.count("test_sent")){
        cerr << "Testing sentence-level attentional model..." << endl;
        if (vm.count("rescore"))//e.g., compute perplexity scores
            Test_Rescore(model, dmt, testing_sent);
        else { // test/decode
            Test_Decode(model, dmt, vm["test_sent"].as<string>(), use_joint_vocab);
        }
        cerr << "Sentence-level decoding completed!" << endl;
    }
    else if (!vm.count("test_doc") && vm.count("train_doc")) {
        if (src_mem && !trg_mem)
            TrainDocMTSrcModel_Batch(model, dmt, tsent_repi, dsent_repi, training_doc, devel_doc, *sgd, fname,
                                     vm["epochs"].as<int>(), vm["lr_epochs"].as<int>());
        else if (!src_mem && trg_mem)
            TrainDocMTTrgModel_Batch(model, dmt, training_doc, devel_doc, *sgd, fname, vm["epochs"].as<int>(), vm["lr_epochs"].as<int>(), use_gold);
        else if (src_mem && trg_mem)
            TrainDocMTSrcTrgModel_Batch(model, dmt, tsent_repi, dsent_repi, training_doc, devel_doc, *sgd, fname,
                                        vm["epochs"].as<int>(), vm["lr_epochs"].as<int>(), use_gold);
    }
    else if (vm.count("test_doc")){
        if (src_mem && !trg_mem) {
            cerr << "Testing Document-level attentional model with source memory..." << endl;
            if (vm.count("rescore"))//e.g., compute perplexity scores
                TestDocSrc_Rescore(model, dmt, Tsent_repi, testing_doc);
            else { // test/decode
                TestDocSrc_Decode(model, dmt, Tsent_repi, vm["test_doc"].as<string>(), use_joint_vocab);
            }
            cerr << "Document-level decoding using source memory completed!" << endl;
        }
        else if (!src_mem && trg_mem) {
            cerr << "Testing Document-level attentional model with only target memory..." << endl;
            if (vm.count("rescore"))//e.g., compute perplexity scores
                TestDocTrg_Rescore(model, dmt, testing_doc);
            else { // test/decode
                TestDocTrg_Decode(model, dmt, vm["test_doc"].as<string>(), iter_decode, use_joint_vocab);
            }
            cerr << "Document-level decoding using target memory completed!" << endl;
        }
        else if (src_mem && trg_mem) {
            cerr << "Testing Document-level attentional model with source and target memory..." << endl;
            if (vm.count("rescore"))//e.g., compute perplexity scores
                TestDocSrcTrg_Rescore(model, dmt, Tsent_repi, testing_doc);
            else { // test/decode
                TestDocSrcTrg_Decode(model, dmt, Tsent_repi, vm["test_doc"].as<string>(), iter_decode, use_joint_vocab);
            }
            cerr << "Document-level decoding using source and target memory completed!" << endl;
        }
    }

    cerr << "Cleaning up..." << endl;
	delete sgd;
	//dynet::cleanup();

	return EXIT_SUCCESS;
}

void LoadRepresentations(vector<vector<vector<dynet::real>>> &sent_rep, const string &rep_file)
{
    ifstream in(rep_file);
    boost::archive::text_iarchive ia(in);

    ia >> sent_rep;
    in.close();
}

template <class DMT_t>
void Test_Rescore(Model &model, DMT_t &dmt, Corpus &testing)
{
	Sentence ssent, tsent;
	int docid;
	ModelStats tstats;
	for (unsigned i = 0; i < testing.size(); ++i) {
		tie(ssent, tsent, docid) = testing[i];

	    ComputationGraph cg;
		auto i_xent = dmt.BuildSentMTGraph(ssent, tsent, cg, tstats);

	    double loss = as_scalar(cg.forward(i_xent));
		cout << i << " |||";
	    for (auto &w: ssent)
		    cout << " " << sd.convert(w);
	    cout << " |||";
	    for (auto &w: tsent)
		    cout << " " << td.convert(w);
	    cout << " ||| " << (loss / (tsent.size()-1)) << endl;

	    tstats.loss += loss;

	}

	cout << "\n***TEST E=" << (tstats.loss / tstats.words_tgt) << " ppl=" << exp(tstats.loss / tstats.words_tgt) << ' ' << endl;
	return;
}

template <class DMT_t>
void Test_Decode(Model &model, DMT_t &dmt, string test_file, bool use_joint_vocab)
{
    cerr << "Reading sentence-level test examples from " << test_file << endl;
    SourceDoc testing = Read_TestCorpus(test_file);

    for (unsigned i = 0; i < testing.size(); ++i) {
        ComputationGraph cg;
        Sentence source = testing[i];
        std::vector<int> target;
        target = dmt.Greedy_Decode(source, cg, td);

        bool first = true;
        for (auto &w: target) {
            if (!first) cout << " ";
            cout << td.convert(w);
            first = false;
        }
        cout << endl;

    }
}

template <class DMT_t>
void TestDocSrc_Rescore(Model &model, DMT_t &dmt, vector<vector<vector<dynet::real>>> &Tsent_repi, DocCorpus &testing_doc)
{
	vector<Sentence> vssent, vtsent;
	ModelStats tstats;

	for (unsigned i = 0; i < testing_doc.size(); ++i) {
		// build graph for this document
		const unsigned tdlen = testing_doc[i].size();
		vector<vector<dynet::real>> vtsent_repi = Tsent_repi[i];

        for (unsigned dl = 0; dl < tdlen; ++dl){
			vssent.push_back(get<0>(testing_doc[i].at(dl)));
			vtsent.push_back(get<1>(testing_doc[i].at(dl)));
		}
        
        cout << "\n*Document " << i << endl;
		double doc_loss = 0;
		unsigned doc_words_tgt = 0;
		for (unsigned j = 0; j < vssent.size(); ++j){
			ComputationGraph cg;

            auto i_xent = dmt.BuildDocMTSrcGraph(vssent[j], vtsent[j], vtsent_repi, j, cg, tstats);
            double loss = as_scalar(cg.forward(i_xent));

			cout << j << " |||";
			for (auto &w: vssent[j])
				cout << " " << sd.convert(w);
			cout << " |||";
			for (auto &w: vtsent[j])
				cout << " " << td.convert(w);
			cout << " ||| " << (loss / (vtsent[j].size()-1)) << endl;

			doc_loss += loss;
			doc_words_tgt += vtsent[j].size() - 1;
		}

		cout << "*Test document " << i << " E= " << (doc_loss / doc_words_tgt) << endl;
		tstats.loss += doc_loss;

		vssent.clear();
		vtsent.clear();
		//if (verbose)
		//	cerr << "chug " << lno++ << "\r" << flush;
	}

	cout << "\n***TEST E=" << (tstats.loss / tstats.words_tgt) << " ppl=" << exp(tstats.loss / tstats.words_tgt) << ' ' << endl;
	return;
}

template <class DMT_t>
void TestDocSrc_Decode(Model &model, DMT_t &dmt, vector<vector<vector<dynet::real>>> &Tsent_repi, string test_file, bool use_joint_vocab)
{
    cerr << "Reading test examples from " << test_file << endl;
    SourceCorpus testing = Read_TestDocCorpus(test_file, use_joint_vocab);
    SourceDocCorpus testing_doc = Read_TestDocCorpus(testing);

    vector<Sentence> vssent;
    //get translations based on document-level model
    for (unsigned i = 0; i < testing_doc.size(); ++i) {
        cout << "<d>" << endl;
        const unsigned tdlen = testing_doc[i].size();
		vector<vector<dynet::real>> vtsent_repi = Tsent_repi[i];

        for (unsigned dl = 0; dl < tdlen; ++dl)
            vssent.push_back(testing_doc[i].at(dl));

        for (unsigned dl = 0; dl < vssent.size(); ++dl){
            ComputationGraph cg;
            Sentence source = vssent[dl];
            std::vector<int> target;
            target = dmt.GreedyDocSrc_Decode(source, vtsent_repi, dl, cg, td);

            bool first = true;
            for (auto &w: target) {
                if (!first) cout << " ";
                cout << td.convert(w);
                first = false;
            }
            cout << endl;
        }

        vssent.clear();
    }
    return;
}

template <class DMT_t>
void TestDocTrg_Rescore(Model &model, DMT_t &dmt, DocCorpus &testing_doc)
{
    vector<Sentence> vssent, vtsent;
    ModelStats tstats;

    vector<vector<vector<dynet::real>>> test_trgsent_rep = ComputeDocTrgRep(model, dmt, testing_doc);

    for (unsigned i = 0; i < testing_doc.size(); ++i) {
        // build graph for this document
        const unsigned tdlen = testing_doc[i].size();
        vector<vector<dynet::real>> vtrgsent_rep = test_trgsent_rep[i];

        for (unsigned dl = 0; dl < tdlen; ++dl){
            vssent.push_back(get<0>(testing_doc[i].at(dl)));
            vtsent.push_back(get<1>(testing_doc[i].at(dl)));
        }

        cout << "\n*Document " << i << endl;
        double doc_loss = 0;
        unsigned doc_words_tgt = 0;
        for (unsigned j = 0; j < vssent.size(); ++j){
            ComputationGraph cg;

            auto i_xent = dmt.BuildDocMTTrgGraph(vssent[j], vtsent[j], vtrgsent_rep, j, cg, tstats);
            double loss = as_scalar(cg.forward(i_xent));

            cout << j << " |||";
            for (auto &w: vssent[j])
                cout << " " << sd.convert(w);
            cout << " |||";
            for (auto &w: vtsent[j])
                cout << " " << td.convert(w);
            cout << " ||| " << (loss / (vtsent[j].size()-1)) << endl;

            doc_loss += loss;
            doc_words_tgt += vtsent[j].size() - 1;
        }

        cout << "*Test document " << i << " E= " << (doc_loss / doc_words_tgt) << endl;
        tstats.loss += doc_loss;

        vssent.clear();
        vtsent.clear();
        //if (verbose)
        //	cerr << "chug " << lno++ << "\r" << flush;
    }

    cout << "\n***TEST E=" << (tstats.loss / tstats.words_tgt) << " ppl=" << exp(tstats.loss / tstats.words_tgt) << ' ' << endl;
    return;
}

template <class DMT_t>
void TestDocTrg_Decode(Model &model, DMT_t &dmt, string test_file, bool iter_decode, bool use_joint_vocab)
{
	cerr << "Reading test examples from " << test_file << endl;
	SourceCorpus testing = Read_TestDocCorpus(test_file, use_joint_vocab);
	SourceDocCorpus testing_doc = Read_TestDocCorpus(testing);

	vector<Sentence> vssent;
	vector<vector<Sentence>> target_doc;
	vector<vector<vector<dynet::real>>> test_trgsent_rep;

    cout << "Iteration 1" << endl;
	//get representations for the targets
	for (unsigned i = 0; i < testing_doc.size(); ++i) {
		vector<vector<dynet::real>> trgdoc_rep;

		const unsigned tdlen = testing_doc[i].size();
		for (unsigned dl = 0; dl < tdlen; ++dl)
			vssent.push_back(testing_doc[i].at(dl));

		for (unsigned dl = 0; dl < vssent.size(); ++dl) {
			ComputationGraph cg;
			Sentence source = vssent[dl];
			trgdoc_rep.push_back(dmt.GetTrgRepresentations(source, cg, td));
		}

		vssent.clear();
		test_trgsent_rep.push_back(trgdoc_rep);
	}

	//get translations based on document-level model
	for (unsigned i = 0; i < testing_doc.size(); ++i) {
		cout << "<d>" << endl;
		const unsigned tdlen = testing_doc[i].size();
		vector<vector<dynet::real>> vtrgsent_rep = test_trgsent_rep[i];
		vector<Sentence> temp_trg;

		for (unsigned dl = 0; dl < tdlen; ++dl)
			vssent.push_back(testing_doc[i].at(dl));

		for (unsigned dl = 0; dl < vssent.size(); ++dl) {
			ComputationGraph cg;
			Sentence source = vssent[dl];
			std::vector<int> target;
			target = dmt.GreedyDocTrg_Decode(source, vtrgsent_rep, dl, cg, td);
			temp_trg.push_back(target);

			bool first = true;
			for (auto &w : target) {
				if (!first) cout << " ";
				cout << td.convert(w);
				first = false;
			}
			cout << endl;
		}

		vssent.clear();
		target_doc.push_back(temp_trg);
	}

	if (iter_decode) {
		unsigned iter = 1;
		while (iter < 10) {
			vector<Sentence> vssent;
			vector<Sentence> vtsent;
			vector<vector<vector<dynet::real>>> test_trgsent_rep;

			cout << "Iteration " << (iter + 1) << endl;
			//get representations for the new targets
			for (unsigned i = 0; i < testing_doc.size(); ++i) {
				vector<vector<dynet::real>> trgdoc_rep;

				const unsigned tdlen = testing_doc[i].size();
				for (unsigned dl = 0; dl < tdlen; ++dl) {
					vssent.push_back(testing_doc[i].at(dl));
					vtsent.push_back(target_doc[i].at(dl));
				}
					
				for (unsigned dl = 0; dl < vssent.size(); ++dl) {
					ComputationGraph cg;
					Sentence source = vssent[dl];
					Sentence target = vtsent[dl];
					trgdoc_rep.push_back(dmt.GetTrueTrgRepresentations(source, target, cg));
				}

				vssent.clear();
				test_trgsent_rep.push_back(trgdoc_rep);
			}

			//get translations based on document-level model
			for (unsigned i = 0; i < testing_doc.size(); ++i) {
				cout << "<d>" << endl;
				const unsigned tdlen = testing_doc[i].size();
				vector<vector<dynet::real>> vtrgsent_rep = test_trgsent_rep[i];
				vector<Sentence> temp_trg;

				for (unsigned dl = 0; dl < tdlen; ++dl)
					vssent.push_back(testing_doc[i].at(dl));

				for (unsigned dl = 0; dl < vssent.size(); ++dl) {
					ComputationGraph cg;
					Sentence source = vssent[dl];
					std::vector<int> target;
					target = dmt.GreedyDocTrg_Decode(source, vtrgsent_rep, dl, cg, td);
					temp_trg.push_back(target);

					bool first = true;
					for (auto &w : target) {
						if (!first) cout << " ";
						cout << td.convert(w);
						first = false;
					}
					cout << endl;
				}

				vssent.clear();
				target_doc[i] = temp_trg;
			}
			iter++;
		}
	}
	return;
}

template <class DMT_t>
void TestDocSrcTrg_Rescore(Model &model, DMT_t &dmt, vector<vector<vector<dynet::real>>> &Tsent_repi, DocCorpus &testing_doc)
{
    vector<Sentence> vssent, vtsent;
    ModelStats tstats;

    vector<vector<vector<dynet::real>>> test_trgsent_rep = ComputeDocTrg_SrcRep(model, dmt, Tsent_repi, testing_doc);

    for (unsigned i = 0; i < testing_doc.size(); ++i) {
        // build graph for this document
        const unsigned tdlen = testing_doc[i].size();
        vector<vector<dynet::real>> vtsent_repi = Tsent_repi[i];
        vector<vector<dynet::real>> vtrgsent_rep = test_trgsent_rep[i];

        for (unsigned dl = 0; dl < tdlen; ++dl){
            vssent.push_back(get<0>(testing_doc[i].at(dl)));
            vtsent.push_back(get<1>(testing_doc[i].at(dl)));
        }

        cout << "\n*Document " << i << endl;
        double doc_loss = 0;
        unsigned doc_words_tgt = 0;
        for (unsigned j = 0; j < vssent.size(); ++j){
            ComputationGraph cg;

            auto i_xent = dmt.BuildDocMTSrcTrgGraph(vssent[j], vtsent[j], vtsent_repi, vtrgsent_rep, j, cg, tstats);
            double loss = as_scalar(cg.forward(i_xent));

            cout << j << " |||";
            for (auto &w: vssent[j])
                cout << " " << sd.convert(w);
            cout << " |||";
            for (auto &w: vtsent[j])
                cout << " " << td.convert(w);
            cout << " ||| " << (loss / (vtsent[j].size()-1)) << endl;

            doc_loss += loss;
            doc_words_tgt += vtsent[j].size() - 1;
        }

        cout << "*Test document " << i << " E= " << (doc_loss / doc_words_tgt) << endl;
        tstats.loss += doc_loss;

        vssent.clear();
        vtsent.clear();
        //if (verbose)
        //	cerr << "chug " << lno++ << "\r" << flush;
    }

    cout << "\n***TEST E=" << (tstats.loss / tstats.words_tgt) << " ppl=" << exp(tstats.loss / tstats.words_tgt) << ' ' << endl;
    return;
}

template <class DMT_t>
void TestDocSrcTrg_Decode(Model &model, DMT_t &dmt, vector<vector<vector<dynet::real>>> &Tsent_repi, string test_file, bool iter_decode, bool use_joint_vocab)
{
	cerr << "Reading test examples from " << test_file << endl;
	SourceCorpus testing = Read_TestDocCorpus(test_file, use_joint_vocab);
	SourceDocCorpus testing_doc = Read_TestDocCorpus(testing);

	vector<Sentence> vssent;
	vector<vector<Sentence>> target_doc;
	vector<vector<vector<dynet::real>>> test_trgsent_rep;

    cout << "Iteration 1" << endl;
	//get representations for the targets
	for (unsigned i = 0; i < testing_doc.size(); ++i) {
		vector<vector<dynet::real>> trgdoc_rep;

		const unsigned tdlen = testing_doc[i].size();
		for (unsigned dl = 0; dl < tdlen; ++dl)
			vssent.push_back(testing_doc[i].at(dl));
		vector<vector<dynet::real>> vtsent_repi = Tsent_repi[i];

		for (unsigned dl = 0; dl < vssent.size(); ++dl) {
			ComputationGraph cg;
			Sentence source = vssent[dl];
			trgdoc_rep.push_back(dmt.GetTrg_SrcRepresentations(source, vtsent_repi, dl, cg, td));
		}

		vssent.clear();
		test_trgsent_rep.push_back(trgdoc_rep);
	}

	//get translations based on document-level model
	for (unsigned i = 0; i < testing_doc.size(); ++i) {
		cout << "<d>" << endl;
		const unsigned tdlen = testing_doc[i].size();
		vector<vector<dynet::real>> vtsent_repi = Tsent_repi[i];
		vector<vector<dynet::real>> vtrgsent_rep = test_trgsent_rep[i];
		vector<Sentence> temp_trg;

		for (unsigned dl = 0; dl < tdlen; ++dl)
			vssent.push_back(testing_doc[i].at(dl));

		for (unsigned dl = 0; dl < vssent.size(); ++dl) {
			ComputationGraph cg;
			Sentence source = vssent[dl];
			std::vector<int> target;
			target = dmt.GreedyDocSrcTrg_Decode(source, vtsent_repi, vtrgsent_rep, dl, cg, td);
			temp_trg.push_back(target);

			bool first = true;
			for (auto &w : target) {
				if (!first) cout << " ";
				cout << td.convert(w);
				first = false;
			}
			cout << endl;
		}

		vssent.clear();
		target_doc.push_back(temp_trg);
	}

	if (iter_decode) {
		unsigned iter = 1;
		while (iter < 10) {
			vector<Sentence> vssent;
			vector<Sentence> vtsent;
			vector<vector<vector<dynet::real>>> test_trgsent_rep;

			cout << "Iteration " << (iter + 1) << endl;
			//get representations for the new targets
			for (unsigned i = 0; i < testing_doc.size(); ++i) {
				vector<vector<dynet::real>> trgdoc_rep;

				const unsigned tdlen = testing_doc[i].size();
				for (unsigned dl = 0; dl < tdlen; ++dl) {
					vssent.push_back(testing_doc[i].at(dl));
					vtsent.push_back(target_doc[i].at(dl));
				}
				vector<vector<dynet::real>> vtsent_repi = Tsent_repi[i];

				for (unsigned dl = 0; dl < vssent.size(); ++dl) {
					ComputationGraph cg;
					Sentence source = vssent[dl];
					Sentence target = vtsent[dl];
					trgdoc_rep.push_back(dmt.GetTrueTrg_SrcRepresentations(source, target, vtsent_repi, dl, cg));
				}

				vssent.clear();
				test_trgsent_rep.push_back(trgdoc_rep);
			}

			//get translations based on document-level model
			for (unsigned i = 0; i < testing_doc.size(); ++i) {
				cout << "<d>" << endl;
				const unsigned tdlen = testing_doc[i].size();
				vector<vector<dynet::real>> vtsent_repi = Tsent_repi[i];
				vector<vector<dynet::real>> vtrgsent_rep = test_trgsent_rep[i];
				vector<Sentence> temp_trg;

				for (unsigned dl = 0; dl < tdlen; ++dl)
					vssent.push_back(testing_doc[i].at(dl));

				for (unsigned dl = 0; dl < vssent.size(); ++dl) {
					ComputationGraph cg;
					Sentence source = vssent[dl];
					std::vector<int> target;
					target = dmt.GreedyDocSrcTrg_Decode(source, vtsent_repi, vtrgsent_rep, dl, cg, td);
					temp_trg.push_back(target);

					bool first = true;
					for (auto &w : target) {
						if (!first) cout << " ";
						cout << td.convert(w);
						first = false;
					}
					cout << endl;
				}

				vssent.clear();
				target_doc[i] = temp_trg;
			}
			iter++;
		}
	}
	return;
}

//----------------------------------------------------------------------------------------------------------------
template <class DMT_t>
void TrainSentMTModel(Model &model, DMT_t &dmt, Corpus &training, Corpus &devel,
	Trainer &sgd, string out_file, bool curriculum, int max_epochs, int lr_epochs)
{
	double best_loss = 9e+99;
	
	unsigned report_every_i = TREPORT;//50;
	unsigned dev_every_i_reports = DREPORT;//500; 
	
	unsigned si = 0;//training.size();
	vector<unsigned> order(training.size());
	for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

	vector<vector<unsigned>> order_by_length; 
	const unsigned curriculum_steps = 10;
	if (curriculum) {
		// simple form of curriculum learning: for the first K epochs, use only
		// the shortest examples from the training set. E.g., K=10, then in
		// epoch 0 using the first decile, epoch 1 use the first & second
		// deciles etc. up to the full dataset in k >= 9.
		multimap<size_t, unsigned> lengths;
		for (unsigned i = 0; i < training.size(); ++i) 
			lengths.insert(make_pair(get<0>(training[i]).size(), i));

		order_by_length.resize(curriculum_steps);
		unsigned i = 0;
		for (auto& landi: lengths) {
			for (unsigned k = i * curriculum_steps / lengths.size(); k < curriculum_steps; ++k)  
			order_by_length[k].push_back(landi.second);
			++i;
		}
	}

	unsigned report = 0;
	unsigned lines = 0;
	unsigned epoch = 0;
	Sentence ssent, tsent;
	int docid;

	cerr << "**SHUFFLE\n";
	shuffle(order.begin(), order.end(), *rndeng);

	Timer timer_epoch("completed in"), timer_iteration("completed in");

	while (sgd.epoch < max_epochs) {
		ModelStats tstats;

		dmt.Enable_Dropout();// enable dropout

		for (unsigned iter = 0; iter < report_every_i; ++iter) {
			if (si == training.size()) {
				//timing
				cerr << "***Epoch " << sgd.epoch << " is finished. ";
				timer_epoch.Show();

				si = 0;

				if (lr_epochs == 0)
					sgd.update_epoch(); 
				else sgd.update_epoch(1, lr_epochs); // @vhoang2: learning rate annealing (after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay.

				if (sgd.epoch >= max_epochs) break;

				cerr << "**SHUFFLE\n";
				shuffle(order.begin(), order.end(), *rndeng);

				// for curriculum learning
				if (curriculum && epoch < order_by_length.size()) {
					order = order_by_length[epoch++];
					cerr << "Curriculum learning, with " << order.size() << " examples\n";
				} 

				timer_epoch.Reset();
			}

			// build graph for this instance
			tie(ssent, tsent, docid) = training[order[si % order.size()]];
			ComputationGraph cg;
			if (DEBUGGING_FLAG){// see http://dynet.readthedocs.io/en/latest/debugging.html
				cg.set_immediate_compute(true);
				cg.set_check_validity(true);
			}
			
			++si;

			Expression i_xent = dmt.BuildSentMTGraph(ssent, tsent, cg, tstats);

			Expression i_objective = i_xent;

			// perform forward computation for aggregate objective
			cg.forward(i_objective);

			// grab the parts of the objective
			tstats.loss += as_scalar(cg.get_value(i_xent.i));

			cg.backward(i_objective);
			sgd.update();

			++lines;

		}

		if (sgd.epoch >= max_epochs) continue;
	
		sgd.status();
		double elapsed = timer_iteration.Elapsed();
		cerr << "sents=" << si << " src_unks=" << tstats.words_src_unk << " trg_unks=" << tstats.words_tgt_unk << " E=" << (tstats.loss / tstats.words_tgt) << " ppl=" << exp(tstats.loss / tstats.words_tgt) << ' ';
		cerr << "[time_elapsed=" << elapsed << "(msec)" << " (" << tstats.words_tgt / elapsed << " words/msec)]" << endl;

		timer_iteration.Reset();	

		// show score on dev data?
		report += report_every_i;
		if (report % dev_every_i_reports == 0) {
			dmt.Disable_Dropout();// disable dropout for evaluating on dev data

			ModelStats dstats;
			for (unsigned i = 0; i < devel.size(); ++i) {
				tie(ssent, tsent, docid) = devel[i];
				ComputationGraph cg;
				auto i_xent = dmt.BuildSentMTGraph(ssent, tsent, cg, dstats);
				dstats.loss += as_scalar(cg.forward(i_xent));
			}
			if (dstats.loss < best_loss) {
				best_loss = dstats.loss;
				//ofstream out(out_file, ofstream::out);
				//boost::archive::text_oarchive oa(out);
				//oa << model;
				dynet::save_dynet_model(out_file, &model);// FIXME: use binary streaming instead for saving disk spaces
			}
            
			cerr << "--------------------------------------------------------------------------------------------------------" << endl;
			cerr << "***DEV [epoch=" << (lines / (double)training.size()) << " eta=" << sgd.eta << "]" << " sents=" << devel.size() << " src_unks=" << dstats.words_src_unk << " trg_unks=" << dstats.words_tgt_unk << " E=" << (dstats.loss / dstats.words_tgt) << " ppl=" << exp(dstats.loss / dstats.words_tgt) << ' ';
			timer_iteration.Show();	
			cerr << "--------------------------------------------------------------------------------------------------------" << endl;
		}

		timer_iteration.Reset();
	}

	cerr << endl << "Training completed in" << sgd.epoch << "epochs!" << endl;
}

struct DoubleLength
{
    DoubleLength(const Corpus & cor_) : cor(cor_) { }
    bool operator() (int i1, int i2);
    const Corpus & cor;
};

bool DoubleLength::operator() (int i1, int i2) {
    if(std::get<0>(cor[i2]).size() != std::get<0>(cor[i1]).size()) return (std::get<0>(cor[i2]).size() < std::get<0>(cor[i1]).size());
    return (std::get<1>(cor[i2]).size() < std::get<1>(cor[i1]).size());
}

inline size_t Calc_Size(const Sentence & src, const Sentence & trg) {
    return src.size()+trg.size();
}

inline size_t Create_MiniBatches(const Corpus& cor
        , size_t max_size
        , std::vector<std::vector<Sentence> > & train_src_minibatch
        , std::vector<std::vector<Sentence> > & train_trg_minibatch
        , std::vector<size_t> & train_ids_minibatch)
{
    train_src_minibatch.clear();
    train_trg_minibatch.clear();

    std::vector<size_t> train_ids(cor.size());
    std::iota(train_ids.begin(), train_ids.end(), 0);
    if(max_size > 1)
        sort(train_ids.begin(), train_ids.end(), DoubleLength(cor));

    std::vector<Sentence> train_src_next;
    std::vector<Sentence> train_trg_next;

    size_t max_len = 0;
    for(size_t i = 0; i < train_ids.size(); i++) {
        max_len = std::max(max_len, Calc_Size(std::get<0>(cor[train_ids[i]]), std::get<1>(cor[train_ids[i]])));
        train_src_next.push_back(std::get<0>(cor[train_ids[i]]));
        train_trg_next.push_back(std::get<1>(cor[train_ids[i]]));

        if((train_trg_next.size()+1) * max_len > max_size) {
            train_src_minibatch.push_back(train_src_next);
            train_src_next.clear();
            train_trg_minibatch.push_back(train_trg_next);
            train_trg_next.clear();
            max_len = 0;
        }
    }

    if(train_trg_next.size()) {
        train_src_minibatch.push_back(train_src_next);
        train_trg_minibatch.push_back(train_trg_next);
    }

    // Create a sentence list for this minibatch
    train_ids_minibatch.resize(train_src_minibatch.size());
    std::iota(train_ids_minibatch.begin(), train_ids_minibatch.end(), 0);

    return train_ids.size();
}

template <class DMT_t>
void TrainSentMTModel_Batch(Model &model, DMT_t &dmt, Corpus &training, Corpus &devel,
                            Trainer &sgd, string out_file, bool curriculum, int max_epochs, int lr_epochs)
{
    if (MINIBATCH_SIZE == 1){
        TrainSentMTModel(model, dmt, training, devel, sgd, out_file, curriculum, max_epochs, lr_epochs);
        return;
    }

    // Create minibatches
    vector<vector<Sentence> > train_src_minibatch;
    vector<vector<Sentence> > train_trg_minibatch;
    vector<size_t> train_ids_minibatch;
    size_t minibatch_size = MINIBATCH_SIZE;
    Create_MiniBatches(training, minibatch_size, train_src_minibatch, train_trg_minibatch, train_ids_minibatch);

    double best_loss = 9e+99;

    unsigned report_every_i = TREPORT;//50;
    unsigned dev_every_i_reports = DREPORT;//500;

    // shuffle minibatches
    cerr << "***SHUFFLE\n";
    std::shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *dynet::rndeng);

    unsigned sid = 0, id = 0, last_print = 0;
    Timer timer_epoch("completed in"), timer_iteration("completed in");

    while (sgd.epoch < max_epochs) {
        ModelStats tstats;

        dmt.Enable_Dropout();// enable dropout

        for (unsigned iter = 0; iter < dev_every_i_reports;) {
            if (id == train_ids_minibatch.size()) {
                //timing
                cerr << "***Epoch " << sgd.epoch << " is finished. ";
                timer_epoch.Show();

                if (lr_epochs == 0)
                    sgd.update_epoch();
                else sgd.update_epoch(1, lr_epochs); // @vhoang2: learning rate annealing (after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay.

                if (sgd.epoch >= max_epochs) break;

                // Shuffle the access order
                cerr << "***SHUFFLE\n";
                std::shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *dynet::rndeng);

                id = 0;
                sid = 0;
                last_print = 0;

                timer_epoch.Reset();
            }

            // build graph for this instance
            ComputationGraph cg;
            if (DEBUGGING_FLAG){//http://dynet.readthedocs.io/en/latest/debugging.html
                cg.set_immediate_compute(true);
                cg.set_check_validity(true);
            }

            Expression i_xent = dmt.BuildSentMTGraph_Batch(train_src_minibatch[train_ids_minibatch[id]], train_trg_minibatch[train_ids_minibatch[id]]
                    , cg, tstats);

            Expression i_objective = i_xent;

            // perform forward computation for aggregate objective
            cg.forward(i_objective);

            // grab the parts of the objective
            tstats.loss += as_scalar(cg.get_value(i_xent.i));

            cg.backward(i_objective);
            sgd.update();

            sid += train_trg_minibatch[train_ids_minibatch[id]].size();
            iter += train_trg_minibatch[train_ids_minibatch[id]].size();

            if (sid / report_every_i != last_print || iter >= dev_every_i_reports || sgd.epoch >= max_epochs){
                last_print = sid / report_every_i;

                float elapsed = timer_iteration.Elapsed();

                sgd.status();
                cerr << "sents=" << sid << " ";
                cerr /*<< "loss=" << tstats.loss*/ << "src_unks=" << tstats.words_src_unk << " trg_unks=" << tstats.words_tgt_unk << " E=" << (tstats.loss / tstats.words_tgt) << " ppl=" << exp(tstats.loss / tstats.words_tgt) << ' ';
                cerr /*<< "time_elapsed=" << elapsed*/ << "(" << tstats.words_tgt * 1000 / elapsed << " words/sec)" << endl;

                if (sgd.epoch >= max_epochs) break;
            }

            ++id;
        }

        timer_iteration.Reset();

        // show score on dev data?
        dmt.Disable_Dropout();// disable dropout for evaluating dev data

        ModelStats dstats;
        for (unsigned i = 0; i < devel.size(); ++i) {
            Sentence ssent, tsent;
            int docid;
            tie(ssent, tsent, docid) = devel[i];

            ComputationGraph cg;
            auto i_xent = dmt.BuildSentMTGraph(ssent, tsent, cg, dstats);
            dstats.loss += as_scalar(cg.forward(i_xent));
        }

        if (dstats.loss < best_loss) {
            best_loss = dstats.loss;
            //ofstream out(out_file, ofstream::out);
            //boost::archive::text_oarchive oa(out);
            //oa << model;
            dynet::save_dynet_model(out_file, &model);// FIXME: use binary streaming instead for saving disk spaces
        }

        cerr << "--------------------------------------------------------------------------------------------------------" << endl;
        cerr << "***DEV [epoch=" << (float)sgd.epoch + (float)sid/(float)training.size() << " eta=" << sgd.eta << "]" << " sents=" << devel.size() << " src_unks=" << dstats.words_src_unk << " trg_unks=" << dstats.words_tgt_unk << " E=" << (dstats.loss / dstats.words_tgt) << " ppl=" << exp(dstats.loss / dstats.words_tgt) << ' ';
        timer_iteration.Show();
        cerr << "--------------------------------------------------------------------------------------------------------" << endl;

        timer_iteration.Reset();
    }

    cerr << endl << "Training completed!" << endl;
}

//------------------------------------------------------------------------------------------------------------------
template <class DMT_t>
void TrainDocMTSrcModel(Model &model, DMT_t &dmt, vector<vector<vector<dynet::real>>> &tsent_repi, vector<vector<vector<dynet::real>>> &dsent_repi,
                     DocCorpus &training_doc, DocCorpus &devel_doc, Trainer &sgd, string out_file, int max_epochs, int lr_epochs)
{
    double best_loss = 9e+99;

    unsigned report_every_i = DTREPORT;//10;
    unsigned dev_every_i_reports = DDREPORT;//50;

    unsigned si = 0;//training.size();
    vector<unsigned> order(training_doc.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    unsigned report = 0;
    //unsigned epoch = 0;
    vector<Sentence> vssent, vtsent;
    vector<int> docid;

    cout << "Starting the TRAINING process!" << endl;
	cerr << "**SHUFFLE\n";
    shuffle(order.begin(), order.end(), *rndeng);

    Timer timer_epoch("completed in"), timer_iteration("completed in");

    while (sgd.epoch < max_epochs) {
        ModelStats tstats;
        unsigned lines = 0;

        dmt.Enable_Dropout();// enable dropout
        dmt.Enable_Dropout_DocRNN();

        for (unsigned iter = 0; iter < report_every_i; ++iter) {
            if (si == training_doc.size()) {
                //timing
                cerr << "***Epoch " << sgd.epoch << " is finished. ";
                timer_epoch.Show();

                si = 0;

                if (lr_epochs == 0)
                    sgd.update_epoch();

                else sgd.update_epoch(1, lr_epochs); // @vhoang2: learning rate annealing (after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay). @smaruf: the learning rate will be multiplied by the factor if it is less than 1.

                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);

                if (sgd.epoch >= max_epochs) break;

                timer_epoch.Reset();
            }

            // build graph for this document
            const unsigned sdlen = training_doc[order[si % order.size()]].size();
            for (unsigned dl = 0; dl < sdlen; ++dl){
                vssent.push_back(get<0>(training_doc[order[si % order.size()]].at(dl)));
                vtsent.push_back(get<1>(training_doc[order[si % order.size()]].at(dl)));
            }
            vector<vector<dynet::real>> vsent_repi = tsent_repi[order[si % order.size()]];

            ++si;

            for (unsigned i = 0; i < vssent.size(); ++i){
                ComputationGraph cg;

                Expression i_xent = dmt.BuildDocMTSrcGraph(vssent[i], vtsent[i], vsent_repi, i, cg, tstats);
                Expression i_objective = i_xent;

                // perform forward computation for aggregate objective
                cg.forward(i_objective);

                // grab the parts of the objective
                tstats.loss += as_scalar(cg.get_value(i_xent.i));

                cg.backward(i_objective);
            }
            sgd.update();

            lines+=sdlen;
            vssent.clear();
            vtsent.clear();
        }

        if (sgd.epoch >= max_epochs) continue;

        sgd.status();
        double elapsed = timer_iteration.Elapsed();
        cerr << "docs=" << si << " src_unks=" << tstats.words_src_unk << " trg_unks=" << tstats.words_tgt_unk << " E=" << (tstats.loss / tstats.words_tgt) << " ppl=" << exp(tstats.loss / tstats.words_tgt) << ' ';
        cerr << "[time_elapsed=" << elapsed << "(msec)" << " (" << tstats.words_tgt * 1000 / elapsed << " words/sec)]" << endl;

        timer_iteration.Reset();

        // show score on dev data?
        report += report_every_i;
        if (report % dev_every_i_reports == 0) {
            dmt.Disable_Dropout();// disable dropout for evaluating dev data
            dmt.Disable_Dropout_DocRNN();

            ModelStats dstats;
            for (unsigned i = 0; i < devel_doc.size(); ++i) {
                const unsigned ddlen = devel_doc[i].size();
                for (unsigned dl = 0; dl < ddlen; ++dl){
                    vssent.push_back(get<0>(devel_doc[i].at(dl)));
                    vtsent.push_back(get<1>(devel_doc[i].at(dl)));
                }
                vector<vector<dynet::real>> vdsent_repi = dsent_repi[i];

                for (unsigned j = 0; j < vssent.size(); ++j){
                    ComputationGraph cg;

                    auto i_xent = dmt.BuildDocMTSrcGraph(vssent[j], vtsent[j], vdsent_repi, j, cg, dstats);
                    dstats.loss += as_scalar(cg.forward(i_xent));
                }

                vssent.clear();
                vtsent.clear();
            }
            if (dstats.loss < best_loss) {
                best_loss = dstats.loss;
                //ofstream out(out_file, ofstream::out);
                //boost::archive::text_oarchive oa(out);
                //oa << model;
                dynet::save_dynet_model(out_file, &model);// FIXME: use binary streaming instead for saving disk spaces
            }


            cerr << "--------------------------------------------------------------------------------------------------------" << endl;
            cerr << "***DEV [ eta=" << sgd.eta << "]" << " docs=" << devel_doc.size() << " src_unks=" << dstats.words_src_unk << " trg_unks=" << dstats.words_tgt_unk << " E=" << (dstats.loss / dstats.words_tgt) << " ppl=" << exp(dstats.loss / dstats.words_tgt) << ' ';
            timer_iteration.Show();
            cerr << "--------------------------------------------------------------------------------------------------------" << endl;
        }

        timer_iteration.Reset();
    }

    cerr << endl << "Training completed in " << sgd.epoch << " epochs!" << endl;
}

struct DoubleDocLength
{
    DoubleDocLength(const Document & doc_) : doc(doc_) { }
    bool operator() (int i1, int i2);
    const Document & doc;
};

bool DoubleDocLength::operator() (int i1, int i2) {
    if(std::get<0>(doc[i2]).size() != std::get<0>(doc[i1]).size()) return (std::get<0>(doc[i2]).size() < std::get<0>(doc[i1]).size());
    return (std::get<1>(doc[i2]).size() < std::get<1>(doc[i1]).size());
}

inline void Create_DocMiniBatch(const Document& doc, size_t max_size,
                                std::vector<std::vector<Sentence>>& train_src_minidoc, std::vector<std::vector<Sentence>>& train_trg_minidoc,
                                std::vector<std::vector<unsigned int>>& train_mini_sids)
{
    train_src_minidoc.clear();
    train_trg_minidoc.clear();
    train_mini_sids.clear();

    std::vector<size_t> train_sids(doc.size());
    std::iota(train_sids.begin(), train_sids.end(), 0);
    sort(train_sids.begin(), train_sids.end(), DoubleDocLength(doc));

    std::vector<Sentence> train_src_next, train_trg_next;
    std::vector<unsigned int> sids;

    size_t max_len = 0;
    for(size_t i = 0; i < train_sids.size(); i++) {
        max_len = std::max(max_len, Calc_Size(std::get<0>(doc[train_sids[i]]), std::get<1>(doc[train_sids[i]])));
        train_src_next.push_back(std::get<0>(doc[train_sids[i]]));
        train_trg_next.push_back(std::get<1>(doc[train_sids[i]]));

        sids.push_back(train_sids[i]);  //to keep track of the actual sentence ID in the document

        if((train_trg_next.size()+1)*max_len > max_size){
            train_src_minidoc.push_back(train_src_next);
            train_src_next.clear();
            train_trg_minidoc.push_back(train_trg_next);
            train_trg_next.clear();
            train_mini_sids.push_back(sids);
            sids.clear();
            max_len=0;
        }
    }

    if(train_trg_next.size()){
        train_src_minidoc.push_back(train_src_next);
        train_trg_minidoc.push_back(train_trg_next);
        train_mini_sids.push_back(sids);
    }
}

template <class DMT_t>
void TrainDocMTSrcModel_Batch(Model &model, DMT_t &dmt, vector<vector<vector<dynet::real>>> &tsent_repi, vector<vector<vector<dynet::real>>> &dsent_repi,
                              DocCorpus &training_doc, DocCorpus &devel_doc, Trainer &sgd, string out_file, int max_epochs, int lr_epochs)
{
    if (DOCMINIBATCH_SIZE == 1){
        TrainDocMTSrcModel(model, dmt, tsent_repi, dsent_repi, training_doc, devel_doc, sgd, out_file, max_epochs, lr_epochs);
        return;
    }

    double best_loss = 9e+99;

    unsigned report_every_i = DTREPORT;//10;
    unsigned dev_every_i_reports = DDREPORT;//50;

    unsigned si = 0;//training.size();
    vector<unsigned> order(training_doc.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    unsigned report = 0;
    size_t docminibatch_size = DOCMINIBATCH_SIZE;
    vector<vector<Sentence>> train_src_minidoc, train_trg_minidoc;
    vector<Sentence> vssent, vtsent;

    cout << "Starting the TRAINING process!" << endl;
    cerr << "**SHUFFLE\n";
    shuffle(order.begin(), order.end(), *rndeng);

    Timer timer_epoch("completed in"), timer_iteration("completed in");

    while (sgd.epoch < max_epochs) {
        ModelStats tstats;
        unsigned lines = 0;

        dmt.Enable_Dropout();// enable dropout
        dmt.Enable_Dropout_DocRNN();

        for (unsigned iter = 0; iter < report_every_i; ++iter) {
            if (si == training_doc.size()) {
                //timing
                cerr << "***Epoch " << sgd.epoch << " is finished. ";
                timer_epoch.Show();

                si = 0;

                if (lr_epochs == 0)
                    sgd.update_epoch();

                else sgd.update_epoch(1, lr_epochs); // @vhoang2: learning rate annealing (after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay.

                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);

                if (sgd.epoch >= max_epochs) break;

                timer_epoch.Reset();
            }

            // build graph for this document
            const unsigned tdlen = training_doc[order[si % order.size()]].size();
            vector<vector<unsigned int>> train_mini_sids;

            Create_DocMiniBatch(training_doc[order[si % order.size()]], docminibatch_size,
                                train_src_minidoc, train_trg_minidoc, train_mini_sids);

            vector<vector<dynet::real>> vsent_repi = tsent_repi[order[si % order.size()]];

            //if (DEBUGGING_FLAG){// see http://dynet.readthedocs.io/en/latest/debugging.html
            //    cg.set_immediate_compute(true);
            //    cg.set_check_validity(true);
            //}

            ++si;

            for (unsigned i = 0; i < train_src_minidoc.size(); ++i){
                ComputationGraph cg;

                Expression i_xent = dmt.BuildDocMTSrcGraph_Batch(train_src_minidoc[i], train_trg_minidoc[i],
                                                                 vsent_repi, train_mini_sids[i], cg, tstats);
                Expression i_objective = i_xent;

                // perform forward computation for aggregate objective
                cg.forward(i_objective);

                // grab the parts of the objective
                tstats.loss += as_scalar(cg.get_value(i_xent.i));

                cg.backward(i_objective);
                //sgd.update();
            }
            sgd.update();
            lines+=tdlen;
        }

        sgd.status();
        double elapsed = timer_iteration.Elapsed();
        cerr << "docs=" << si << " sents=" << lines << " src_unks=" << tstats.words_src_unk << " trg_unks=" << tstats.words_tgt_unk << " E=" << (tstats.loss / tstats.words_tgt) << " ppl=" << exp(tstats.loss / tstats.words_tgt) << ' ';
        cerr << "[time_elapsed=" << elapsed << "(msec)" << " (" << tstats.words_tgt * 1000 / elapsed << " words/sec)]" << endl;

        if (sgd.epoch >= max_epochs) break;

        timer_iteration.Reset();

        // show score on dev data?
        report += report_every_i;
        if (report % dev_every_i_reports == 0) {
            dmt.Disable_Dropout();// disable dropout for evaluating dev data
            dmt.Disable_Dropout_DocRNN();

            ModelStats dstats;
            for (unsigned i = 0; i < devel_doc.size(); ++i) {
                const unsigned ddlen = devel_doc[i].size();
                for (unsigned dl = 0; dl < ddlen; ++dl){
                    vssent.push_back(get<0>(devel_doc[i].at(dl)));
                    vtsent.push_back(get<1>(devel_doc[i].at(dl)));
                }
                vector<vector<dynet::real>> vdsent_repi = dsent_repi[i];

                for (unsigned j = 0; j < vssent.size(); ++j){
                    ComputationGraph cg;

                    auto i_xent = dmt.BuildDocMTSrcGraph(vssent[j], vtsent[j], vdsent_repi, j, cg, dstats);
                    dstats.loss += as_scalar(cg.forward(i_xent));
                }
                vssent.clear();
                vtsent.clear();
            }
            if (dstats.loss < best_loss) {
                best_loss = dstats.loss;
                //ofstream out(out_file, ofstream::out);
                //boost::archive::text_oarchive oa(out);
                //oa << model;
                dynet::save_dynet_model(out_file, &model);// FIXME: use binary streaming instead for saving disk spaces
            }

            cerr << "--------------------------------------------------------------------------------------------------------" << endl;
            cerr << "***DEV [eta=" << sgd.eta << "]" << " docs=" << devel_doc.size() << " src_unks=" << dstats.words_src_unk << " trg_unks=" << dstats.words_tgt_unk << " E=" << (dstats.loss / dstats.words_tgt) << " ppl=" << exp(dstats.loss / dstats.words_tgt) << ' ';
            timer_iteration.Show();
            cerr << "--------------------------------------------------------------------------------------------------------" << endl;
        }

        timer_iteration.Reset();
    }

    cerr << endl << "Training completed in " << sgd.epoch << " epochs!" << endl;
}

//------------------------------------------------------------------------------------------------------------------
template <class DMT_t>
vector<vector<vector<dynet::real>>> ComputeDocTrgRep(Model &model, DMT_t &dmt, DocCorpus &training_doc)
{
    vector<Sentence> vssent;
    vector<vector<vector<dynet::real>>> trgdoccorpus_rep;

    //get translations based on document-level model including source memory
    for (unsigned i = 0; i < training_doc.size(); ++i) {
        vector<vector<dynet::real>> trgdoc_rep;

        const unsigned sdlen = training_doc[i].size();
        for (unsigned dl = 0; dl < sdlen; ++dl)
            vssent.push_back(get<0>(training_doc[i].at(dl)));

        for (unsigned dl = 0; dl < vssent.size(); ++dl) {
            ComputationGraph cg;
            Sentence source = vssent[dl];

            trgdoc_rep.push_back(dmt.GetTrgRepresentations(source, cg, td));
        }

        vssent.clear();
        trgdoccorpus_rep.push_back(trgdoc_rep);
    }

    return trgdoccorpus_rep;
}

template <class DMT_t>
vector<vector<vector<dynet::real>>> ComputeDocTrueTrgRep(Model &model, DMT_t &dmt, DocCorpus &training_doc)
{
	vector<Sentence> vssent, vtsent;
	vector<vector<vector<dynet::real>>> trgdoccorpus_rep;

	//get translations based on document-level model including source memory
	for (unsigned i = 0; i < training_doc.size(); ++i) {
		vector<vector<dynet::real>> trgdoc_rep;

		const unsigned sdlen = training_doc[i].size();
		for (unsigned dl = 0; dl < sdlen; ++dl) {
			vssent.push_back(get<0>(training_doc[i].at(dl)));
			vtsent.push_back(get<1>(training_doc[i].at(dl)));
		}			

		for (unsigned dl = 0; dl < vssent.size(); ++dl) {
			ComputationGraph cg;
			Sentence source = vssent[dl];
			Sentence target = vtsent[dl];

			trgdoc_rep.push_back(dmt.GetTrueTrgRepresentations(source, target, cg));
		}

		vssent.clear();
		vtsent.clear();
		trgdoccorpus_rep.push_back(trgdoc_rep);
	}

	return trgdoccorpus_rep;
}

template <class DMT_t>
void TrainDocMTTrgModel(Model &model, DMT_t &dmt, DocCorpus &training_doc, DocCorpus &devel_doc, Trainer &sgd,
                        string out_file, int max_epochs, int lr_epochs, bool use_gold)
{
    double best_loss = 9e+99;

    unsigned report_every_i = DTREPORT;//10;
    unsigned dev_every_i_reports = DDREPORT;//50;

    unsigned si = 0;//training.size();
    vector<unsigned> order(training_doc.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    unsigned report = 0;
    //unsigned epoch = 0;
    vector<Sentence> vssent, vtsent;
    vector<int> docid;
    vector<vector<vector<dynet::real>>> train_trgsent_rep;
    vector<vector<vector<dynet::real>>> dev_trgsent_rep;

    cout << "Starting the TRAINING process!" << endl;
    if (use_gold){
        cout << "Computing the true target sentence representations without using the source memory!" << endl;
        train_trgsent_rep = ComputeDocTrueTrgRep(model, dmt, training_doc);
        dev_trgsent_rep = ComputeDocTrueTrgRep(model, dmt, devel_doc);
    }
    else{
        cout << "Computing the target sentence representations without using the source memory!" << endl;
        train_trgsent_rep = ComputeDocTrgRep(model, dmt, training_doc);
        dev_trgsent_rep = ComputeDocTrgRep(model, dmt, devel_doc);
    }

    cerr << "**SHUFFLE\n";
    shuffle(order.begin(), order.end(), *rndeng);

    Timer timer_epoch("completed in"), timer_iteration("completed in");

    while (sgd.epoch < max_epochs) {
        ModelStats tstats;
        unsigned lines = 0;

        dmt.Enable_Dropout();// enable dropout
        dmt.Enable_Dropout_DocRNN();

        for (unsigned iter = 0; iter < report_every_i; ++iter) {
            if (si == training_doc.size()) {
                //timing
                cerr << "***Epoch " << sgd.epoch << " is finished. ";
                timer_epoch.Show();

                si = 0;

                if (lr_epochs == 0)
                    sgd.update_epoch();

                else sgd.update_epoch(1, lr_epochs); // @vhoang2: learning rate annealing (after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay). @smaruf: the learning rate will be multiplied by the factor if it is less than 1.

	            cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);

                if (sgd.epoch >= max_epochs) break;

                timer_epoch.Reset();
            }

            // build graph for this document
            const unsigned sdlen = training_doc[order[si % order.size()]].size();
            for (unsigned dl = 0; dl < sdlen; ++dl){
                vssent.push_back(get<0>(training_doc[order[si % order.size()]].at(dl)));
                vtsent.push_back(get<1>(training_doc[order[si % order.size()]].at(dl)));
            }
            vector<vector<dynet::real>> vtrgsent_rep = train_trgsent_rep[order[si % order.size()]];

            ++si;

            for (unsigned i = 0; i < vssent.size(); ++i){
                ComputationGraph cg;

                Expression i_xent = dmt.BuildDocMTTrgGraph(vssent[i], vtsent[i], vtrgsent_rep, i, cg, tstats);
                Expression i_objective = i_xent;

                // perform forward computation for aggregate objective
                cg.forward(i_objective);

                // grab the parts of the objective
                tstats.loss += as_scalar(cg.get_value(i_xent.i));

                cg.backward(i_objective);
            }
            sgd.update();

            lines+=sdlen;
            vssent.clear();
            vtsent.clear();
        }

        if (sgd.epoch >= max_epochs) continue;

        sgd.status();
        double elapsed = timer_iteration.Elapsed();
        cerr << "docs=" << si << " src_unks=" << tstats.words_src_unk << " trg_unks=" << tstats.words_tgt_unk << " E=" << (tstats.loss / tstats.words_tgt) << " ppl=" << exp(tstats.loss / tstats.words_tgt) << ' ';
        cerr << "[time_elapsed=" << elapsed << "(msec)" << " (" << tstats.words_tgt * 1000 / elapsed << " words/sec)]" << endl;

        timer_iteration.Reset();

        // show score on dev data?
        report += report_every_i;
        if (report % dev_every_i_reports == 0) {
            dmt.Disable_Dropout();// disable dropout for evaluating dev data
            dmt.Disable_Dropout_DocRNN();

            ModelStats dstats;
            for (unsigned i = 0; i < devel_doc.size(); ++i) {
                const unsigned ddlen = devel_doc[i].size();
                for (unsigned dl = 0; dl < ddlen; ++dl){
                    vssent.push_back(get<0>(devel_doc[i].at(dl)));
                    vtsent.push_back(get<1>(devel_doc[i].at(dl)));
                }
                vector<vector<dynet::real>> vdtrgsent_rep = dev_trgsent_rep[i];

                for (unsigned j = 0; j < vssent.size(); ++j){
                    ComputationGraph cg;

                    auto i_xent = dmt.BuildDocMTTrgGraph(vssent[j], vtsent[j], vdtrgsent_rep, j, cg, dstats);
                    dstats.loss += as_scalar(cg.forward(i_xent));
                }

                vssent.clear();
                vtsent.clear();
            }
            if (dstats.loss < best_loss) {
                best_loss = dstats.loss;
                //ofstream out(out_file, ofstream::out);
                //boost::archive::text_oarchive oa(out);
                //oa << model;
                dynet::save_dynet_model(out_file, &model);// FIXME: use binary streaming instead for saving disk spaces
            }


            cerr << "--------------------------------------------------------------------------------------------------------" << endl;
            cerr << "***DEV [ eta=" << sgd.eta << "]" << " docs=" << devel_doc.size() << " src_unks=" << dstats.words_src_unk << " trg_unks=" << dstats.words_tgt_unk << " E=" << (dstats.loss / dstats.words_tgt) << " ppl=" << exp(dstats.loss / dstats.words_tgt) << ' ';
            timer_iteration.Show();
            cerr << "--------------------------------------------------------------------------------------------------------" << endl;
        }

        timer_iteration.Reset();
    }

    cerr << endl << "Training completed in " << sgd.epoch << " epochs!" << endl;
}

template <class DMT_t>
void TrainDocMTTrgModel_Batch(Model &model, DMT_t &dmt, DocCorpus &training_doc, DocCorpus &devel_doc, Trainer &sgd,
                              string out_file, int max_epochs, int lr_epochs, bool use_gold)
{
    if (DOCMINIBATCH_SIZE == 1){
        TrainDocMTTrgModel(model, dmt, training_doc, devel_doc, sgd, out_file, max_epochs, lr_epochs, use_gold);
        return;
    }

    double best_loss = 9e+99;

    unsigned report_every_i = DTREPORT;//10;
    unsigned dev_every_i_reports = DDREPORT;//50;

    unsigned si = 0;//training.size();
    vector<unsigned> order(training_doc.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    unsigned report = 0;
    size_t docminibatch_size = DOCMINIBATCH_SIZE;
    vector<vector<Sentence>> train_src_minidoc, train_trg_minidoc;
    vector<Sentence> vssent, vtsent;
    vector<vector<vector<dynet::real>>> train_trgsent_rep;
    vector<vector<vector<dynet::real>>> dev_trgsent_rep;

    cout << "Starting the TRAINING process!" << endl;
    //compute the representations of the target sentences by doing greedy decoding
    if (use_gold){
        cout << "Computing the true target sentence representations without using the source memory!" << endl;
        train_trgsent_rep = ComputeDocTrueTrgRep(model, dmt, training_doc);
        dev_trgsent_rep = ComputeDocTrueTrgRep(model, dmt, devel_doc);
    }
    else{
        cout << "Computing the target sentence representations without using the source memory!" << endl;
        train_trgsent_rep = ComputeDocTrgRep(model, dmt, training_doc);
        dev_trgsent_rep = ComputeDocTrgRep(model, dmt, devel_doc);
    }

    cerr << "**SHUFFLE\n";
    shuffle(order.begin(), order.end(), *rndeng);

    Timer timer_epoch("completed in"), timer_iteration("completed in");

    while (sgd.epoch < max_epochs) {
        ModelStats tstats;
        unsigned lines = 0;

        dmt.Enable_Dropout();// enable dropout
        dmt.Enable_Dropout_DocRNN();

        for (unsigned iter = 0; iter < report_every_i; ++iter) {
            if (si == training_doc.size()) {
                //timing
                cerr << "***Epoch " << sgd.epoch << " is finished. ";
                timer_epoch.Show();

                si = 0;

                if (lr_epochs == 0)
                    sgd.update_epoch();

                else sgd.update_epoch(1, lr_epochs); // @vhoang2: learning rate annealing (after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay.

                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);

                if (sgd.epoch >= max_epochs) break;

                timer_epoch.Reset();
            }

            // build graph for this document
            const unsigned tdlen = training_doc[order[si % order.size()]].size();
            vector<vector<unsigned int>> train_mini_sids;

            Create_DocMiniBatch(training_doc[order[si % order.size()]], docminibatch_size,
                                train_src_minidoc, train_trg_minidoc, train_mini_sids);

            vector<vector<dynet::real>> vtrgsent_rep = train_trgsent_rep[order[si % order.size()]];

            ++si;

            for (unsigned i = 0; i < train_src_minidoc.size(); ++i){
                ComputationGraph cg;

                Expression i_xent = dmt.BuildDocMTTrgGraph_Batch(train_src_minidoc[i], train_trg_minidoc[i],
                                                                 vtrgsent_rep, train_mini_sids[i], cg, tstats);
                Expression i_objective = i_xent;

                // perform forward computation for aggregate objective
                cg.forward(i_objective);

                // grab the parts of the objective
                tstats.loss += as_scalar(cg.get_value(i_xent.i));

                cg.backward(i_objective);
                //sgd.update();
            }
            sgd.update();
            lines+=tdlen;
        }

        sgd.status();
        double elapsed = timer_iteration.Elapsed();
        cerr << "docs=" << si << " sents=" << lines << " src_unks=" << tstats.words_src_unk << " trg_unks=" << tstats.words_tgt_unk << " E=" << (tstats.loss / tstats.words_tgt) << " ppl=" << exp(tstats.loss / tstats.words_tgt) << ' ';
        cerr << "[time_elapsed=" << elapsed << "(msec)" << " (" << tstats.words_tgt * 1000 / elapsed << " words/sec)]" << endl;

        if (sgd.epoch >= max_epochs) break;

        timer_iteration.Reset();

        // show score on dev data?
        report += report_every_i;
        if (report % dev_every_i_reports == 0) {
            dmt.Disable_Dropout();// disable dropout for evaluating dev data
            dmt.Disable_Dropout_DocRNN();

            ModelStats dstats;
            for (unsigned i = 0; i < devel_doc.size(); ++i) {
                const unsigned ddlen = devel_doc[i].size();
                for (unsigned dl = 0; dl < ddlen; ++dl){
                    vssent.push_back(get<0>(devel_doc[i].at(dl)));
                    vtsent.push_back(get<1>(devel_doc[i].at(dl)));
                }
                vector<vector<dynet::real>> vdtrgsent_rep = dev_trgsent_rep[i];

                for (unsigned j = 0; j < vssent.size(); ++j){
                    ComputationGraph cg;

                    auto i_xent = dmt.BuildDocMTTrgGraph(vssent[j], vtsent[j], vdtrgsent_rep, j, cg, dstats);
                    dstats.loss += as_scalar(cg.forward(i_xent));
                }
                vssent.clear();
                vtsent.clear();
            }
            if (dstats.loss < best_loss) {
                best_loss = dstats.loss;
                //ofstream out(out_file, ofstream::out);
                //boost::archive::text_oarchive oa(out);
                //oa << model;
                dynet::save_dynet_model(out_file, &model);// FIXME: use binary streaming instead for saving disk spaces
            }

            cerr << "--------------------------------------------------------------------------------------------------------" << endl;
            cerr << "***DEV [eta=" << sgd.eta << "]" << " docs=" << devel_doc.size() << " src_unks=" << dstats.words_src_unk << " trg_unks=" << dstats.words_tgt_unk << " E=" << (dstats.loss / dstats.words_tgt) << " ppl=" << exp(dstats.loss / dstats.words_tgt) << ' ';
            timer_iteration.Show();
            cerr << "--------------------------------------------------------------------------------------------------------" << endl;
        }

        timer_iteration.Reset();
    }

    cerr << endl << "Training completed in " << sgd.epoch << " epochs!" << endl;
}

//----------------------------------------------------------------------------------------------------------------
template <class DMT_t>
vector<vector<vector<dynet::real>>> ComputeDocTrg_SrcRep(Model &model, DMT_t &dmt, vector<vector<vector<dynet::real>>> &sent_repi,
                                                         DocCorpus &training_doc)
{
    vector<Sentence> vssent;
    vector<vector<vector<dynet::real>>> trgdoccorpus_rep;

    //get translations based on document-level model including source memory
    for (unsigned i = 0; i < training_doc.size(); ++i) {
        vector<vector<dynet::real>> trgdoc_rep;

        const unsigned sdlen = training_doc[i].size();
        for (unsigned dl = 0; dl < sdlen; ++dl)
            vssent.push_back(get<0>(training_doc[i].at(dl)));

        vector<vector<dynet::real>> vsent_repi = sent_repi[i];

        for (unsigned dl = 0; dl < vssent.size(); ++dl) {
            ComputationGraph cg;
            Sentence source = vssent[dl];

            trgdoc_rep.push_back(dmt.GetTrg_SrcRepresentations(source, vsent_repi, dl, cg, td));
        }

        vssent.clear();
		trgdoccorpus_rep.push_back(trgdoc_rep);
    }

    return trgdoccorpus_rep;
}

template <class DMT_t>
vector<vector<vector<dynet::real>>> ComputeDocTrueTrg_SrcRep(Model &model, DMT_t &dmt, vector<vector<vector<dynet::real>>> &sent_repi,
                                                             DocCorpus &training_doc)
{
	vector<Sentence> vssent, vtsent;
	vector<vector<vector<dynet::real>>> trgdoccorpus_rep;

	//get translations based on document-level model including source memory
	for (unsigned i = 0; i < training_doc.size(); ++i) {
		vector<vector<dynet::real>> trgdoc_rep;

		const unsigned sdlen = training_doc[i].size();
		for (unsigned dl = 0; dl < sdlen; ++dl) {
			vssent.push_back(get<0>(training_doc[i].at(dl)));
			vtsent.push_back(get<1>(training_doc[i].at(dl)));
		}

		vector<vector<dynet::real>> vsent_repi = sent_repi[i];

		for (unsigned dl = 0; dl < vssent.size(); ++dl) {
			ComputationGraph cg;
			Sentence source = vssent[dl];
			Sentence target = vtsent[dl];

			trgdoc_rep.push_back(dmt.GetTrueTrg_SrcRepresentations(source, target, vsent_repi, dl, cg));
		}

		vssent.clear();
		vtsent.clear();
		trgdoccorpus_rep.push_back(trgdoc_rep);
	}

	return trgdoccorpus_rep;
}

template <class DMT_t>
void TrainDocMTSrcTrgModel(Model &model, DMT_t &dmt, vector<vector<vector<dynet::real>>> &tsent_repi, vector<vector<vector<dynet::real>>> &dsent_repi,
                           DocCorpus &training_doc, DocCorpus &devel_doc, Trainer &sgd, string out_file, int max_epochs, int lr_epochs, bool use_gold)
{
    double best_loss = 9e+99;

    unsigned report_every_i = DTREPORT;//10;
    unsigned dev_every_i_reports = DDREPORT;//50;

    unsigned si = 0;//training.size();
    vector<unsigned> order(training_doc.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    unsigned report = 0;
    //unsigned epoch = 0;
    vector<Sentence> vssent, vtsent;
    vector<int> docid;
    vector<vector<vector<dynet::real>>> train_trgsent_rep;
    vector<vector<vector<dynet::real>>> dev_trgsent_rep;

    cout << "Starting the TRAINING process!" << endl;
    if (use_gold){
        cout << "Computing the true target sentence representations using the source memory!" << endl;
        train_trgsent_rep = ComputeDocTrueTrg_SrcRep(model, dmt, tsent_repi, training_doc);
        dev_trgsent_rep = ComputeDocTrueTrg_SrcRep(model, dmt, dsent_repi, devel_doc);
    }
    else{
        cout << "Computing the target sentence representations using the source memory!" << endl;
        train_trgsent_rep = ComputeDocTrg_SrcRep(model, dmt, tsent_repi, training_doc);
        dev_trgsent_rep = ComputeDocTrg_SrcRep(model, dmt, dsent_repi, devel_doc);
    }

    cerr << "**SHUFFLE\n";
    shuffle(order.begin(), order.end(), *rndeng);

    Timer timer_epoch("completed in"), timer_iteration("completed in");

    while (sgd.epoch < max_epochs) {
        ModelStats tstats;
        unsigned lines = 0;

        dmt.Enable_Dropout();// enable dropout
        dmt.Enable_Dropout_DocRNN();

        for (unsigned iter = 0; iter < report_every_i; ++iter) {
            if (si == training_doc.size()) {
                //timing
                cerr << "***Epoch " << sgd.epoch << " is finished. ";
                timer_epoch.Show();

                si = 0;

                if (lr_epochs == 0)
                    sgd.update_epoch();

                else sgd.update_epoch(1, lr_epochs); // @vhoang2: learning rate annealing (after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay). @smaruf: the learning rate will be multiplied by the factor if it is less than 1.

                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);

                if (sgd.epoch >= max_epochs) break;

                timer_epoch.Reset();
            }

            // build graph for this document
            const unsigned sdlen = training_doc[order[si % order.size()]].size();
            for (unsigned dl = 0; dl < sdlen; ++dl){
                vssent.push_back(get<0>(training_doc[order[si % order.size()]].at(dl)));
                vtsent.push_back(get<1>(training_doc[order[si % order.size()]].at(dl)));
            }
            vector<vector<dynet::real>> vsent_repi = tsent_repi[order[si % order.size()]];
            vector<vector<dynet::real>> vtrgsent_rep = train_trgsent_rep[order[si % order.size()]];

            ++si;

            for (unsigned i = 0; i < vssent.size(); ++i){
                ComputationGraph cg;

                Expression i_xent = dmt.BuildDocMTSrcTrgGraph(vssent[i], vtsent[i], vsent_repi, vtrgsent_rep, i, cg, tstats);
                Expression i_objective = i_xent;

                // perform forward computation for aggregate objective
                cg.forward(i_objective);

                // grab the parts of the objective
                tstats.loss += as_scalar(cg.get_value(i_xent.i));

                cg.backward(i_objective);
            }
            sgd.update();

            lines+=sdlen;
            vssent.clear();
            vtsent.clear();
        }

        if (sgd.epoch >= max_epochs) continue;

        sgd.status();
        double elapsed = timer_iteration.Elapsed();
        cerr << "docs=" << si << " src_unks=" << tstats.words_src_unk << " trg_unks=" << tstats.words_tgt_unk << " E=" << (tstats.loss / tstats.words_tgt) << " ppl=" << exp(tstats.loss / tstats.words_tgt) << ' ';
        cerr << "[time_elapsed=" << elapsed << "(msec)" << " (" << tstats.words_tgt * 1000 / elapsed << " words/sec)]" << endl;

        timer_iteration.Reset();

        // show score on dev data?
        report += report_every_i;
        if (report % dev_every_i_reports == 0) {
            dmt.Disable_Dropout();// disable dropout for evaluating dev data
            dmt.Disable_Dropout_DocRNN();

            ModelStats dstats;
            for (unsigned i = 0; i < devel_doc.size(); ++i) {
                const unsigned ddlen = devel_doc[i].size();
                for (unsigned dl = 0; dl < ddlen; ++dl){
                    vssent.push_back(get<0>(devel_doc[i].at(dl)));
                    vtsent.push_back(get<1>(devel_doc[i].at(dl)));
                }
                vector<vector<dynet::real>> vdsent_repi = dsent_repi[i];
                vector<vector<dynet::real>> vdtrgsent_rep = dev_trgsent_rep[i];

                for (unsigned j = 0; j < vssent.size(); ++j){
                    ComputationGraph cg;

                    auto i_xent = dmt.BuildDocMTSrcTrgGraph(vssent[j], vtsent[j], vdsent_repi, vdtrgsent_rep, j, cg, dstats);
                    dstats.loss += as_scalar(cg.forward(i_xent));
                }

                vssent.clear();
                vtsent.clear();
            }
            if (dstats.loss < best_loss) {
                best_loss = dstats.loss;
                //ofstream out(out_file, ofstream::out);
                //boost::archive::text_oarchive oa(out);
                //oa << model;
                dynet::save_dynet_model(out_file, &model);// FIXME: use binary streaming instead for saving disk spaces
            }


            cerr << "--------------------------------------------------------------------------------------------------------" << endl;
            cerr << "***DEV [ eta=" << sgd.eta << "]" << " docs=" << devel_doc.size() << " src_unks=" << dstats.words_src_unk << " trg_unks=" << dstats.words_tgt_unk << " E=" << (dstats.loss / dstats.words_tgt) << " ppl=" << exp(dstats.loss / dstats.words_tgt) << ' ';
            timer_iteration.Show();
            cerr << "--------------------------------------------------------------------------------------------------------" << endl;
        }

        timer_iteration.Reset();
    }

    cerr << endl << "Training completed in " << sgd.epoch << " epochs!" << endl;
}

template <class DMT_t>
void TrainDocMTSrcTrgModel_Batch(Model &model, DMT_t &dmt, vector<vector<vector<dynet::real>>> &tsent_repi, vector<vector<vector<dynet::real>>> &dsent_repi,
                              DocCorpus &training_doc, DocCorpus &devel_doc, Trainer &sgd, string out_file, int max_epochs, int lr_epochs, bool use_gold)
{
    if (DOCMINIBATCH_SIZE == 1){
        TrainDocMTSrcTrgModel(model, dmt, tsent_repi, dsent_repi, training_doc, devel_doc, sgd, out_file, max_epochs, lr_epochs, use_gold);
        return;
    }

    double best_loss = 9e+99;

    unsigned report_every_i = DTREPORT;//10;
    unsigned dev_every_i_reports = DDREPORT;//50;

    unsigned si = 0;//training.size();
    vector<unsigned> order(training_doc.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    unsigned report = 0;
    size_t docminibatch_size = DOCMINIBATCH_SIZE;
    vector<vector<Sentence>> train_src_minidoc, train_trg_minidoc;
    vector<Sentence> vssent, vtsent;
    vector<vector<vector<dynet::real>>> train_trgsent_rep;
    vector<vector<vector<dynet::real>>> dev_trgsent_rep;

    cout << "Starting the TRAINING process!" << endl;
    //compute the representations of the target sentences by doing greedy decoding
    if (use_gold){
        cout << "Computing the true target sentence representations using the source memory!" << endl;
        train_trgsent_rep = ComputeDocTrueTrg_SrcRep(model, dmt, tsent_repi, training_doc);
        dev_trgsent_rep = ComputeDocTrueTrg_SrcRep(model, dmt, dsent_repi, devel_doc);
    }
    else{
        cout << "Computing the target sentence representations using the source memory!" << endl;
        train_trgsent_rep = ComputeDocTrg_SrcRep(model, dmt, tsent_repi, training_doc);
        dev_trgsent_rep = ComputeDocTrg_SrcRep(model, dmt, dsent_repi, devel_doc);
    }

    cerr << "**SHUFFLE\n";
    shuffle(order.begin(), order.end(), *rndeng);

    Timer timer_epoch("completed in"), timer_iteration("completed in");

    while (sgd.epoch < max_epochs) {
        ModelStats tstats;
        unsigned lines = 0;

        dmt.Enable_Dropout();// enable dropout
        dmt.Enable_Dropout_DocRNN();

        for (unsigned iter = 0; iter < report_every_i; ++iter) {
            if (si == training_doc.size()) {
                //timing
                cerr << "***Epoch " << sgd.epoch << " is finished. ";
                timer_epoch.Show();

                si = 0;

                if (lr_epochs == 0)
                    sgd.update_epoch();

                else sgd.update_epoch(1, lr_epochs); // @vhoang2: learning rate annealing (after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay.

                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);

                if (sgd.epoch >= max_epochs) break;

                timer_epoch.Reset();
            }

            // build graph for this document
            const unsigned tdlen = training_doc[order[si % order.size()]].size();
            vector<vector<unsigned int>> train_mini_sids;

            Create_DocMiniBatch(training_doc[order[si % order.size()]], docminibatch_size,
                                train_src_minidoc, train_trg_minidoc, train_mini_sids);

            vector<vector<dynet::real>> vsent_repi = tsent_repi[order[si % order.size()]];
            vector<vector<dynet::real>> vtrgsent_rep = train_trgsent_rep[order[si % order.size()]];
            //if (DEBUGGING_FLAG){// see http://dynet.readthedocs.io/en/latest/debugging.html
            //    cg.set_immediate_compute(true);
            //    cg.set_check_validity(true);
            //}

            ++si;

            for (unsigned i = 0; i < train_src_minidoc.size(); ++i){
                ComputationGraph cg;

                Expression i_xent = dmt.BuildDocMTSrcTrgGraph_Batch(train_src_minidoc[i], train_trg_minidoc[i],
                                                                 vsent_repi, vtrgsent_rep, train_mini_sids[i],
                                                                 cg, tstats);
                Expression i_objective = i_xent;

                // perform forward computation for aggregate objective
                cg.forward(i_objective);

                // grab the parts of the objective
                tstats.loss += as_scalar(cg.get_value(i_xent.i));

                cg.backward(i_objective);
                //sgd.update();
            }
            sgd.update();
            lines+=tdlen;
        }

        sgd.status();
        double elapsed = timer_iteration.Elapsed();
        cerr << "docs=" << si << " sents=" << lines << " src_unks=" << tstats.words_src_unk << " trg_unks=" << tstats.words_tgt_unk << " E=" << (tstats.loss / tstats.words_tgt) << " ppl=" << exp(tstats.loss / tstats.words_tgt) << ' ';
        cerr << "[time_elapsed=" << elapsed << "(msec)" << " (" << tstats.words_tgt * 1000 / elapsed << " words/sec)]" << endl;

        if (sgd.epoch >= max_epochs) break;

        timer_iteration.Reset();

        // show score on dev data?
        report += report_every_i;
        if (report % dev_every_i_reports == 0) {
            dmt.Disable_Dropout();// disable dropout for evaluating dev data
            dmt.Disable_Dropout_DocRNN();

            ModelStats dstats;
            for (unsigned i = 0; i < devel_doc.size(); ++i) {
                const unsigned ddlen = devel_doc[i].size();
                for (unsigned dl = 0; dl < ddlen; ++dl){
                    vssent.push_back(get<0>(devel_doc[i].at(dl)));
                    vtsent.push_back(get<1>(devel_doc[i].at(dl)));
                }
                vector<vector<dynet::real>> vdsent_repi = dsent_repi[i];
                vector<vector<dynet::real>> vdtrgsent_rep = dev_trgsent_rep[i];

                for (unsigned j = 0; j < vssent.size(); ++j){
                    ComputationGraph cg;

                    auto i_xent = dmt.BuildDocMTSrcTrgGraph(vssent[j], vtsent[j], vdsent_repi, vdtrgsent_rep, j, cg, dstats);
                    dstats.loss += as_scalar(cg.forward(i_xent));
                }
                vssent.clear();
                vtsent.clear();
            }
            if (dstats.loss < best_loss) {
                best_loss = dstats.loss;
                //ofstream out(out_file, ofstream::out);
                //boost::archive::text_oarchive oa(out);
                //oa << model;
                dynet::save_dynet_model(out_file, &model);// FIXME: use binary streaming instead for saving disk spaces
            }

            cerr << "--------------------------------------------------------------------------------------------------------" << endl;
            cerr << "***DEV [eta=" << sgd.eta << "]" << " docs=" << devel_doc.size() << " src_unks=" << dstats.words_src_unk << " trg_unks=" << dstats.words_tgt_unk << " E=" << (dstats.loss / dstats.words_tgt) << " ppl=" << exp(dstats.loss / dstats.words_tgt) << ' ';
            timer_iteration.Show();
            cerr << "--------------------------------------------------------------------------------------------------------" << endl;
        }

        timer_iteration.Reset();
    }

    cerr << endl << "Training completed in " << sgd.epoch << " epochs!" << endl;
}

// --------------------------------------------------------------------------------------------------------------------------------
Corpus Read_Corpus(const string &filename, bool use_joint_vocab)
{
    ifstream in(filename);
    assert(in);
    Corpus corpus;
    string line;
    int lc = 0, stoks = 0, ttoks = 0;
    vector<int> identifiers({ -1 });
    while (getline(in, line)) {
        Sentence source, target;
        if (!use_joint_vocab)
            read_sentence_pair(line, source, sd, target, td);
        else
            read_sentence_pair(line, source, sd, target, sd);

        corpus.push_back(SentencePairID(source, target, identifiers[0]));

        stoks += source.size();
        ttoks += target.size();

        if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
            (target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
            cerr << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
            abort();
        }

        ++lc;
    }

    if (use_joint_vocab)	td = sd;

    if (!use_joint_vocab)
        cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sd.size() << " & " << td.size() << " types\n";
    else
        cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sd.size() << " joint s & t types\n";

    return corpus;
}

SourceDoc Read_TestCorpus(const string &filename)
{
    SourceDoc sourcecor;

    std::ifstream f(filename);

    string line;
    int lc = 0, toks = 0;
    while (std::getline(f, line))
    {
        Sentence source;
        source = read_sentence(line, sd);
        sourcecor.push_back(Sentence(source));

        toks += source.size();

        if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS))
        {
            cerr << "Sentence in " << filename << ":" << lc << " didn't start or end with <s> and </s>\n";
            abort();
        }

        lc++;
    }

    cerr << lc << " lines, " << toks << " tokens, " << sd.size() << " types\n";

    return sourcecor;
}

//function to read the corpus with docid's. Output is a bilingual parallel corpus with docid
Corpus Read_DocCorpus(const string &filename, bool use_joint_vocab)
{
	ifstream in(filename);
	assert(in);
	Corpus corpus;
	string line;
	int lc = 0, stoks = 0, ttoks = 0;
	vector<int> identifiers({ -1 });
	while (getline(in, line)) {
		++lc;
		Sentence source, target;
        if (!use_joint_vocab)
            Read_Numbered_Sentence_Pair(line, &source, &sd, &target, &td, identifiers);
        else
            Read_Numbered_Sentence_Pair(line, &source, &sd, &target, &sd, identifiers);

        corpus.push_back(SentencePairID(source, target, identifiers[0]));

		stoks += source.size();
		ttoks += target.size();

		if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
				(target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
			cerr << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
			abort();
		}
	}

    if (!use_joint_vocab)
        cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sd.size() << " & " << td.size() << " types\n";
    else
        cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sd.size() << " joint s & t types\n";

    return corpus;
}

//function to convert the bilingual parallel corpus with docid to document-level corpus
DocCorpus Read_DocCorpus(Corpus &corpus)
{
    //for loop to create a document level corpus
    Document document;
    DocCorpus doccorpus;
    int docid = 0, prev_docid = 1;
    for (unsigned int index = 0; index < corpus.size(); ++index)
    {
        docid = get<2>(corpus.at(index));
        if (index > 0)
            prev_docid = get<2>(corpus.at(index - 1));
        else
            prev_docid = docid;
        if (docid == prev_docid)
            document.push_back(SentencePair(get<0>(corpus.at(index)),get<1>(corpus.at(index))));
        else{
            doccorpus.push_back(document);
            document.clear();
            document.push_back(SentencePair(get<0>(corpus.at(index)),get<1>(corpus.at(index))));
        }
    }
    doccorpus.push_back(document);	//push the last document read onto the doccorpus
    cerr << doccorpus.size() << " # of documents\n";

    return doccorpus;
}

//function to read the source corpus with docid's. Output is a monolingual corpus with docid
SourceCorpus Read_TestDocCorpus(const string &filename, bool use_joint_vocab)
{
    ifstream in(filename);
    assert(in);
    SourceCorpus scorpus;
    string line;
    int lc = 0, stoks = 0;
    vector<int> identifiers({ -1 });
    while (getline(in, line)) {
        ++lc;
        Sentence source;
        Read_Numbered_Sentence(line, &source, &sd, identifiers);
        scorpus.push_back(SentenceID(source, identifiers[0]));

        stoks += source.size();

        if (source.front() != kSRC_SOS && source.back() != kSRC_EOS) {
            cerr << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
            abort();
        }
    }

    if (!use_joint_vocab)
        cerr << lc << " lines, " << stoks << " tokens (s), " << sd.size() << " & " << td.size() << " types\n";
    else
        cerr << lc << " lines, " << stoks << " tokens (s), " << sd.size() << " joint s & t types\n";

    return scorpus;
}

//function to convert the monolingual corpus with docid to document-level corpus
SourceDocCorpus Read_TestDocCorpus(SourceCorpus &scorpus)
{
    //for loop to create a document level corpus
    SourceDoc sdoc;
    SourceDocCorpus sdoccorpus;
    int docid = 0, prev_docid = 1;
    for (unsigned int index = 0; index < scorpus.size(); ++index)
    {
        docid = get<1>(scorpus.at(index));
        if (index > 0)
            prev_docid = get<1>(scorpus.at(index - 1));
        else
            prev_docid = docid;
        if (docid == prev_docid)
            sdoc.push_back(get<0>(scorpus.at(index)));
        else{
            sdoccorpus.push_back(sdoc);
            sdoc.clear();
            sdoc.push_back(get<0>(scorpus.at(index)));
        }
    }
    sdoccorpus.push_back(sdoc);	//push the last document read onto the sdoccorpus
    cerr << sdoccorpus.size() << " # of documents\n";

    return sdoccorpus;
}

void Read_Numbered_Sentence(const std::string& line, std::vector<int>* s, Dict* sd, vector<int> &identifiers) {
	std::istringstream in(line);
	std::string word;
    std::vector<int>* v = s;
    std::string sep = "|||";
	if (in) {
		identifiers.clear();
		while (in >> word) {
			if (!in || word.empty()) break;
			if (word == sep) break;
			identifiers.push_back(atoi(word.c_str()));
		}
	}

	while(in) {
		in >> word;
		if (!in || word.empty()) break;
		v->push_back(sd->convert(word));
	}

}

void Read_Numbered_Sentence_Pair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, vector<int> &identifiers) 
{
	std::istringstream in(line);
	std::string word;
	std::string sep = "|||";
	Dict* d = sd;
	std::vector<int>* v = s; 

	if (in) {
		identifiers.clear();
		while (in >> word) {
			if (!in || word.empty()) break;
			if (word == sep) break;
			identifiers.push_back(atoi(word.c_str()));
		}
	}

	while(in) {
		in >> word;
		if (!in) break;
		if (word == sep) { d = td; v = t; continue; }
		v->push_back(d->convert(word));
	}
}

void Initialise(Model &model, const string &filename)
{
	cerr << "Initialising model parameters from file: " << filename << endl;
	//ifstream in(filename, ifstream::in);
	//boost::archive::text_iarchive ia(in);
	//ia >> model;
	dynet::load_dynet_model(filename, &model);// FIXME: use binary streaming instead for saving disk spaces
}

const Sentence* GetContext(const Corpus &corpus, unsigned i)
{
	if (i > 0) {
		int docid = get<2>(corpus.at(i));
		int prev_docid = get<2>(corpus.at(i-1));
		if (docid == prev_docid) 
			return &get<0>(corpus.at(i-1));
	} 

	return nullptr;
}