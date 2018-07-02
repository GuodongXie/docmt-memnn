#include "sentrnnlm.h"
#include "math-utils.h"

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

unsigned LAYERS = 1;
unsigned HIDDEN_DIM = 64;  // 1024
unsigned VOCAB_SIZE = 0;    //same as SRC_VOCAB_SIZE but declared again for LM purposes
unsigned MINIBATCH_SIZE = 1;

unsigned TREPORT = 5000;
unsigned DREPORT = 20000;

dynet::Dict d;

bool verbose;

typedef vector<int> Sentence;
typedef pair<Sentence, int> SentenceID;
typedef vector<SentenceID> SourceCorpus;   //used for getting representations

typedef vector<Sentence> Corpus;	//sentence-level corpus
typedef vector<Corpus> SourceDocCorpus;  //document-level corpus

void Initialise(Model &model, const string &filename);

inline size_t Calc_Size(const Sentence & src);
inline void Create_MiniBatches(const Corpus& traincor, size_t max_size,
                               std::vector<std::vector<Sentence> > & traincor_minibatch);

template <class RNNLM_t>
void TrainSentRNNModel(Model &model, RNNLM_t &srnn, Corpus &traincor, Corpus &devcor
        , Trainer &sgd, const string out_file, int max_epochs, int lr_epochs);
template <class RNNLM_t>
void Train_BwdSentRNNModel(Model &model, RNNLM_t &srnn, Corpus &traincor, Corpus &devcor
        , Trainer &sgd, const string out_file, int max_epochs, int lr_epochs);
template <class RNNLM_t>
void TrainSentRNNModel_Batch(Model &model, RNNLM_t &srnn, Corpus &traincor, Corpus &devcor
        , Trainer &sgd, const string out_file, int max_epochs, int lr_epochs);
template <class RNNLM_t>
void Train_BwdSentRNNModel_Batch(Model &model, RNNLM_t &srnn, Corpus &traincor, Corpus &devcor
        , Trainer &sgd, const string out_file, int max_epochs, int lr_epochs);

template <class RNNLM_t> void TestModel(Model &model, RNNLM_t &srnn, const Corpus &testcor);
template <class RNNLM_t> void Test_BwdModel(Model &model, RNNLM_t &srnn, const Corpus &testcor);

template <class RNNLM_t> void GetRepresentations(Model &model, RNNLM_t &srnn, SourceDocCorpus &corpus_doc, const string& out_file);

Corpus Read_Corpus(const string &filename);//read corpus for language modelling i.e. contains only source sentence
void Read_Numbered_Sentence(const std::string& line, std::vector<int>* s, Dict* sd, vector<int> &identifiers);
SourceCorpus Read_DocCorpus(const string &filename);
SourceDocCorpus Read_DocCorpus(SourceCorpus &scorpus);

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
		//options for sentence RNN training
        ("train,t", value<string>(), "file containing training sentences with each line consisting of only source sentence")
        ("devel,d", value<string>(), "file containing development sentences")
        ("test,T", value<string>(), "file containing testing source sentences for computing perplexity scores")
		("get_rep", "get sentence representations using model")
		//-----------------------------------------
		("minibatch_size", value<unsigned>()->default_value(1), "impose the minibatch size for training (support both GPU and CPU); no by default")
        //-----------------------------------------
		("sgd_trainer", value<unsigned>()->default_value(0), "use specific SGD trainer (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam)")
		("sparse_updates", value<bool>()->default_value(true), "enable/disable sparse update(s) for lookup parameter(s); true by default")
        ("r2l_target", "use right-to-left direction for target for training the backward RNN; default not")
		//-----------------------------------------
		("initialise", value<string>(), "load initial parameters for sentence RNN from file")
		("parameters", value<string>(), "save best parameters for sentence RNN to this file")
		("representations", value<string>(), "save representations for sentence RNN to this file")
		//-----------------------------------------
		("layers", value<unsigned>()->default_value(LAYERS), "use <num> layers for sentence RNN components")
		("hidden,h", value<unsigned>()->default_value(HIDDEN_DIM), "use <num> dimensions for recurrent hidden states")
		//-----------------------------------------
		("dropout_f", value<float>()->default_value(0.f), "apply dropout technique (Gal et al., 2015) for forward RNN")
		("dropout_b", value<float>()->default_value(0.f), "apply dropout technique (Gal et al., 2015) for backward RNN")
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
		("treport", value<unsigned>()->default_value(5000), "no. of training instances for reporting current model status on training data")
		("dreport", value<unsigned>()->default_value(20000), "no. of training instances for reporting current model status on development data (dreport = N * treport)")
		//-----------------------------------------
        ("verbose,v", "be extremely chatty")
	;
	store(parse_command_line(argc, argv, opts), vm);
	notify(vm);

	cerr << "PID=" << ::getpid() << endl;
	
	if(!vm.count("get_rep")){
		if (vm.count("help") || vm.count("train") != 1 || (vm.count("devel") != 1 && vm.count("test") != 1)) {
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
	verbose = vm.count("verbose");

	LAYERS = vm["layers"].as<unsigned>();
    HIDDEN_DIM = vm["hidden"].as<unsigned>();

	TREPORT = vm["treport"].as<unsigned>(); 
	DREPORT = vm["dreport"].as<unsigned>(); 
	if (DREPORT % TREPORT != 0) assert("dreport must be divisible by treport.");// to ensure the reporting on development data

	MINIBATCH_SIZE = vm["minibatch_size"].as<unsigned>();

	bool reverse = vm.count("r2l_target");
	
	string flavour = "RNN";
	if (vm.count("lstm"))
	flavour = "LSTM";
	else if (vm.count("gru"))
	flavour = "GRU";

    kSOS = d.convert("<s>");
    kEOS = d.convert("</s>");

	Corpus training, devel, testing;
    SourceCorpus training_src, devel_src, testing_src;
	SourceDocCorpus training_doc, devel_doc, testing_doc;

	if (!vm.count("get_rep")) {
		cerr << "Reading source training data from " << vm["train"].as<string>() << "...\n";
		training = Read_Corpus(vm["train"].as<string>());//contains sentence-level source corpus
		kUNK = d.convert("<unk>");// add <unk> if does not exist!
		d.freeze(); // no new word types allowed
		VOCAB_SIZE = d.size();

		if (vm.count("devel")) {
			cerr << "Reading source dev data from " << vm["devel"].as<string>() << "...\n";
			devel = Read_Corpus(vm["devel"].as<string>());
		}

		if (vm.count("test")) {
			cerr << "Reading source test data from " << vm["test"].as<string>() << "...\n";
			testing = Read_Corpus(vm["test"].as<string>());
		}
	}
	else {
		cerr << "Reading source training data from " << vm["train"].as<string>() << "...\n";
		training_src = Read_DocCorpus(vm["train"].as<string>());//contains sentence-level source corpus with document ID
		training_doc = Read_DocCorpus(training_src);//contains document-level source corpus
		kUNK = d.convert("<unk>");// add <unk> if does not exist!
		d.freeze(); // no new word types allowed
		VOCAB_SIZE = d.size();

		if (vm.count("devel")) {
			cerr << "Reading source dev data from " << vm["devel"].as<string>() << "...\n";
			devel_src = Read_DocCorpus(vm["devel"].as<string>());
			devel_doc = Read_DocCorpus(devel_src);
		}

		if (vm.count("test")) {
			cerr << "Reading source test data from " << vm["test"].as<string>() << "...\n";
			testing_src = Read_DocCorpus(vm["test"].as<string>());
			testing_doc = Read_DocCorpus(testing_src);
		}
	}

    string sfname;
    if (vm.count("parameters"))
        sfname = vm["parameters"].as<string>();
    else {
        ostringstream os;
        os << "lm"
           << '_' << LAYERS
           << '_' << HIDDEN_DIM
           << '_' << flavour
           << "_b" << reverse
           << "-pid" << getpid() << ".params";
        sfname = os.str();
    }

	string repfname;
	if (vm.count("representations"))
		repfname = vm["representations"].as<string>();
	else if (!vm.count("parameters")){
        cout << "Filename to save representations not provided. " << endl;
        return 1;
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
	sgd->eta_decay = vm["lr_eta_decay"].as<float>();
	sgd->sparse_updates_enabled = vm["sparse_updates"].as<bool>();
	if (!sgd->sparse_updates_enabled)
		cerr << "Sparse updates for lookup parameter(s) to be disabled!" << endl;

	cerr << "%% Using " << flavour << " recurrent units" << endl;
	SentRNNLM<rnn_t> srnn(&model, VOCAB_SIZE, LAYERS, HIDDEN_DIM, reverse);

    if(!vm.count("get_rep")){
	if(!reverse){
            cerr << "Sentence forward RNN parameters will be written to: " << sfname << endl;
            srnn.Set_Dropout_SentRNNf(vm["dropout_f"].as<float>());
        }
        else{
            cerr << "Sentence backward RNN parameters will be written to: " << sfname << endl;
            srnn.Set_Dropout_SentRNNb(vm["dropout_b"].as<float>());
        }
    }

    //train reverse/test sentence RNN with initial values
    if (reverse || vm.count("test") || vm.count("get_rep"))
        srnn.LoadModel(model, vm["initialise"].as<string>());
    else
        assert("Forward RNN parameters must be learnt first");

    cerr << "Count of model parameters: " << model.parameter_count() << endl << endl;

	if (vm.count("get_rep")) {
		if (!vm.count("devel") && !vm.count("test")){
                        cout << "Computing sentence representations for training set!" << endl;	
			GetRepresentations(model, srnn, training_doc, repfname);
                }
		else if (vm.count("devel")){
                        cout << "Computing sentence representations for dev set!" << endl;	
			GetRepresentations(model, srnn, devel_doc, repfname);
                }
		else if (vm.count("test")){
                        cout << "Computing sentence representations for test set!" << endl;	
			GetRepresentations(model, srnn, testing_doc, repfname);
                }
	}
	else{
    if (!vm.count("test")) {
        if (!reverse)
            TrainSentRNNModel_Batch(model, srnn, training, devel, *sgd, sfname,
                                       vm["epochs"].as<int>(), vm["lr_epochs"].as<int>());
        else
            Train_BwdSentRNNModel_Batch(model, srnn, training, devel, *sgd, sfname,
                                          vm["epochs"].as<int>(), vm["lr_epochs"].as<int>());
    }
    else{
        if (!reverse){
            cerr << "Testing forward sentence-level RNN model..." << endl;
            TestModel(model, srnn, testing);
        }
        else{
            cerr << "Testing backward sentence-level RNN model..." << endl;
            Test_BwdModel(model, srnn, testing);
        }
    }
}
	cerr << "Cleaning up..." << endl;
	delete sgd;
	//dynet::cleanup();

	return EXIT_SUCCESS;
}

template <class RNNLM_t>
void TestModel(Model &model, RNNLM_t &srnn, const Corpus &testcor)
{
    double dloss = 0;
    unsigned dtokens = 0;
    unsigned id = 0;
    unsigned unk_dtokens = 0;
    for (auto& sent: testcor)
    {
        ComputationGraph cg;
        unsigned tokens = 0, unk_tokens = 0;
        Expression i_xent = srnn.BuildSentRNNGraph(sent, tokens, unk_tokens, cg);

        //more score details for each sentence
        double loss = as_scalar(cg.forward(i_xent));
        //int tokens = sent.size() - 1;
        cerr << id++ << "\t" << loss << "\t" << tokens << "\t" << exp(loss / tokens) << endl;

        dloss += loss;
        dtokens += tokens;
        unk_dtokens += unk_tokens;
    }

    cerr << "-------------------------------------------------------------------------" << endl;
    cerr << "***TEST " << "sentences=" << testcor.size() << " unks=" << unk_dtokens << " E=" << (dloss / dtokens) << " ppl=" << exp(dloss / dtokens) << ' ';
    cerr << "\n-------------------------------------------------------------------------\n" << endl;
}

//flag "r2l_target" should be set for this
template <class RNNLM_t>
void Test_BwdModel(Model &model, RNNLM_t &srnn, const Corpus &testcor)
{
    double dloss = 0;
    unsigned dtokens = 0;
    unsigned id = 0;
    unsigned unk_dtokens = 0;
    for (auto& sent: testcor)
    {
        ComputationGraph cg;
        unsigned tokens = 0, unk_tokens = 0;
        Expression i_xent = srnn.BuildSentRNNGraph(sent, tokens, unk_tokens, cg);

        //more score details for each sentence
        double loss = as_scalar(cg.forward(i_xent));
        //int tokens = sent.size() - 1;
        cerr << id++ << "\t" << loss << "\t" << tokens << "\t" << exp(loss / tokens) << endl;

        dloss += loss;
        dtokens += tokens;
        unk_dtokens += unk_tokens;
    }

    cerr << "-------------------------------------------------------------------------" << endl;
    cerr << "***TEST " << "sentences=" << testcor.size() << " unks=" << unk_dtokens << " E=" << (dloss / dtokens) << " ppl=" << exp(dloss / dtokens) << ' ';
    cerr << "\n-------------------------------------------------------------------------\n" << endl;
}

template <class RNNLM_t>
void GetRepresentations(Model &model, RNNLM_t &srnn, SourceDocCorpus &corpus_doc, const string& out_file)
{
	vector<vector<vector<dynet::real>>> srcsent_rep;
	vector<Sentence> vssent;

	for (unsigned sd = 0; sd < corpus_doc.size(); sd++) {
		const unsigned sdlen = corpus_doc[sd].size();
		for (unsigned dl = 0; dl < sdlen; ++dl) 
			vssent.push_back(corpus_doc[sd].at(dl));
			
		srcsent_rep.push_back(srnn.ComputeSrcSentRepresentations(vssent));		
		vssent.clear();
	}

	ofstream out(out_file);
	boost::archive::text_oarchive oa(out);
	oa << srcsent_rep;
    out.close();
}

template <class RNNLM_t>
void TrainSentRNNModel(Model &model, RNNLM_t &srnn, Corpus &traincor, Corpus &devcor
        , Trainer &sgd, const string out_file, int max_epochs, int lr_epochs)
{
    unsigned report_every_i = TREPORT;
    unsigned devcor_every_i_reports = DREPORT;
    double best = 9e+99;

    vector<unsigned> order((traincor.size() + MINIBATCH_SIZE - 1) / MINIBATCH_SIZE);
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i * MINIBATCH_SIZE;

    cerr << "**SHUFFLE\n";
    shuffle(order.begin(), order.end(), *rndeng);

    unsigned si = 0;//order.size();
    Timer timer_epoch("completed in"), timer_iteration("completed in");

    int report = 0;
    unsigned lines = 0;
    while (sgd.epoch < max_epochs) {
        srnn.Enable_Dropout_SentRNNf();

        double loss = 0;
        unsigned tokens = 0, unk_tokens = 0;
        for (unsigned i = 0; i < report_every_i; ++i, ++si) {
            if (si == order.size()) {
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

                timer_epoch.Reset();
            }

            // build graph for this instance
            ComputationGraph cg;
            unsigned c1 = 0, c2 = 0;
            Expression i_xent = srnn.BuildSentRNNGraph(traincor[order[si]], c1/*tokens*/, c2/*unk_tokens*/, cg);

            float closs = as_scalar(cg.forward(i_xent));// consume the loss
			if (!is_valid(closs)) {
				std::cerr << "***Warning***: nan or -nan values occurred!" << std::endl;
				continue;
			}

            loss += closs;
            tokens += c1;
            unk_tokens += c2;

            cg.backward(i_xent);
            sgd.update();

            lines++;
        }

        if (sgd.epoch >= max_epochs) continue;

        sgd.status();
        cerr << "sents=" << lines << " unks=" << unk_tokens << " E=" << (loss / tokens) << " ppl=" << exp(loss / tokens) << ' ';
        double elapsed = timer_iteration.Elapsed();
        cerr << "[time_elapsed=" << elapsed << "(msec)" << " (" << tokens * 1000.f / elapsed << " words/sec)]" << endl;
        timer_iteration.Reset();

        // show score on devcor data?
        report += report_every_i;
        if (report % devcor_every_i_reports == 0) {
            srnn.Disable_Dropout_SentRNNf();

            double dloss = 0;
            unsigned dtokens = 0, unk_dtokens = 0;
            for (auto& sent: devcor){
                ComputationGraph cg;
                Expression i_xent = srnn.BuildSentRNNGraph(sent, dtokens, unk_dtokens, cg);
                dloss += as_scalar(cg.forward(i_xent));
            }

            if (dloss < best) {
                best = dloss;
                dynet::save_dynet_model(out_file, &model);
            }

            cerr << "\n***DEV [epoch=" << (lines / (double)traincor.size()) << " eta=" << sgd.eta << "]" << " sents=" << devcor.size() << " unks=" << unk_dtokens << " E=" << (dloss / dtokens) << " ppl=" << exp(dloss / dtokens) << ' ';
            timer_iteration.Show();
            timer_iteration.Reset();
        }
    }

    cerr << endl << "Training of forward Sentence RNN completed!" << endl;
}

template <class RNNLM_t>
void Train_BwdSentRNNModel(Model &model, RNNLM_t &srnn, Corpus &traincor, Corpus &devcor
        , Trainer &sgd, const string out_file, int max_epochs, int lr_epochs)
{
    unsigned report_every_i = TREPORT;
    unsigned devcor_every_i_reports = DREPORT;
    double best = 9e+99;

    vector<unsigned> order((traincor.size() + MINIBATCH_SIZE - 1) / MINIBATCH_SIZE);
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i * MINIBATCH_SIZE;

    cerr << "**SHUFFLE\n";
    shuffle(order.begin(), order.end(), *rndeng);

    unsigned si = 0;//order.size();
    Timer timer_epoch("completed in"), timer_iteration("completed in");

    int report = 0;
    unsigned lines = 0;
    while (sgd.epoch < max_epochs) {
        srnn.Enable_Dropout_SentRNNb();

        double loss = 0;
        unsigned tokens = 0, unk_tokens = 0;
        for (unsigned i = 0; i < report_every_i; ++i, ++si) {
            if (si == order.size()) {
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

                timer_epoch.Reset();
            }

            // build graph for this instance
            ComputationGraph cg;
            unsigned c1 = 0, c2 = 0;
            Expression i_xent = srnn.BuildSentRNNGraph(traincor[order[si]], c1/*tokens*/, c2/*unk_tokens*/, cg);

            float closs = as_scalar(cg.forward(i_xent));// consume the loss
            if (!is_valid(closs)) {
                std::cerr << "***Warning***: nan or -nan values occurred!" << std::endl;
                continue;
            }

            loss += closs;
            tokens += c1;
            unk_tokens += c2;

            cg.backward(i_xent);
            sgd.update();

            lines++;
        }

        if (sgd.epoch >= max_epochs) continue;

        sgd.status();
        cerr << "sents=" << lines << " unks=" << unk_tokens << " E=" << (loss / tokens) << " ppl=" << exp(loss / tokens) << ' ';
        double elapsed = timer_iteration.Elapsed();
        cerr << "[time_elapsed=" << elapsed << "(msec)" << " (" << tokens * 1000.f / elapsed << " words/sec)]" << endl;
        timer_iteration.Reset();

        // show score on devcor data?
        report += report_every_i;
        if (report % devcor_every_i_reports == 0) {
            srnn.Disable_Dropout_SentRNNb();

            double dloss = 0;
            unsigned dtokens = 0, unk_dtokens = 0;
            for (auto& sent: devcor){
                ComputationGraph cg;
                Expression i_xent = srnn.BuildSentRNNGraph(sent, dtokens, unk_dtokens, cg);
                dloss += as_scalar(cg.forward(i_xent));
            }

            if (dloss < best) {
                best = dloss;
                dynet::save_dynet_model(out_file, &model);
            }

            cerr << "\n***DEV [epoch=" << (lines / (double)traincor.size()) << " eta=" << sgd.eta << "]" << " sents=" << devcor.size() << " unks=" << unk_dtokens << " E=" << (dloss / dtokens) << " ppl=" << exp(dloss / dtokens) << ' ';
            timer_iteration.Show();
            timer_iteration.Reset();
        }
    }

    cerr << endl << "Training of backward Sentence RNN completed!" << endl;
}

//---------------------------------------------------------------------------------------------------------------------
// The following codes are referenced from lamtram toolkit (https://github.com/neubig/lamtram).
struct SingleLength
{
    SingleLength(const vector<Sentence> & v) : vec(v) { }
    inline bool operator() (int i1, int i2)
    {
        return (vec[i2].size() < vec[i1].size());
    }
    const vector<Sentence> & vec;
};

inline size_t Calc_Size(const Sentence & src) {
    return src.size()+1;
}

inline void Create_MiniBatches(const Corpus& traincor, size_t max_size,
                               std::vector<std::vector<Sentence> > & traincor_minibatch)
{
    std::vector<int> train_ids(traincor.size());
    std::iota(train_ids.begin(), train_ids.end(), 0);

    if(max_size > 1)
        sort(train_ids.begin(), train_ids.end(), SingleLength(traincor));

    std::vector<Sentence> traincor_next;
    size_t first_size = 0;
    for(size_t i = 0; i < train_ids.size(); i++) {
        if (traincor_next.size() == 0)
            first_size = traincor[train_ids[i]].size();

        traincor_next.push_back(traincor[train_ids[i]]);

        if ((traincor_next.size()+1) * first_size > max_size) {
            traincor_minibatch.push_back(traincor_next);
            traincor_next.clear();
        }
    }

    if (traincor_next.size()) traincor_minibatch.push_back(traincor_next);
}
// --------------------------------------------------------------------------------------------------------------------------------

template <class RNNLM_t>
void TrainSentRNNModel_Batch(Model &model, RNNLM_t &srnn, Corpus &traincor, Corpus &devcor
        , Trainer &sgd, const string out_file, int max_epochs, int lr_epochs)
{
    if (MINIBATCH_SIZE == 1){
        TrainSentRNNModel(model, srnn, traincor, devcor, sgd, out_file, max_epochs, lr_epochs);
        return;
    }

    // create minibatches
    std::vector<std::vector<Sentence> > train_minibatch;
    size_t minibatch_size = MINIBATCH_SIZE;
    Create_MiniBatches(traincor, minibatch_size, train_minibatch);

    std::vector<int> train_ids_minibatch(train_minibatch.size());
    std::iota(train_ids_minibatch.begin(), train_ids_minibatch.end(), 0);
    cerr << "**SHUFFLE\n";
    std::shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *dynet::rndeng);

    unsigned report_every_i = TREPORT;
    unsigned dev_every_i_reports = DREPORT;
    double best = 9e+99;
    unsigned si = 0, last_print = 0, lines = 0;
    Timer timer_epoch("completed in"), timer_iteration("completed in");
    while (sgd.epoch < max_epochs) {
        srnn.Enable_Dropout_SentRNNf();

        double loss = 0;
        unsigned tokens = 0, unk_tokens = 0;
        for (unsigned i = 0; i < dev_every_i_reports; ++si) {
            if (si == train_ids_minibatch.size()) {
                //timing
                cerr << "***Epoch " << sgd.epoch << " is finished. ";
                timer_epoch.Show();

                si = 0;
                last_print = 0;
                lines = 0;

                if (lr_epochs == 0)
                    sgd.update_epoch();
                else sgd.update_epoch(1, lr_epochs); // @vhoang2: learning rate annealing (after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay.

                if (sgd.epoch >= max_epochs) break;

                cerr << "**SHUFFLE\n";
                shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *dynet::rndeng);

                timer_epoch.Reset();
            }

            // build graph for this instance
            ComputationGraph cg;
            unsigned c1 = 0, c2 = 0;// intermediate variables
            Expression i_xent = srnn.BuildSentRNNGraphBatch(train_minibatch[train_ids_minibatch[si]], c1/*tokens*/, c2/*unk_tokens*/, cg);

            cg.forward(i_xent);// forward step
            float closs = as_scalar(cg.get_value(i_xent.i));// consume the loss
			if (!is_valid(closs)) {
				std::cerr << "***Warning***: nan or -nan values occurred!" << std::endl;
				continue;
			}

            // update returned values
            loss += closs;
            tokens += c1;
            unk_tokens += c2;

            cg.backward(i_xent);// backward step
            sgd.update();// SGD update step

            lines += train_minibatch[train_ids_minibatch[si]].size();
            i += train_minibatch[train_ids_minibatch[si]].size();

            if (lines / report_every_i != last_print
                || i >= dev_every_i_reports /*|| sgd.epoch >= max_epochs*/
                || si + 1 == train_ids_minibatch.size()){
                last_print = lines / report_every_i;

                sgd.status();
                cerr << "sents=" << lines << " unks=" << unk_tokens << " E=" << (loss / tokens) << " ppl=" << exp(loss / tokens) << ' ';
                double elapsed = timer_iteration.Elapsed();
                cerr << "[time_elapsed=" << elapsed << "(msec)" << " (" << tokens * 1000.f / elapsed << " words/sec)]" << endl;
            }
        }

        timer_iteration.Reset();

        if (sgd.epoch >= max_epochs) continue;

        // show score on devcor data?
        srnn.Disable_Dropout_SentRNNf();

        double dloss = 0;
        unsigned dtokens = 0, unk_dtokens = 0;
        for (auto& sent: devcor){
            ComputationGraph cg;
            Expression i_xent = srnn.BuildSentRNNGraph(sent, dtokens, unk_dtokens, cg);
            dloss += as_scalar(cg.forward(i_xent));
        }

        if (dloss < best) {
            best = dloss;
            dynet::save_dynet_model(out_file, &model);
        }

        cerr << "--------------------------------------------------------------------------------------------------------" << endl;
        cerr << "***DEV [epoch=" << sgd.epoch + lines / (double)traincor.size() << " eta=" << sgd.eta << "]" << " sents=" << devcor.size() << " unks=" << unk_dtokens << " E=" << (dloss / dtokens) << " ppl=" << exp(dloss / dtokens) << ' ';
        timer_iteration.Show();
        cerr << "--------------------------------------------------------------------------------------------------------" << endl;
        timer_iteration.Reset();
    }

    cerr << endl << "Training of forward Sentence RNN completed!" << endl;
}

template <class RNNLM_t>
void Train_BwdSentRNNModel_Batch(Model &model, RNNLM_t &srnn, Corpus &traincor, Corpus &devcor
        , Trainer &sgd, const string out_file, int max_epochs, int lr_epochs)
{
    if (MINIBATCH_SIZE == 1){
        Train_BwdSentRNNModel(model, srnn, traincor, devcor, sgd, out_file, max_epochs, lr_epochs);
        return;
    }

    // create minibatches
    std::vector<std::vector<Sentence> > train_minibatch;
    size_t minibatch_size = MINIBATCH_SIZE;
    Create_MiniBatches(traincor, minibatch_size, train_minibatch);

    std::vector<int> train_ids_minibatch(train_minibatch.size());
    std::iota(train_ids_minibatch.begin(), train_ids_minibatch.end(), 0);
    cerr << "**SHUFFLE\n";
    std::shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *dynet::rndeng);

    unsigned report_every_i = TREPORT;
    unsigned dev_every_i_reports = DREPORT;
    double best = 9e+99;
    unsigned si = 0, last_print = 0, lines = 0;
    Timer timer_epoch("completed in"), timer_iteration("completed in");
    while (sgd.epoch < max_epochs) {
        srnn.Enable_Dropout_SentRNNb();

        double loss = 0;
        unsigned tokens = 0, unk_tokens = 0;
        for (unsigned i = 0; i < dev_every_i_reports; ++si) {
            if (si == train_ids_minibatch.size()) {
                //timing
                cerr << "***Epoch " << sgd.epoch << " is finished. ";
                timer_epoch.Show();

                si = 0;
                last_print = 0;
                lines = 0;

                if (lr_epochs == 0)
                    sgd.update_epoch();
                else sgd.update_epoch(1, lr_epochs); // @vhoang2: learning rate annealing (after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay.

                if (sgd.epoch >= max_epochs) break;

                cerr << "**SHUFFLE\n";
                shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *dynet::rndeng);

                timer_epoch.Reset();
            }

            // build graph for this instance
            ComputationGraph cg;
            unsigned c1 = 0, c2 = 0;// intermediate variables
            Expression i_xent = srnn.BuildSentRNNGraphBatch(train_minibatch[train_ids_minibatch[si]], c1/*tokens*/, c2/*unk_tokens*/, cg);

            cg.forward(i_xent);// forward step
            float closs = as_scalar(cg.get_value(i_xent.i));// consume the loss
            if (!is_valid(closs)) {
                std::cerr << "***Warning***: nan or -nan values occurred!" << std::endl;
                continue;
            }

            // update returned values
            loss += closs;
            tokens += c1;
            unk_tokens += c2;

            cg.backward(i_xent);// backward step
            sgd.update();// SGD update step

            lines += train_minibatch[train_ids_minibatch[si]].size();
            i += train_minibatch[train_ids_minibatch[si]].size();

            if (lines / report_every_i != last_print
                || i >= dev_every_i_reports /*|| sgd.epoch >= max_epochs*/
                || si + 1 == train_ids_minibatch.size()){
                last_print = lines / report_every_i;

                sgd.status();
                cerr << "sents=" << lines << " unks=" << unk_tokens << " E=" << (loss / tokens) << " ppl=" << exp(loss / tokens) << ' ';
                double elapsed = timer_iteration.Elapsed();
                cerr << "[time_elapsed=" << elapsed << "(msec)" << " (" << tokens * 1000.f / elapsed << " words/sec)]" << endl;
            }
        }

        timer_iteration.Reset();

        if (sgd.epoch >= max_epochs) continue;

        // show score on devcor data?
        srnn.Disable_Dropout_SentRNNb();

        double dloss = 0;
        unsigned dtokens = 0, unk_dtokens = 0;
        for (auto& sent: devcor){
            ComputationGraph cg;
            Expression i_xent = srnn.BuildSentRNNGraph(sent, dtokens, unk_dtokens, cg);
            dloss += as_scalar(cg.forward(i_xent));
        }

        if (dloss < best) {
            best = dloss;
            dynet::save_dynet_model(out_file, &model);
        }

        cerr << "--------------------------------------------------------------------------------------------------------" << endl;
        cerr << "***DEV [epoch=" << sgd.epoch + lines / (double)traincor.size() << " eta=" << sgd.eta << "]" << " sents=" << devcor.size() << " unks=" << unk_dtokens << " E=" << (dloss / dtokens) << " ppl=" << exp(dloss / dtokens) << ' ';
        timer_iteration.Show();
        cerr << "--------------------------------------------------------------------------------------------------------" << endl;
        timer_iteration.Reset();
    }

    cerr << endl << "Training of backward Sentence RNN completed!" << endl;
}

//---------------------------------------------------------------------------------------------------------------------
//to read the source file to train Sentence RNN
Corpus Read_Corpus(const string &filename)
{
    Corpus sourcecor;

    std::ifstream f(filename);

    string line;
    int lc = 0, toks = 0;
    while (std::getline(f, line))
    {
        Sentence source;
        source = read_sentence(line, d);
        sourcecor.push_back(Sentence(source));

        toks += source.size();

        if ((source.front() != kSOS && source.back() != kEOS))
        {
            cerr << "Sentence in " << filename << ":" << lc << " didn't start or end with <s> and </s>\n";
            abort();
        }

        lc++;
    }

    cerr << lc << " lines, " << toks << " tokens, " << d.size() << " types\n";

    return sourcecor;
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

SourceCorpus Read_DocCorpus(const string &filename)
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
		Read_Numbered_Sentence(line, &source, &d, identifiers);
		scorpus.push_back(SentenceID(source, identifiers[0]));

		stoks += source.size();

		if (source.front() != kSOS && source.back() != kEOS) {
			cerr << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
			abort();
		}
	}

	cerr << lc << " lines, " << stoks << " tokens (s), " << d.size() << " types\n";
	return scorpus;
}

//function to convert the monolingual corpus with docid to document-level corpus
SourceDocCorpus Read_DocCorpus(SourceCorpus &scorpus)
{
	//for loop to create a document level corpus
	Corpus sdoc;
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
		else {
			sdoccorpus.push_back(sdoc);
			sdoc.clear();
			sdoc.push_back(get<0>(scorpus.at(index)));
		}
	}
	sdoccorpus.push_back(sdoc);	//push the last document read onto the sdoccorpus
	cerr << sdoccorpus.size() << " # of documents\n";

	return sdoccorpus;
}



