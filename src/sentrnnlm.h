#pragma once

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
//#include "dynet/dglstm.h" // FIXME: add this to dynet?
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/globals.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/range/irange.hpp>

int kSOS;
int kEOS;
int kUNK;

using namespace std;

namespace dynet {

template <class Builder>
struct SentRNNLM {
	explicit SentRNNLM(dynet::Model* model,
		unsigned _vocab_size,
        unsigned layers,
        unsigned hidden_dim, bool _reverse,
        LookupParameter* _p_c_fwd=nullptr, LookupParameter* _p_c_bwd=nullptr);

    Expression BuildSentRNNGraph(const vector<int>& sent, unsigned & tokens/*token count*/,
                                      unsigned & unk_tokens/*<unk> token count*/, ComputationGraph& cg);
    Expression BuildSentRNNGraphBatch(const vector<vector<int>>& sents, unsigned & tokens/*token count*/,
                                            unsigned & unk_tokens/*<unk> token count*/, ComputationGraph& cg);

    // enable/disable dropout for Sentence RNNs following Gal et al., 2016 and sentence RNNLM input and output
	void Set_Dropout_SentRNNf(float do_f);
    void Enable_Dropout_SentRNNf();
    void Disable_Dropout_SentRNNf();
    void Set_Dropout_SentRNNb(float do_b);
    void Enable_Dropout_SentRNNb();
    void Disable_Dropout_SentRNNb();
    //---------------------------------------------------------------------------------------------
    //parameters for sentence RNN for input and output memory
    LookupParameter p_c_fwd;
    LookupParameter p_c_bwd;
    Parameter p_R_fwd;
    Parameter p_bias_fwd;
    Parameter p_R_bwd;
    Parameter p_bias_bwd;

	Builder builder_srnn_fwd;
    Builder builder_srnn_bwd;

	bool reverse;

	unsigned vocab_size;

	float dropout_f;
    float dropout_b;

    //for computing sentence representations for docMT model
	std::vector<std::vector<float>> ComputeSrcSentRepresentations(const std::vector<std::vector<int>>& sources);

    void LoadModel(Model &model, const string &filename);

    // Intermediate expressions for sentence-RNNs
    Expression i_R_fwd;
    Expression i_bias_fwd;
    Expression i_R_bwd;
    Expression i_bias_bwd;

};

template <class Builder>
SentRNNLM<Builder>::SentRNNLM(dynet::Model* model,
	unsigned _vocab_size
	, unsigned layers
	, unsigned hidden_dim
	, bool _reverse
	, LookupParameter* _p_c_fwd, LookupParameter* _p_c_bwd)
: builder_srnn_fwd(layers, hidden_dim, hidden_dim, *model),
  builder_srnn_bwd(layers, hidden_dim, hidden_dim, *model),
  reverse(_reverse),
  vocab_size(_vocab_size)
{
    // Add embedding parameters of forward sentence RNNLM to the model
    p_c_fwd = (_p_c_fwd==nullptr)?model->add_lookup_parameters(vocab_size, {hidden_dim}):*_p_c_fwd;
    p_R_fwd = model->add_parameters({vocab_size, hidden_dim});
    p_bias_fwd = model->add_parameters({vocab_size});

    // Add embedding parameters of backward sentence RNNLM to the model
    p_c_bwd = (_p_c_bwd==nullptr)?model->add_lookup_parameters(vocab_size, {hidden_dim}):*_p_c_bwd;
    p_R_bwd = model->add_parameters({vocab_size, hidden_dim});
    p_bias_bwd = model->add_parameters({vocab_size});

	dropout_f = 0.f;
    dropout_b = 0.f;
}

// enable/disable dropout for input and output RNNs
template <class Builder>
void SentRNNLM<Builder>::Set_Dropout_SentRNNf(float do_f)
{
    dropout_f = do_f;
}

template <class Builder>
void SentRNNLM<Builder>::Enable_Dropout_SentRNNf()
{
    builder_srnn_fwd.set_dropout(dropout_f);
}

template <class Builder>
void SentRNNLM<Builder>::Disable_Dropout_SentRNNf()
{
    builder_srnn_fwd.disable_dropout();
}

template <class Builder>
void SentRNNLM<Builder>::Set_Dropout_SentRNNb(float do_b)
{
    dropout_b = do_b;
}

template <class Builder>
void SentRNNLM<Builder>::Enable_Dropout_SentRNNb()
{
    builder_srnn_bwd.set_dropout(dropout_b);
}

template <class Builder>
void SentRNNLM<Builder>::Disable_Dropout_SentRNNb()
{
    builder_srnn_bwd.disable_dropout();
}

template <class Builder>
Expression SentRNNLM<Builder>::BuildSentRNNGraph(const vector<int>& sent, unsigned & tokens/*token count*/,
                                                 unsigned & unk_tokens/*<unk> token count*/, ComputationGraph& cg)
{
    const unsigned slen = sent.size();
    std::vector<int> rsent = sent;

    // Initialize variables for batch errors
    vector<Expression> errs;

    if (!reverse) {
        // Initialize the RNN for a new computation graph
        builder_srnn_fwd.new_graph(cg);
        // Prepare for new sequence (essentially set hidden states to 0)
        builder_srnn_fwd.start_new_sequence();
        // Instantiate embedding parameters in the computation graph
        // output -> word rep parameters (matrix + bias)
        i_R_fwd = parameter(cg, p_R_fwd);
        i_bias_fwd = parameter(cg, p_bias_fwd);

        for (unsigned t = 0; t < slen - 1; ++t) {
            // Count non-EOS words
            tokens++;
            if (rsent[t] == kUNK) unk_tokens++;

            // Embed the current tokens
            Expression i_x_t = lookup(cg, p_c_fwd, rsent[t]);
            // Run one step of the rnn : y_t = RNN(x_t)
            Expression i_y_t = builder_srnn_fwd.add_input(i_x_t);

            // Project to the token space using an affine transform
            Expression i_r_t = i_bias_fwd + i_R_fwd * i_y_t;

            // Compute error for each member of the batch
            Expression i_err = pickneglogsoftmax(i_r_t, rsent[t + 1]);

            errs.push_back(i_err);
        }
    }
    else {
        std::reverse(rsent.begin() + 1/*BOS*/, rsent.end() - 1/*EOS*/);

        // Initialize the RNN for a new computation graph
        builder_srnn_bwd.new_graph(cg);
        // Prepare for new sequence (essentially set hidden states to 0)
        builder_srnn_bwd.start_new_sequence();
        // Instantiate embedding parameters in the computation graph
        // output -> word rep parameters (matrix + bias)
        i_R_bwd = parameter(cg, p_R_bwd);
        i_bias_bwd = parameter(cg, p_bias_bwd);

        for (unsigned t = 0; t < slen - 1; ++t) {
            // Count non-EOS words
            tokens++;
            if (rsent[t] == kUNK) unk_tokens++;

            // Embed the current tokens
            Expression i_x_t = lookup(cg, p_c_bwd, rsent[t]);
            // Run one step of the rnn : y_t = RNN(x_t)
            Expression i_y_t = builder_srnn_bwd.add_input(i_x_t);

            // Project to the token space using an affine transform
            Expression i_r_t = i_bias_bwd + i_R_bwd * i_y_t;

            // Compute error for each member of the batch
            Expression i_err = pickneglogsoftmax(i_r_t, rsent[t + 1]);

            errs.push_back(i_err);
        }
    }
    // Add all errors
    Expression i_nerr = sum(errs);
    return i_nerr;
}

template <class Builder>
Expression SentRNNLM<Builder>::BuildSentRNNGraphBatch(const vector<vector<int>>& sents, unsigned & tokens/*token count*/,
                                                      unsigned & unk_tokens/*<unk> token count*/, ComputationGraph& cg)
{
    std::vector<Expression> errs;

    const unsigned len = sents[0].size() - 1;
    std::vector<unsigned> next_words(sents.size()), words(sents.size());

    if (!reverse) {
        // Initialize the RNN for a new computation graph
        builder_srnn_fwd.new_graph(cg);
        // Prepare for new sequence (essentially set hidden states to 0)
        builder_srnn_fwd.start_new_sequence();
        // Instantiate embedding parameters in the computation graph
        // output -> word rep parameters (matrix + bias)
        i_R_fwd = parameter(cg, p_R_fwd);
        i_bias_fwd = parameter(cg, p_bias_fwd);

        for (unsigned t = 0; t < len; t++){
            for (unsigned bs = 0; bs < sents.size(); ++bs) {
                words[bs] = (t < sents[bs].size()) ? (unsigned) sents[bs][t] : kEOS;
                next_words[bs] = ((t + 1) < sents[bs].size()) ? (unsigned) sents[bs][t + 1] : kEOS;
                if (t < sents[bs].size()) {
                    tokens++;
                    if (sents[bs][t] == kUNK) unk_tokens++;
                }
            }
            // Embed the current tokens
            Expression i_x_t = lookup(cg, p_c_fwd, words);

            // Run one step of the rnn : {y_t} = RNN({x_t})
            Expression i_y_t = builder_srnn_fwd.add_input(i_x_t);

            // Project to the token space using an affine transform
            Expression i_r_t = i_bias_fwd + i_R_fwd * i_y_t;

            // Compute error for each member of the batch
            Expression i_err = pickneglogsoftmax(i_r_t, next_words);

            errs.push_back(i_err);
        }
    }
    else {
        // Initialize the RNN for a new computation graph
        builder_srnn_bwd.new_graph(cg);
        // Prepare for new sequence (essentially set hidden states to 0)
        builder_srnn_bwd.start_new_sequence();
        // Instantiate embedding parameters in the computation graph
        // output -> word rep parameters (matrix + bias)
        i_R_bwd = parameter(cg, p_R_bwd);
        i_bias_bwd = parameter(cg, p_bias_bwd);

        for (unsigned t = 0; t < len; t++){
            auto ct = len - t - 1;
            for(size_t bs = 0; bs < sents.size(); bs++){
                words[bs] = (t < sents[bs].size()) ? ((t != 0/*BOS*/) ? (unsigned)sents[bs][ct+1] : (unsigned)sents[bs][t]) : kEOS;
                next_words[bs] = ((t + 1) < sents[bs].size()) ? ((ct != 0/*EOS*/) ? (unsigned)sents[bs][ct] : (unsigned)sents[bs][t+1]) : kEOS;
                if (t < sents[bs].size()) {
                    tokens++;
                    if (sents[bs][t] == kUNK) unk_tokens++;
                }
            }
            // Embed the current tokens
            Expression i_x_t = lookup(cg, p_c_bwd, words);

            // Run one step of the rnn : {y_t} = RNN({x_t})
            Expression i_y_t = builder_srnn_bwd.add_input(i_x_t);

            // Project to the token space using an affine transform
            Expression i_r_t = i_bias_bwd + i_R_bwd * i_y_t;

            // Compute error for each member of the batch
            Expression i_err = pickneglogsoftmax(i_r_t, next_words);

            errs.push_back(i_err);
        }
    }

    // Add all errors
    Expression i_nerr = sum_batches(sum(errs));
    return i_nerr;
}

//------------------------------------------------------------------------------------------------------------

//compute the hidden representation of source document using Sentence RNN
template <class Builder>
std::vector<std::vector<float>> SentRNNLM<Builder>::ComputeSrcSentRepresentations(const std::vector<std::vector<int>>& sources)
{
    std::vector<std::vector<float>> srcsent_rep;
    Expression i_yfwd_t, i_ybwd_t;

    for (unsigned i = 0; i < sources.size(); i++) {
        dynet::ComputationGraph cg;
        const unsigned slen = sources[i].size();
        std::vector<int> rsent = sources[i];

        builder_srnn_fwd.new_graph(cg);
        builder_srnn_fwd.start_new_sequence();

        for (unsigned t = 0; t < slen - 1; ++t) {
            // Embed the current tokens
            Expression i_x_t = lookup(cg, p_c_fwd, rsent[t]);
            // Run one step of the rnn : y_t = RNN(x_t)
            i_yfwd_t = builder_srnn_fwd.add_input(i_x_t);
        }

        if (reverse) {
            std::reverse(rsent.begin() + 1/*BOS*/, rsent.end() - 1/*EOS*/);

            builder_srnn_bwd.new_graph(cg);
            builder_srnn_bwd.start_new_sequence();

            for (unsigned t = 0; t < slen - 1; ++t) {
                // Embed the current tokens
                Expression i_x_t = lookup(cg, p_c_bwd, rsent[t]);
                // Run one step of the rnn : y_t = RNN(x_t)
                i_ybwd_t = builder_srnn_bwd.add_input(i_x_t);
            }
        }

        //save the last hidden state of each sentence
		if (!reverse) {
			vector<dynet::real> yfwd_t = as_vector(cg.forward(i_yfwd_t));
			srcsent_rep.push_back(yfwd_t);
		}
		else {
			vector<dynet::real> yfwd_t = as_vector(cg.forward(i_yfwd_t));
			vector<dynet::real> ybwd_t = as_vector(cg.forward(i_ybwd_t));
			yfwd_t.insert(yfwd_t.end(), ybwd_t.begin(), ybwd_t.end());
			srcsent_rep.push_back(yfwd_t);
		}
    }

    return srcsent_rep;
}

template <class Builder>
void SentRNNLM<Builder>::LoadModel(Model &model, const string &filename)
{
    cerr << "Initialising model parameters from file: " << filename << endl;
    //ifstream in(filename, ifstream::in);
    //boost::archive::text_iarchive ia(in);
    //ia >> model;
    dynet::load_dynet_model(filename, &model);// FIXME: use binary streaming instead for saving disk spaces
}

}; // namespace dynet