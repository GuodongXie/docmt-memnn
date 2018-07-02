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

//#define RNN_H0_IS_ZERO

int kSRC_SOS;
int kSRC_EOS;
int kSRC_UNK;
int kTGT_SOS;
int kTGT_EOS;
int kTGT_UNK;

using namespace std;

namespace dynet {

struct ModelStats {
	double loss = 0.0f;
	unsigned words_src = 0;
	unsigned words_tgt = 0;
	unsigned words_src_unk = 0;
	unsigned words_tgt_unk = 0;

	ModelStats(){}
};

template <class Builder>
struct DocMTMemNNModel {
	explicit DocMTMemNNModel(dynet::Model* model,
		unsigned _src_vocab_size, unsigned _tgt_vocab_size,
        unsigned slayers, unsigned tlayers,
        unsigned hidden_dim, unsigned align_dim,
        bool _rnn_src_embeddings, bool _rnn_sent_embeddings,
        bool _doc_src_mem, bool _doc_trg_mem, bool _local_mem,
        bool _mem_to_ctx, bool _mem_to_op,
		LookupParameter* _p_cs=nullptr, LookupParameter* _p_ct=nullptr);

    Expression BuildSentMTGraph(const std::vector<int>& source, const std::vector<int>& target,	ComputationGraph& cg, ModelStats& tstats);
    Expression BuildDocMTSrcGraph(const std::vector<int>& source, const std::vector<int>& target,
                                  const std::vector<std::vector<dynet::real>>& srcsent_repi, unsigned i, ComputationGraph& cg, ModelStats& tstats);
    Expression BuildDocMTTrgGraph(const std::vector<int>& source, const std::vector<int>& target,
                                  const std::vector<std::vector<dynet::real>>& trgsent_rep, unsigned i, ComputationGraph& cg, ModelStats& tstats);
    Expression BuildDocMTSrcTrgGraph(const std::vector<int>& source, const std::vector<int>& target,
								     const std::vector<std::vector<dynet::real>>& srcsent_repi, const std::vector<std::vector<dynet::real>>& trgsent_rep,
                                     unsigned i, ComputationGraph& cg, ModelStats& tstats);
    // for supporting mini-batch training
    Expression BuildSentMTGraph_Batch(const std::vector<std::vector<int>>& sources, const std::vector<std::vector<int>>& targets,
                                      ComputationGraph& cg, ModelStats& tstats);
    Expression BuildDocMTSrcGraph_Batch(const std::vector<std::vector<int>>& sources, const std::vector<std::vector<int>>& targets,
                                        const std::vector<std::vector<dynet::real>>& srcsent_repi,
                                        const std::vector<unsigned int> sids, ComputationGraph& cg, ModelStats& tstats);
    Expression BuildDocMTTrgGraph_Batch(const std::vector<std::vector<int>>& sources, const std::vector<std::vector<int>>& targets,
                                        const std::vector<std::vector<dynet::real>>& trgsent_rep, const std::vector<unsigned int> sids, ComputationGraph& cg, ModelStats& tstats);
    Expression BuildDocMTSrcTrgGraph_Batch(const std::vector<std::vector<int>>& sources, const std::vector<std::vector<int>>& targets,
                                           const std::vector<std::vector<dynet::real>>& srcsent_repi, const std::vector<std::vector<dynet::real>>& trgsent_rep,
                                           const std::vector<unsigned int> sids, ComputationGraph& cg, ModelStats& tstats);

    // enable/disable dropout for source and target RNNs following Gal et al., 2016
	void Set_Dropout(float do_enc, float do_dec);
	void Enable_Dropout();
	void Disable_Dropout();
    // enable/disable dropout for Document RNNs following Gal et al., 2016
    void Set_Dropout_DocRNN(float do_df, float do_db);
    void Enable_Dropout_DocRNN();
    void Disable_Dropout_DocRNN();
    //---------------------------------------------------------------------------------------------

	std::vector<int> Greedy_Decode(const std::vector<int> &source, ComputationGraph& cg, Dict &tdict);
    std::vector<int> GreedyDocSrc_Decode(const std::vector<int> &source, const std::vector<std::vector<dynet::real>>& srcsent_repi,
                                         unsigned i, ComputationGraph& cg, dynet::Dict &tdict);
    std::vector<int> GreedyDocTrg_Decode(const std::vector<int> &source, const std::vector<std::vector<dynet::real>>& trgsent_rep,
                                         unsigned i, ComputationGraph& cg, dynet::Dict &tdict);
    std::vector<int> GreedyDocSrcTrg_Decode(const std::vector<int> &source, const std::vector<vector<dynet::real>>& srcsent_repi,
                                            const std::vector<std::vector<dynet::real>>& trgsent_rep, unsigned i, ComputationGraph& cg, dynet::Dict &tdict);

	LookupParameter p_cs;// source vocabulary lookup
	LookupParameter p_ct;// target vocabulary lookup
	Parameter p_R;
	Parameter p_Q;
	Parameter p_P;
	Parameter p_S;
	Parameter p_bias;
	Parameter p_Wa;
	//std::vector<Parameter> p_Wh0;
	Parameter p_Ua;
	Parameter p_va;
    Parameter p_Ws;//for transforming the output response from source memory layer
    Parameter p_Wt;
    Parameter p_Ust;
    Parameter p_bias_t;

    Builder builder;
	Builder builder_src_fwd;
	Builder builder_src_bwd;
    Builder builder_drnn_fwd;
    Builder builder_drnn_bwd;

    bool rnn_src_embeddings;
    bool rnn_sent_embeddings;

	bool src_mem;
    bool trg_mem;
    bool loco_mem;

    bool mem_to_ctx;
    bool mem_to_op;

	unsigned src_vocab_size;
	unsigned tgt_vocab_size;
    const unsigned int h_dim;

	float dropout_dec;
	float dropout_enc;
    float dropout_df;
    float dropout_db;

	// statefull functions for incrementally creating computation graph, one target word at a time
	void StartNewInstance(const std::vector<int> &source, ComputationGraph &cg, ModelStats& tstats);
	void StartNewInstance(const std::vector<int> &source, ComputationGraph &cg);
	void StartNewInstance_Batch(const std::vector<std::vector<int>> &sources, ComputationGraph &cg, ModelStats& tstats);// for supporting mini-batch training

    Expression AddInput(unsigned tgt_tok, unsigned t, ComputationGraph &cg, RNNPointer *prev_state=0);
	Expression AddInput_Batch(const std::vector<unsigned>& tgt_tok, unsigned t, ComputationGraph &cg);// for supporting mini-batch training
    Expression AddDocInput(unsigned trg_tok, unsigned t, ComputationGraph &cg, Expression &i_c_src, Expression &i_c_trg, RNNPointer *prev_state = 0);
	Expression AddDocInput_Batch(const std::vector<unsigned>& trg_words, unsigned t, ComputationGraph &cg, Expression &i_c_src, Expression &i_c_trg);

    void ComputeSrcDocRepresentations(const std::vector<std::vector<dynet::real>>& srcsent_rep, unsigned i, ComputationGraph &cg);
    void ComputeSrcDocRepresentations_Batch(const std::vector<std::vector<dynet::real>>& srcsent_rep,
                                                 const std::vector<unsigned int> sids, ComputationGraph &cg);
    void ComputeTrgDocRepresentations(const std::vector<std::vector<dynet::real>>& trgsent_rep, unsigned i, ComputationGraph &cg);
    void ComputeTrgDocRepresentations_Batch(const std::vector<std::vector<dynet::real>>& trgsent_rep, const std::vector<unsigned int> sids, ComputationGraph &cg);

    std::vector<dynet::real> GetTrgRepresentations(const std::vector<int> &source, ComputationGraph& cg, dynet::Dict &tdict);
	std::vector<dynet::real> GetTrueTrgRepresentations(const std::vector<int> &source, const std::vector<int> &target, ComputationGraph& cg);
	//to get decoder hidden states w.r.t source memory
    std::vector<dynet::real> GetTrg_SrcRepresentations(const std::vector<int> &source, const std::vector<vector<dynet::real>>& srcsent_repi,
                                                       unsigned i, ComputationGraph& cg, dynet::Dict &tdict);
	std::vector<dynet::real> GetTrueTrg_SrcRepresentations(const std::vector<int> &source, const std::vector<int> &target,
                                                           const std::vector<vector<dynet::real>>& srcsent_repi, unsigned i, ComputationGraph& cg);

    // state variables used in the above two methods
	Expression src;
    Expression src_rep;
    Expression i_R;
	Expression i_Q;
	Expression i_P;
	Expression i_bias;
	Expression i_uax;
	Expression i_va;
	Expression i_Wa;
	Expression i_Ua;
    Expression i_Ws;//parameter for source memory transformation
    Expression i_Wt;//parameter for target memory transformation
    Expression i_Ust;
    Expression i_bias_t;

    Expression i_zsrc;
    Expression i_zsrc_rep; //used for sentence-level model training in batch mode
    Expression i_ztrg;
    Expression i_ztrg_rep;

    Expression src_doco_rep;
    Expression src_loco_rep;
    Expression trg_rep;
    Expression trg_loco_rep;
    Expression trg_doco_rep;

	unsigned slen; // source sentence length
};

#define WTF(expression) \
	std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;
#define KTHXBYE(expression) \
	std::cout << *cg.get_value(expression.i) << std::endl;

#define LOLCAT(expression) \
	WTF(expression) \
	KTHXBYE(expression) 

template <class Builder>
DocMTMemNNModel<Builder>::DocMTMemNNModel(dynet::Model* model,
	unsigned _src_vocab_size, unsigned _tgt_vocab_size
	, unsigned slayers, unsigned tlayers
	, unsigned hidden_dim, unsigned align_dim
	, bool _rnn_src_embeddings, bool _rnn_sent_embeddings
	, bool _doc_src_mem, bool _doc_trg_mem, bool _local_mem
    , bool _mem_to_ctx, bool _mem_to_op
	, LookupParameter* _p_cs, LookupParameter* _p_ct)
: builder_src_fwd(slayers, hidden_dim, hidden_dim, *model),
  builder_src_bwd(slayers, hidden_dim, hidden_dim, *model),
  rnn_src_embeddings(_rnn_src_embeddings),
  rnn_sent_embeddings(_rnn_sent_embeddings),
  src_mem(_doc_src_mem),
  trg_mem(_doc_trg_mem),
  loco_mem(_local_mem),
  mem_to_ctx(_mem_to_ctx),
  mem_to_op(_mem_to_op),
  src_vocab_size(_src_vocab_size),
  tgt_vocab_size(_tgt_vocab_size),
  h_dim(hidden_dim)
{
    if (mem_to_ctx)
        builder = Builder(tlayers, (_rnn_src_embeddings) ? 6 * hidden_dim : 4 * hidden_dim, hidden_dim, *model);
    else
        builder = Builder(tlayers, (_rnn_src_embeddings) ? 3 * hidden_dim : 2 * hidden_dim, hidden_dim, *model);

	p_cs = (_p_cs==nullptr)?model->add_lookup_parameters(src_vocab_size, {hidden_dim}):*_p_cs;
	p_ct = (_p_ct==nullptr)?model->add_lookup_parameters(tgt_vocab_size, {hidden_dim}):*_p_ct;
    p_R = model->add_parameters({tgt_vocab_size, hidden_dim});
	p_P = model->add_parameters({hidden_dim, hidden_dim});
	p_bias = model->add_parameters({tgt_vocab_size});
	p_Wa = model->add_parameters({align_dim, tlayers*hidden_dim});
    //parameters for source memory
    builder_drnn_fwd = Builder(slayers, (_rnn_sent_embeddings) ? 2 * hidden_dim : hidden_dim, hidden_dim, *model);
    builder_drnn_bwd = Builder(slayers, (_rnn_sent_embeddings) ? 2 * hidden_dim : hidden_dim, hidden_dim, *model);
    //parameters for target memory
	p_Ust = model->add_parameters({ hidden_dim, 2 * hidden_dim });
	p_bias_t = model->add_parameters({ 1 });

    if (mem_to_op)
        p_Wt = model->add_parameters({ tgt_vocab_size, hidden_dim });

    if (rnn_src_embeddings) {
		p_Ua = model->add_parameters({align_dim, 2*hidden_dim});
		p_Q = model->add_parameters({hidden_dim, 2*hidden_dim});
        if (mem_to_op)
            p_Ws = model->add_parameters({tgt_vocab_size, 2*hidden_dim});
	} 
	else {
		p_Ua = model->add_parameters({align_dim, hidden_dim});
		p_Q = model->add_parameters({hidden_dim, hidden_dim});
        if (mem_to_op)
            p_Ws = model->add_parameters({tgt_vocab_size, hidden_dim});
    }

	p_va = model->add_parameters({align_dim});

    //int hidden_layers = builder.num_h0_components();
	//for (int l = 0; l < hidden_layers; ++l) {
	//if (rnn_src_embeddings)
	//	p_Wh0.push_back(model->add_parameters({hidden_dim, 2*hidden_dim}));
	//else
	//	p_Wh0.push_back(model->add_parameters({hidden_dim, hidden_dim}));
	//}

	dropout_dec = 0.f;
	dropout_enc = 0.f;
    dropout_df = 0.f;
    dropout_db = 0.f;
}

// enable/disable dropout for source and target RNNs
template <class Builder>
void DocMTMemNNModel<Builder>::Set_Dropout(float do_enc, float do_dec)
{
	dropout_dec = do_dec;
	dropout_enc = do_enc;
}

template <class Builder>
void DocMTMemNNModel<Builder>::Enable_Dropout()
{
	builder.set_dropout(dropout_dec);
	builder_src_fwd.set_dropout(dropout_enc);
	builder_src_bwd.set_dropout(dropout_enc);
}

template <class Builder>
void DocMTMemNNModel<Builder>::Disable_Dropout()
{
	builder.disable_dropout();
	builder_src_fwd.disable_dropout();
	builder_src_bwd.disable_dropout();
}

// enable/disable dropout for input and output Document RNNs
template <class Builder>
void DocMTMemNNModel<Builder>::Set_Dropout_DocRNN(float do_df, float do_db)
{
    dropout_df = do_df;
    dropout_db = do_db;
}

template <class Builder>
void DocMTMemNNModel<Builder>::Enable_Dropout_DocRNN()
{
    builder_drnn_fwd.set_dropout(dropout_df);
    builder_drnn_bwd.set_dropout(dropout_db);
}

template <class Builder>
void DocMTMemNNModel<Builder>::Disable_Dropout_DocRNN()
{
    builder_drnn_fwd.disable_dropout();
    builder_drnn_bwd.disable_dropout();
}

template <class Builder>
void DocMTMemNNModel<Builder>::StartNewInstance(const std::vector<int> &source, ComputationGraph &cg)
{
	slen = source.size();
	std::vector<Expression> source_embeddings;
	if (!rnn_src_embeddings) {
		for (unsigned s = 0; s < slen; ++s)
			source_embeddings.push_back(lookup(cg, p_cs, source[s]));
        if (src_mem || trg_mem)
            src_rep = source_embeddings[slen - 1];
        if (!src_mem && mem_to_ctx)  i_zsrc = zeroes(cg, { h_dim });
	} 
	else {
		// run a RNN backward and forward over the source sentence
		// and stack the top-level hidden states from each model as 
		// the representation at each position
		std::vector<Expression> src_fwd(slen);
		builder_src_fwd.new_graph(cg);
		builder_src_fwd.start_new_sequence();
		for (unsigned i = 0; i < slen; ++i)
			src_fwd[i] = builder_src_fwd.add_input(lookup(cg, p_cs, source[i]));

		std::vector<Expression> src_bwd(slen);
		builder_src_bwd.new_graph(cg);
		builder_src_bwd.start_new_sequence();
		for (int i = slen-1; i >= 0; --i) {
			// offset by one position to the right, to catch </s> and generally
			// not duplicate the w_t already captured in src_fwd[t]
			src_bwd[i] = builder_src_bwd.add_input(lookup(cg, p_cs, source[i]));
		}

		for (unsigned i = 0; i < slen; ++i) 
			source_embeddings.push_back(concatenate(std::vector<Expression>({src_fwd[i], src_bwd[i]})));

        if (src_mem || trg_mem)
            src_rep = concatenate(std::vector<Expression>({src_fwd[slen - 1], src_bwd[0]}));
        if (!src_mem && mem_to_ctx)  i_zsrc = zeroes(cg, { 2*h_dim });
    }
    src = concatenate_cols(source_embeddings);
    if (!trg_mem && mem_to_ctx)    i_ztrg = zeroes(cg, { h_dim });

	// now for the target sentence
	i_R = parameter(cg, p_R); // hidden -> word rep parameter
	i_Q = parameter(cg, p_Q);
	i_P = parameter(cg, p_P);
	i_bias = parameter(cg, p_bias);  // word bias
	i_Wa = parameter(cg, p_Wa); 
	i_Ua = parameter(cg, p_Ua);
	i_va = parameter(cg, p_va);
    i_uax = i_Ua * src;

    if (mem_to_op){
        if (src_mem)   i_Ws = parameter(cg, p_Ws);
        if (trg_mem)   i_Wt = parameter(cg, p_Wt);
    }

    if (trg_mem){
        if (!loco_mem){
            i_Ust = parameter(cg, p_Ust);
            i_bias_t = parameter(cg, p_bias_t);
        }
    }

	// initialise h from global information of the source sentence
    /*
#ifndef RNN_H0_IS_ZERO
	std::vector<Expression> h0;
	Expression i_src = average(source_embeddings); // try max instead?

	int hidden_layers = builder.num_h0_components();
	for (int l = 0; l < hidden_layers; ++l) {
		Expression i_Wh0 = parameter(cg, p_Wh0[l]);
		h0.push_back(tanh(i_Wh0 * i_src));
	}

	builder.new_graph(cg); 
	builder.start_new_sequence(h0);
#else*/
	builder.new_graph(cg); 
	builder.start_new_sequence();
//#endif

}

template <class Builder>
void DocMTMemNNModel<Builder>::StartNewInstance(const std::vector<int> &source, ComputationGraph &cg, ModelStats& tstats)
{
	tstats.words_src += source.size() - 1;

	slen = source.size(); 
	std::vector<Expression> source_embeddings;
	if (!rnn_src_embeddings) {
		for (unsigned s = 0; s < slen; ++s){
			if (source[s] == kSRC_UNK) tstats.words_src_unk++;
			source_embeddings.push_back(lookup(cg, p_cs, source[s]));
		}
        if (src_mem || trg_mem)
            src_rep = source_embeddings[slen - 1];
        if (!src_mem && mem_to_ctx)  i_zsrc = zeroes(cg, { h_dim });
	} 
	else {
		// run a RNN backward and forward over the source sentence
		// and stack the top-level hidden states from each model as 
		// the representation at each position
		std::vector<Expression> src_fwd(slen);
		builder_src_fwd.new_graph(cg);
		builder_src_fwd.start_new_sequence();
		for (unsigned i = 0; i < slen; ++i){ 
			if (source[i] == kSRC_UNK) tstats.words_src_unk++;		
			src_fwd[i] = builder_src_fwd.add_input(lookup(cg, p_cs, source[i]));
		}

		std::vector<Expression> src_bwd(slen);
		builder_src_bwd.new_graph(cg);
		builder_src_bwd.start_new_sequence();
		for (int i = slen-1; i >= 0; --i) {
			// offset by one position to the right, to catch </s> and generally
			// not duplicate the w_t already captured in src_fwd[t]
			src_bwd[i] = builder_src_bwd.add_input(lookup(cg, p_cs, source[i]));
		}

		for (unsigned i = 0; i < slen; ++i) 
			source_embeddings.push_back(concatenate(std::vector<Expression>({src_fwd[i], src_bwd[i]})));

        if (src_mem || trg_mem)
            src_rep = concatenate(std::vector<Expression>({src_fwd[slen - 1], src_bwd[0]}));
        if (!src_mem && mem_to_ctx)  i_zsrc = zeroes(cg, { 2*h_dim });
	}
    src = concatenate_cols(source_embeddings);
    if (!trg_mem && mem_to_ctx)    i_ztrg = zeroes(cg, { h_dim });

	// now for the target sentence
	i_R = parameter(cg, p_R); // hidden -> word rep parameter
	i_Q = parameter(cg, p_Q);
	i_P = parameter(cg, p_P);
	i_bias = parameter(cg, p_bias);  // word bias
	i_Wa = parameter(cg, p_Wa); 
	i_Ua = parameter(cg, p_Ua);
	i_va = parameter(cg, p_va);
    i_uax = i_Ua * src;

    if (mem_to_op){
        if (src_mem)   i_Ws = parameter(cg, p_Ws);
        if (trg_mem)   i_Wt = parameter(cg, p_Wt);
    }

    if (trg_mem){
        if (!loco_mem){
            i_Ust = parameter(cg, p_Ust);
            i_bias_t = parameter(cg, p_bias_t);
        }
    }
    // initialise h from global information of the source sentence
    /*
#ifndef RNN_H0_IS_ZERO
	std::vector<Expression> h0;
	Expression i_src = average(source_embeddings); // try max instead?

	int hidden_layers = builder.num_h0_components();
	for (int l = 0; l < hidden_layers; ++l) {
		Expression i_Wh0 = parameter(cg, p_Wh0[l]);
		h0.push_back(tanh(i_Wh0 * i_src));
	}

	builder.new_graph(cg); 
	builder.start_new_sequence(h0);
#else*/
	builder.new_graph(cg); 
	builder.start_new_sequence();
//#endif

}

template <class Builder>
void DocMTMemNNModel<Builder>::StartNewInstance_Batch(const std::vector<std::vector<int>> &sources
		, ComputationGraph &cg, ModelStats& tstats)
{
	// Get the max size
	size_t max_len = sources[0].size();
	for(size_t i = 1; i < sources.size(); i++) max_len = std::max(max_len, sources[i].size());

	slen = max_len;
	std::vector<unsigned> words(sources.size());
	std::vector<Expression> source_embeddings, zero_batch, zerotrg_batch;
	//cerr << "(1a) embeddings" << endl;
	if (!rnn_src_embeddings) {
		for (unsigned l = 0; l < max_len; l++){
			for (unsigned bs = 0; bs < sources.size(); ++bs){
				words[bs] = (l < sources[bs].size()) ? (unsigned)sources[bs][l] : kSRC_EOS;
				if (l < sources[bs].size()){ 
					tstats.words_src++; 
					if (sources[bs][l] == kSRC_UNK) tstats.words_src_unk++;
				}
			}
			source_embeddings.push_back(lookup(cg, p_cs, words));
		}
        if (src_mem || trg_mem)
            src_rep = source_embeddings[max_len - 1];
        if (!src_mem && mem_to_ctx)  i_zsrc = zeroes(cg, { h_dim });
	}
	else {
		// run a RNN backward and forward over the source sentence
		// and stack the top-level hidden states from each model as 
		// the representation at each position
		std::vector<Expression> src_fwd(max_len);
		builder_src_fwd.new_graph(cg);
		builder_src_fwd.start_new_sequence();
		for (unsigned l = 0; l < max_len; l++){
			for (unsigned bs = 0; bs < sources.size(); ++bs){
				words[bs] = (l < sources[bs].size()) ? (unsigned)sources[bs][l] : kSRC_EOS;
				if (l < sources[bs].size()){ 
					tstats.words_src++; 
					if (sources[bs][l] == kSRC_UNK) tstats.words_src_unk++;
				}
			}
			src_fwd[l] = builder_src_fwd.add_input(lookup(cg, p_cs, words));
		}

		std::vector<Expression> src_bwd(max_len);
		builder_src_bwd.new_graph(cg);
		builder_src_bwd.start_new_sequence();
		for (int l = max_len - 1; l >= 0; --l) { // int instead of unsigned for negative value of l
			// offset by one position to the right, to catch </s> and generally
			// not duplicate the w_t already captured in src_fwd[t]
			for (unsigned bs = 0; bs < sources.size(); ++bs) 
				words[bs] = ((unsigned)l < sources[bs].size()) ? (unsigned)sources[bs][l] : kSRC_EOS;
			src_bwd[l] = builder_src_bwd.add_input(lookup(cg, p_cs, words));
		}

		for (unsigned l = 0; l < max_len; ++l) 
			source_embeddings.push_back(concatenate(std::vector<Expression>({src_fwd[l], src_bwd[l]})));

        if (src_mem || trg_mem)
            src_rep = concatenate(std::vector<Expression>({src_fwd[max_len - 1], src_bwd[0]}));
        if (!src_mem && mem_to_ctx)  i_zsrc = zeroes(cg, { 2*h_dim });
	}
    if (!trg_mem && mem_to_ctx)    i_ztrg = zeroes(cg, { h_dim });

	/*
    //compute zero vector in batch mode for memory context representation
    if (mem_to_ctx){
        if (!src_mem){
            for (unsigned l = 0; l < max_len; ++l) {
                std::vector<Expression> temp_zero;
                for (unsigned bs = 0; bs < sources.size(); ++bs)
                    temp_zero.push_back(i_zsrc);
                zero_batch.push_back(concatenate_to_batch(temp_zero));
            }
            i_zsrc_rep = concatenate_cols(zero_batch);
        }
        if (!trg_mem){
            for (unsigned l = 0; l < max_len; ++l) {
                std::vector<Expression> temp_zerotrg;
                for (unsigned bs = 0; bs < sources.size(); ++bs)
                    temp_zerotrg.push_back(i_ztrg);
                zerotrg_batch.push_back(concatenate_to_batch(temp_zerotrg));
            }
            i_ztrg_rep = concatenate_cols(zerotrg_batch);
        }
    }
	*/

	src = concatenate_cols(source_embeddings);
	// now for the target sentence
	i_R = parameter(cg, p_R); // hidden -> word rep parameter
	i_Q = parameter(cg, p_Q);
	i_P = parameter(cg, p_P);
	i_bias = parameter(cg, p_bias);  // word bias
	i_Wa = parameter(cg, p_Wa); 
	i_Ua = parameter(cg, p_Ua);
	i_va = parameter(cg, p_va);
    i_uax = i_Ua * src;

    if (mem_to_op){
        if (src_mem)   i_Ws = parameter(cg, p_Ws);
        if (trg_mem)   i_Wt = parameter(cg, p_Wt);
    }
    if (trg_mem){
        if (!loco_mem){
            i_Ust = parameter(cg, p_Ust);
            i_bias_t = parameter(cg, p_bias_t);
        }
    }

	// initialise h from global information of the source sentence
	//cerr << "(1e) init builder" << endl;
    /*
#ifndef RNN_H0_IS_ZERO
	std::vector<Expression> h0;
	Expression i_src = average(source_embeddings); // try max instead?
	int hidden_layers = builder.num_h0_components();

	for (int l = 0; l < hidden_layers; ++l) {
		Expression i_Wh0 = parameter(cg, p_Wh0[l]);
		h0.push_back(tanh(i_Wh0 * i_src));
	}

	builder.new_graph(cg); 
	builder.start_new_sequence(h0);
#else*/
	builder.new_graph(cg); 
	builder.start_new_sequence();
//#endif
}

template <class Builder>
Expression DocMTMemNNModel<Builder>::AddInput(unsigned trg_tok, unsigned t, ComputationGraph &cg, RNNPointer *prev_state)
{
	// alignment input 
	Expression i_wah_rep;
	if (t > 0) {
		Expression i_h_tm1;
		if (prev_state)
			i_h_tm1 = concatenate(builder.get_h(*prev_state));// This is required for beam search decoding implementation.
		else
			i_h_tm1 = concatenate(builder.final_h());

		Expression i_wah = i_Wa * i_h_tm1;
	
		// want numpy style broadcasting, but have to do this manually
		i_wah_rep = concatenate_cols(std::vector<Expression>(slen, i_wah));
	}

	Expression i_e_t;
	if (t > 0)
		i_e_t = transpose(tanh(i_wah_rep + i_uax)) * i_va;
	else
		i_e_t = transpose(tanh(i_uax)) * i_va;

	Expression i_alpha_t = softmax(i_e_t); // FIXME: consider summing to less than one?
	Expression i_c_t = src * i_alpha_t; // FIXME: effectively summing here, consider maxing?

	// word input
	Expression i_x_t = lookup(cg, p_ct, trg_tok);
    Expression input;
    if (mem_to_ctx)
        input = concatenate(std::vector<Expression>({i_x_t, i_c_t, i_zsrc, i_ztrg}));
    else
        input = concatenate(std::vector<Expression>({i_x_t, i_c_t}));

	// y_t = RNN([x_t, a_t])
	Expression i_y_t;
	if (prev_state)
	   i_y_t = builder.add_input(*prev_state, input);
	else
	   i_y_t = builder.add_input(input);
	
#ifndef VANILLA_TARGET_LSTM
	// Bahdanau does a max-out thing here; I do a tanh. Tomaatos tomateos.
	Expression i_tildet_t = tanh(affine_transform({i_y_t, i_Q, i_c_t, i_P, i_x_t}));
	Expression i_r_t = affine_transform({i_bias, i_R, i_tildet_t}); 
#else
	Expression i_r_t = affine_transform({i_bias, i_R, i_y_t}); 
#endif

	return i_r_t;
}

template <class Builder>
Expression DocMTMemNNModel<Builder>::AddInput_Batch(const std::vector<unsigned>& trg_words, unsigned t, ComputationGraph &cg)
{
	// alignment input 
	Expression i_wah_rep;
	if (t > 0) {
		Expression i_h_tm1 = concatenate(builder.final_h());
		Expression i_wah = i_Wa * i_h_tm1;

		// want numpy style broadcasting, but have to do this manually
		i_wah_rep = concatenate_cols(std::vector<Expression>(slen, i_wah));
	}

	Expression i_e_t;
	if (t > 0)
		i_e_t = transpose(tanh(i_wah_rep + i_uax)) * i_va;
	else
		i_e_t = transpose(tanh(i_uax)) * i_va;

	Expression i_alpha_t = softmax(i_e_t); // FIXME: consider summing to less than one?
	Expression i_c_t = src * i_alpha_t; // FIXME: effectively summing here, consider maxing?

    // target word inputs
	Expression i_x_t = lookup(cg, p_ct, trg_words);
    Expression input;
    if (mem_to_ctx)
        input = concatenate(std::vector<Expression>({i_x_t, i_c_t, i_zsrc, i_ztrg}));
    else
        input = concatenate(std::vector<Expression>({i_x_t, i_c_t}));

	// y_t = RNN([x_t, a_t])
	Expression i_y_t = builder.add_input(input);

#ifndef VANILLA_TARGET_LSTM
	// Bahdanau does a max-out thing here; I do a tanh. Tomaatos tomateos.
	Expression i_tildet_t = tanh(affine_transform({ i_y_t, i_Q, i_c_t, i_P, i_x_t }));
	Expression i_r_t = affine_transform({ i_bias, i_R, i_tildet_t });
#else
	Expression i_r_t = affine_transform({ i_bias, i_R, i_y_t });
#endif

	return i_r_t;
}

template <class Builder>
Expression DocMTMemNNModel<Builder>::BuildSentMTGraph(const std::vector<int> &source, const std::vector<int>& target,
                                                          ComputationGraph& cg, ModelStats& tstats)
{
	//std::cout << "source sentence length: " << source.size() << " target: " << target.size() << std::endl;
	StartNewInstance(source, cg, tstats);

	std::vector<Expression> errs;
	const unsigned tlen = target.size() - 1; 
	for (unsigned t = 0; t < tlen; ++t) {
		tstats.words_tgt++;
		if (target[t] == kTGT_UNK) tstats.words_tgt_unk++;

		Expression i_r_t = AddInput(target[t], t, cg);
		Expression i_err = pickneglogsoftmax(i_r_t, target[t+1]);
		errs.push_back(i_err);
	}

	Expression i_nerr = sum(errs);
	return i_nerr;
}


template <class Builder>
Expression DocMTMemNNModel<Builder>::BuildSentMTGraph_Batch(const std::vector<std::vector<int>> &sources, const std::vector<std::vector<int>>& targets,
                                                                ComputationGraph& cg, ModelStats& tstats)
{
	StartNewInstance_Batch(sources, cg, tstats);

	std::vector<Expression> errs;

	const unsigned tlen = targets[0].size() - 1;
	std::vector<unsigned> next_words(targets.size()), words(targets.size());

	for (unsigned t = 0; t < tlen; ++t) {
		for (size_t bs = 0; bs < targets.size(); bs++) {
			words[bs] = (targets[bs].size() > t) ? (unsigned)targets[bs][t] : kTGT_EOS;
			next_words[bs] = (targets[bs].size() >(t + 1)) ? (unsigned)targets[bs][t + 1] : kTGT_EOS;
			if (targets[bs].size() > t) {
				tstats.words_tgt++;
				if (targets[bs][t] == kTGT_UNK) tstats.words_tgt_unk++;
			}
		}

		Expression i_r_t = AddInput_Batch(words, t, cg);
		Expression i_err = pickneglogsoftmax(i_r_t, next_words);

		errs.push_back(i_err);
	}

	Expression i_nerr = sum_batches(sum(errs));
	return i_nerr;
}

//------------------------------------------------------------------------------------------------------
//compute the hidden representation using input Document RNN
template <class Builder>
void DocMTMemNNModel<Builder>::ComputeSrcDocRepresentations(const std::vector<std::vector<dynet::real>>& srcsent_rep,
	unsigned i, ComputationGraph &cg)
{
	std::vector<Expression> srcdoc_rep;
	const unsigned dlen = srcsent_rep.size();
	std::vector<Expression> document_embeddings;
	std::vector<Expression> localsent_embeddings;

	// run a RNN backward and forward over the source sentence
	// and stack the top-level hidden states from each model as
	// the representation at each position
	std::vector<Expression> src_fwd(dlen);
	builder_drnn_fwd.new_graph(cg);
	builder_drnn_fwd.start_new_sequence();

	for (unsigned t = 0; t < dlen; ++t) {
		const unsigned hdim = srcsent_rep[t].size();
		Expression i_x_t = input(cg, { hdim }, srcsent_rep[t]);
		src_fwd[t] = builder_drnn_fwd.add_input(i_x_t);
	}

	std::vector<Expression> src_bwd(dlen);
	builder_drnn_bwd.new_graph(cg);
	builder_drnn_bwd.start_new_sequence();

	for (int t = dlen - 1; t >= 0; --t) { // int instead of unsigned for negative value of l
										  // offset by one position to the right, to catch </s> and generally
										  // not duplicate the w_t already captured in src_fwd[t]
		const unsigned hdim = srcsent_rep[t].size();
		Expression i_x_t = input(cg, { hdim }, srcsent_rep[t]);
		src_bwd[t] = builder_drnn_bwd.add_input(i_x_t);
	}

	//create a vector of hidden representations excluding the current sentence
	for (unsigned t = 0; t < dlen; ++t) {
		if (t != i)
			document_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[t], src_bwd[t] })));
	}
	//only select the embeddings of previous and next sentence if considering local context
	if (loco_mem) {
		if (i == 0 || (i == dlen - 1))
			localsent_embeddings.push_back(concatenate(std::vector<Expression>({ zeroes(cg,{ h_dim }), zeroes(cg,{ h_dim }) })));
		for (unsigned t = 0; t < dlen - 1; ++t) {
			if (t == i - 1 || t == i)
				localsent_embeddings.push_back(document_embeddings[t]);
		}
		src_loco_rep = average(localsent_embeddings);
	}

	src_doco_rep = concatenate_cols(document_embeddings);
}

template <class Builder>
void DocMTMemNNModel<Builder>::ComputeSrcDocRepresentations_Batch(const std::vector<std::vector<dynet::real>>& srcsent_rep,
	const std::vector<unsigned int> sids, ComputationGraph &cg)
{
	std::vector<Expression> srcdoc_rep;
	const unsigned dlen = srcsent_rep.size();
	const unsigned int bs = sids.size();//gives the number of sentences in the batch
	std::vector<std::vector<Expression>> temp_doc_embeddings(bs);
	std::vector<std::vector<Expression>> temp_local_embeddings(bs);
	std::vector<Expression> document_embeddings, localsent_embeddings;

	// run a RNN backward and forward over the source sentence
	// and stack the top-level hidden states from each model as
	// the representation at each position
	std::vector<Expression> src_fwd(dlen);
	builder_drnn_fwd.new_graph(cg);
	builder_drnn_fwd.start_new_sequence();

	for (unsigned t = 0; t < dlen; ++t) {
		const unsigned hdim = srcsent_rep[t].size();
		Expression i_x_t = input(cg, { hdim }, srcsent_rep[t]);
		src_fwd[t] = builder_drnn_fwd.add_input(i_x_t);
	}

	std::vector<Expression> src_bwd(dlen);
	builder_drnn_bwd.new_graph(cg);
	builder_drnn_bwd.start_new_sequence();

	for (int t = dlen - 1; t >= 0; --t) { // int instead of unsigned for negative value of l
										  // offset by one position to the right, to catch </s> and generally
										  // not duplicate the w_t already captured in src_fwd[t]
		const unsigned hdim = srcsent_rep[t].size();
		Expression i_x_t = input(cg, { hdim }, srcsent_rep[t]);
		src_bwd[t] = builder_drnn_bwd.add_input(i_x_t);
	}

	//create a vector of hidden representations excluding the current sentence
	for (unsigned i = 0; i < bs; ++i) {
		for (unsigned t = 0; t < dlen; ++t) {
			if (t != sids[i])
				temp_doc_embeddings[i].push_back(concatenate(std::vector<Expression>({ src_fwd[t], src_bwd[t] })));
		}
	}

	//only select the embeddings of previous and next sentence if considering local context
	if (loco_mem) {
		for (unsigned i = 0; i < bs; ++i) {
			if (sids[i] == 0 || (sids[i] == dlen - 1))
				temp_local_embeddings[i].push_back(concatenate(std::vector<Expression>({ zeroes(cg,{ h_dim }), zeroes(cg,{ h_dim }) })));
			for (unsigned t = 0; t < dlen - 1; ++t) {
				if (t == sids[i] - 1 || t == sids[i])
					temp_local_embeddings[i].push_back(temp_doc_embeddings[i][t]);
			}
			localsent_embeddings.push_back(average(temp_local_embeddings[i]));
		}
	}

	for (unsigned t = 0; t < dlen - 1; ++t) {
		vector<Expression> temp_sent;
		for (unsigned i = 0; i < bs; ++i)
			temp_sent.push_back(temp_doc_embeddings[i][t]);

		document_embeddings.push_back(concatenate_to_batch(temp_sent));
	}

	if (loco_mem) {
		src_loco_rep = concatenate_to_batch(localsent_embeddings);
	}

	src_doco_rep = concatenate_cols(document_embeddings);
}

template <class Builder>
void DocMTMemNNModel<Builder>::ComputeTrgDocRepresentations(const std::vector<std::vector<dynet::real>>& trgsent_rep,
	unsigned i, ComputationGraph &cg)
{
	const unsigned dlen = trgsent_rep.size();
	std::vector<Expression> document_embeddings;
	std::vector<Expression> localsent_embeddings;

	//create a vector of hidden representations excluding the current sentence
	for (unsigned t = 0; t < dlen; ++t) {
		const unsigned hdim = trgsent_rep[t].size();
		Expression i_x_t = input(cg, { hdim }, trgsent_rep[t]);
		if (t != i)
			document_embeddings.push_back(i_x_t);
		else
			trg_rep = i_x_t;
	}

	//only select the embeddings of previous and next sentence if considering local context
	if (loco_mem) {
		if (i == 0 || (i == dlen - 1))
			localsent_embeddings.push_back(zeroes(cg, { h_dim }));
		for (unsigned t = 0; t < dlen - 1; ++t) {
			if (t == i - 1 || t == i)
				localsent_embeddings.push_back(document_embeddings[t]);
		}
		trg_loco_rep = average(localsent_embeddings);
	}

	trg_doco_rep = concatenate_cols(document_embeddings);
}

template <class Builder>
void DocMTMemNNModel<Builder>::ComputeTrgDocRepresentations_Batch(const std::vector<std::vector<dynet::real>>& trgsent_rep,
	const std::vector<unsigned int> sids, ComputationGraph &cg)
{
	const unsigned dlen = trgsent_rep.size();
	const unsigned int bs = sids.size();//gives the number of sentences in the batch
	std::vector<std::vector<Expression>> temp_doc_embeddings(bs);
	std::vector<std::vector<Expression>> temp_local_embeddings(bs);
	std::vector<Expression> document_embeddings, localsent_embeddings, temp_present_embeddings;

	//create a vector of hidden representations excluding the current sentence
	for (unsigned i = 0; i < bs; ++i) {
		for (unsigned t = 0; t < dlen; ++t) {
			const unsigned hdim = trgsent_rep[t].size();
			Expression i_x_t = input(cg, { hdim }, trgsent_rep[t]);
			if (t != sids[i])
				temp_doc_embeddings[i].push_back(i_x_t);
			else
				temp_present_embeddings.push_back(i_x_t);
		}
	}

	//only select the embeddings of previous and next sentence if considering local context
	if (loco_mem) {
		for (unsigned i = 0; i < bs; ++i) {
			if (sids[i] == 0 || (sids[i] == dlen - 1))
				temp_local_embeddings[i].push_back({ zeroes(cg,{ h_dim }) });
			for (unsigned t = 0; t < dlen - 1; ++t) {
				if (t == sids[i] - 1 || t == sids[i])
					temp_local_embeddings[i].push_back(temp_doc_embeddings[i][t]);
			}
			localsent_embeddings.push_back(average(temp_local_embeddings[i]));
		}
	}

	for (unsigned t = 0; t < dlen - 1; ++t) {
		vector<Expression> temp_sent;
		for (unsigned i = 0; i < bs; ++i)
			temp_sent.push_back(temp_doc_embeddings[i][t]);

		document_embeddings.push_back(concatenate_to_batch(temp_sent));
	}

	if (loco_mem) {
		trg_loco_rep = concatenate_to_batch(localsent_embeddings);
	}

	trg_rep = concatenate_to_batch(temp_present_embeddings);
	trg_doco_rep = concatenate_cols(document_embeddings);
}
//----------------------------------------------------------------------------------------------------------------

template <class Builder>
Expression DocMTMemNNModel<Builder>::AddDocInput(unsigned trg_tok, unsigned t, ComputationGraph &cg, Expression &i_c_src,
                                                 Expression &i_c_trg, RNNPointer *prev_state)
{
    // alignment input
    Expression i_wah_rep;
    if (t > 0) {
        Expression i_h_tm1;
        if (prev_state)
            i_h_tm1 = concatenate(builder.get_h(*prev_state));// This is required for beam search decoding implementation.
        else
            i_h_tm1 = concatenate(builder.final_h());

        Expression i_wah = i_Wa * i_h_tm1;

        // want numpy style broadcasting, but have to do this manually
        i_wah_rep = concatenate_cols(std::vector<Expression>(slen, i_wah));
    }

    Expression i_e_t;
    if (t > 0)
        i_e_t = transpose(tanh(i_wah_rep + i_uax)) * i_va;
    else
        i_e_t = transpose(tanh(i_uax)) * i_va;

    Expression i_alpha_t = softmax(i_e_t); // FIXME: consider summing to less than one?
    Expression i_c_t = src * i_alpha_t; // FIXME: effectively summing here, consider maxing?

    // word input
    Expression i_x_t = lookup(cg, p_ct, trg_tok);
    Expression input;
    if (mem_to_ctx)
        input = concatenate(std::vector<Expression>({i_x_t, i_c_t, i_c_src, i_c_trg}));
    else
        input = concatenate(std::vector<Expression>({i_x_t, i_c_t}));

    // y_t = RNN([x_t, a_t])
    Expression i_y_t;
    if (prev_state)
        i_y_t = builder.add_input(*prev_state, input);
    else
        i_y_t = builder.add_input(input);

    Expression i_r_t;
#ifndef VANILLA_TARGET_LSTM
    // Bahdanau does a max-out thing here; I do a tanh. Tomaatos tomateos.
    Expression i_tildet_t = tanh(affine_transform({i_y_t, i_Q, i_c_t, i_P, i_x_t}));
    if (mem_to_op){
        if (src_mem && trg_mem)
            i_r_t = affine_transform({i_bias, i_R, i_tildet_t, i_Ws, i_c_src, i_Wt, i_c_trg});
        else if (src_mem && !trg_mem)
            i_r_t = affine_transform({i_bias, i_R, i_tildet_t, i_Ws, i_c_src});
        else if (!src_mem && trg_mem)
            i_r_t = affine_transform({i_bias, i_R, i_tildet_t, i_Wt, i_c_trg});
    }
    else
        i_r_t = affine_transform({i_bias, i_R, i_tildet_t});
#else
    if (mem_to_op){
        if (src_mem && trg_mem)
            i_r_t = affine_transform({i_bias, i_R, i_y_t, i_Ws, i_c_src, i_Wt, i_c_trg});
        else if (src_mem && !trg_mem)
            i_r_t = affine_transform({i_bias, i_R, i_y_t, i_Ws, i_c_src});
        else if (!src_mem && trg_mem)
            i_r_t = affine_transform({i_bias, i_R, i_y_t, i_Wt, i_c_trg});
    }
    else
        i_r_t = affine_transform({i_bias, i_R, i_y_t});
#endif

    return i_r_t;
}

template <class Builder>
Expression DocMTMemNNModel<Builder>::AddDocInput_Batch(const std::vector<unsigned>& trg_words, unsigned t, ComputationGraph &cg, Expression &i_c_src, Expression &i_c_trg)
{
    // alignment input
    Expression i_wah_rep;
    if (t > 0) {
        Expression i_h_tm1 = concatenate(builder.final_h());
        Expression i_wah = i_Wa * i_h_tm1;

        // want numpy style broadcasting, but have to do this manually
        i_wah_rep = concatenate_cols(std::vector<Expression>(slen, i_wah));
    }

    Expression i_e_t;
    if (t > 0)
        i_e_t = transpose(tanh(i_wah_rep + i_uax)) * i_va;
    else
        i_e_t = transpose(tanh(i_uax)) * i_va;

    Expression i_alpha_t = softmax(i_e_t); // FIXME: consider summing to less than one?
    Expression i_c_t = src * i_alpha_t; // FIXME: effectively summing here, consider maxing?

    // target word inputs
    Expression i_x_t = lookup(cg, p_ct, trg_words);
    Expression input;
    if (mem_to_ctx)
        input = concatenate(std::vector<Expression>({i_x_t, i_c_t, i_c_src, i_c_trg}));
    else
        input = concatenate(std::vector<Expression>({i_x_t, i_c_t}));

    // y_t = RNN([x_t, a_t])
    Expression i_y_t = builder.add_input(input);

    Expression i_r_t;
#ifndef VANILLA_TARGET_LSTM
    // Bahdanau does a max-out thing here; I do a tanh. Tomaatos tomateos.
    Expression i_tildet_t = tanh(affine_transform({i_y_t, i_Q, i_c_t, i_P, i_x_t}));
    if (mem_to_op){
        if (src_mem && trg_mem)
            i_r_t = affine_transform({i_bias, i_R, i_tildet_t, i_Ws, i_c_src, i_Wt, i_c_trg});
        else if (src_mem && !trg_mem)
            i_r_t = affine_transform({i_bias, i_R, i_tildet_t, i_Ws, i_c_src});
        else if (!src_mem && trg_mem)
            i_r_t = affine_transform({i_bias, i_R, i_tildet_t, i_Wt, i_c_trg});
    }
    else
        i_r_t = affine_transform({i_bias, i_R, i_tildet_t});
#else
    if (mem_to_op){
        if (src_mem && trg_mem)
            i_r_t = affine_transform({i_bias, i_R, i_y_t, i_Ws, i_c_src, i_Wt, i_c_trg});
        else if (src_mem && !trg_mem)
            i_r_t = affine_transform({i_bias, i_R, i_y_t, i_Ws, i_c_src});
        else if (!src_mem && trg_mem)
            i_r_t = affine_transform({i_bias, i_R, i_y_t, i_Wt, i_c_trg});
    }
    else
        i_r_t = affine_transform({i_bias, i_R, i_y_t});
#endif

    return i_r_t;
}

//----------------------------------------------------------------------------------------------------------------
template <class Builder>
Expression DocMTMemNNModel<Builder>::BuildDocMTSrcGraph(const std::vector<int>& source, const std::vector<int>& target,
                                                        const std::vector<std::vector<dynet::real>>& srcsent_repi, unsigned i, ComputationGraph& cg, ModelStats& tstats)
{
    Expression c_src;
    //compute the document representations using Document RNNs
    ComputeSrcDocRepresentations(srcsent_repi, i, cg);

    //computation for the MT model
    StartNewInstance(source, cg, tstats);

    //no need to compute output memory if only using local context
    if (!loco_mem){
		Expression p_t = softmax(transpose(src_doco_rep) * src_rep);
        c_src = src_doco_rep * p_t;
    }
    else
        c_src = src_loco_rep;

    std::vector<Expression> errs;
    const unsigned tlen = target.size() - 1;
    for (unsigned t = 0; t < tlen; ++t) {
        tstats.words_tgt++;
        if (target[t] == kTGT_UNK) tstats.words_tgt_unk++;

        Expression i_r_t = AddDocInput(target[t], t, cg, c_src, i_ztrg);
        Expression i_err = pickneglogsoftmax(i_r_t, target[t+1]);
        errs.push_back(i_err);
    }

    Expression i_nerr = sum(errs);
    return i_nerr;
}

template <class Builder>
Expression DocMTMemNNModel<Builder>::BuildDocMTSrcGraph_Batch(const std::vector<std::vector<int>>& sources, const std::vector<std::vector<int>>& targets,
	const std::vector<std::vector<dynet::real>>& srcsent_repi, const std::vector<unsigned int> sids, ComputationGraph& cg, ModelStats& tstats)
{
	Expression c_src;
	//compute the document representations using Document RNNs
	ComputeSrcDocRepresentations_Batch(srcsent_repi, sids, cg);

	//computation for the MT model
	StartNewInstance_Batch(sources, cg, tstats);

	//no need to compute output memory if only using local context
    if (!loco_mem){
        Expression p_t = softmax(transpose(src_doco_rep) * src_rep);
        c_src = src_doco_rep * p_t;
    }
    else
        c_src = src_loco_rep;

	std::vector<Expression> errs;

	const unsigned tlen = targets[0].size() - 1;
	std::vector<unsigned> next_words(targets.size()), words(targets.size());

	for (unsigned t = 0; t < tlen; ++t) {
		for (size_t bs = 0; bs < targets.size(); bs++) {
			words[bs] = (targets[bs].size() > t) ? (unsigned)targets[bs][t] : kTGT_EOS;
			next_words[bs] = (targets[bs].size() >(t + 1)) ? (unsigned)targets[bs][t + 1] : kTGT_EOS;
			if (targets[bs].size() > t) {
				tstats.words_tgt++;
				if (targets[bs][t] == kTGT_UNK) tstats.words_tgt_unk++;
			}
		}

		Expression i_r_t = AddDocInput_Batch(words, t, cg, c_src, i_ztrg_rep);
		Expression i_err = pickneglogsoftmax(i_r_t, next_words);

		errs.push_back(i_err);
	}

	Expression i_nerr = sum_batches(sum(errs));
	return i_nerr;
}

//----------------------------------------------------------------------------------------------------------------

template <class Builder>
Expression DocMTMemNNModel<Builder>::BuildDocMTTrgGraph(const std::vector<int>& source, const std::vector<int>& target,
                                                        const std::vector<std::vector<dynet::real>>& trgsent_rep, unsigned i, ComputationGraph& cg, ModelStats& tstats)
{
    Expression c_trg;
    ComputeTrgDocRepresentations(trgsent_rep, i, cg);

    //computation for the MT model
    StartNewInstance(source, cg, tstats);

    //no need to compute output memory if only using local context
    if (!loco_mem){
        const unsigned trgdoc_len = trgsent_rep.size();
        Expression i_bt_rep = transpose(concatenate_cols(std::vector<Expression>(trgdoc_len - 1 , i_bias_t)));
        //generate the output response from memory
        Expression a_t = softmax(transpose(trg_doco_rep) * (trg_rep + i_Ust * src_rep) + i_bt_rep);
        c_trg = trg_doco_rep * a_t;
    }
    else
        c_trg = trg_loco_rep;

    std::vector<Expression> errs;
    const unsigned tlen = target.size() - 1;
    for (unsigned t = 0; t < tlen; ++t) {
        tstats.words_tgt++;
        if (target[t] == kTGT_UNK) tstats.words_tgt_unk++;

        Expression i_r_t = AddDocInput(target[t], t, cg, i_zsrc, c_trg);
        Expression i_err = pickneglogsoftmax(i_r_t, target[t+1]);
        errs.push_back(i_err);
    }

    Expression i_nerr = sum(errs);
    return i_nerr;
}

template <class Builder>
Expression DocMTMemNNModel<Builder>::BuildDocMTTrgGraph_Batch(const std::vector<std::vector<int>>& sources, const std::vector<std::vector<int>>& targets,
                                                                  const std::vector<std::vector<dynet::real>>& trgsent_rep, const std::vector<unsigned int> sids, ComputationGraph& cg, ModelStats& tstats)
{
    Expression c_trg;
    ComputeTrgDocRepresentations_Batch(trgsent_rep, sids, cg);

    //computation for the MT model
    StartNewInstance_Batch(sources, cg, tstats);

    //no need to compute output memory if only using local context
    if (!loco_mem) {
        const unsigned trgdoc_len = trgsent_rep.size();
        Expression i_bt_rep = transpose(concatenate_cols(std::vector<Expression>(trgdoc_len - 1 , i_bias_t)));
        //generate the output response from memory
        Expression a_t = softmax(transpose(trg_doco_rep) * (trg_rep + i_Ust * src_rep) + i_bt_rep);
        c_trg = trg_doco_rep * a_t;
    }
    else
        c_trg = trg_loco_rep;

    std::vector<Expression> errs;
    const unsigned tlen = targets[0].size() - 1;
    std::vector<unsigned> next_words(targets.size()), words(targets.size());

    for (unsigned t = 0; t < tlen; ++t) {
        for (size_t bs = 0; bs < targets.size(); bs++) {
            words[bs] = (targets[bs].size() > t) ? (unsigned)targets[bs][t] : kTGT_EOS;
            next_words[bs] = (targets[bs].size() >(t + 1)) ? (unsigned)targets[bs][t + 1] : kTGT_EOS;
            if (targets[bs].size() > t) {
                tstats.words_tgt++;
                if (targets[bs][t] == kTGT_UNK) tstats.words_tgt_unk++;
            }
        }

        Expression i_r_t = AddDocInput_Batch(words, t, cg, i_zsrc_rep, c_trg);
        Expression i_err = pickneglogsoftmax(i_r_t, next_words);

        errs.push_back(i_err);
    }

    Expression i_nerr = sum_batches(sum(errs));
    return i_nerr;
}

//----------------------------------------------------------------------------------------------------

template <class Builder>
Expression DocMTMemNNModel<Builder>::BuildDocMTSrcTrgGraph(const std::vector<int>& source, const std::vector<int>& target,
                                                           const std::vector<std::vector<dynet::real>>& srcsent_repi, const std::vector<std::vector<dynet::real>>& trgsent_rep,
                                                           unsigned i, ComputationGraph& cg, ModelStats& tstats)
{
    Expression c_src, c_trg;
    //compute the document representations using Document RNNs
    ComputeSrcDocRepresentations(srcsent_repi, i, cg);
    ComputeTrgDocRepresentations(trgsent_rep, i, cg);

    //computation for the MT model
    StartNewInstance(source, cg, tstats);

    //no need to compute output memory if only using local context
    if (!loco_mem){
        if (src_mem){
            Expression p_t = softmax(transpose(src_doco_rep) * src_rep);
            c_src = src_doco_rep * p_t;
        }
        if (trg_mem){
            const unsigned trgdoc_len = trgsent_rep.size();
            Expression i_bt_rep = transpose(concatenate_cols(std::vector<Expression>(trgdoc_len - 1 , i_bias_t)));
            //generate the output response from memory
            Expression a_t = softmax(transpose(trg_doco_rep) * (trg_rep + i_Ust * src_rep) + i_bt_rep);
            c_trg = trg_doco_rep * a_t;
        }
    }
    else{
        if (src_mem)   c_src = src_loco_rep;
        if (trg_mem)   c_trg = trg_loco_rep;
    }

    std::vector<Expression> errs;
    const unsigned tlen = target.size() - 1;
    for (unsigned t = 0; t < tlen; ++t) {
        tstats.words_tgt++;
        if (target[t] == kTGT_UNK) tstats.words_tgt_unk++;

        Expression i_r_t = AddDocInput(target[t], t, cg, c_src, c_trg);
        Expression i_err = pickneglogsoftmax(i_r_t, target[t+1]);
        errs.push_back(i_err);
    }

    Expression i_nerr = sum(errs);
    return i_nerr;
}


template <class Builder>
Expression DocMTMemNNModel<Builder>::BuildDocMTSrcTrgGraph_Batch(const std::vector<std::vector<int>>& sources, const std::vector<std::vector<int>>& targets,
                                                                 const std::vector<std::vector<dynet::real>>& srcsent_repi, const std::vector<std::vector<dynet::real>>& trgsent_rep,
                                                                 const std::vector<unsigned int> sids, ComputationGraph& cg, ModelStats& tstats)
{
	Expression c_src, c_trg;
	//compute the document representations using Document RNNs
	ComputeSrcDocRepresentations_Batch(srcsent_repi, sids, cg);
	ComputeTrgDocRepresentations_Batch(trgsent_rep, sids, cg);

	//computation for the MT model
	StartNewInstance_Batch(sources, cg, tstats);

	//no need to compute output memory if only using local context
    if (!loco_mem){
        if (src_mem){
            Expression p_t = softmax(transpose(src_doco_rep) * src_rep);
            c_src = src_doco_rep * p_t;
        }
        if (trg_mem){
            const unsigned trgdoc_len = trgsent_rep.size();
            Expression i_bt_rep = transpose(concatenate_cols(std::vector<Expression>(trgdoc_len - 1 , i_bias_t)));
            //generate the output response from memory
            Expression a_t = softmax(transpose(trg_doco_rep) * (trg_rep + i_Ust * src_rep) + i_bt_rep);
            c_trg = trg_doco_rep * a_t;
        }
    }
    else{
        if (src_mem)   c_src = src_loco_rep;
        if (trg_mem)   c_trg = trg_loco_rep;
    }

	std::vector<Expression> errs;
	const unsigned tlen = targets[0].size() - 1;
	std::vector<unsigned> next_words(targets.size()), words(targets.size());

	for (unsigned t = 0; t < tlen; ++t) {
		for (size_t bs = 0; bs < targets.size(); bs++) {
			words[bs] = (targets[bs].size() > t) ? (unsigned)targets[bs][t] : kTGT_EOS;
			next_words[bs] = (targets[bs].size() >(t + 1)) ? (unsigned)targets[bs][t + 1] : kTGT_EOS;
			if (targets[bs].size() > t) {
				tstats.words_tgt++;
				if (targets[bs][t] == kTGT_UNK) tstats.words_tgt_unk++;
			}
		}

		Expression i_r_t = AddDocInput_Batch(words, t, cg, c_src, c_trg);
		Expression i_err = pickneglogsoftmax(i_r_t, next_words);

		errs.push_back(i_err);
	}

	Expression i_nerr = sum_batches(sum(errs));
	return i_nerr;
}

//---------------------------------------------------------------------------------------------
template <class Builder>
std::vector<dynet::real> DocMTMemNNModel<Builder>::GetTrgRepresentations(const std::vector<int> &source, ComputationGraph& cg, dynet::Dict &tdict)
{
    const int sos_sym = tdict.convert("<s>");
    const int eos_sym = tdict.convert("</s>");

    std::vector<int> target;
    target.push_back(sos_sym);

    //computation for the MT model
    StartNewInstance(source, cg);

    unsigned t = 0;
    while (target.back() != eos_sym) {
        Expression i_scores = AddInput(target.back(), t, cg);
        Expression ydist = softmax(i_scores); // compiler warning, but see below

        // find the argmax next word (greedy)
        unsigned w = 0;
        auto dist = as_vector(cg.incremental_forward(ydist));
        auto pr_w = dist[w];
        for (unsigned x = 1; x < dist.size(); ++x) {
            if (dist[x] > pr_w) {
                w = x;
                pr_w = dist[x];
            }
        }

        // break potential infinite loop
        if (t > 2 * source.size()) {
            w = eos_sym;
            pr_w = dist[w];
        }

        //std::cerr << " " << tdict.convert(w) << " [p=" << pr_w << "]";
        t += 1;
        target.push_back(w);
    }

    std::vector<Expression> trg_rep = builder.final_h();//includes final hidden states of all layers
    vector<dynet::real> trgsent_rep = as_vector(cg.forward(trg_rep.back()));

    return trgsent_rep;
}

template <class Builder>
std::vector<dynet::real> DocMTMemNNModel<Builder>::GetTrueTrgRepresentations(const std::vector<int> &source, const std::vector<int> &target, ComputationGraph& cg)
{
    StartNewInstance(source, cg);

    const unsigned tlen = target.size() - 1;
    for (unsigned t = 0; t < tlen; ++t)
        Expression i_r_t = AddInput(target[t], t, cg);

    std::vector<Expression> trg_rep = builder.final_h();//includes final hidden states of all layers
    vector<dynet::real> trgsent_rep = as_vector(cg.forward(trg_rep.back()));

    return trgsent_rep;
}

template <class Builder>
std::vector<dynet::real> DocMTMemNNModel<Builder>::GetTrg_SrcRepresentations(const std::vector<int> &source, const std::vector<vector<dynet::real>>& srcsent_repi,
                                                                             unsigned i, ComputationGraph& cg, dynet::Dict &tdict)
{
    const int sos_sym = tdict.convert("<s>");
    const int eos_sym = tdict.convert("</s>");

    std::vector<int> target;
    target.push_back(sos_sym);

    Expression c_src;
    //compute the document representations using Document RNNs
    ComputeSrcDocRepresentations(srcsent_repi, i, cg);

    //computation for the MT model
    StartNewInstance(source, cg);

    //no need to compute output memory if only using local context
    if (!loco_mem){
        Expression p_t = softmax(transpose(src_doco_rep) * src_rep);
        c_src = src_doco_rep * p_t;
    }
    else
        c_src = src_loco_rep;

    unsigned t = 0;
    while (target.back() != eos_sym) {
        Expression i_scores = AddDocInput(target.back(), t, cg, c_src, i_ztrg);
        Expression ydist = softmax(i_scores); // compiler warning, but see below

        // find the argmax next word (greedy)
        unsigned w = 0;
        auto dist = as_vector(cg.incremental_forward(ydist));
        auto pr_w = dist[w];
        for (unsigned x = 1; x < dist.size(); ++x) {
            if (dist[x] > pr_w) {
                w = x;
                pr_w = dist[x];
            }
        }

        // break potential infinite loop
        if (t > 2 * source.size()) {
            w = eos_sym;
            pr_w = dist[w];
        }

        //std::cerr << " " << tdict.convert(w) << " [p=" << pr_w << "]";
        t += 1;
        target.push_back(w);
    }

    std::vector<Expression> trg_rep = builder.final_h();//includes final hidden states of all layers
    vector<dynet::real> trgsent_rep = as_vector(cg.forward(trg_rep.back()));

    return trgsent_rep;
}

template <class Builder>
std::vector<dynet::real> DocMTMemNNModel<Builder>::GetTrueTrg_SrcRepresentations(const std::vector<int> &source, const std::vector<int> &target,
                                                                                 const std::vector<vector<dynet::real>>& srcsent_repi, unsigned i, ComputationGraph& cg)
{
	Expression c_src;
	//compute the document representations using Document RNNs
	ComputeSrcDocRepresentations(srcsent_repi, i, cg);

	//computation for the MT model
	StartNewInstance(source, cg);

	//no need to compute output memory if only using local context
	if (!loco_mem) {
		Expression p_t = softmax(transpose(src_doco_rep) * src_rep);
        c_src = src_doco_rep * p_t;
	}
	else
        c_src = src_loco_rep;

	const unsigned tlen = target.size() - 1;
	for (unsigned t = 0; t < tlen; ++t)
		Expression i_r_t = AddDocInput(target[t], t, cg, c_src, i_ztrg);

	std::vector<Expression> trg_rep = builder.final_h();//includes final hidden states of all layers
	vector<dynet::real> trgsent_rep = as_vector(cg.forward(trg_rep.back()));

	return trgsent_rep;
}

//---------------------------------------------------------------------------------------------

template <class Builder>
std::vector<int>
DocMTMemNNModel<Builder>::Greedy_Decode(const std::vector<int> &source, ComputationGraph& cg, dynet::Dict &tdict)
{
	const int sos_sym = tdict.convert("<s>");
	const int eos_sym = tdict.convert("</s>");

	std::vector<int> target;
	target.push_back(sos_sym); 

	//std::cerr << tdict.convert(target.back());
	unsigned t = 0;
	StartNewInstance(source, cg);
	while (target.back() != eos_sym) 
	{
		Expression i_scores = AddInput(target.back(), t, cg);
		Expression ydist = softmax(i_scores); // compiler warning, but see below

		// find the argmax next word (greedy)
		unsigned w = 0;
		auto dist = as_vector(cg.incremental_forward(ydist));
		auto pr_w = dist[w];
		for (unsigned x = 1; x < dist.size(); ++x) {
			if (dist[x] > pr_w) {
				w = x;
				pr_w = dist[x];
			}
		}

		// break potential infinite loop
		if (t > 2*source.size()) {
			w = eos_sym;
			pr_w = dist[w];
		}

		//std::cerr << " " << tdict.convert(w) << " [p=" << pr_w << "]";
		t += 1;
		target.push_back(w);
	}
	//std::cerr << std::endl;

	return target;
}

template <class Builder>
std::vector<int>
DocMTMemNNModel<Builder>::GreedyDocSrc_Decode(const std::vector<int> &source, const std::vector<vector<dynet::real>>& srcsent_repi,
                                              unsigned i, ComputationGraph& cg, dynet::Dict &tdict)
{
    const int sos_sym = tdict.convert("<s>");
    const int eos_sym = tdict.convert("</s>");

    std::vector<int> target;
    target.push_back(sos_sym);

    Expression c_src;
    //compute the document representations using Document RNNs
    ComputeSrcDocRepresentations(srcsent_repi, i, cg);

    //computation for the MT model
    StartNewInstance(source, cg);

    //no need to compute output memory if only using local context
    if (!loco_mem){
		Expression p_t = softmax(transpose(src_doco_rep) * src_rep);
        c_src = src_doco_rep * p_t;
    }
    else
        c_src = src_loco_rep;

    unsigned t = 0;
    while (target.back() != eos_sym) {
        Expression i_scores = AddDocInput(target.back(), t, cg, c_src, i_ztrg);
        Expression ydist = softmax(i_scores); // compiler warning, but see below

        // find the argmax next word (greedy)
        unsigned w = 0;
        auto dist = as_vector(cg.incremental_forward(ydist));
        auto pr_w = dist[w];
        for (unsigned x = 1; x < dist.size(); ++x) {
            if (dist[x] > pr_w) {
                w = x;
                pr_w = dist[x];
            }
        }

        // break potential infinite loop
        if (t > 2 * source.size()) {
            w = eos_sym;
            pr_w = dist[w];
        }

        //std::cerr << " " << tdict.convert(w) << " [p=" << pr_w << "]";
        t += 1;
        target.push_back(w);
    }

    return target;
}

template <class Builder>
std::vector<int>
DocMTMemNNModel<Builder>::GreedyDocTrg_Decode(const std::vector<int> &source, const std::vector<std::vector<dynet::real>>& trgsent_rep,
                                                  unsigned i, ComputationGraph& cg, dynet::Dict &tdict)
{
    const int sos_sym = tdict.convert("<s>");
    const int eos_sym = tdict.convert("</s>");

    std::vector<int> target;
    target.push_back(sos_sym);

    Expression c_trg;
    //compute the document representations using Document RNNs
    ComputeTrgDocRepresentations(trgsent_rep, i, cg);

    //computation for the MT model
    StartNewInstance(source, cg);

    //no need to compute output memory if only using local context
    if (!loco_mem){
        const unsigned trgdoc_len = trgsent_rep.size();
        Expression i_bt_rep = transpose(concatenate_cols(std::vector<Expression>(trgdoc_len - 1 , i_bias_t)));
        //generate the output response from memory
        Expression a_t = softmax(transpose(trg_doco_rep) * (trg_rep + i_Ust * src_rep) + i_bt_rep);
        c_trg = trg_doco_rep * a_t;
    }
    else
        c_trg = trg_loco_rep;

    unsigned t = 0;
    while (target.back() != eos_sym) {
        Expression i_scores = AddDocInput(target.back(), t, cg, i_zsrc, c_trg);
        Expression ydist = softmax(i_scores); // compiler warning, but see below

        // find the argmax next word (greedy)
        unsigned w = 0;
        auto dist = as_vector(cg.incremental_forward(ydist));
        auto pr_w = dist[w];
        for (unsigned x = 1; x < dist.size(); ++x) {
            if (dist[x] > pr_w) {
                w = x;
                pr_w = dist[x];
            }
        }

        // break potential infinite loop
        if (t > 2 * source.size()) {
            w = eos_sym;
            pr_w = dist[w];
        }

        //std::cerr << " " << tdict.convert(w) << " [p=" << pr_w << "]";
        t += 1;
        target.push_back(w);
    }

    return target;
}

template <class Builder>
std::vector<int>
DocMTMemNNModel<Builder>::GreedyDocSrcTrg_Decode(const std::vector<int> &source, const std::vector<vector<dynet::real>>& srcsent_repi,
                                                 const std::vector<std::vector<dynet::real>>& trgsent_rep, unsigned i, ComputationGraph& cg, dynet::Dict &tdict)
{
    const int sos_sym = tdict.convert("<s>");
    const int eos_sym = tdict.convert("</s>");

    std::vector<int> target;
    target.push_back(sos_sym);

    Expression c_src, c_trg;
    //compute the document representations using Document RNNs
    ComputeSrcDocRepresentations(srcsent_repi, i, cg);
    ComputeTrgDocRepresentations(trgsent_rep, i, cg);

    //computation for the MT model
    StartNewInstance(source, cg);

    //no need to compute output memory if only using local context
    if (!loco_mem){
        if (src_mem){
            Expression p_t = softmax(transpose(src_doco_rep) * src_rep);
            c_src = src_doco_rep * p_t;
        }
        if (trg_mem){
            const unsigned trgdoc_len = trgsent_rep.size();
            Expression i_bt_rep = transpose(concatenate_cols(std::vector<Expression>(trgdoc_len - 1 , i_bias_t)));
            //generate the output response from memory
            Expression a_t = softmax(transpose(trg_doco_rep) * (trg_rep + i_Ust * src_rep) + i_bt_rep);
            c_trg = trg_doco_rep * a_t;
        }
    }
    else{
        if (src_mem)   c_src = src_loco_rep;
        if (trg_mem)   c_trg = trg_loco_rep;
    }

    unsigned t = 0;
    while (target.back() != eos_sym) {
        Expression i_scores = AddDocInput(target.back(), t, cg, c_src, c_trg);
        Expression ydist = softmax(i_scores); // compiler warning, but see below

        // find the argmax next word (greedy)
        unsigned w = 0;
        auto dist = as_vector(cg.incremental_forward(ydist));
        auto pr_w = dist[w];
        for (unsigned x = 1; x < dist.size(); ++x) {
            if (dist[x] > pr_w) {
                w = x;
                pr_w = dist[x];
            }
        }

        // break potential infinite loop
        if (t > 2 * source.size()) {
            w = eos_sym;
            pr_w = dist[w];
        }

        //std::cerr << " " << tdict.convert(w) << " [p=" << pr_w << "]";
        t += 1;
        target.push_back(w);
    }

    return target;
}

#undef WTF
#undef KTHXBYE
#undef LOLCAT

}; // namespace dynet