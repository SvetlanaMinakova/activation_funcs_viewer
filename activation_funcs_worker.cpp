#include "activation_funcs_worker.h"


activation_funcs_worker::~activation_funcs_worker(void)
{
}

std::shared_ptr<activation_funcs_worker> activation_funcs_worker::create_activation_funcs_worker(activation_funcs_type af_type)
{
	switch(af_type)
	{
		case activation_funcs_type::SIGM :
			return std::make_shared<sigm_af>();
		case activation_funcs_type::THN :
			return std::make_shared<thn_af>();
		case activation_funcs_type::RELU :
			return std::make_shared<relu_af>();
		case activation_funcs_type::SOFTMAX :
			return std::make_shared<softmax_af>();
		default:
			return std::make_shared<sigm_af>();
	}
}

std::shared_ptr<activation_funcs_worker> activation_funcs_worker::create_activation_funcs_worker(activation_funcs_type af_type, double param)
{
    std::shared_ptr<activation_funcs_worker> af;
	switch(af_type)
	{
		case activation_funcs_type::SIGM :
			af= std::make_shared<sigm_af_p>();
			break;
		case activation_funcs_type::THN :
			af= std::make_shared<thn_af_p>();
			break;
		case activation_funcs_type::RELU :
			af= std::make_shared<relu_af_p>();
			break;
		case activation_funcs_type::SOFTMAX :
			af= std::make_shared<softmax_af_p>();
			break;
		default:
			af= std::make_shared<sigm_af_p>();
			break;
	}

	af->set_param(param);
	return af;
}

//scale factor is 1 by default
double activation_funcs_worker::get_scale_factor_for_weights_init(int &neurons_num)
{
	return 1;
}

