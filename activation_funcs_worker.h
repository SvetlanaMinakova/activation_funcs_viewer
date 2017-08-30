#pragma once
#include <vector>
#include "activation_funcs.h"
#include <memory>

enum activation_funcs_type {SIGM,THN,RELU,SOFTMAX};

class activation_funcs_worker
{
public:
    virtual ~activation_funcs_worker(void);
    virtual void apply(std::vector<double> &input, std::vector<double> &result)=0;
    virtual void apply_d(std::vector<double> &input, std::vector<double> &result)=0;
    static std::shared_ptr<activation_funcs_worker> create_activation_funcs_worker (activation_funcs_type af_type);
    static std::shared_ptr<activation_funcs_worker> create_activation_funcs_worker (activation_funcs_type af_type, double param);
    double parameter;
    void set_param(double param)
	{parameter=param;}
    virtual double get_scale_factor_for_weights_init(int &neurons_num);
};

class sigm_af : public activation_funcs_worker
{
public:
    void apply(std::vector<double> &input, std::vector<double> &result)
    {activation_funcs::apply_act_f_to_vec(activation_funcs::sigm,input,result);}

    void apply_d(std::vector<double> &input, std::vector<double> &result)
    {activation_funcs::apply_act_f_to_vec(activation_funcs::sigm_deriv,input,result);}

	 //weights in [-4* sqrt(6/(inputs_num+neurons_num)) ; 4* sqrt(6/(inputs_num+neurons_num))] range are
//(optimal for sigmoidal activation func)
    double get_scale_factor_for_weights_init(double &inputs_num)
	{
        return 4* std::sqrt(6/(double)inputs_num);
	}
};

class sigm_af_p : public activation_funcs_worker
{
public:
    void apply(std::vector<double> &input, std::vector<double> &result)
    {activation_funcs::apply_act_f_to_vec(activation_funcs::sigm,input,result,this->parameter);}

    void apply_d(std::vector<double> &input, std::vector<double> &result)
    {activation_funcs::apply_act_f_to_vec(activation_funcs::sigm_deriv,input,result, this->parameter);}

		 //weights in [-4* sqrt(6/(inputs_num+neurons_num)) ; 4* sqrt(6/(inputs_num+neurons_num))] range are
//(optimal for sigmoidal activation func)
        double get_scale_factor_for_weights_init(double &inputs_num)
	{
        return 4* std::sqrt(6/(double)inputs_num);
	}

};

class thn_af : public activation_funcs_worker
{
public:
    void apply(std::vector<double> &input, std::vector<double> &result)
    {activation_funcs::apply_act_f_to_vec(activation_funcs::thn,input,result);}

    void apply_d(std::vector<double> &input, std::vector<double> &result)
    {activation_funcs::apply_act_f_to_vec(activation_funcs::thn_deriv,input,result);}

		 //weights in [-1* sqrt(6/(inputs_num+neurons_num)) ; 1* sqrt(6/(inputs_num+neurons_num))] range are
//(optimal for thn activation func)
double get_scale_factor_for_weights_init(double &inputs_num)
	{
        return 4* std::sqrt(6/(double)inputs_num);
	}

};

class thn_af_p : public activation_funcs_worker
{
public:
    void apply(std::vector<double> &input, std::vector<double> &result)
    {activation_funcs::apply_act_f_to_vec(activation_funcs::thn,input,result,this->parameter);}

    void apply_d(std::vector<double> &input, std::vector<double> &result)
    {activation_funcs::apply_act_f_to_vec(activation_funcs::thn_deriv,input,result, this->parameter);}

			 //weights in [-1* sqrt(6/(inputs_num+neurons_num)) ; 1* sqrt(6/(inputs_num+neurons_num))] range are
//(optimal for thn activation func)
double get_scale_factor_for_weights_init(double &inputs_num)
	{
        return std::sqrt(6/(double)inputs_num);
	}
};

class relu_af : public activation_funcs_worker
{
public:
    void apply(std::vector<double> &input, std::vector<double> &result)
    {activation_funcs::apply_act_f_to_vec(activation_funcs::relu,input,result);}

    void apply_d(std::vector<double> &input, std::vector<double> &result)
    {activation_funcs::apply_act_f_to_vec(activation_funcs::relu_deriv,input,result);}

double get_scale_factor_for_weights_init(double &inputs_num)
	{
        return std::sqrt(6/(double)inputs_num);
	}
};

class relu_af_p : public activation_funcs_worker
{
public:
    void apply(std::vector<double> &input, std::vector<double> &result)
    {activation_funcs::apply_act_f_to_vec(activation_funcs::relu,input,result,this->parameter);}

    void apply_d(std::vector<double> &input, std::vector<double> &result)
    {activation_funcs::apply_act_f_to_vec(activation_funcs::relu_deriv,input,result, this->parameter);}
};

class softmax_af : public activation_funcs_worker
{
public:
    void apply(std::vector<double> &input, std::vector<double> &result)
	{activation_funcs::apply_softmax_to_vec(input,result);}

    void apply_d(std::vector<double> &input, std::vector<double> &result)
	{activation_funcs::apply_softmax_to_vec(input,result);}

};

class softmax_af_p : public activation_funcs_worker
{
public:
    void apply(std::vector<double> &input, std::vector<double> &result)
	{activation_funcs::apply_softmax_to_vec(input,result,this->parameter);}

    void apply_d(std::vector<double> &input, std::vector<double> &result)
	{activation_funcs::apply_softmax_to_vec(input,result,this->parameter);}

};
