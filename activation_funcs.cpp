#include "activation_funcs.h"
#include <math.h>


activation_funcs::activation_funcs(void)
{
}


activation_funcs::~activation_funcs(void)
{
}

void activation_funcs::apply_act_f_to_vec(double (*act_f)(double &inp),std::vector<double> &inp, std::vector<double> &result)
{
	std::transform(inp.begin(),inp.end(),result.begin(),act_f);	
}

void activation_funcs::apply_act_f_to_vec(double (*act_f)(double &inp),std::vector<double> &inp, std::vector<double> &result, double &param)
{
	int id=0;
    double parametred_inp_el;
    std::for_each(inp.begin(),inp.end(),[&](double &_inp_el)
	{   
		parametred_inp_el=_inp_el*param;
		result[id]=act_f(parametred_inp_el);
		id++;
	});
}

void activation_funcs::apply_softmax_to_vec(std::vector<double> &inp, std::vector<double> &result)
{
    double softmax_summ = get_softmax_summ(inp);
	int id=0;
    std::for_each(inp.begin(),inp.end(),[&](double &_inp_el)
	{   
		result[id]=softmax(_inp_el,softmax_summ);
		id++;
	});
}

void activation_funcs::apply_softmax_to_vec(std::vector<double> &inp, std::vector<double> &result, double &param)
{
    double softmax_summ = get_softmax_summ(inp);
	int id=0;
    double parametred_inp_el;
    std::for_each(inp.begin(),inp.end(),[&](double &_inp_el)
	{   
		parametred_inp_el=_inp_el*param;
		result[id]=softmax(parametred_inp_el,softmax_summ);
		id++;
	});
}

void activation_funcs::apply_softmax_deriv_to_vec(std::vector<double> &inp, std::vector<double> &result)
{
    double softmax_summ = get_softmax_summ(inp);
	int id=0;
    std::for_each(inp.begin(),inp.end(),[&](double &_inp_el)
	{   
		result[id]=softmax_deriv(_inp_el,softmax_summ);
		id++;
	});
}

void activation_funcs::apply_softmax_deriv_to_vec(std::vector<double> &inp, std::vector<double> &result, double &param)
{
    double softmax_summ = get_softmax_summ(inp);
	int id=0;
    double parametred_inp_el;
    std::for_each(inp.begin(),inp.end(),[&](double &_inp_el)
	{   
		parametred_inp_el=_inp_el*param;
		result[id]=softmax_deriv(_inp_el,softmax_summ);
		id++;
	});
}

double activation_funcs::sigm(double &inp)
{
     return  (double)(1/(1+exp((-1)*inp)));
}

double activation_funcs::sigm_deriv(double &inp)
{ 
	return sigm(inp)*(1-sigm(inp));
} 

double activation_funcs::thn(double &inp)
{
    return  (double)((exp(2*inp -1)/(exp(2*inp) + 1)));
}


double activation_funcs::thn_deriv(double &inp)
{	 double ch =0;
     ch=(double)((exp(inp) + exp((-1)*inp))/2);
	 return 1/(pow(ch,2));
}

double activation_funcs::relu(double &inp)
{
	return logf(1+exp(inp));
}

double activation_funcs::relu_deriv(double &inp)
{
	return 1/(1+exp((-1)*inp));
}


double activation_funcs::softmax(double &inp, double &softmax_summ)
{
	return exp(inp)/softmax_summ;
}

double activation_funcs::softmax_deriv(double &inp,double &softmax_summ)
{
    double y=softmax(inp,softmax_summ);
	return y*(1-y);
}

double activation_funcs::get_softmax_summ(double* non_activated_stages, int &non_activated_stages_len)
{
    double softmax_summ=0;
	for (int i=0;i<non_activated_stages_len;i++)
		softmax_summ+=exp(non_activated_stages[i]);
	return softmax_summ;
}

double activation_funcs::get_softmax_summ(std::vector <double> &non_activated_stages)
{
    double softmax_summ = 0;
    for( double i : non_activated_stages)
		softmax_summ+=exp(i);
	return softmax_summ;
}
