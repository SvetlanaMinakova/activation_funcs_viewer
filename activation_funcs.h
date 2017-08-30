#pragma once
#include <vector>
#include <algorithm>
//activation function "presses" its input and produces output for the neuron
class activation_funcs
{
public:
	activation_funcs(void);
	~activation_funcs(void);

    static double sigm(double &inp);
    static double sigm_deriv(double &inp);
    static double thn(double &inp);
    static double thn_deriv(double &inp);
    static double relu(double &inp);
    static double relu_deriv(double &inp);
    static double softmax(double &inp, double &softmax_summ);
    static double softmax_deriv(double &inp,double &softmax_summ);

    static double get_softmax_summ(double* non_activated_stages, int &non_activated_stages_len);
    static double get_softmax_summ(std::vector<double> &non_activated_stages);

    static void apply_act_f_to_vec(double (*act_f)(double &inp),std::vector<double> &input, std::vector<double> &result);
    static void apply_act_f_to_vec(double (*act_f)(double &inp),std::vector<double> &input, std::vector<double> &result, double &par);
    static void apply_softmax_to_vec(std::vector<double> &input, std::vector<double> &result);
    static void apply_softmax_to_vec(std::vector<double> &input, std::vector<double> &result, double &par);
    static void apply_softmax_deriv_to_vec(std::vector<double> &input, std::vector<double> &result);
    static void apply_softmax_deriv_to_vec(std::vector<double> &input, std::vector<double> &result, double &par);
};


