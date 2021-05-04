#include "robust_correlation.hpp"
#include "robust_estimators.hpp"
#include "ddc.hpp"

#include <algorithm>

extern "C"
{
void cor_wrap_c(double* ptr, int n_rows, int n_cols, double* out_ptr)
{
    arma::mat mat(ptr, n_rows, n_cols, false);
    arma::mat cor = cor_wrap(mat);
    std::copy(cor.begin(), cor.end(), out_ptr);
}


void estimate_params_c(double* ptr, int n_rows, int n_cols, double* locs_out_ptr, double* scales_out_ptr)
{
    arma::mat mat(ptr, n_rows, n_cols, false);
    arma::vec locs;
    arma::vec scales;
    std::tie(locs, scales) = estimate_params(mat);
    std::copy(locs.begin(), locs.end(), locs_out_ptr);
    std::copy(scales.begin(), scales.end(), scales_out_ptr);
}


auto standardise_c(double* ptr, int n_rows, int n_cols, double* out_ptr)
{
    arma::mat mat(ptr, n_rows, n_cols, false);
    arma::mat std = standardise(mat);
    std = std.t();
    std::copy(std.begin(), std.end(), out_ptr);
}


auto predict_univariate_c(double* ptr, int n_rows, int n_cols, double* out_ptr)
{
    arma::mat mat(ptr, n_rows, n_cols, false);
    arma::mat res = predict_univariate(mat);
    res = res.t();
    std::copy(res.begin(), res.end(), out_ptr);
}


auto ddc_c(double* ptr, int n_rows, int n_cols, double n_cor, double p, double min_cor, double* out_ptr)
{
    arma::mat mat(ptr, n_rows, n_cols, false);
    arma::mat res = ddc(mat, n_cor, p, min_cor);
    res = res.t();
    std::copy(res.begin(), res.end(), out_ptr);
}


}

