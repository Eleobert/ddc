#include <armadillo>


auto predict_univariate(arma::mat x) -> arma::mat;
auto ddc(arma::mat x, int n_cor, double p, double min_cor) -> arma::mat;