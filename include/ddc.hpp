#include <armadillo>


auto predict_univariate(arma::mat x) -> arma::mat;
auto ddc(arma::mat x, double p, double min_cor) -> arma::mat;