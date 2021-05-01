#pragma once

#include <armadillo>


/* @brief Compute the location estimate.
 * @param sx sample sorted
 * @param u initial estimate μ_0 (for instance, the sample median)
 * @param epsilon the tolerance parameter. The algorithm stops iterating when |μ_{k+1} - μ_k| < epsilon * σ^
 */
auto estimate_loc(const arma::vec& sx, double u, double epsilon) -> double;


/*
 * @brief Compute the scale estimate.
 * @param sx sample sorted.
 * @param u location.
 * @param s initial dispersion estimate (for instance, the normalized mad)
 * @param epsilon the tolerance parameter. The algorithm stops iterating when |σ_{k+1} / σ_k - 1| <= epsilon
 */
auto estimate_scale(arma::vec sx, double u, double s, double epsilon, int max_iter = 100) -> double;

auto estimate_params(arma::vec x, double loc_tol = 0.01, double scale_tol = 0.01, int max_iter = 100) 
    -> std::tuple<double, double>;

auto estimate_params(arma::mat x, double loc_tol = 0.01, double scale_tol = 0.01, int max_iter = 100) 
    -> std::tuple<arma::vec, arma::vec>;

auto standardise(arma::mat x) -> arma::mat;