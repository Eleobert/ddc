#include "robust_estimators.hpp"

#include <armadillo>



// @brief calculate the slope of the line with intercept at 0
auto slope(const arma::vec& x, const arma::vec& y)
{
    return arma::sum(x % y) / arma::sum(x % x);
}


// @brief calculate the robust slope of the line with intercept at 0
auto slope_robust(arma::vec x, arma::vec y, double c)
{
    arma::uvec finites = arma::find_finite(x + y);
    x = x(finites);
    y = y(finites);
    arma::uvec non_zeros = arma::find(x != 0);
    if(non_zeros.size() == x.size())
    {
        return arma::datum::nan;
    }
    auto m = arma::median(y(non_zeros)) / arma::median(x(non_zeros));
    arma::vec r = y - m * x;
    auto [loc, scale] = estimate_params(r);

    arma::uvec idx = arma::find(arma::abs(r) <= c * scale);

    return slope(x(idx), y(idx));
}