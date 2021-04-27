#include "robust_estimators.hpp"
#include "robust_correlation.hpp"

#include <armadillo>

auto univariate(arma::mat x, double c)
{
    x = standardise(x);
    auto cut = [c](double val)
    {
        return (val <= c) ? val: arma::datum::nan;
    };
    x.transform(cut);
    return x;
}