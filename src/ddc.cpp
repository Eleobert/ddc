#include <armadillo>
#include <boost/math/distributions/chi_squared.hpp>

#include "robust_estimators.hpp"

auto predict_univariate(arma::mat x) -> arma::mat
{
    x = standardise(x);

    auto dist = boost::math::chi_squared(1);
    
    auto pred = [dist] (double e)
    {
        if(arma::is_finite(e)) [[likely]]
        {
            return boost::math::cdf(dist, e * e);
        }
        return e;
    };
    x.transform(pred);
    return x;
}