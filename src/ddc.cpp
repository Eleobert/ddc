#include <algorithm>

#include <armadillo>
#include <boost/math/distributions/chi_squared.hpp>

#include "robust_estimators.hpp"
#include "robust_correlation.hpp"
#include "regression.hpp"


auto predict_univariate(arma::mat x) -> arma::mat
{
    x = standardise(x);

    auto dist = boost::math::chi_squared(1);
    
    auto pred = [dist] (double e)
    {
        if(arma::is_finite(e)) // [[likely]]
        {
            return boost::math::cdf(dist, e * e);
        }
        return e;
    };
    x.transform(pred);
    return x;
}


auto cutoff(double p)
{
    using boost::math::chi_squared;
    using boost::math::quantile;
    return std::sqrt(quantile(chi_squared(1), p));
}


template<typename Pred>
auto fill(const arma::mat x, Pred pred, double c)
{
    arma::mat res = x;
    res.elem(arma::find(pred)).fill(c);
    return res;
}


/* Compute slopes for pairs with correlation
 */
auto get_slopes(const arma::mat& x, const arma::mat& cor, double c)
{
    assert(x.n_cols == cor.n_cols);
    assert(cor.n_rows == cor.n_cols);

    arma::mat slopes(x.n_cols, x.n_cols);

    // TODO: cache optimization 
    for(auto j = 0ull; j < x.n_cols; j++)
    {
        for(auto i = 0ull; i < x.n_cols; i++)
        {
            slopes(i, j) = std::isfinite(cor(i, j))? slope_robust(x.col(i), x.col(j), c): arma::datum::nan;
        }
    }
    return slopes;
}


auto get_slopes_2(const arma::mat& x, const arma::mat& y, double c)
{
    assert(x.n_cols == y.n_cols);
    assert(x.n_rows == y.n_rows);

    arma::rowvec slopes(x.n_cols);

    // TODO: cache optimization 
    for(auto i = 0ull; i < x.n_cols; i++)
    {
        slopes(i) = slope_robust(x.col(i), y.col(i), c);
    }
    return slopes;
}


auto combine(arma::mat preds, const arma::rowvec& weight)
{
    arma::vec means(preds.n_rows);

    preds.each_row() %= weight;

    for(auto j = 0; j < preds.n_rows; j++)
    {
        arma::rowvec row = preds.row(j);
        //std::cout << row << std::endl;
        row = row(arma::find_finite(row)).t();
        means(j) = (row.empty())? arma::datum::nan: arma::mean(row);
    }
    return means;
}


auto predict(const arma::mat& x, const arma::mat& cor, const arma::mat& slopes)
{
    arma::mat z(x.n_rows, x.n_cols);

    for(auto h = 0ull; h < x.n_cols; h++)
    {
        arma::mat predictions_h = x.each_row() % slopes.row(h);
        arma::uvec correlated   = arma::find_finite(slopes.col(h)); // slopes is symetric
        arma::rowvec weight     = cor.row(h);
        // why I have to transpose the second argument is still a mistery for me.
        z.col(h) = combine(predictions_h.cols(correlated), weight(correlated).t());
    }
    return z;
}


auto standardised_residuals(const arma::mat& x, const arma::mat& y) -> arma::mat
{
    arma::mat r = x - y;
    arma::vec loc;
    arma::vec scale;
    std::tie(loc, scale) = estimate_params(r);
    r.each_row() /= scale.t();
    return r;
}


auto find_outlier_rows(arma::mat r, double c)
{
    r %= r;
    arma::uvec res(r.n_rows);
    for(auto i = 0ull; i < r.n_rows; i++)
    {
        arma::vec row = r.row(i).t();
        row = row(arma::find_finite(row));
        res(i) = row.empty() || arma::mean(row) > c;
    }
    return res;
}


auto scale(arma::mat x, const arma::vec& loc, const arma::vec& scale)
{
    x.each_row() -= loc.t();
    x.each_row() /= scale.t();
    return x;
}


auto unscale(arma::mat x, const arma::vec& loc, const arma::vec& scale)
{
    x.each_row() %= scale.t();
    x.each_row() += loc.t();
    return x;
}


auto ddc(arma::mat x, double p, double min_cor) -> arma::mat
{    
    auto c = cutoff(p);

    auto [locs, scales] = estimate_params(x);

    // univariate outlier detection
    arma::mat z = scale(x, locs, scales);
    arma::mat u = fill(z, arma::abs(z) > c, arma::datum::nan);
    
    arma::mat cor = cor_wrap(u);
    // cor < min_cor are not good enought. We will ignore it.
    cor = fill(cor, cor < min_cor, arma::datum::nan);

    arma::mat slopes = get_slopes(u, cor, c);
    arma::mat preds  = predict(u, cor, slopes);
    preds.each_row() %= get_slopes_2(preds, z, c);
    
    arma::mat r = standardised_residuals(u, preds);

    u = fill(z, arma::abs(r) > c, arma::datum::nan);
    u.rows(find_outlier_rows(r, c)).fill(arma::datum::nan);

    preds = unscale(preds, locs, scales);

    x(arma::find_nonfinite(u)) = preds(arma::find_nonfinite(u));

    return x;
}