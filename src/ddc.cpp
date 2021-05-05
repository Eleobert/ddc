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


auto get_slopes(const arma::mat& x, const arma::umat& top_cor, double c) -> arma::mat
{
    assert(x.n_cols == top_cor.n_cols); 
    assert(x.n_cols >= top_cor.n_rows); 

    arma::mat slopes(top_cor.n_rows, top_cor.n_cols);
    slopes.fill(arma::datum::nan);

    for(auto j = 0ull; j < top_cor.n_cols; j++)
    {
        for(auto i = 0ull; i < top_cor.n_rows; i++)
        {
            if(top_cor(i, j) < x.n_cols)
            {
                slopes(i, j) = slope_robust(x.col(top_cor(i, j)), x.col(j), c);
            }
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

    auto trans = [](double x)
    {
        return std::pow(x, 14);
    };

    for(auto j = 0ull; j < preds.n_rows; j++)
    {
        arma::rowvec row = preds.row(j);
        arma::rowvec row_weight = weight(arma::find_finite(row)).t();

        row = row(arma::find_finite(row)).t();

        if(row.empty())
        {
            means(j) = arma::datum::nan;
            continue;
        }
        row_weight /= arma::max(row_weight);
        row_weight.transform(trans);
        row_weight /= arma::sum(row_weight);
        means(j) = arma::sum(row % row_weight);
    }
    return means;
}


auto predict(const arma::mat& x, const arma::mat& cor, const arma::umat& top_cor, const arma::mat& slopes)
{
    arma::mat z(x.n_rows, x.n_cols);

    for(auto h = 0ull; h < x.n_cols; h++)
    {
        arma::uvec contributors = top_cor.col(h);
        arma::mat predictions_h = x.cols(contributors);
        predictions_h.each_row() %= slopes.col(h).t();
        arma::vec weight = cor.col(h);
        weight = arma::abs(weight(contributors));
        z.col(h) = combine(predictions_h, weight.t());
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


auto deshrink(arma::mat x, arma::mat y, double c)
{
    x.each_row() %= get_slopes_2(x, y, c);
    return x;
}


auto top_n(const arma::mat& x, int n)
{
    arma::umat res(n, x.n_cols);
    for(auto i = 0ull; i < x.n_cols; i++)
    {        
        arma::vec vec = arma::abs(x.col(i));
        vec(arma::find_nonfinite(vec)).fill(-arma::datum::inf);
        vec(i) = -arma::datum::inf;
        arma::uvec idx = arma::sort_index(vec, "descending");
        idx(arma::find_nonfinite(vec)).fill(n); // for when total of correlated cols < n
        std::copy(idx.begin(), idx.begin() + n, res.begin_col(i));
    }
    return res;
}


auto ddc(arma::mat x, int n_cor, double p, double min_cor) -> arma::mat
{    
    auto c = cutoff(p);

    auto [locs, scales] = estimate_params(x);

    // univariate outlier detection
    arma::mat z = scale(x, locs, scales);
    arma::mat u = fill(z, arma::abs(z) > c, arma::datum::nan);
    
    arma::mat cor = cor_wrap(u);
    arma::umat top_cor = top_n(cor, n_cor);
    // cor < min_cor are not good enought. We will ignore it.
    cor = fill(cor, cor < min_cor, arma::datum::nan);

    arma::mat slopes = get_slopes(u, top_cor, c);
    arma::mat preds  = predict(u, cor, top_cor, slopes);

    preds = deshrink(preds, u, c);
    
    arma::mat r = standardised_residuals(u, preds);

    u = fill(z, arma::abs(r) > c, arma::datum::nan);
    u.rows(find_outlier_rows(r, c)).fill(arma::datum::nan);

    preds = unscale(preds, locs, scales);

    x(arma::find_nonfinite(u)) = preds(arma::find_nonfinite(u));

    return x;
}