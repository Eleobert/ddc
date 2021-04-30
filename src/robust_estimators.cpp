#include <cassert>

#include "basic_stats.hpp"
#include "psi.hpp"
#include "psi.hpp"

#include <armadillo>


/* @brief Compute the location estimate.
 * @param sx sample sorted
 * @param u initial estimate μ_0 (for instance, the sample median)
 * @param epsilon the tolerance parameter. The algorithm stops iterating when |μ_{k+1} - μ_k| < epsilon * σ^
 */

auto estimate_loc(const arma::vec& sx, double u, double epsilon)
{
#ifdef NDEBUG
    assert(sx.is_sorted());
#endif
    auto pu = arma::datum::nan;   // previous u
    auto  s = madn_sorted(sx, u); // normalized mad

    arma::vec w(sx.n_elem);

    auto weight = [](const arma::vec& x) -> arma::vec
    {
    arma::vec w(x.n_elem);
    std::transform(x.begin(), x.end(), w.begin(),[](double x)
    {
    return (x != 0)? psi::huber(x) / x: 0;
    });
    return w;
    };

    do
    {
        pu = u;
        w  = weight((sx - u) / s);
        u  = arma::sum(w % sx) / arma::sum(w);;

    } while(std::abs(pu - u) > epsilon * s);

    return u;
}


/*
 * @brief Compute the scale estimate.
 * @param sx sample sorted.
 * @param u location.
 * @param s initial dispersion estimate (for instance, the normalized mad)
 * @param epsilon the tolerance parameter. The algorithm stops iterating when |σ_{k+1} / σ_k - 1| <= epsilon
 */
auto estimate_scale(arma::vec sx, double u, double s, double epsilon)
{
#ifdef NDEBUG
    assert(sx.is_sorted());
#endif
    sx -= u;

    // return the squared weight
    auto weight = [](const arma::vec& x) -> arma::vec
    {
        arma::vec w(x.n_elem);
        std::transform(x.begin(), x.end(), w.begin(), [](double x)
        {
            return (x != 0)? psi::huber_rho(x) / (x * x): 0;
        });
        return w;
    };

    auto ps = 0.0; // previous s
    const auto n = static_cast<int>(sx.n_elem);

    do
    {
        ps = s;
        arma::vec w = weight(sx / s);
        s = std::sqrt(arma::sum(w % sx % sx) / n);

    } while(std::abs(s / ps  - 1) > epsilon);

    return s;
}


auto estimate_params(arma::vec x, double loc_tol = 0.01, double scale_tol = 0.01)
{
    x = x(arma::find_finite(x));
    x = arma::sort(x);
    auto mad = madn(x);
    auto med = ::median(x);
    auto loc = estimate_loc(x, med, loc_tol);
    auto scale = estimate_scale(x, loc, mad, scale_tol);
    return std::make_tuple(loc, scale);
}


auto estimate_params(arma::mat x, double loc_tol = 0.01, double scale_tol = 0.01)
{
    arma::vec locs(x.n_cols);
    arma::vec scales(x.n_cols);

    for(auto i = 0ull; i < x.n_cols; i++)
    {
        arma::vec col = x.col(i);
        col = col(arma::find_finite(col));
        auto med = ::median(col);
        auto mad = madn(col);
        locs(i) = estimate_loc(col, med, loc_tol);
        scales(i) = estimate_scale(col, locs(i), mad, scale_tol);
        locs(i) = med;
        scales(i) = mad;
    }

    return std::make_tuple(locs, scales);
}


auto standardise(arma::mat x)
{
    arma::vec locations;
    arma::vec scales;

    std::tie(locations, scales) = estimate_params(x);

    x = x.each_row() - locations.t();
    x = x.each_row() / scales.t();

    return x;
}