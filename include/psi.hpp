#pragma once

#include <armadillo>

namespace psi
{
    constexpr
    auto huber(double x, double k = 1.345)
    {
        return (std::abs(x) <= k)? x: arma::sign(x) * k;
    }

    constexpr
    auto huber_rho(double x, double k = 1.345)
    {
        return (std::abs(x) <= k)? x*x: 2 * k * std::abs(x) - k * k;
    }

};