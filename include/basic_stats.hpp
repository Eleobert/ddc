#pragma once

#include <optional>
#include <cmath>
#include <algorithm>

// quantile for already sorted data
template<typename RandomAcessContainer>
auto quantile_sorted(const RandomAcessContainer& vec, typename RandomAcessContainer::value_type p)
-> typename RandomAcessContainer::value_type
{
    auto t = p * static_cast<float>(vec.size() - 1);
    const auto low = std::floor(t);
    const auto hig = std::ceil(t);
    t = t - low;

    // return linear interpolation
    return (1 - t) * vec[low] + t * vec[hig];
}


template<typename RandomAcessContainer>
auto quantile(const RandomAcessContainer& vec, typename RandomAcessContainer::value_type p)
-> typename RandomAcessContainer::value_type
{
    RandomAcessContainer svec = vec;
    std::sort(svec.begin(), svec.end());
    return quantile_sorted(svec, p);
}


// normalized mad
template<typename RandomAcessContainer>
auto madn_sorted(RandomAcessContainer vec,  std::optional<double> center = std::nullopt) -> double
{
    constexpr auto sn = 1.482602218505602;

    const auto c = center.value_or(quantile_sorted(vec, 0.5));

    for(auto& e: vec)
    {
        e = std::abs(e - c);
    }

    return sn * quantile(vec, 0.5);
}

// normalized mad
template<typename RandomAcessContainer>
auto madn(RandomAcessContainer vec, std::optional<double> center = std::nullopt) -> double
{
    std::sort(vec.begin(), vec.end());

    return madn_sorted(vec, center);
}


template<typename RandomAcessContainer>
auto median_sorted(const RandomAcessContainer& vec) -> double
{
    return quantile_sorted(vec, 0.5);
}


template<typename RandomAcessContainer>
auto median(const RandomAcessContainer& vec) -> double
{
    return quantile(vec, 0.5);
}

