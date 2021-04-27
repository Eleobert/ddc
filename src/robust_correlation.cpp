#include <armadillo>

auto wrap(double z) -> double
{
    constexpr auto b = 1.5;
    constexpr auto c = 4.0;

    // TODO: derive these from a and c
    constexpr auto m  = 0.7532528; // A
    constexpr auto n  = 0.8430849; // B
    constexpr auto k  = 4.1517212;
    constexpr auto q1 = 1.540793;
    constexpr auto q2 = 0.8622731;

    if(std::abs(z) < c)
    {
        return (std::abs(z) < b)? z: (q1 * std::tanh(q2 * (c - std::abs(z)))) * arma::sign(z);
    }

    // return 0 both for values outside [-b, b] and nan
    return 0.0;
}


auto cor_wrap(arma::mat x) -> arma::mat
{
    // non finite will be transformed to 0
    x.transform(wrap);
    //std::cout << x << "\n\n";
    //return x;
    return arma::cor(x);
}
