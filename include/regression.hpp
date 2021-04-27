#pragma once

// @brief calculate the slope of the line with intercept at 0
auto slope(const arma::vec& x, const arma::vec& y) -> double;

// @brief calculate the robust slope of the line with intercept at 0
auto slope_robust(arma::vec x, arma::vec y, double c) -> double;