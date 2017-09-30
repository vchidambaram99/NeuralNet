/*
 * CostFunctions.hpp
 *
 *  Created on: Sep 28, 2017
 *      Author: vchid
 */

#ifndef NEURALNETS_COSTFUNCTIONS_HPP_
#define NEURALNETS_COSTFUNCTIONS_HPP_
#include <Eigen/Dense>

Eigen::MatrixXd quadraticCostDerivative(Eigen::MatrixXd input, Eigen::MatrixXd &answer);
Eigen::MatrixXd crossEntropyCostDerivative(Eigen::MatrixXd input, Eigen::MatrixXd &answer);


#endif /* NEURALNETS_COSTFUNCTIONS_HPP_ */
