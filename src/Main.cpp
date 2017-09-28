/*
 * Main.cpp
 *
 *  Created on: Sep 28, 2017
 *      Author: vchid
 */
#include <Eigen/Dense>
#include <iostream>
using namespace std;

int main(){
	Eigen::MatrixXd a = Eigen::MatrixXd::Zero(7,5);
	a(3,4) = 1000;
	cout<<a<<endl;
	return 0;
}


