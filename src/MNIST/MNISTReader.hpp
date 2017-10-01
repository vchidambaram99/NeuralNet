/*
 * MNISTReader.h
 *
 *  Created on: Sep 26, 2017
 *      Author: vchid
 */

#ifndef MNIST_MNISTREADER_H_
#define MNIST_MNISTREADER_H_
#include <vector>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>

std::vector<Eigen::MatrixXd> readLabels(std::string filename){//vector of Matrices representing answers
	std::vector<Eigen::MatrixXd> d;
	std::ifstream data(filename,std::ios::binary|std::ios::in|std::ios::ate);
	if(data.is_open()){
		int size = data.tellg();
		unsigned char * memblock = new unsigned char[size]; //allocates memory for contents of file
		data.seekg (0, std::ios::beg); //moves to beginning of file
		data.read ((char*)memblock, size); //reads entire file
		data.close(); //closes file

		unsigned int magicNum = 0;
		for(int i = 0;i<4;i++){
			magicNum = (magicNum<<8) + memblock[i];
		}
		if(magicNum!=2049){
			std::cout<<"Not a valid MNIST label file (magic number is supposed to be 2049, but it is "<<magicNum<<")."<<std::endl;
			throw;
		}

		unsigned int count = 0;
		for(int i = 4;i<8;i++){
			count = (count<<8) + memblock[i];
		}
		d = std::vector<Eigen::MatrixXd>(count);
		for(int i = 0;i<count;i++){
			d[i] = Eigen::MatrixXd::Zero(10,1);
			d[i]((int)memblock[i+8]) = 1;
		}
		delete[] memblock; //deletes dynamically allocated memblock
	}else{
		std::cout<<"Failed to open MNIST label file"<<std::endl;
		throw;
	}
	return d;
}
std::vector<Eigen::MatrixXd> readImages(std::string filename){//vector of Matrices containing image
	std::vector<Eigen::MatrixXd> d;
	std::ifstream data(filename,std::ios::binary|std::ios::in|std::ios::ate);
	if(data.is_open()){
		int size = data.tellg();
		unsigned char * memblock = new unsigned char[size]; //allocates memory for contents of file
		data.seekg (0, std::ios::beg); //moves to beginning of file
		data.read ((char*)memblock, size); //reads entire file
		data.close(); //closes file

		unsigned int magicNum = 0;
		for(int i = 0;i<4;i++){
			magicNum = (magicNum<<8) + memblock[i];
		}
		if(magicNum!=2051){
			std::cout<<"Not a valid MNIST label file (magic number is supposed to be 2051, but it is "<<magicNum<<")."<<std::endl;
			throw;
		}

		unsigned int count = 0;
		for(int i = 4;i<8;i++){
			count = (count<<8) + memblock[i];
		}
		unsigned int rows = 0, cols = 0;
		for(int i = 8;i<12;i++){
			rows = (rows<<8) + memblock[i];
		}
		for(int i = 12;i<16;i++){
			cols = (cols<<8) + memblock[i];
		}
		d = std::vector<Eigen::MatrixXd>(count);//constructs vector for storing images
		for(int i = 0;i<count;i++){
			d[i] = Eigen::MatrixXd::Zero(rows*cols,1);
			for(int j = 0;j<rows*cols;j++){
				d[i](j) = ((double)memblock[(i*rows*cols)+j+16])/256;
			}
		}
		delete[] memblock; //deletes dynamically allocated memblock
	}else{
		std::cout<<"Failed to open MNIST image file"<<std::endl;
		throw;
	}
	return d;
}


#endif /* MNIST_MNISTREADER_H_ */
