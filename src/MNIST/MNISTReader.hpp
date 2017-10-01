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

std::vector<int> readLabels(std::string filename){
	std::vector<int> d;
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
		d = std::vector<int>(count);
		for(int i = 0;i<count;i++){
			d[i] = memblock[i+8];
		}
		delete[] memblock; //deletes dynamically allocated memblock
	}else{
		std::cout<<"Failed to open MNIST label file"<<std::endl;
		throw;
	}
	return d;
}
std::vector<std::vector<double>> readImages(std::string filename){
	std::vector<std::vector<double>> d;
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
		d = std::vector<std::vector<double>>(count,std::vector<double>(rows*cols));//constructs vector for storing images
		for(int i = 0;i<count;i++){
			for(int j = 0;j<rows*cols;j++){
				d[i][j] = ((double)memblock[(i*rows*cols)+j+16])/256;
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
