//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
//#include <sstream>
#include <cctype>

#include "firmware/parameters.h"
#include "firmware/myproject.h"

//#define CHECKPOINT 5000
#define CHECKPOINT 1

// This function is written to avoid stringstream, which is
// not supported in cosim 20.1, and because strtok
// requires a const_cast or allocation to use with std::strings.
// This function returns the next float (by argument) at position pos,
// updating pos. True is returned if conversion done, false if the string
// has ended, and std::invalid_argument exception if the sting was bad.
bool nextToken(const std::string& str, std::size_t& pos, float& val)
{
  while (pos < str.size() && std::isspace(static_cast<unsigned char>(str[pos]))) {
    pos++;
  }
  if (pos >= str.size()) {
    return false;
  }
  std::size_t offset = 0;
  val = std::stof(str.substr(pos), &offset);
  pos += offset;
  return true;
}

int main(int argc, char **argv)
{
  //load input data from text file
  std::ifstream fcount("/atlas/chiedde/Documents/CPPM_Nemer/cppm-nnlar/hls4ml/quartus-test-cppm/tb_data/data_new.txt");
  std::ifstream fin("/atlas/chiedde/Documents/CPPM_Nemer/cppm-nnlar/hls4ml/quartus-test-cppm/tb_data/data_new.txt");
  //load predictions from text file
  std::ifstream fpr("/atlas/chiedde/Documents/CPPM_Nemer/cppm-nnlar/hls4ml/quartus-test-cppm/tb_data/true_lauri.txt");
  std::cout << "Openning files for simulations" << std::endl;

  std::string RESULTS_LOG = "tb_data/results_test.log";
  std::ofstream fout(RESULTS_LOG);

  std::string iline;
  std::string pline;

  int e = 0;

  int num_iterations = 1000;
  //std::count(std::istreambuf_iterator<char>(fcount), std::istreambuf_iterator<char>(), '\n');
  //int num_iterations = std::count(std::istreambuf_iterator<char>(fin),
  //                 std::istreambuf_iterator<char>(), '\n');
  if (fin.is_open() && fpr.is_open()) {
    //hls-fpga-machine-learning insert component-io
    std::vector<std::vector<float>> pr(num_iterations,std::vector<float>());
    while ( std::getline(fin,iline) && std::getline (fpr,pline) ) {
      if (e % CHECKPOINT == 0) std::cout << "Processing input " << e << std::endl;
      char* cstr=const_cast<char*>(iline.c_str());
      char* current;
      std::vector<float> in;
      current=strtok(cstr," ");
      while(current!=NULL) {
        in.push_back(atof(current));
        current=strtok(NULL," ");
      }
      cstr=const_cast<char*>(pline.c_str());

      current=strtok(cstr," ");
      while(current!=NULL) {
        pr[e].push_back(atof(current));
        current=strtok(NULL," ");
      }

      //hls-fpga-machine-learning insert data

      e++;
      if (num_iterations == e) break;
      //hls-fpga-machine-learning insert top-level-function
    for(int j = 0; j < e; j++) {
      //hls-fpga-machine-learning insert tb-output
      if (j % CHECKPOINT == 0) {
        std::cout << "Predictions" << std::endl;
        //hls-fpga-machine-learning insert predictions
        std::cout << "Quantized predictions" << std::endl;
        //hls-fpga-machine-learning insert quantized
      }
    }
    fin.close();
    fpr.close();
  } else {
    num_iterations = 10;
    std::cout << "INFO: Unable to open input/predictions file, using default input with " << num_iterations << " invocations." << std::endl;
    //hls-fpga-machine-learning insert zero

    for (int i = 0; i < num_iterations; i++) {
      //hls-fpga-machine-learning insert second-top-level-function

      for (int j = 0; j < num_iterations; j++) {
        //hls-fpga-machine-learning insert output

       //hls-fpga-machine-learning insert tb-output
    }
  }
  fout.close();
  std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

  return 0;
}
