#ifndef VERIFY_OUTPUT_HPP
#define VERIFY_OUTPUT_HPP

#include <iostream>
#include <vector>
#include <string>

int verify_output(const std::vector<int8_t>& hw_output,
		  const std::string& golden_file,
		  int H, int W, int C, int stride, int flatten);

int verify_gap_output(const std::vector<int8_t>& hw_output,
		      const std::string& golden_file,
		      int total_channels);


#endif
