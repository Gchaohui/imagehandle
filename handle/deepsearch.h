#ifndef DEEPSEARCH_H
#define DEEPSEARCH_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utils/trace.hpp> 
#include <opencv2/dnn.hpp>

#include <iostream>
#include <stdio.h>
#include <map>
#include <vector>
#include <sys/time.h>

using namespace cv;
using namespace cv::dnn; 

#define BYTE unsigned char

struct Result
{
	std::string desc;
	BYTE *ret;
};

namespace ImageHandle {
	class DeepSearch{
	public:
	DeepSearch(){};
	~DeepSearch(){};
	void DeepLearning(Result &_return, const std::string& image);
	//保存
	std::map<std::string, std::string> m_loadpath;
	std::vector<std::string> m_labels;
	cv::Mat image;
	};

}// namespace

#endif
