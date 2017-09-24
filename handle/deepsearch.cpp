#include "deepsearch.h"

namespace ImageHandle {
    void DeepSearch::DeepLearning(Result &_return, const std::string& aimage){
       //The path to the input image
        m_loadpath["image"] = aimage;
        //The path to the Caffe “deploy” prototxt file.
        m_loadpath["prototxt"] = "bvlc_googlenet.prototxt";
        //The pre-trained Caffe model (i.e,. the network weights themselves).
        m_loadpath["model"] = "bvlc_googlenet.caffemodel";
        //The path to ImageNet labels (i.e., “syn-sets”).
        m_loadpath["labels"] = "synset_words.txt";
        //m_loadpath["labels"] = "1.txt";

        image = cv::imread(m_loadpath["image"]);
     
        char strLine[512];
        FILE *fp = fopen(m_loadpath["labels"].c_str(),  "r");
        if (fp  == NULL){
            std::cout<<"open labels error"<<std::endl;
            return ;
        }
        std::string fin;
        char * f;
        char *p;
        while ( !feof(fp)){
            fgets(strLine, 512, fp);
            if((f = strchr(strLine,'\n')) != NULL)  
                *f = '\0'; 
            p = strtok(strLine, " ");
            if (p == NULL){
                continue;
            }
            fin = strtok(NULL, ",");

           m_labels.push_back(fin);
        }
        fclose(fp);
        fp = NULL; 

        Mat inputBlob = blobFromImage(image, 1, Size(224, 224), Scalar(104, 117, 123));
        // create googlenet with caffemodel text and bin   加载训练好的模型
        std::cout<< " Load  model"<< std::endl;
        Net net = readNetFromCaffe(m_loadpath["prototxt"], m_loadpath["model"]);
        if (net.empty()){
            std::cout<<" can't load prototxt and model" <<std::endl;
        }
     
        struct timeval startTime, endTime;
        Mat prob;
        for (int i = 0; i < 10; i++)   
        {  
            CV_TRACE_REGION("forward"); 
             net.setInput(inputBlob, "data");  
            gettimeofday(&startTime, NULL);
            prob = net.forward("prob");
            gettimeofday(&endTime, NULL);
        } 
        float time_use = (endTime.tv_sec - startTime.tv_sec) * 1000000 + (endTime.tv_usec-startTime.tv_usec);
        std::cout<< "time : "<< time_use <<" us"<<std::endl;

        Mat probMat = prob.reshape(1, 1);
        Point classNumber;  
        double classProb;  
        minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber); // 可能性最大的一个  

        int classIdx = classNumber.x; // 分类索引号  
        char text[128];
        sprintf(text," %s :%.3f %%",m_labels.at(classIdx).c_str(), classProb * 100);
        putText(image, text, Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2, 8);        

        imshow("Image", image);

        waitKey(0);        

        _return.desc = text;
        _return.ret = image.ptr();
    }
}//end namespace

int  main(int argc, char const *argv[])
{
    Result re;
    ImageHandle::DeepSearch deep;
    deep.DeepLearning(re, "jemma.png");
    //deep.load();
    return 0;
}