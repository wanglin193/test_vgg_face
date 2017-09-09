//some code from caffe\examples\cpp_classification\classification.cpp
//wanglin193 at gmail.com

#include <caffe/caffe.hpp>
#include "opencv2/opencv.hpp" 
 
#include <iostream>
#include <string>
#include <vector>

using namespace caffe;   
using namespace std;
 
//#define CPU_ONLY 

bool parse_csv(const string& list_name, vector<string>& IDs, vector<string>& im_names, vector<string>& bin_names)
{
  ifstream in;
  in.open(list_name.c_str());
  if (in.bad())
  {
    cout << "no such file: " << list_name << endl;
    return false;
  }

  cout << "Parsing  " << list_name << endl; 
  std::string line;
  vector<string> fname;
  std::getline(in, line);//first line
  while (std::getline(in, line))
  {
    fname.clear();
    std::istringstream iss(line);
    std::string col;
    while (std::getline(iss, col, ','))
      fname.push_back(col);
    if (fname.size() != 3) cout << "Item num error in csv." << endl;
    //cout << fname[0] << fname[1] << fname[2] << endl;
    IDs.push_back(fname[0]);
    im_names.push_back(fname[1]);
    bin_names.push_back(fname[2]);
  }
  in.close();
  
  return true;
}

namespace wl_VGG_FACE
{
  typedef std::pair<int, float> Prediction;//<index of items, match value>
  struct FR
  {
    //vgg face caffe model 
    boost::shared_ptr<Net<float> > net;
    cv::Size input_geometry;//net input  sample size
    int num_channels; //net input sample channek
    cv::Mat input_mean;//input image need subtract this 
 
    //gallery data
    string gpath;//data path
    string gcsv_name;//data list 
    vector<string> mLabels;
    vector<string> mSampleFiles;
    vector<string> mFeatureFiles;
    vector<cv::Mat> mSamples; //croped face sample 
    vector<vector<float> > mvFeatures;//feature vectors
    
    int num_item; //item number before add new items
    
    //face detecition
    cv::CascadeClassifier face_cascade;
    cv::Mat mNoFace;
    FR(void)
    {  
      face_cascade.load("vgg_face_caffe/haarcascade_frontalface_alt2.xml"); 
    }

    //--------- 5 functions from classification.cpp --------
    void load_model(const string& model_file, const string& trained_file)
    { 

#ifdef CPU_ONLY
      Caffe::set_mode(Caffe::CPU);
#else
      int count;
      CUDA_CHECK(cudaGetDeviceCount(&count));
      if(count>0)
        std::cout << "cudaGetDevice OK. GPU device count: "<<count << std::endl;
      else
        std::cout << "NO GPU found, pls use CPU_ONLY. " << std::endl;

      Caffe::set_mode(Caffe::GPU);
#endif

      /* Load the network. */
      net.reset(new Net<float>(model_file, TEST));
      net->CopyTrainedLayersFrom(trained_file);
      std::cout << "Load model done." << std::endl;

      CHECK_EQ(net->num_inputs(), 1) << "Network should have exactly one input.";
      CHECK_EQ(net->num_outputs(), 1) << "Network should have exactly one output.";

      Blob<float>* input_layer = net->input_blobs()[0];
      num_channels = input_layer->channels();
      CHECK(num_channels == 3 || num_channels == 1)
        << "Input layer should have 1 or 3 channels.";
      input_geometry= cv::Size(input_layer->width(), input_layer->height());

      //set mean of input channel 
      cv::Scalar channel_mean = cv::Scalar(129.1863, 104.7624, 93.5940, 0);
      input_mean = cv::Mat(input_geometry, CV_32FC3, channel_mean);
      if (input_mean.channels() != num_channels)
      {
        std::cout << "channels of mean image: " << input_mean.channels() << " error  input_mean  .\n";
      }
      //cv::imwrite("ss.png",input_mean);
    } 
    static bool PairCompare(const std::pair<float,int>& lhs, const std::pair<float, int>& rhs)
    {
      return lhs.first > rhs.first;
    }
    static std::vector<int> Argmax(const std::vector<float>& v, int N)
    {
      std::vector<std::pair<float, int> > pairs;
      for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
      std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

      std::vector<int> result;
      for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
      return result;
    }  
    //from classification.cpp
    void WrapInputLayer(std::vector<cv::Mat>* input_channels)
    {
      Blob<float>* input_layer = net->input_blobs()[0];

      int width = input_layer->width();
      int height = input_layer->height();

      float* input_data = input_layer->mutable_cpu_data(); 
      for (int i = 0; i < input_layer->channels(); ++i)
      {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
      }
    }
    void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels)
    { 
      /* Convert the input image to the input image format of the network. */
      cv::Mat sample;
      if (img.channels() == 3 && num_channels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
      else if (img.channels() == 4 && num_channels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
      else if (img.channels() == 4 && num_channels == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
      else if (img.channels() == 1 && num_channels == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
      else
        sample = img;
 
      cv::Mat sample_resized;
      if (sample.size() != input_geometry)
        cv::resize(sample, sample_resized, input_geometry);
      else
        sample_resized = sample;
 
      cv::Mat sample_float;
      if (num_channels == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
      else
        sample_resized.convertTo(sample_float, CV_32FC1);
 
      cv::Mat sample_normalized;
      cv::subtract(sample_float, input_mean, sample_normalized); 
 
      /* This operation will write the separate BGR planes directly to the
      * input layer of the network because it is wrapped by the cv::Mat
      * objects in input_channels. */
      cv::split(sample_normalized, *input_channels);
 
      CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == net->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
    }
    //------------------------------------------------------- 

    //extract features at fc7
    std::vector<float> extract_feature(const cv::Mat& img)
    {
      Blob<float>* input_layer = net->input_blobs()[0];
      //1*3*224*224
      input_layer->Reshape(1, num_channels, input_geometry.height, input_geometry.width);
      /* Forward dimension change to all layers. */
      net->Reshape();
 
      std::vector<cv::Mat> input_channels;
      WrapInputLayer(&input_channels);  
     
      Preprocess(img, &input_channels);
     
      net->Forward();

      //提取特征用
      boost::shared_ptr<caffe::Blob<float>> fc7 = net->blob_by_name("fc7");
  
      const float* begin = fc7->cpu_data(); 
      const float* end = begin + fc7->channels();
      //归一化
      std::vector<float> v = std::vector<float>(begin, end);
      normlize_unit(v);
      return v;
    }
    void print_net_info()
    {
      std::cout << "Num of inputs: " << net->num_inputs()
        << ", Num of outputs: " << net->num_outputs() << "\n";

      /* for (int i = 0; i < net_->blob_names().size(); i++)
         std::cout << net_->blob_names()[i] << std::endl;
       auto& blob = net_->blob_by_name("fc7");
       std::cout << blob->count() << std::endl;*/

      for (int i = 0; i < net->input_blobs().size(); i++)
      {
        std::cout << net->input_blobs()[i]->width() << " "
          << net->input_blobs()[i]->height() << " "
          << net->input_blobs()[i]->channels() << "\n";
      }

      std::vector<boost::shared_ptr<caffe::Layer<float> > > layers = net->layers();
      std::vector<std::string> layer_names = net->layer_names();
      std::cout << "layer size of net :" << layers.size() << std::endl;
      std::cout << "layer_names size of net :" << layer_names.size() << std::endl;

      for (int i = 0; i < layers.size(); i++)
      {
        std::cout << ">>> Layer " << i << " :\n";
        std::string type_name = layers[i]->type();
        std::cout << "name: " << layer_names[i] << ", " << " type: " << type_name << std::endl;

        vector<boost::shared_ptr<caffe::Blob<float> > > & blobs = layers[i]->blobs();
        //if(blobs.size()==0) std::cout << "no w&b in this layer" << std::endl;
        if (blobs.size() == 2) //weight,bias
        {
          boost::shared_ptr<caffe::Blob<float> >  weight = blobs[0];
          boost::shared_ptr<caffe::Blob<float> >  bias = blobs[1];
          std::cout << "weight num: " << weight->count() << ", bias num: " << bias->count() << std::endl;
          //std::cout << "weight size = "<<weight->height()<< "x" <<weight->width() << std::endl;
          // std::cout << "weight num = " << weight->num() << std::endl;
          std::cout << "weight dim = " << weight->shape().size() << ", [ ";
          for (int j = 0; j < weight->shape().size(); j++) std::cout << weight->shape()[j] << " ";
          std::cout << "]" << std::endl;

          //if(weight->count()<2000)
          //{
          //  for(int j=0;j<weight->count();j++) std::cout << weight->cpu_data()[j] <<",";
          //  for (int j = 0; j<bias->count(); j++)   std::cout << bias->cpu_data()[j] << ",";
          //}
        }
      }
    }
    void normlize_unit(vector<float>& v)
    {
      float sum = 0.f, sum2 = 0.f;
      int n = v.size();
      for (int i = 0; i < n; i++)
      {
        sum += v[i];
        sum2 += v[i]* v[i];
      }
      sum /= (float)n; 
      float var = sum2/(float)n  - sum*sum;
      //cout << sum <<","<< var << endl;
      for (int i = 0; i < n; i++)
      {
        //v[i] = (v[i] - sum) / var; //
        v[i] = v[i] / sqrt(sum2);//norm=1
      }
      //verify the norm
     /* sum = 0; for (int i = 0; i < n; i++)  sum += v[i]* v[i];  cout <<"Norm = " <<sum << endl;*/
    }
    bool load_feature(const string& filename, vector<float>& vM)
    {
      FILE* file = fopen(filename.c_str(), "rb");
      if (file == NULL)
        return false;
      vM.clear();
      int num;// = vM.size();
      fread(&num, sizeof(int), 1, file);
      for (int i = 0; i < num; i++)
      {
        float v;
        fread(&v, sizeof(float), 1, file);
        vM.push_back(v);
      }
      fclose(file);
      
      /*//check norm
      float sum = 0;
      for (int i = 0; i < num; i++) { sum+=vM[i]* vM[i]; }
      cout << "Norm = " << sum << endl;
      */
      return true;
    }
    bool write_feature(const string& filename, vector<float>& vM)
    {
      FILE* file = fopen(filename.c_str(), "wb");
      if (file == NULL )
        return false;

      int num = vM.size();
      fwrite(&num, sizeof(int), 1, file);
      float *p1 = &vM[0];
      fwrite(p1,sizeof(float),num,file);
      fclose(file);
      return true;
    }
    bool load_gallery(const string &gpath_, const string &gcsv_)
    {
      //set database path
      gpath = gpath_;
      gcsv_name = gcsv_;

      //load NOface image
      mNoFace = cv::imread(gpath + "no_face.png");

      //new database
      mLabels.clear();
      mSampleFiles.clear();
      mFeatureFiles.clear();
      mSamples.clear();
      mvFeatures.clear();
      parse_csv(gcsv_name, mLabels, mSampleFiles, mFeatureFiles);
      
      //load images and features:
      for (int i = 0; i < mLabels.size(); i++)
      {
        mSamples.push_back(cv::imread(gpath + mSampleFiles[i]));
        vector<float> vfeature;
        load_feature(gpath + mFeatureFiles[i], vfeature);
        mvFeatures.push_back(vfeature);
      }
      //item number before add new
      num_item = mLabels.size();
      return true;
    }
    bool write_gallery( )
    {
      //save data base
      FILE *fp = fopen(gcsv_name.c_str(),"w");
      
      fprintf(fp, "name,img_name,bin_name\n");
      for (int i = 0; i < mLabels.size(); i++)
      {
        fprintf(fp,"%s,%s,%s\n", mLabels[i].c_str(), mSampleFiles[i].c_str(), mFeatureFiles[i].c_str());
      }
      fclose(fp);
      //save data to disk
      for (int i = num_item; i < mLabels.size(); i++)
      {
        cv::imwrite(gpath + mSampleFiles[i], mSamples[i]);
        write_feature(gpath + mFeatureFiles[i], mvFeatures[i]);
      }
      cout << "Save " << mLabels.size() << " items in database." << endl;
      return true;
    }
   //
    bool process_keyboard(cv::Mat& img,int key, vector<cv::Rect> detected_faces,cv::Mat & roiImg)
    {
      cv::Mat canvas = img.clone();
      
      bool face_found = detected_faces.size() > 0;
      //draw face
      if(face_found)
        cv::rectangle(canvas, detected_faces[0], cv::Scalar(0, 0, 255));

      //default PinP
      cv::Size sz = cv::Size(128, 128);
      cv::Mat thumbnail;
      cv::Mat roi = canvas(cv::Rect(0, 0, sz.width, sz.height));
      cv::resize(mNoFace, thumbnail, sz);
      thumbnail.copyTo(roi);
 
      //press space to do recognize
      if (/*key == ' ' &&*/ face_found)
      { 
        std::vector<float> pv = extract_feature(roiImg); 
        //recog here
        std::vector<Prediction> predictions = compare_vectors(pv, 3);
        //  Print the top N predictions. 
        std::cout << "-- firse 3 item, Name and Score --\n";
        for (size_t i = 0; i < predictions.size(); ++i)
        {
          Prediction p = predictions[i];
          std::cout << mLabels[p.first] << "  " << p.second << std::endl;
        }

         //show first rank face sample:
        if (predictions[0].second > 0.4)
        {
          cv::resize(mSamples[predictions[0].first], thumbnail, sz);
          thumbnail.copyTo(roi);
        }       
        //cv::imshow("first match face ", mSamples[predictions[0].first]);
      }

      //add to face gallery 
      if (key == 'a' & face_found)
      {
        std::string name;
        std::cout << "-----Add new item----\nPlease input name: ";
        std::cin >> name;

        std::vector<float> pv = extract_feature(roiImg);

        mLabels.push_back(name);
        mSampleFiles.push_back(name + ".png");
        mFeatureFiles.push_back(name + ".bin");
        mSamples.push_back(roiImg);
        mvFeatures.push_back(pv);
      }
      //quit
      if (key == 27)
      {
        if (mLabels.size()>num_item)
          write_gallery();
        return false;
      }
      //
      imshow("FR demo.Press 'a' to add new face,'ESC' to save & quit", canvas);

      return true;
    }
    //probe vs. galleries 
    std::vector<Prediction> compare_vectors(std::vector<float>& probe, int N)
    { 
      float mv =  -9999;
      if (probe.size() != mvFeatures[0].size())
      {
        cout << " match length error." << endl;
        return std::vector<Prediction>();
      }
      std::vector<float> v;
      for (int i = 0; i < mvFeatures.size(); i++)
      {
        float sum = 0.0f;
        for (int j = 0; j < probe.size(); j++)
          sum += (probe[j] * mvFeatures[i][j]);
        v.push_back(sum);
      }
      N = std::min<int>(mLabels.size(), N);
      std::vector<int> maxN = Argmax(v, N);
      std::vector<Prediction> predictions;
      for (int i = 0; i < N; ++i)
      {
        int idx = maxN[i];
        predictions.push_back(std::make_pair(idx, v[idx]));
      }
      return predictions;
    }
    int recognition_add_gallery_webcam()
    {
      cv::Size norm_size = input_geometry;//图像尺寸，NN输入决定
      int min_face_size = 80;
      vector<cv::Rect> detected_faces;
      cv::VideoCapture capture;
      capture.open(0);

      if (!capture.isOpened())
      {
        cout << " NO camera input." << endl;
        return 1;
      }

      std::cout << "Capture is opened." << std::endl;
      for (;;)
      {
        cv::Mat img, gray, roiImg;

        capture >> img;
        if (img.empty())
        {
          cout << " capture error.\n";
          continue;
        }
        // resize(img, img, Size(), 0.5, 0.5, CV_INTER_LINEAR);
        cvtColor(img, gray, CV_BGR2GRAY);
        face_cascade.detectMultiScale(gray, detected_faces, 1.2, 4, 0, cv::Size(min_face_size, min_face_size));
         
        //normalize faces
        if (detected_faces.size() > 0)
        {
          //for (int j = 0; j < detected_faces.size(); j++)
          int j = 0; //first one only
          {
            cv::Rect rectface = detected_faces[j];

            int normal_roi_width = norm_size.width;
            auto normal_roi_bord = (float)normal_roi_width*0.13;
            //transform from ori-image to normorlized-sample
            auto scale = ((float)normal_roi_width - 2 * normal_roi_bord) / (float)rectface.width;
            auto shift_x = normal_roi_bord - scale*rectface.x;
            auto shift_y = (normal_roi_bord*0.3) - scale*rectface.y;
            cv::Mat mat2roi = (cv::Mat_<float>(2, 3) << scale, 0, shift_x, 0, scale, shift_y);
            cv::warpAffine(img, roiImg, mat2roi, cv::Size(normal_roi_width, normal_roi_width), 
              cv::INTER_LINEAR, cv::BORDER_CONSTANT,cv::Scalar(128,128,128));
            cv::resize(roiImg, roiImg, norm_size);
          }
        }

        int key = cv::waitKey(10); 
    
        if (false == process_keyboard(img,key, detected_faces, roiImg))
          break;
      }//end for(;;)
      return 0;
    }
  };
}//end namespace

void print_usage()
{
  cout << "Usage: --------------------------------------.\n";
  cout << "      'a' to add new detect face to database.\n";
  cout << "      Esc to save database and quit.\n";
}
void main(int argc, char** argv)
{
  ::google::InitGoogleLogging(argv[0]);
  wl_VGG_FACE::FR mFR;
  mFR.load_model("vgg_face_caffe/VGG_FACE_deploy.prototxt", "vgg_face_caffe/VGG_FACE.caffemodel");
  //load gallery 
  mFR.load_gallery("gallery/", "gallery/namelist.csv");

  print_usage();

  //recog and add items
  mFR.recognition_add_gallery_webcam();
}

