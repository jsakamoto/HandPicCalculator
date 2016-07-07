// HandPicCalculator.cpp : Defines the entry point for the console application.
//
#include <opencv2/opencv.hpp>

//toString template
template <typename T>
std::string inline ToString(T val)
{
    std::stringstream stream;
    stream << val;
    return stream.str();
}

cv::Ptr<std::vector<cv::Mat>> GetFindContourMats(cv::Mat input_mat) {

    cv::Mat taget_image, image_for_contours, contours_image, draw_image;

    contours_image = input_mat.clone();
    cv::imwrite("image_original.png", contours_image);

    cv::cvtColor(contours_image, image_for_contours, cv::COLOR_BGR2GRAY);
    cv::normalize(image_for_contours, image_for_contours, 0, 255, cv::NORM_MINMAX);
    cv::imwrite("image_gray.png", image_for_contours);

    //This parameter may be sensitive, Try THRESH_OTSU
    cv::threshold(image_for_contours, image_for_contours, 70, 255, cv::THRESH_BINARY);

    draw_image = image_for_contours.clone();
    cv::imwrite("image_threshold.png", image_for_contours);

    std::vector<cv::Mat> contours;
    cv::findContours(image_for_contours, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

    cv::Ptr<std::vector<cv::Mat>> ret_mat_vector(new std::vector<cv::Mat>);
    std::vector<cv::Rect> rects;
    for (size_t i = 0; i < contours.size(); i++)
    {
        rects.push_back(cv::boundingRect(contours[i]));
    }

    //x-line sort
    std::sort(rects.begin(), rects.end(),
        [](const cv::Rect& a, const cv::Rect& b) {return (a.x < b.x); }
    );

    int rect_number = 0;
    for (cv::Rect rect : rects)
    {
        //Exclusion small height contour
        if (rect.height < 20) {
            continue;
        }
        //Exclusion too big contour
        if ((rect.width * rect.height) > 600 * 600) {
            continue;
        }

        //clip picture without first contour
        if (rect.x != 1) {
            cv::Mat temp;
            draw_image(rect).copyTo(temp);
            cv::imwrite("clip" + ToString(rect_number) + ".png", temp);
            ret_mat_vector->push_back(temp);
        }

        //frame lines
        cv::line(contours_image, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y), 
            cv::Scalar(0, 128, 0), 2, 0);
        cv::line(contours_image, cv::Point(rect.x + rect.width, rect.y), 
            cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Scalar(0, 128, 0), 2, 0);
        cv::line(contours_image, cv::Point(rect.x + rect.width, rect.y + rect.height), 
            cv::Point(rect.x, rect.y + rect.height), cv::Scalar(0, 128, 0), 2, 0);
        cv::line(contours_image, cv::Point(rect.x, rect.y + rect.height), cv::Point(rect.x, rect.y), 
            cv::Scalar(0, 128, 0), 2, 0);

        rect_number++;
    }

    cv::imwrite("image_contours.png", contours_image);

    return ret_mat_vector;

}

void ResizeKeepRetio(cv::Mat& target_mat, int width, double resize_ratio) {

    cv::Mat tmp_image, convert_mat;

    tmp_image = target_mat.clone();
    cv::Mat base_mat(cv::Size(width, width), CV_8UC1, 255);

    int big = tmp_image.cols > tmp_image.rows ? tmp_image.cols : tmp_image.rows;
    double ratio = ((double)width / (double)big) * resize_ratio;

    //resize
    cv::resize(tmp_image, convert_mat, cv::Size(), ratio, ratio, cv::INTER_NEAREST);

    //Anchor Center
    cv::Mat Roi1(base_mat, cv::Rect((width - convert_mat.cols) / 2, 
        (width - convert_mat.rows) / 2, convert_mat.cols, convert_mat.rows));
    convert_mat.copyTo(Roi1);

    target_mat = base_mat;
}

void ConvertToMl(const std::vector< cv::Mat > & train_samples, cv::Mat& train_data)
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = std::max<int>(train_samples[0].cols, train_samples[0].rows);
    cv::Mat tmp(1, cols, CV_32FC1);
    train_data = cv::Mat(rows, cols, CV_32FC1);

    for (const cv::Mat& mat : train_samples) {

        size_t index = &mat - &train_samples[0];

        CV_Assert(mat.cols == 1 || mat.rows == 1);
        if (mat.cols == 1)
        {
            cv::transpose(mat, tmp);
            tmp.copyTo(train_data.row(index));
        }
        else if (mat.rows == 1)
        {
            mat.copyTo(train_data.row(index));
        }
    }
}

void ConvertToHogVec(int picIndx, cv::Mat& original_train_mat, std::vector< cv::Mat >& gradients_lst) {

    cv::Mat taget_image2, mat_for_vector;
    cv::HOGDescriptor hog;

    hog.winSize = cv::Size(32, 32);
    std::vector<cv::Point> location;
    std::vector< float > descriptors;

    hog.compute(original_train_mat, descriptors, cv::Size(8, 8), cv::Size(0, 0), location);

    gradients_lst.push_back(cv::Mat(descriptors).clone());

}

//==============================================================================
void calc() {

    cv::Mat image = cv::imread("TestPicture.png");

    cv::String KNearest = "KNearestDigit.xml";
    cv::String filenameSvm = "SVMDigit.xml";

    cv::Ptr<cv::ml::KNearest> knn = cv::ml::StatModel::load<cv::ml::KNearest>(KNearest);
    cv::Ptr<cv::ml::SVM> svm = cv::ml::StatModel::load<cv::ml::SVM>(filenameSvm);

    cv::Ptr<std::vector<cv::Mat>> train_mats = GetFindContourMats(image);

    int K = 5;
    cv::Mat response_knn, dist;

    for (cv::Mat train_mat : *train_mats) {

        std::vector< cv::Mat > gradient_list;
        cv::Mat train_vector;

        ResizeKeepRetio(train_mat, 32, 0.65);

        ConvertToHogVec(0, train_mat, gradient_list);
        ConvertToMl(gradient_list, train_vector);

        //Result KNearest
        knn->findNearest(train_vector, K, cv::noArray(), response_knn, dist);
        std::cerr << "KNearest:" << response_knn << std::endl;

        //Result SVM
        int response_Svm = static_cast<int>(svm->predict(train_vector));
        std::cerr << "SVM:" << response_Svm << std::endl;

    }

    system("pause");
}


int main()
{
    calc();
    return 0;
}
