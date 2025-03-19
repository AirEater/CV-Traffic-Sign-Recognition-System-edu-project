//#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//#include <iostream>
//#include <fstream>
//#include <algorithm> // for std::min_element and std::max_element
//
//using namespace cv;
//using namespace std;
//
//// Function to extract Hu moments with black background exclusion
//vector<double> extractHuMomentsWithMask(const Mat& image) {
//    Mat grayImage, mask, maskedGrayImage;
//
//    // Convert image to grayscale
//    cvtColor(image, grayImage, COLOR_BGR2GRAY);
//
//    // Create a mask to exclude the black background
//    inRange(image, Scalar(0, 0, 0), Scalar(20, 20, 20), mask); // Adjust thresholds as necessary
//    bitwise_not(mask, mask); // Invert mask to focus on non-black pixels
//    
//    // Apply the mask to the grayscale image
//    grayImage.copyTo(maskedGrayImage, mask);
//
//    // Compute moments and Hu Moments
//    Moments moments = cv::moments(maskedGrayImage, true);
//    vector<double> huMoments(7);
//    HuMoments(moments, huMoments);
//
//    return huMoments;
//}
//
//// Function to extract Color Histograms with black background exclusion
//vector<float> extractColorHistogramWithMask(const Mat& image, int bins = 256) {
//    vector<Mat> bgr_planes;
//    split(image, bgr_planes);
//
//    vector<float> histData;
//    int histSize = bins;
//    float range[] = { 0, 256 };
//    const float* histRange = { range };
//    bool uniform = true, accumulate = false;
//    Mat b_hist, g_hist, r_hist;
//
//    // Create a mask to exclude the black background
//    Mat mask;
//    inRange(image, Scalar(0, 0, 0), Scalar(20, 20, 20), mask); // Adjust thresholds as necessary
//    bitwise_not(mask, mask); // Invert mask to focus on non-black pixels
//
//    // Calculate histograms for B, G, and R channels with the mask
//    calcHist(&bgr_planes[0], 1, 0, mask, b_hist, 1, &histSize, &histRange, uniform, accumulate);
//    calcHist(&bgr_planes[1], 1, 0, mask, g_hist, 1, &histSize, &histRange, uniform, accumulate);
//    calcHist(&bgr_planes[2], 1, 0, mask, r_hist, 1, &histSize, &histRange, uniform, accumulate);
//
//    // Normalize histograms and add to output vector
//    normalize(b_hist, b_hist, 0, 1, NORM_MINMAX);
//    normalize(g_hist, g_hist, 0, 1, NORM_MINMAX);
//    normalize(r_hist, r_hist, 0, 1, NORM_MINMAX);
//
//    histData.insert(histData.end(), b_hist.begin<float>(), b_hist.end<float>());
//    histData.insert(histData.end(), g_hist.begin<float>(), g_hist.end<float>());
//    histData.insert(histData.end(), r_hist.begin<float>(), r_hist.end<float>());
//
//    return histData;
//}
//
//// Function to normalize a vector (Min-Max Normalization)
//vector<double> minMaxNormalize(const vector<double>& values) {
//    double minVal = *min_element(values.begin(), values.end());
//    double maxVal = *max_element(values.begin(), values.end());
//
//    vector<double> normalizedValues;
//    for (double val : values) {
//        double normVal = (val - minVal) / (maxVal - minVal);
//        normalizedValues.push_back(normVal);
//    }
//    return normalizedValues;
//}
//
//// Function to assign labels based on the filename (simple example, modify as needed)
//int getLabelFromFilename(const string& filename) {
//    if (filename.find("red_sign") != string::npos) {
//        return 1;
//    }
//    else if (filename.find("yellow_sign") != string::npos) {
//        return 2;
//    }
//    else if (filename.find("blue_sign") != string::npos) {
//        return 0;
//    }
//}
//
//// Main function to process images and store features with labels
//void performFeatureExtraction() {
//    String imgPattern("Inputs/Traffic signs/Segmented/*.png");
//    vector<string> imagePath;
//    glob(imgPattern, imagePath, true);
//
//    // Open two CSV files: one for Hu moments and one for color histograms
//    ofstream huMomentsFile("hu_moments_data.csv");
//    ofstream colorHistogramFile("color_histogram_data.csv");
//
//    // Write the header for Hu Moments CSV file
//    huMomentsFile << "FileName,Label,HuMoment1,HuMoment2,HuMoment3,HuMoment4,HuMoment5,HuMoment6,HuMoment7" << endl;
//
//    // Write the header for Color Histograms CSV file
//    colorHistogramFile << "FileName,Label";
//    for (int i = 0; i < 256; ++i) colorHistogramFile << ",ColorHist_B" << i;
//    for (int i = 0; i < 256; ++i) colorHistogramFile << ",ColorHist_G" << i;
//    for (int i = 0; i < 256; ++i) colorHistogramFile << ",ColorHist_R" << i;
//    colorHistogramFile << endl;
//
//    for (size_t i = 0; i < imagePath.size(); ++i) {
//        Mat image = imread(imagePath[i]);
//
//        if (image.empty()) {
//            cout << "The image is not loaded: " << imagePath[i] << endl;
//            continue;
//        }
//
//        // Extract Hu moments with black background exclusion
//        vector<double> huMoments = extractHuMomentsWithMask(image);
//        vector<double> normalizedHuMoments = minMaxNormalize(huMoments);
//
//        // Extract Color Histogram with black background exclusion
//        vector<float> colorHistogram = extractColorHistogramWithMask(image);
//
//        // Get label from filename
//        int label = getLabelFromFilename(imagePath[i]);
//
//        // Write Hu Moments to the Hu Moments CSV file
//        huMomentsFile << imagePath[i] << "," << label;
//        for (double huMoment : normalizedHuMoments) {
//            huMomentsFile << "," << huMoment;
//        }
//        huMomentsFile << endl;
//
//        // Write Color Histogram to the Color Histogram CSV file
//        colorHistogramFile << imagePath[i] << "," << label;
//        for (float colorHistValue : colorHistogram) {
//            colorHistogramFile << "," << colorHistValue;
//        }
//        colorHistogramFile << endl;
//    }
//
//    huMomentsFile.close();
//    colorHistogramFile.close();
//
//    cout << "Feature extraction complete. Data saved to hu_moments_data.csv and color_histogram_data.csv with labels." << endl;
//}
//
//// Main function with menu options
//int main() {
//    int choice = 0;
//
//    while (choice != 2) { // Add an exit option
//        cout << "Menu Options:\n";
//        cout << "1. Feature Extraction (Hu Moments & Color Histograms)\n";
//        cout << "2. Exit\n";
//        cout << "Enter your choice: ";
//        cin >> choice;
//
//        switch (choice) {
//        case 1:
//            cout << "Performing Color Histogram and Hu Moments Extraction..." << endl;
//            performFeatureExtraction();
//            break;
//        case 2:
//            cout << "Exiting program." << endl;
//            break;
//        default:
//            cout << "Invalid option. Please try again." << endl;
//        }
//    }
//
//    return 0;
//}
