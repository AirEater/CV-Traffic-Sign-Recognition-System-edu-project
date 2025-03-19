#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdlib.h>
#include <string>
#include "Supp.h"

using namespace cv;
using namespace std;

void resizeImage(Mat&);

void preprocessImage(Mat&);
double calculateAverageBrightness(const Mat&);
void autoAdjustBrightness(Mat&);
void applyCLAHE(Mat&, double, Size);
void edgeDetect(Mat&, Mat&);

//Mat loadShapeTemplate(const string& templatePath);
//void generateShapeTemplate(vector<Mat>&, vector<string>&);
vector<vector<Point>> displayCreateContours(Mat& image, const string& windowName = "Contours");

Mat colorSegment(Mat&, String = "yes");
Mat segmentRedSign(Mat&);
Mat segmentYellowSign(Mat&);
Mat segmentBlueSign(Mat&);

Mat shapeSegment(const Mat&, const Mat&);

bool isSegmentedImageNearlyEmpty(const Mat&, double thresholdPercentage = 1.0);

int menu() {
    cout << "0. All Signs\n";
    cout << "1. Red Signs\n";
    cout << "2. Yellow Signs\n";
    cout << "3. Blue Signs\n";
    cout << "4. Exit\n";

    int choice = -1;
    cout << "Which signs are you looking for? (0-4): ";
    cin >> choice;
    // Get user input and validate it
    while (choice < 0 || choice > 4) {
        system("cls");
        cout << "0. All Signs\n";
        cout << "1. Red Signs\n";
        cout << "2. Yellow Signs\n";
        cout << "3. Blue Signs\n";
        cout << "4. Exit\n";

        cout << "Invalid choice. Please select a number between 0 and 4.\n";
        cin >> choice;

        if (choice == 4) {
            cout << "Exiting the program...\n";
            system("cls");
            return -1;
        }


    }

    return choice;
}

int main(int argc, char** argv) {
    String allImageInOneFolder("Inputs/Traffic signs/All/*.png");
    String redImgPattern("Inputs/Traffic signs/Red signs/*.png");
    String blueImgPattern("Inputs/Traffic Signs/Blue Signs/*.png");
    String yellowImgPattern("Inputs/Traffic Signs/Yellow Signs/*.png");
    String folderPath;
    vector<string> imagePath;
    String imgPattern[] = { redImgPattern, yellowImgPattern,  blueImgPattern };

    int start = menu();
    int end;
    if (start == -1)
        return 0;
    else if (start == 0)
        end = 0;
    else if (start > 0) // if not looking for all
    {
        end = start;
        start -= 1;
    }

    if (end != 0) {
        for (int patternIndex = start; patternIndex < end; patternIndex++) {
            glob(imgPattern[patternIndex], imagePath, true);
            folderPath = imgPattern[patternIndex];
            for (size_t i = 0; i < imagePath.size(); ++i) {

                Mat image = imread(imagePath[i]);

                if (image.empty()) {
                    cout << "The image is not loaded." << endl;
                    continue;
                }

                // create large window for show the original and segmented result
                Mat resultWin;
                int const noOfImagePerCol = 1, noOfImagePerRow = 2;
                Mat win[noOfImagePerRow * noOfImagePerCol], legend[noOfImagePerRow * noOfImagePerCol];
                createWindowPartition(image, resultWin, win, legend, noOfImagePerCol, noOfImagePerRow);

                // Add labels to the windows
                putText(legend[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
                putText(legend[1], "Sign segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

                image.copyTo(win[0]);

                Mat colorSegmented = colorSegment(image);
                imshow("Color Segmented Image", colorSegmented);

                Mat shapeSegmented = shapeSegment(image, colorSegmented);

                Mat secondShapeSegmented = shapeSegment(image, shapeSegmented);

                Mat segmentedImage = colorSegment(secondShapeSegmented, "no");

                // **Check if the segmentation result is nearly empty using the function**
                if (isSegmentedImageNearlyEmpty(segmentedImage, 1.0)) {  // Using 1% as the threshold
                    cout << "The segmented image is almost empty, using the previously segmented image." << endl;
                    // Fall back to the previous segmentation
                    segmentedImage = secondShapeSegmented;
                }

                segmentedImage.copyTo(win[1]);
                imshow("Segmented Image", resultWin);
                waitKey(0);
                destroyAllWindows();
            }
        }
    }
    else if (end == 0) { // Process all images in the "All" folder
        glob(allImageInOneFolder, imagePath, true);
        for (size_t i = 0; i < imagePath.size(); ++i) {
            Mat image = imread(imagePath[i]);

            if (image.empty()) {
                cout << "The image is not loaded." << endl;
                continue;
            }

            // create large window for show the original and segmented result
            Mat resultWin;
            int const noOfImagePerCol = 1, noOfImagePerRow = 2;
            Mat win[noOfImagePerRow * noOfImagePerCol], legend[noOfImagePerRow * noOfImagePerCol];
            createWindowPartition(image, resultWin, win, legend, noOfImagePerCol, noOfImagePerRow);

            // Add labels to the windows
            putText(legend[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
            putText(legend[1], "Sign segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

            image.copyTo(win[0]);

            Mat colorSegmented = colorSegment(image);
            imshow("Color Segmented Image", colorSegmented);

            Mat shapeSegmented = shapeSegment(image, colorSegmented);

            Mat secondShapeSegmented = shapeSegment(image, shapeSegmented);

            Mat segmentedImage = colorSegment(secondShapeSegmented, "no");

            if (isSegmentedImageNearlyEmpty(segmentedImage, 1.0)) {  // Using 1% as the threshold
                cout << "The segmented image is almost empty, using the previously segmented image." << endl;

                segmentedImage = secondShapeSegmented;
            }

            segmentedImage.copyTo(win[1]);
            imshow("Segmented Image", resultWin);
            waitKey(0);
            destroyAllWindows();
        }
    }

    return 0;
}

// Function to check if the segmented image is nearly empty
bool isSegmentedImageNearlyEmpty(const Mat& segmentedImage, double thresholdPercentage) {
    if (segmentedImage.empty()) {
        return true;  // If the image is empty, consider it "nearly empty"
    }

    Mat graySegmented;

    // Convert to grayscale if the image is not single-channel
    if (segmentedImage.channels() == 3) {
        cvtColor(segmentedImage, graySegmented, COLOR_BGR2GRAY);
    }
    else {
        graySegmented = segmentedImage.clone();
    }

    // Calculate non-zero pixels
    int nonZeroCount = countNonZero(graySegmented);
    double emptyThreshold = (thresholdPercentage / 100.0) * (graySegmented.rows * graySegmented.cols);  // Based on percentage

    // Return true if the non-zero pixels are below the threshold
    return nonZeroCount < emptyThreshold;
}

Mat segmentRedSign(Mat& srcI) {
    Mat threeImages[3], redMask, canvasColor, canvasGray, resultWin;

    // Create a result window and partitions for display
    int const noOfImagePerCol2 = 1, noOfImagePerRow2 = 2;
    Mat win2[noOfImagePerRow2 * noOfImagePerCol2], legend2[noOfImagePerRow2 * noOfImagePerCol2];
    createWindowPartition(srcI, resultWin, win2, legend2, noOfImagePerCol2, noOfImagePerRow2);

    // Add labels to the windows
    putText(legend2[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    putText(legend2[1], "Sign segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

    // Copy original image to the first window
    srcI.copyTo(win2[0]);



    // Prepare canvas for segmentation
    canvasColor.create(srcI.rows, srcI.cols, CV_8UC3);
    canvasGray.create(srcI.rows, srcI.cols, CV_8U);
    canvasColor = Scalar(0, 0, 0);

    // Split the image into RGB channels
    split(srcI, threeImages);

    // Create a red mask based on the red channel being dominant
    redMask = (threeImages[0] * 1.5 < threeImages[2]) & (threeImages[1] * 1.5 < threeImages[2]);

    // Find contours on the red mask
    vector<vector<Point>> contours;
    findContours(redMask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    int index = 0, max = 0;

    // Find the largest contour
    for (int i = 0; i < contours.size(); i++) {
        canvasGray = 0;
        if (max < contours[i].size()) {
            max = contours[i].size();
            index = i;
        }
        drawContours(canvasGray, contours, i, 255); // Draw all contours for visualization (optional)
    }

    // Reset the canvas and draw only the largest contour
    canvasGray = 0;
    drawContours(canvasGray, contours, index, 255);

    // Compute the moments to find the center of the largest contour
    Moments M = moments(canvasGray);
    Point2i center;
    center.x = M.m10 / M.m00;
    center.y = M.m01 / M.m00;

    // Flood fill from the center to segment the area
    floodFill(canvasGray, center, 255);

    // Convert grayscale mask to BGR for visualization
    cvtColor(canvasGray, canvasGray, COLOR_GRAY2BGR);

    // Apply the mask to the original image
    canvasColor = canvasGray & srcI;

    // Copy the segmented result to the second window for display
    canvasColor.copyTo(win2[1]);

    // Show the result window
    //imshow("Traffic sign segmentation", resultWin);
    //waitKey();
    destroyAllWindows();

    // Return the segmented image
    return canvasColor;
}

Mat segmentYellowSign(Mat& srcI) {
    Mat hsv, blurred, yelMask, orangeMask, highGreenMask, yellowMask;
    Mat resize, final, result;

    // Create a result window and partitions for display
    Mat resultWin;
    int const noOfImagePerCol = 1, noOfImagePerRow = 2;
    Mat win[noOfImagePerRow * noOfImagePerCol], legend[noOfImagePerRow * noOfImagePerCol];
    createWindowPartition(srcI, resultWin, win, legend, noOfImagePerCol, noOfImagePerRow);

    // Add labels to the windows
    putText(legend[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    putText(legend[1], "Sign segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

    srcI.copyTo(win[0]);



    //Convert blurred to HSV
    cvtColor(srcI, hsv, COLOR_BGR2HSV);

    //Define yellow HSV range
    Scalar yelLower(12, 70, 80);
    Scalar yelUpper(39, 255, 255);

    // Define the orange HSV range
    Scalar orangeLower(5, 50, 200);
    Scalar orangeUpper(30, 255, 255);

    //Create yellow mask
    inRange(hsv, yelLower, yelUpper, yelMask);
    // Create the orange mask
    inRange(hsv, orangeLower, orangeUpper, orangeMask);

    //Combine yelMask and highGreenMask
    yellowMask = yelMask | orangeMask;

    //Apply morphological
    morphologyEx(yellowMask, yellowMask, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(1, 2)));
    morphologyEx(yellowMask, yellowMask, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));

    //Drwaing contours
    Mat canvasColor(resize.size(), CV_8UC3, Scalar(0, 0, 0));
    Mat canvasGray(resize.size(), CV_8U, Scalar(0));

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(yellowMask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    //int largestContourIndex = -1;
    //double maxArea = 0;

    //// Find largest contours
    //for (int j = 0; j < contours.size(); j++) {
    //	double area = contourArea(contours[j]);
    //	if (area > maxArea) {
    //		maxArea = area;
    //		largestContourIndex = j;
    //	}
    //	drawContours(canvasColor, contours, j, Scalar(0, 0, 255), 2);
    //}

    //if (largestContourIndex != -1) {
    //	vector<Point> hull;
    //	convexHull(contours[largestContourIndex], hull);
    //	final = Mat::zeros(yellowMask.size(), CV_8U);
    //	drawContours(final, contours, largestContourIndex, Scalar(255), 1);
    //	drawContours(final, vector<vector<Point>>{hull}, 0, Scalar(255), FILLED);
    //}
    //else {
    //	final = Mat::zeros(yellowMask.size(), CV_8U);
    //}

    double minArea = 500.0;  // Set a minimum contour area to filter out small noise
    final = Mat::zeros(yellowMask.size(), CV_8U);

    // Draw only the contours that meet the minimum area threshold and apply convexHull
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > minArea) {
            vector<Point> hull;
            // Find convex hull for the contour
            convexHull(contours[i], hull);
            drawContours(final, vector<vector<Point>>{hull}, 0, Scalar(255), FILLED);  // Draw the convex hull
        }
    }

    Mat finalMask;
    cvtColor(final, finalMask, COLOR_GRAY2BGR);

    result = win[0] & finalMask;
    result.copyTo(win[1]);

    //imshow("Traffic sign segmentation", resultWin);
    //waitKey(0);
    destroyAllWindows();

    return result;
}

Mat segmentBlueSign(Mat& srcI) {
    Mat blueMask, canvasColor, canvasGray, hsv;

    // Create a result window and partitions for display
    Mat resultWin;
    int const noOfImagePerCol = 1, noOfImagePerRow = 2;
    Mat win[noOfImagePerRow * noOfImagePerCol], legend[noOfImagePerRow * noOfImagePerCol];
    createWindowPartition(srcI, resultWin, win, legend, noOfImagePerCol, noOfImagePerRow);

    // Add labels to the windows
    putText(legend[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    putText(legend[1], "Sign segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

    srcI.copyTo(win[0]);

    // Blue color range in HSV
    Scalar blueLower(70, 50, 50);
    Scalar blueUpper(140, 255, 255);

    // Create canvases for drawing
    canvasColor.create(srcI.rows, srcI.cols, CV_8UC3);
    canvasGray.create(srcI.rows, srcI.cols, CV_8U);
    canvasColor = Scalar(0, 0, 0);

    // Convert to HSV and threshold for blue
    cvtColor(srcI, hsv, COLOR_BGR2HSV);
    inRange(hsv, blueLower, blueUpper, blueMask);

    // Apply Morphological operations to clean up the mask
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(blueMask, blueMask, MORPH_CLOSE, kernel);
    morphologyEx(blueMask, blueMask, MORPH_OPEN, kernel);

    // Get contours of the blue regions
    vector<vector<Point>> contours;
    findContours(blueMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    double maxArea = 0;
    int largestContourIndex = -1;
    Point2i center;

    // Find the contour with the largest area
    for (int j = 0; j < contours.size(); j++) {
        double area = contourArea(contours[j]);
        if (area > maxArea) {
            maxArea = area;
            largestContourIndex = j;
        }
    }

    // If no valid contours were found, return an empty Mat
    if (largestContourIndex < 0) {
        return Mat();
    }

    // Draw the largest contour and create a mask
    canvasGray = 0;
    drawContours(canvasGray, contours, largestContourIndex, 255, FILLED);

    // Calculate the center of the mask
    Moments M = moments(canvasGray);
    center.x = M.m10 / M.m00;
    center.y = M.m01 / M.m00;

    // Generate mask for the sign
    floodFill(canvasGray, center, 255);
    cvtColor(canvasGray, canvasGray, COLOR_GRAY2BGR);

    // Use the mask to segment the color portion from the image
    canvasColor = canvasGray & srcI;
    canvasColor.copyTo(win[1]);

    // Show the result window
    //imshow("Traffic sign segmentation", resultWin);
    //waitKey(0);
    destroyAllWindows();

    // Return the segmented image
    return canvasColor;
}

Mat colorSegment(Mat& image, String preprocessing) {
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);  // Convert image to HSV

    // Define HSV color ranges for Red, Yellow, and Blue
    Scalar redLower1(0, 70, 50), redUpper1(10, 255, 255);
    Scalar redLower2(160, 70, 50), redUpper2(180, 255, 255);

    Scalar blueLower(90, 50, 50), blueUpper(128, 255, 255);

    Scalar yelLower(12, 70, 80), yelUpper(39, 255, 255);
    Scalar orangeLower(5, 50, 200), orangeUpper(30, 255, 255);

    // Create masks for each color
    Mat redMask1, redMask2, redMask, blueMask, yelMask, orangeMask, yellowMask;
    inRange(hsv, redLower1, redUpper1, redMask1);   // Lower red mask
    inRange(hsv, redLower2, redUpper2, redMask2);   // Upper red mask
    redMask = redMask1 | redMask2;   // Combine both red masks

    inRange(hsv, blueLower, blueUpper, blueMask);  // Blue mask
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(blueMask, blueMask, MORPH_CLOSE, kernel);
    morphologyEx(blueMask, blueMask, MORPH_OPEN, kernel);

    inRange(hsv, yelLower, yelUpper, yelMask);
    inRange(hsv, orangeLower, orangeUpper, orangeMask);
    yellowMask = yelMask | orangeMask;  // Combine both yellow masks

    // Calculate the percentage of the image that each color occupies
    double redPercentage = (double)countNonZero(redMask) / (image.rows * image.cols) * 100;
    double yellowPercentage = (double)countNonZero(yellowMask) / (image.rows * image.cols) * 100;
    double bluePercentage = (double)countNonZero(blueMask) / (image.rows * image.cols) * 100;

    cout << "Red Percentage: " << redPercentage << "%" << endl;
    cout << "Yellow Percentage: " << yellowPercentage << "%" << endl;
    cout << "Blue Percentage: " << bluePercentage << "%" << endl;

    // Set a threshold to determine whether the second segmentation should be done
    double minColorThreshold = 6.0;  // Minimum percentage to consider a color as dominant

    // **Sky Detection Logic**
    bool isSky = false;
    if (bluePercentage >= yellowPercentage && bluePercentage >= redPercentage) {  // Adjust threshold based on the image
        cout << "Blue is dominant, checking if it is likely to be sky..." << endl;

        // Check if the blue region is mostly in the upper half of the image
        Mat upperHalf = blueMask(Rect(0, 0, blueMask.cols, blueMask.rows / 2));
        double upperHalfBlue = (double)countNonZero(upperHalf) / (blueMask.rows * blueMask.cols / 2) * 100;

        // Check if blue has low saturation (typical of sky)
        Mat sat;
        extractChannel(hsv, sat, 1);  // Extract the saturation channel
        Scalar avgSaturation = mean(sat, blueMask);

        // Check if the blue area is uniform (low number of contours)
        vector<vector<Point>> contours;
        findContours(blueMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (upperHalfBlue > 30.0 && avgSaturation[0] < 140 && contours.size() < 5) {
            isSky = true;
        }

        if (isSky) {
            cout << "Sky detected." << endl;
            // deprioritize blue by lowering the blue percentage if there is other color in the image
            if (yellowPercentage >= minColorThreshold || redPercentage >= minColorThreshold) {
                cout << "\n It's not likely blue traffic sign, adjusting the segmentation strategy." << endl;
                bluePercentage = 0;  // Lower blue percentage so it's not chosen for segmentation
            }
        }
    }


    Mat firstSegmented, pre = image.clone();
    string segmentStatus;

    // Perform first segmentation based on the dominant color
    if (redPercentage > yellowPercentage && redPercentage > bluePercentage) {
        cout << "Dominant color: Red" << endl;
        if (preprocessing != "no")
            preprocessImage(pre);
        firstSegmented = segmentRedSign(pre);
        segmentStatus = "red";
    }
    else if (yellowPercentage > redPercentage && yellowPercentage > bluePercentage) {
        cout << "Dominant color: Yellow" << endl;
        //Apply Gaussian Blur

        if (preprocessing != "no") {
            GaussianBlur(image, pre, Size(5, 5), 0);
            preprocessImage(pre);
        }
        firstSegmented = segmentYellowSign(pre);
        segmentStatus = "yellow";
    }
    else if (bluePercentage > redPercentage && bluePercentage > yellowPercentage) {
        cout << "Dominant color: Blue" << endl;
        firstSegmented = segmentBlueSign(image);
        segmentStatus = "blue";
    }
    else {
        // If no dominant color is detected, return the original image
        cout << "No dominant color detected." << endl;
        return image;
    }

    // Recalculate color percentages on the first segmented image
    Mat hsvSegmented;
    cvtColor(firstSegmented, hsvSegmented, COLOR_BGR2HSV);

    // Recalculate masks on the first segmented image
    inRange(hsvSegmented, redLower1, redUpper1, redMask1);
    inRange(hsvSegmented, redLower2, redUpper2, redMask2);
    redMask = redMask1 | redMask2;

    inRange(hsvSegmented, blueLower, blueUpper, blueMask);

    inRange(hsvSegmented, yelLower, yelUpper, yelMask);
    inRange(hsvSegmented, orangeLower, orangeUpper, orangeMask);
    yellowMask = yelMask | orangeMask;

    // Recalculate the percentage of the image that each color occupies in the segmented image
    redPercentage = (double)countNonZero(redMask) / (image.rows * image.cols) * 100;
    yellowPercentage = (double)countNonZero(yellowMask) / (image.rows * image.cols) * 100;
    bluePercentage = (double)countNonZero(blueMask) / (image.rows * image.cols) * 100;

    cout << "After first segmentation - Red Percentage: " << redPercentage << "%" << endl;
    cout << "After first segmentation - Yellow Percentage: " << yellowPercentage << "%" << endl;
    cout << "After first segmentation - Blue Percentage: " << bluePercentage << "%" << endl;

    // Consolidated logic for second segmentation
    if (segmentStatus == "red") {
        if (yellowPercentage > bluePercentage && yellowPercentage > minColorThreshold) {
            cout << "\nFinal Color Segment: Yellow" << endl;
            return segmentYellowSign(firstSegmented);
        }
        else if (bluePercentage > yellowPercentage && bluePercentage > minColorThreshold) {
            cout << "\nFinal Color Segment: blue" << endl;
            return segmentBlueSign(firstSegmented);
        }
    }
    else if (segmentStatus == "yellow") {
        if (redPercentage > yellowPercentage && redPercentage > bluePercentage && redPercentage > minColorThreshold) {
            cout << "\nFinal Color Segment: Red" << endl;
            return segmentRedSign(firstSegmented);
        }
        else if (bluePercentage > yellowPercentage && bluePercentage > redPercentage && bluePercentage > minColorThreshold) {
            cout << "\nFinal Color Segment: blue" << endl;
            return segmentBlueSign(firstSegmented);
        }
    }
    else if (segmentStatus == "blue") {
        if (redPercentage > yellowPercentage && redPercentage > minColorThreshold) {
            cout << "\nFinal Color Segment: Red" << endl;
            return segmentRedSign(firstSegmented);
        }
        else if (yellowPercentage > redPercentage && yellowPercentage > minColorThreshold) {
            cout << "\nFinal Color Segment: Yellow" << endl;
            return segmentYellowSign(firstSegmented);
        }
    }

    // If no other dominant color is detected, return the first segmented image
    if (segmentStatus == "red") {
        cout << "\nFinal Color Segment: Red" << endl;
    }
    else if (segmentStatus == "yellow") {
        cout << "\nFinal Color Segment: Yellow" << endl;
    }
    else if (segmentStatus == "blue") {
        cout << "\nFinal Color Segment: Blue" << endl;
    }

    return firstSegmented;
}






void resizeImage(Mat& image) {
    int const imageWidth = 300, imageHeight = 270;
    resize(image, image, Size(imageWidth, imageHeight));
}

double calculateAverageBrightness(const Mat& img) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Scalar meanValue = mean(gray);

    return meanValue[0];
}

void autoAdjustBrightness(Mat& inputImage) {
    double brightness = calculateAverageBrightness(inputImage);
    const double  targetBrightness = 100;
    double adjustFactor = targetBrightness / brightness;

    inputImage *= adjustFactor;

    normalize(inputImage, inputImage, 0, 255, cv::NORM_MINMAX);

    cout << "Brightness: " << brightness << endl;
}

void applyCLAHE(Mat& inputImage, double clipLimit = 2.0, Size tileGridSize = Size(8, 8)) {

    Mat LABImage;
    // convert image to LAB color space
    cvtColor(inputImage, LABImage, COLOR_BGR2Lab);

    vector<Mat> LABChannels(3);
    split(LABImage, LABChannels); // split the LAB image into individual channles

    // create CLAHE object and initialise the object for applying CLAHE to the L channel
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(clipLimit);
    clahe->setTilesGridSize(tileGridSize);
    clahe->apply(LABChannels[0], LABChannels[0]);

    Mat claheImage;
    merge(LABChannels, claheImage);

    cvtColor(claheImage, inputImage, COLOR_Lab2BGR);

}

void preprocessImage(Mat& inputImage) {
    double brightness = calculateAverageBrightness(inputImage);

    // autoAdjustBrightness(inputImage);

    double clipLimit = brightness > 150 ? 2.0 : 3.0;
    Size tileGridSize = brightness > 150 ? Size(8, 8) : Size(12, 12);
    applyCLAHE(inputImage, clipLimit, tileGridSize);
}

void edgeDetect(Mat& inputImage, Mat& outputImage) {
    // Canny Edge Detection
    Mat grayImage, edgeImage;
    int lowThreshold = 30, ratio = 7, kernelSize = 3; // set parameters for Canny

    // Check if input image is not grayscale
    if (inputImage.channels() != 1) {
        cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);
    }
    else {
        // If already grayscale, just clone the input
        grayImage = inputImage.clone();
    }

    // Apply Canny edge detection
    Canny(grayImage, edgeImage, lowThreshold, lowThreshold * ratio, kernelSize);

    // Convert the binary edge image to BGR for visualization
    cvtColor(edgeImage, outputImage, COLOR_GRAY2BGR);
    imshow("canny", outputImage);
}


Mat shapeSegment(const Mat& originalImage, const Mat& colorSegmented) {
    Mat gray, shapeMask, outputImage;

    // Clone the original image for output
    outputImage = originalImage.clone();

    // Step 1: Convert colorSegmented to grayscale if necessary
    if (colorSegmented.channels() == 3) {
        cvtColor(colorSegmented, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = colorSegmented.clone();
    }

    // Apply adaptive thresholding
    Mat thresholded, inverted;
    adaptiveThreshold(gray, thresholded, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 15, 2);
    bitwise_not(thresholded, inverted);

    // Find contours with hierarchy
    vector<vector<Point>> contours = displayCreateContours(inverted);

    // Create a mask to store the segmented shapes
    shapeMask = Mat::zeros(colorSegmented.size(), CV_8U);  // Initialize with black background

    bool areaSimilarityFound = false; // Flag to track if any area similarity is found

    // Step 4: Loop through the detected contours to approximate shapes and check area similarity
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);

        // Ignore small areas
        if (area > 1000) {
            vector<Point> approx;
            double perimeter = arcLength(contours[i], true);
            approxPolyDP(contours[i], approx, 0.02 * perimeter, true); // Simplify the contour

            string shapeType = "Unknown";  // Default to unknown
            bool areaSimilar = false;      // Flag for area similarity match
            double shapeArea = 0.0;

            // Check for Circle first using circularity (don't use vertices)
            double circularity = 4 * CV_PI * area / (perimeter * perimeter);
            if (circularity > 0.85) {
                shapeType = "Circle";
                areaSimilar = true;
            }

            // Check for Triangle (area similarity)
            if (!areaSimilar) {
                double sideLengthTriangle = perimeter / 3.0;
                shapeArea = (sqrt(3) / 4) * sideLengthTriangle * sideLengthTriangle;
                if (abs(shapeArea - area) < 0.2 * shapeArea) {  // 20% tolerance
                    shapeType = "Triangle";
                    areaSimilar = true;
                }
            }

            // Check for Rectangle or Square (area similarity)
            if (!areaSimilar) {
                Rect boundingBox = boundingRect(approx);
                double aspectRatio = (double)boundingBox.width / boundingBox.height;

                // Stricter check for square aspect ratio
                if (aspectRatio >= 0.95 && aspectRatio <= 1.05) {
                    double sideLengthSquare = perimeter / 4.0;
                    shapeArea = sideLengthSquare * sideLengthSquare;

                    if (abs(shapeArea - area) < 0.05 * shapeArea) {  // 5% tolerance
                        shapeType = "Square";
                        areaSimilar = true;
                    }
                }
            }

            // Check for Octagon (area similarity)
            if (!areaSimilar) {
                double sideLengthOctagon = perimeter / 8.0;
                shapeArea = 2 * (1 + sqrt(2)) * sideLengthOctagon * sideLengthOctagon;
                if (abs(shapeArea - area) < 0.2 * shapeArea) {  // 20% tolerance
                    shapeType = "Octagon";
                    areaSimilar = true;
                }

                // Reclassify as a circle if the circularity is high
                if (circularity > 0.9) {
                    shapeType = "Circle";  // Reclassify as a circle if the circularity is high
                    areaSimilar = true;
                }
            }

            // Step 5: Segmentation based on area similarity
            if (areaSimilar) {
                areaSimilarityFound = true;  // Area similarity is found for this contour
                drawContours(shapeMask, contours, (int)i, Scalar(255), FILLED);  // Fill the shape in white
                putText(outputImage, shapeType + " (Area)", boundingRect(approx).tl(), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
                cout << "Detected shape based on area similarity: " << shapeType << endl;
            }
        }
    }

    // If no area similarity is found for any contour, use the approximate contours to segment
    if (!areaSimilarityFound) {
        cout << "No area similarity found, using approximate contours for segmentation." << endl;
        for (size_t i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area > 1000) {
                // Use convex hull to smooth or fix broken contours
                vector<Point> hull;
                convexHull(contours[i], hull);

                // Draw the convex hull as the segmented shape
                drawContours(shapeMask, vector<vector<Point>>{hull}, -1, Scalar(255), FILLED);  // Use the convex hull to segment
            }
        }
    }

    // Use the mask to segment the original image (only segment shapes with similarity or approximate contours)
    Mat segmentedOutput;
    originalImage.copyTo(segmentedOutput, shapeMask);  // Copy the parts of the original image where the mask is white

    return segmentedOutput;  // Return the segmented image
}







// Function to display and create contours
vector<vector<Point>> displayCreateContours(Mat& image, const string& windowName) {
    Mat gray;

    // Convert to grayscale if necessary
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = image.clone();
    }

    // Find contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(gray, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

    // Display the contours (for visualization)
    Mat contourImage;
    if (gray.channels() == 1) {
        cvtColor(gray, contourImage, COLOR_GRAY2BGR); // Convert back to BGR for visualization
    }
    else {
        contourImage = gray.clone();
    }
    drawContours(contourImage, contours, -1, Scalar(0, 255, 0), 1);
    //imshow(windowName, contourImage);
    //waitKey(0);  // Wait for a key press before closing

    // Return the found contours
    return contours;
}


Mat loadShapeTemplate(const string& templatePath) {
    Mat templateImage = imread(templatePath, IMREAD_GRAYSCALE);
    if (templateImage.empty()) {
        cerr << "Error: Template image not found at " << templatePath << endl;
        return Mat();  // Return empty on failure
    }
    int const imageWidth = 220, imageHeight = 160;
    resize(templateImage, templateImage, Size(imageWidth, imageHeight));

    Mat canny;
    edgeDetect(templateImage, canny);
    cvtColor(canny, canny, COLOR_BGR2GRAY);
    // Find contours of the template
    vector<vector<Point>> templateContours;
    findContours(canny, templateContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    double maxLength = 0.0;
    vector<Point> longestContour;
    for (const auto& contour : templateContours) {
        double length = arcLength(contour, true);  // Compute the perimeter of the contour
        if (length > maxLength) {
            maxLength = length;
            longestContour = contour;  // Update to the longest contour
        }
    }
    // Create a blank mask (same size as the resized template)
    Mat shapeMask = Mat::zeros(templateImage.size(), CV_8UC1);  // Black background
    drawContours(shapeMask, vector<vector<Point>>{longestContour}, -1, Scalar(255), FILLED);  // Fill the contour with white
    imshow("Template Mask", shapeMask);
    return shapeMask;  // Return the filled shape mask
}

// Function to generate the reference shape templates
void generateShapeTemplate(vector<Mat>& shapeMasks, vector<string>& shapeNames) {
    string templateFolder = "Inputs/Traffic signs/Shape Template/";
    vector<string> templateFiles = { "circle.png", "triangle.png", "square.png", "octagon.png" };

    // Shape names corresponding to the files, ensuring they match the reference shapes
    shapeNames = { "Circle", "Triangle", "Square", "Octagon" };

    for (const string& fileName : templateFiles) {
        string fullPath = templateFolder + fileName;

        // Load the template mask
        Mat shapeMask = loadShapeTemplate(fullPath);
        if (shapeMask.empty()) {
            cerr << "Error: Could not load shape mask for " << fileName << endl;
            continue;  // Skip if mask loading fails
        }

        // Store the filled shape mask
        shapeMasks.push_back(shapeMask);

        waitKey(0);  // Wait for a key press to move to the next template
    }

    destroyAllWindows();

}

//
//// Convert reference shape masks to contours
//vector<vector<Point>> referenceContours;
//for (const auto& mask : referenceShapes) {
//    vector<vector<Point>> maskContours;
//    findContours(mask, maskContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//
//    if (!maskContours.empty()) {
//        // We assume that the largest contour is the desired shape
//        double maxLength = 0.0;
//        vector<Point> longestContour;
//        for (const auto& contour : maskContours) {
//            double length = arcLength(contour, true);  // Compute the perimeter of the contour
//            if (length > maxLength) {
//                maxLength = length;
//                longestContour = contour;  // Update to the longest contour
//            }
//        }
//        referenceContours.push_back(longestContour);  // Store the longest contour for each shape mask
//    }
//}

//// Loop through the detected contours to match with reference shapes
//for (size_t i = 0; i < contours.size(); i++) {
//    // Create a mask for each detected shape
//    Mat detectedShapeMask = Mat::zeros(colorSegmented.size(), CV_8U);
//    drawContours(detectedShapeMask, contours, (int)i, Scalar(255), FILLED);  // Fill the detected shape
//    imshow("Detected Shape Mask", detectedShapeMask);
//    double bestMatchScore = -1;  // Initialize with a very low value
//    int bestMatchIndex = -1;     // Track the best matching shape index
//
//    // Loop through the detected contours
//    for (size_t i = 0; i < contours.size(); i++) {
//        double bestMatchScore = 1e10;  // Initialize with a large value
//        int bestMatchIndex = -1;       // Track the best matching shape index
//
//        // Compare the detected contour with all reference shape contours using matchShapes()
//        for (size_t j = 0; j < referenceContours.size(); j++) {
//            // Use matchShapes to compare contours (Hu Moments-based shape comparison)
//            double matchScore = matchShapes(contours[i], referenceContours[j], CONTOURS_MATCH_I1, 0);
//
//            // Check if this match is the best one
//            if (matchScore < bestMatchScore) {
//                bestMatchScore = matchScore;
//                bestMatchIndex = j;
//            }
//        }
//
//        // Based on the best match, segment the shape
//        if (bestMatchIndex != -1 && bestMatchScore < 0.3) {  // Threshold for similarity
//            cout << "Shape Matched! Best match index: " << bestMatchIndex << " with score: " << bestMatchScore << endl;
//            cout << "Detected Shape: " << shapeNames[bestMatchIndex] << endl;
//
//            // Step 7: Use the matching contour to segment out the shape on the mask
//            drawContours(shapeMask, contours, (int)i, Scalar(255), FILLED);  // Segment the best-matching shape
//
//            // Display the matched reference shape
//            Mat referenceShapeDisplay = Mat::zeros(300, 300, CV_8UC3);  // Create a blank image for visualization
//            drawContours(referenceShapeDisplay, vector<vector<Point>>{referenceContours[bestMatchIndex]}, -1, Scalar(0, 255, 0), 2);
//            imshow("Best Matched Reference Shape", referenceShapeDisplay);
//            waitKey(0);  // Wait for a key press to proceed
//        }
//    }
//}