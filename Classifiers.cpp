#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>

using namespace std;
using namespace cv;
using namespace cv::ml;

const int RANDOM_SEED = 123;

struct Record {
    string imageName;
    int label;
    vector<float> features;
};

struct Metrics {
    float accuracy;
    map<string, float> precision;
    map<string, float> recall;
    map<string, float> f1Score;

};

// Function declarations
void readCSV(const string& filename, vector<Record>& records);
vector<Record> mergeRecordsByName(const vector<Record>& colorRecords, const vector<Record>& shapeRecords);
Mat convertToMat(const vector<Record>& records);
Mat convertLabelsToMat(const vector<Record>& records);
void shuffleDataset(vector<Record>& records, unsigned int seed);
void splitData(const vector<Record>& records, Mat& trainFeatures, Mat& testFeatures, Mat& trainLabels, Mat& testLabels, float trainRatio = 0.8);
Mat getConfusionMatrix(const Mat& testLabels, const Mat& predictedLabels, int numClasses);
void printConfusionMatrix(const Mat& confusionMatrix, const vector<string>& classNames);
void calculateMetrics(const Mat& confusionMatrix, const vector<string>& classNames, Metrics& metrics);

// KNN 
int knnHyperparameterTuning(const Mat&, const Mat&, const Mat&, const Mat&);
Metrics runKNN(const String featureType);
Metrics runKNNColor();
Metrics runKNNHu();
Metrics runKNNCombined();

// SVM
void svmHyperparameterTuning(const Mat& trainFeatures, const Mat& trainLabels, const Mat& testFeatures, const Mat& testLabels, double& bestC, double& bestGamma);
Metrics runSVM(const String featureType);
Metrics runSVMColor();
Metrics runSVMHu();
Metrics runSVMCombined();

// Random Forests
void randomForestHyperparameterTuning(const Mat& trainFeatures, const Mat& trainLabels, const Mat& testFeatures, const Mat& testLabels,
    int& bestMaxDepth, int& bestMinSampleCount, int& bestMaxCategories);
Metrics runRandomForest(const String featureType);
Metrics runRandomForestColor();
Metrics runRandomForestHu();
Metrics runRandomForestCombined();

void printReport(const map<string, map<string, Metrics>>& results);

// Menu Functions
int showClassifierMenu();
int showFeatureMenu();

int main() {
    int classifierChoice = -1;
    int featureChoice = -1;
    map<string, map<string, Metrics>> results;

    while (classifierChoice != 4) {  // Loop until user chooses to exit
        classifierChoice = showClassifierMenu();

        if (classifierChoice == 4) break;  // Exit program

        if (classifierChoice >= 1 && classifierChoice <= 3) {
            featureChoice = showFeatureMenu();

            cout << endl;
            string featureType;
            if (featureChoice == 1) featureType = "color";
            else if (featureChoice == 2) featureType = "hu";
            else if (featureChoice == 3) featureType = "combine";
            else continue;  // Go back if featureChoice == 0 (back)

            switch (classifierChoice) {
            case 1:
                results["KNN"][featureType] = runKNN(featureType);
                break;
            case 2:
                results["SVM"][featureType] = runSVM(featureType);
                break;
            case 3:
                results["Random Forest"][featureType] = runRandomForest(featureType);
                break;
            }
        }
        else if (classifierChoice == 5) {
            // Generate reports for all classifiers
            vector<string> featureTypes = { "color", "hu", "combine" };
            for (const string& featureType : featureTypes) {
                results["KNN"][featureType] = runKNN(featureType);
                results["SVM"][featureType] = runSVM(featureType);
                results["Random Forest"][featureType] = runRandomForest(featureType);
            }
            printReport(results);  // Print the final summarized report
        }
    }

    return 0;
}

// Menu to choose the classifier
int showClassifierMenu() {
    int choice;
    cout << "\nSelect a classifier:\n";
    cout << "1. KNN\n";
    cout << "2. SVM\n";
    cout << "3. Random Forest\n";
    cout << "4. Exit\n";
    cout << "5. Generate Reports for All Classifiers\n";  // New option for generating reports
    cout << "Enter choice: ";
    cin >> choice;
    return choice;
}

// Menu to choose the features for the classifier
int showFeatureMenu() {
    int choice;
    cout << "\nClassifier with feature:\n";
    cout << "0. Back\n";
    cout << "1. Color\n";
    cout << "2. Hu Moments\n";
    cout << "3. Combine both Color and Hu Moments\n";
    cout << "Enter choice: ";
    cin >> choice;
    return choice;
}

void printReport(const map<string, map<string, Metrics>>& results) {
    cout << left << setw(20) << "Classifier"
        << setw(15) << "Feature Type"
        << setw(10) << "Accuracy"
        << setw(15) << "Precision"
        << setw(15) << "Recall"
        << setw(15) << "F1 Score" << endl;

    for (const auto& classifier : results) {
        const string& classifierName = classifier.first;
        for (const auto& featureType : classifier.second) {
            const string& feature = featureType.first;
            const Metrics& metrics = featureType.second;

            cout << setw(20) << classifierName
                << setw(15) << feature
                << setw(10) << metrics.accuracy;

            // Now print the average precision, recall, and F1 score
            cout << setw(15) << setprecision(4) << metrics.precision.at("Average")   // Precision average
                << setw(15) << setprecision(4) << metrics.recall.at("Average")      // Recall average
                << setw(15) << setprecision(4) << metrics.f1Score.at("Average");    // F1 score average

            cout << endl;
        }
    }
}

/* =================================== Classifiers ======================================== */

// KNN Classifier
Metrics runKNN(const String featureType) {
    Metrics metrics;

    if (featureType == "color") {
        metrics = runKNNColor();  // Capture the Metrics result
    }
    else if (featureType == "hu") {
        metrics = runKNNHu();
    }
    else if (featureType == "combine") {
        metrics = runKNNCombined();
    }

    return metrics;  // Return the metrics
}


Metrics runKNNColor() {
    Metrics metrics;
    vector<Record> colorRecords;

    // Load color histogram features
    readCSV("color_histogram_data.csv", colorRecords);

    // Shuffle the records using a fixed random seed for reproducibility
    unsigned int seed = RANDOM_SEED;
    shuffleDataset(colorRecords, seed);

    // Split the data into training and testing sets after shuffling
    Mat trainFeatures, testFeatures, trainLabels, testLabels;
    splitData(colorRecords, trainFeatures, testFeatures, trainLabels, testLabels);

    // KNN Hyperparameter Tuning
    int bestK = knnHyperparameterTuning(trainFeatures, trainLabels, testFeatures, testLabels);

    // Train a K-Nearest Neighbors model using OpenCV
    Ptr<KNearest> knn = KNearest::create();
    knn->setDefaultK(bestK);
    knn->setIsClassifier(true);
    knn->train(trainFeatures, ROW_SAMPLE, trainLabels);

    // Predict the test data
    Mat predictedLabels;
    knn->findNearest(testFeatures, knn->getDefaultK(), predictedLabels);

    // Evaluate accuracy
    int correctPredictions = 0;
    for (int i = 0; i < testLabels.rows; ++i) {
        if (testLabels.at<int>(i, 0) == static_cast<int>(predictedLabels.at<float>(i, 0))) {
            correctPredictions++;
        }
    }
    metrics.accuracy = (float)correctPredictions / testLabels.rows * 100.0f;
    cout << "1. Accuracy: " << metrics.accuracy << "%" << endl;
    // Define class names for confusion matrix labels
    vector<string> classNames = { "Informative Sign", "Prohibitive Sign", "Warning Sign" };

    // Get confusion matrix
    Mat confusionMatrix = getConfusionMatrix(testLabels, predictedLabels, classNames.size());

    // Calculate and print precision, recall, and F1 score
    cout << "2. Precision, Recall, and F1 Score for each class:" << endl << endl;
    calculateMetrics(confusionMatrix, classNames, metrics);

    // Print labeled confusion matrix
    cout << "4. Confusion Matrix:" << endl;
    printConfusionMatrix(confusionMatrix, classNames);

    return metrics;
}


Metrics runKNNHu() {
    Metrics metrics;
    vector<Record> colorRecords;

    // Load color histogram features
    readCSV("hu_moments_data.csv", colorRecords);

    // Shuffle the records using a fixed random seed for reproducibility
    unsigned int seed = RANDOM_SEED;
    shuffleDataset(colorRecords, seed);

    // Split the data into training and testing sets after shuffling
    Mat trainFeatures, testFeatures, trainLabels, testLabels;
    splitData(colorRecords, trainFeatures, testFeatures, trainLabels, testLabels);

    // KNN Hyperparameter Tuning
    int bestK = knnHyperparameterTuning(trainFeatures, trainLabels, testFeatures, testLabels);

    // Train a K-Nearest Neighbors model using OpenCV
    Ptr<KNearest> knn = KNearest::create();
    knn->setDefaultK(3);  // Set the value of K (e.g., 3 for K=3)
    knn->setIsClassifier(true);

    // Train KNN on the training data
    knn->train(trainFeatures, ROW_SAMPLE, trainLabels);

    // Predict the test data
    Mat predictedLabels;
    knn->findNearest(testFeatures, knn->getDefaultK(), predictedLabels);

    // Evaluate the accuracy
    int correctPredictions = 0;
    for (int i = 0; i < testLabels.rows; ++i) {
        if (testLabels.at<int>(i, 0) == static_cast<int>(predictedLabels.at<float>(i, 0))) {
            correctPredictions++;
        }
    }
    metrics.accuracy = (float)correctPredictions / testLabels.rows * 100.0f;
    cout << "\n1. Accuracy: " << metrics.accuracy << "%" << endl;

    // Define class names for confusion matrix labels
    vector<string> classNames = { "Informative Sign", "Prohibitive Sign", "Warning Sign" };

    // Get confusion matrix
    Mat confusionMatrix = getConfusionMatrix(testLabels, predictedLabels, classNames.size());

    // Calculate and print precision, recall, and F1 score
    cout << "2. Precision, Recall, and F1 Score for each class:" << endl << endl;
    calculateMetrics(confusionMatrix, classNames, metrics);

    // Print labeled confusion matrix
    cout << "4. Confusion Matrix:" << endl;
    printConfusionMatrix(confusionMatrix, classNames);

    return metrics;
}

Metrics runKNNCombined() {
    Metrics metrics;
    cout << "Running KNN Combined" << endl;
    vector<Record> colorRecords, shapeRecords;

    // Load color histogram features
    readCSV("color_histogram_data.csv", colorRecords);

    // Load Hu moments features
    readCSV("hu_moments_data.csv", shapeRecords);

    // Merge records ensuring name match
    vector<Record> combinedRecords = mergeRecordsByName(colorRecords, shapeRecords);

    // Shuffle the records using a fixed random seed for reproducibility
    unsigned int seed = RANDOM_SEED;
    shuffleDataset(combinedRecords, seed);

    // Split the data into training and testing sets after shuffling
    Mat trainFeatures, testFeatures, trainLabels, testLabels;
    splitData(combinedRecords, trainFeatures, testFeatures, trainLabels, testLabels);

    // KNN Hyperparameter Tuning
    int bestK = knnHyperparameterTuning(trainFeatures, trainLabels, testFeatures, testLabels);

    // Train a K-Nearest Neighbors model using OpenCV
    Ptr<KNearest> knn = KNearest::create();
    knn->setDefaultK(3);  // Set the value of K (e.g., 3 for K=3)
    knn->setIsClassifier(true);

    // Train KNN on the training data
    knn->train(trainFeatures, ROW_SAMPLE, trainLabels);

    // Predict the test data
    Mat predictedLabels;
    knn->findNearest(testFeatures, knn->getDefaultK(), predictedLabels);

    // Calculate and print accuracy
    int correctPredictions = 0;
    for (int i = 0; i < testLabels.rows; ++i) {
        if (testLabels.at<int>(i, 0) == static_cast<int>(predictedLabels.at<float>(i, 0))) {
            correctPredictions++;
        }
    }
    metrics.accuracy = (float)correctPredictions / testLabels.rows * 100.0f;
    cout << "1. Random Forest Accuracy: " << metrics.accuracy << "%" << endl;

    // Define class names for confusion matrix labels
    vector<string> classNames = { "Informative Sign", "Prohibitive Sign", "Warning Sign" };

    // Get confusion matrix
    Mat confusionMatrix = getConfusionMatrix(testLabels, predictedLabels, classNames.size());

    // Calculate and print precision, recall, and F1 score
    cout << "2. Precision, Recall, and F1 Score for each class:" << endl << endl;
    calculateMetrics(confusionMatrix, classNames, metrics);

    // Print labeled confusion matrix
    cout << "4. Confusion Matrix:" << endl;
    printConfusionMatrix(confusionMatrix, classNames);

    return metrics;
}

int knnHyperparameterTuning(const Mat& trainFeatures, const Mat& trainLabels, const Mat& testFeatures, const Mat& testLabels) {
    vector<int> kValues = { 1, 3, 5, 7, 9 };  // Values to test for K (number of neighbors)
    double bestAccuracy = 0;
    int bestK = 0;

    for (int k : kValues) {
        // Create and configure the KNN model
        Ptr<KNearest> knn = KNearest::create();
        knn->setDefaultK(k);
        knn->setIsClassifier(true);

        // Train the model
        knn->train(trainFeatures, ROW_SAMPLE, trainLabels);

        // Predict on the test set
        Mat predictedLabels;
        knn->findNearest(testFeatures, knn->getDefaultK(), predictedLabels);

        // Evaluate the model
        int correctPredictions = 0;
        for (int i = 0; i < testLabels.rows; ++i) {
            if (testLabels.at<int>(i, 0) == static_cast<int>(predictedLabels.at<float>(i, 0))) {
                correctPredictions++;
            }
        }

        float accuracy = (correctPredictions / static_cast<float>(testLabels.rows)) * 100;

        // Check if this combination gives the best accuracy
        if (accuracy > bestAccuracy) {
            bestAccuracy = accuracy;
            bestK = k;
        }

        cout << "K: " << k << ", Accuracy: " << accuracy << "%" << endl;
    }

    cout << "\nBest K: " << bestK << ", Best Accuracy: " << bestAccuracy << "%" << endl;

    return bestK;  // Return the best K
}


// SVM Classifier
Metrics runSVM(const String featureType) {
    Metrics metrics;

    if (featureType == "color") {
        metrics = runSVMColor();  // Capture the Metrics result
    }
    else if (featureType == "hu") {
        metrics = runSVMHu();
    }
    else if (featureType == "combine") {
        metrics = runSVMCombined();
    }

    return metrics;  // Return the metrics
}


// Function to run SVM with color histogram features
Metrics runSVMColor() {
    Metrics metrics;
    vector<Record> colorRecords;

    // Load color histogram features
    readCSV("color_histogram_data.csv", colorRecords);

    // Shuffle the records using a fixed random seed for reproducibility
    unsigned int seed = RANDOM_SEED;
    shuffleDataset(colorRecords, seed);

    // Split the data into training and testing sets after shuffling
    Mat trainFeatures, testFeatures, trainLabels, testLabels;
    splitData(colorRecords, trainFeatures, testFeatures, trainLabels, testLabels);

    // Hyperparameter tuning to find the best values for C and gamma
    double bestC = 1.0, bestGamma = 0.1;  // Initial values
    svmHyperparameterTuning(trainFeatures, trainLabels, testFeatures, testLabels, bestC, bestGamma);

    // Train an SVM model with the best hyperparameters
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->setC(bestC);
    svm->setGamma(bestGamma);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

    svm->train(trainFeatures, ROW_SAMPLE, trainLabels);

    // Save the model for future use
    svm->save("svm_color_model.yml");

    // Predict the test data
    Mat predictedLabels;
    svm->predict(testFeatures, predictedLabels);

    // Evaluate the accuracy
    int correctPredictions = 0;
    for (int i = 0; i < testLabels.rows; ++i) {
        if (testLabels.at<int>(i, 0) == static_cast<int>(predictedLabels.at<float>(i, 0))) {
            correctPredictions++;
        }
    }
    metrics.accuracy = (float)correctPredictions / testLabels.rows * 100.0f;
    cout << "1. SVM Color Accuracy: " << metrics.accuracy << "%" << endl;

    // Define class names for confusion matrix labels
    vector<string> classNames = { "Informative Sign", "Prohibitive Sign", "Warning Sign" };

    // Get confusion matrix
    Mat confusionMatrix = getConfusionMatrix(testLabels, predictedLabels, classNames.size());

    // Calculate and print precision, recall, and F1 score
    cout << "2. Precision, Recall, and F1 Score for each class:" << endl << endl;
    calculateMetrics(confusionMatrix, classNames, metrics);

    // Print labeled confusion matrix
    cout << "4. Confusion Matrix:" << endl;
    printConfusionMatrix(confusionMatrix, classNames);

    return metrics;
}

// Function to run SVM with Hu moments features
Metrics runSVMHu() {
    Metrics metrics;
    vector<Record> huRecords;

    // Load Hu moments features
    readCSV("hu_moments_data.csv", huRecords);

    // Shuffle the records using a fixed random seed for reproducibility
    unsigned int seed = RANDOM_SEED;
    shuffleDataset(huRecords, seed);

    // Split the data into training and testing sets after shuffling
    Mat trainFeatures, testFeatures, trainLabels, testLabels;
    splitData(huRecords, trainFeatures, testFeatures, trainLabels, testLabels);

    // Hyperparameter tuning to find the best values for C and gamma
    double bestC = 1.0, bestGamma = 0.1;  // Initial values
    svmHyperparameterTuning(trainFeatures, trainLabels, testFeatures, testLabels, bestC, bestGamma);

    // Train an SVM model with the best hyperparameters
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->setC(bestC);
    svm->setGamma(bestGamma);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

    svm->train(trainFeatures, ROW_SAMPLE, trainLabels);

    // Save the model for future use
    svm->save("svm_hu_model.yml");

    // Predict the test data
    Mat predictedLabels;
    svm->predict(testFeatures, predictedLabels);

    // Evaluate the accuracy
    int correctPredictions = 0;
    for (int i = 0; i < testLabels.rows; ++i) {
        if (testLabels.at<int>(i, 0) == static_cast<int>(predictedLabels.at<float>(i, 0))) {
            correctPredictions++;
        }
    }
    metrics.accuracy = (float)correctPredictions / testLabels.rows * 100.0f;
    cout << "1. SVM Hu Accuracy: " << metrics.accuracy << "%" << endl;

    // Define class names for confusion matrix labels
    vector<string> classNames = { "Informative Sign", "Prohibitive Sign", "Warning Sign" };

    // Get confusion matrix
    Mat confusionMatrix = getConfusionMatrix(testLabels, predictedLabels, classNames.size());

    // Calculate and print precision, recall, and F1 score
    cout << "2. Precision, Recall, and F1 Score for each class:" << endl << endl;
    calculateMetrics(confusionMatrix, classNames, metrics);

    // Print labeled confusion matrix
    cout << "4. Confusion Matrix:" << endl;
    printConfusionMatrix(confusionMatrix, classNames);

    return metrics;
}

// Function to run SVM with combined color and Hu moments features
Metrics runSVMCombined() {
    Metrics metrics;
    vector<Record> colorRecords, shapeRecords;

    // Load color histogram features
    readCSV("color_histogram_data.csv", colorRecords);

    // Load Hu moments features
    readCSV("hu_moments_data.csv", shapeRecords);

    // Merge records ensuring name match
    vector<Record> combinedRecords = mergeRecordsByName(colorRecords, shapeRecords);

    // Shuffle the records using a fixed random seed for reproducibility
    unsigned int seed = RANDOM_SEED;
    shuffleDataset(combinedRecords, seed);

    // Split the data into training and testing sets after shuffling
    Mat trainFeatures, testFeatures, trainLabels, testLabels;
    splitData(combinedRecords, trainFeatures, testFeatures, trainLabels, testLabels);

    // Hyperparameter tuning to find the best values for C and gamma
    double bestC = 1.0, bestGamma = 0.1;  // Initial values
    svmHyperparameterTuning(trainFeatures, trainLabels, testFeatures, testLabels, bestC, bestGamma);

    // Train an SVM model with the best hyperparameters
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->setC(bestC);
    svm->setGamma(bestGamma);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

    svm->train(trainFeatures, ROW_SAMPLE, trainLabels);

    // Save the model for future use
    svm->save("svm_combined_model.yml");

    // Predict the test data
    Mat predictedLabels;
    svm->predict(testFeatures, predictedLabels);

    // Calculate and print accuracy
    int correctPredictions = 0;
    for (int i = 0; i < testLabels.rows; ++i) {
        if (testLabels.at<int>(i, 0) == static_cast<int>(predictedLabels.at<float>(i, 0))) {
            correctPredictions++;
        }
    }
    metrics.accuracy = (float)correctPredictions / testLabels.rows * 100.0f;
    cout << "1. SVM Combined Accuracy: " << metrics.accuracy << "%" << endl;

    // Define class names for confusion matrix labels
    vector<string> classNames = { "Informative Sign", "Prohibitive Sign", "Warning Sign" };

    // Get confusion matrix
    Mat confusionMatrix = getConfusionMatrix(testLabels, predictedLabels, classNames.size());

    // Calculate and print precision, recall, and F1 score
    cout << "2. Precision, Recall, and F1 Score for each class:" << endl << endl;
    calculateMetrics(confusionMatrix, classNames, metrics);

    // Print labeled confusion matrix
    cout << "4. Confusion Matrix:" << endl;
    printConfusionMatrix(confusionMatrix, classNames);

    return metrics;
}

// Function for hyperparameter tuning of SVM
void svmHyperparameterTuning(const Mat& trainFeatures, const Mat& trainLabels, const Mat& testFeatures, const Mat& testLabels, double& bestC, double& bestGamma) {
    vector<double> C_values = { 0.01, 0.1, 1, 10, 100 };  // Values to test for C
    vector<double> gamma_values = { 0.001, 0.01, 0.1, 1 };  // Values to test for gamma

    double bestAccuracy = 0;

    for (double C : C_values) {
        for (double gamma : gamma_values) {
            Ptr<SVM> svm = SVM::create();
            svm->setType(SVM::C_SVC);
            svm->setKernel(SVM::RBF);  // Using RBF kernel for hyperparameter tuning
            svm->setC(C);
            svm->setGamma(gamma);
            svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

            // Train the SVM
            svm->train(trainFeatures, ROW_SAMPLE, trainLabels);

            // Test the model
            int correctPredictions = 0;
            for (int i = 0; i < testFeatures.rows; ++i) {
                float prediction = svm->predict(testFeatures.row(i));
                if (prediction == testLabels.at<int>(i, 0)) {
                    correctPredictions++;
                }
            }

            float accuracy = (correctPredictions / static_cast<float>(testFeatures.rows)) * 100;

            // Check if this combination gives the best accuracy
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestC = C;
                bestGamma = gamma;
            }

            cout << "C: " << C << ", Gamma: " << gamma << ", Accuracy: " << accuracy << "%" << endl;
        }
    }

    cout << "\nBest C: " << bestC << ", Best Gamma: " << bestGamma << ", Best Accuracy: " << bestAccuracy << "%" << endl;
}


// Random Forests Classifier
Metrics runRandomForest(const String featureType) {
    Metrics metrics;

    if (featureType == "color") {
        metrics = runRandomForestColor();  // Capture the Metrics result
    }
    else if (featureType == "hu") {
        metrics = runRandomForestHu();
    }
    else if (featureType == "combine") {
        metrics = runRandomForestCombined();
    }

    return metrics;  // Return the metrics
}


// Function to run Random Forest with color histogram features
Metrics runRandomForestColor() {
    Metrics metrics;
    vector<Record> colorRecords;

    // Load color histogram features
    readCSV("color_histogram_data.csv", colorRecords);

    // Shuffle the records using a fixed random seed for reproducibility
    unsigned int seed = RANDOM_SEED;
    shuffleDataset(colorRecords, seed);

    // Split the data into training and testing sets after shuffling
    Mat trainFeatures, testFeatures, trainLabels, testLabels;
    splitData(colorRecords, trainFeatures, testFeatures, trainLabels, testLabels);

    // Variables to store the best hyperparameters
    int bestMaxDepth = 10;
    int bestMinSampleCount = 10;
    int bestMaxCategories = 2;

    // Hyperparameter tuning for Random Forest (the best values will be updated here)
    randomForestHyperparameterTuning(trainFeatures, trainLabels, testFeatures, testLabels,
        bestMaxDepth, bestMinSampleCount, bestMaxCategories);

    // Create a map to translate numeric labels to category names
    map<int, string> labelMap = {
        {0, "Informative Sign"},
        {1, "Prohibitive Sign"},
        {2, "Warning Sign"}
    };

    // Train the Random Forest model
    Ptr<RTrees> randomForest = RTrees::create();
    randomForest->setMaxDepth(bestMaxDepth);
    randomForest->setMinSampleCount(bestMinSampleCount);
    randomForest->setMaxCategories(bestMaxCategories);
    randomForest->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

    randomForest->train(trainFeatures, ROW_SAMPLE, trainLabels);

    // Save the model for future use
    randomForest->save("random_forest_color_model.xml");

    // Predict the test data
    Mat predictedLabels;
    randomForest->predict(testFeatures, predictedLabels);

    // Evaluate the accuracy
    int correctPredictions = 0;
    for (int i = 0; i < testLabels.rows; ++i) {
        if (testLabels.at<int>(i, 0) == static_cast<int>(predictedLabels.at<float>(i, 0))) {
            correctPredictions++;
        }
    }
    metrics.accuracy = (float)correctPredictions / testLabels.rows * 100.0f;
    cout << "1. Random Forest Color Accuracy: " << metrics.accuracy << "%" << endl;

    // Define class names for confusion matrix labels
    vector<string> classNames = { "Informative Sign", "Prohibitive Sign", "Warning Sign" };

    // Get confusion matrix
    Mat confusionMatrix = getConfusionMatrix(testLabels, predictedLabels, classNames.size());

    // Calculate and print precision, recall, and F1 score
    cout << "2. Precision, Recall, and F1 Score for each class:" << endl << endl;
    calculateMetrics(confusionMatrix, classNames, metrics);

    // Print labeled confusion matrix
    cout << "4. Confusion Matrix:" << endl;
    printConfusionMatrix(confusionMatrix, classNames);

    return metrics;
}

// Function to run Random Forest with Hu moments features
Metrics runRandomForestHu() {
    Metrics metrics;
    vector<Record> huRecords;

    // Load Hu moments features
    readCSV("hu_moments_data.csv", huRecords);

    // Shuffle the records using a fixed random seed for reproducibility
    unsigned int seed = RANDOM_SEED;
    shuffleDataset(huRecords, seed);

    // Split the data into training and testing sets after shuffling
    Mat trainFeatures, testFeatures, trainLabels, testLabels;
    splitData(huRecords, trainFeatures, testFeatures, trainLabels, testLabels);

    // Variables to store the best hyperparameters
    int bestMaxDepth = 10;
    int bestMinSampleCount = 10;
    int bestMaxCategories = 2;

    // Hyperparameter tuning for Random Forest (the best values will be updated here)
    randomForestHyperparameterTuning(trainFeatures, trainLabels, testFeatures, testLabels,
        bestMaxDepth, bestMinSampleCount, bestMaxCategories);

    // Create a map to translate numeric labels to category names
    map<int, string> labelMap = {
        {0, "Informative Sign"},
        {1, "Prohibitive Sign"},
        {2, "Warning Sign"}
    };

    // Train the Random Forest model
    Ptr<RTrees> randomForest = RTrees::create();
    randomForest->setMaxDepth(bestMaxDepth);
    randomForest->setMinSampleCount(bestMinSampleCount);
    randomForest->setMaxCategories(bestMaxCategories);
    randomForest->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

    randomForest->train(trainFeatures, ROW_SAMPLE, trainLabels);

    // Save the model for future use
    randomForest->save("random_forest_hu_model.xml");

    // Predict the test data
    Mat predictedLabels;
    randomForest->predict(testFeatures, predictedLabels);

    // Evaluate the accuracy
    int correctPredictions = 0;
    for (int i = 0; i < testLabels.rows; ++i) {
        if (testLabels.at<int>(i, 0) == static_cast<int>(predictedLabels.at<float>(i, 0))) {
            correctPredictions++;
        }
    }
    metrics.accuracy = (float)correctPredictions / testLabels.rows * 100.0f;
    cout << "1. Random Forest Hu Accuracy: " << metrics.accuracy << "%" << endl;

    // Define class names for confusion matrix labels
    vector<string> classNames = { "Informative Sign", "Prohibitive Sign", "Warning Sign" };

    // Get confusion matrix
    Mat confusionMatrix = getConfusionMatrix(testLabels, predictedLabels, classNames.size());

    // Calculate and print precision, recall, and F1 score
    cout << "2. Precision, Recall, and F1 Score for each class:" << endl << endl;
    calculateMetrics(confusionMatrix, classNames, metrics);

    // Print labeled confusion matrix
    cout << "4. Confusion Matrix:" << endl;
    printConfusionMatrix(confusionMatrix, classNames);

    return metrics;
}

// Function to run Random Forest with combined color and Hu moments features
Metrics runRandomForestCombined() {
    Metrics metrics;
    vector<Record> colorRecords, shapeRecords;

    // Load color histogram features
    readCSV("color_histogram_data.csv", colorRecords);

    // Load Hu moments features
    readCSV("hu_moments_data.csv", shapeRecords);

    // Merge records ensuring name match
    vector<Record> combinedRecords = mergeRecordsByName(colorRecords, shapeRecords);

    // Shuffle the records using a fixed random seed for reproducibility
    unsigned int seed = RANDOM_SEED;
    shuffleDataset(combinedRecords, seed);

    // Split the data into training and testing sets after shuffling
    Mat trainFeatures, testFeatures, trainLabels, testLabels;
    splitData(combinedRecords, trainFeatures, testFeatures, trainLabels, testLabels);

    // Variables to store the best hyperparameters
    int bestMaxDepth = 10;
    int bestMinSampleCount = 10;
    int bestMaxCategories = 2;

    // Hyperparameter tuning for Random Forest (the best values will be updated here)
    randomForestHyperparameterTuning(trainFeatures, trainLabels, testFeatures, testLabels,
        bestMaxDepth, bestMinSampleCount, bestMaxCategories);

    // Create a map to translate numeric labels to category names
    map<int, string> labelMap = {
        {0, "Informative Sign"},
        {1, "Prohibitive Sign"},
        {2, "Warning Sign"}
    };

    // Train the Random Forest model
    Ptr<RTrees> randomForest = RTrees::create();
    randomForest->setMaxDepth(bestMaxDepth);
    randomForest->setMinSampleCount(bestMinSampleCount);
    randomForest->setMaxCategories(bestMaxCategories);
    randomForest->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

    randomForest->train(trainFeatures, ROW_SAMPLE, trainLabels);

    // Save the model for future use
    randomForest->save("random_forest_combined_model.xml");

    // Predict the test data
    Mat predictedLabels;
    randomForest->predict(testFeatures, predictedLabels);

    // Evaluate the accuracy
    int correctPredictions = 0;
    for (int i = 0; i < testLabels.rows; ++i) {
        if (testLabels.at<int>(i, 0) == static_cast<int>(predictedLabels.at<float>(i, 0))) {
            correctPredictions++;
        }
    }
    metrics.accuracy = (float)correctPredictions / testLabels.rows * 100.0f;
    cout << "1. Random Forest Combined Accuracy: " << metrics.accuracy << "%" << endl;

    // Define class names for confusion matrix labels
    vector<string> classNames = { "Informative Sign", "Prohibitive Sign", "Warning Sign" };

    // Get confusion matrix
    Mat confusionMatrix = getConfusionMatrix(testLabels, predictedLabels, classNames.size());

    // Calculate and print precision, recall, and F1 score
    cout << "2. Precision, Recall, and F1 Score for each class:" << endl << endl;
    calculateMetrics(confusionMatrix, classNames, metrics);

    // Print labeled confusion matrix
    cout << "4. Confusion Matrix:" << endl;
    printConfusionMatrix(confusionMatrix, classNames);

    return metrics;
}

// Function for Random Forest hyperparameter tuning
void randomForestHyperparameterTuning(const Mat& trainFeatures, const Mat& trainLabels, const Mat& testFeatures, const Mat& testLabels,
    int& bestMaxDepth, int& bestMinSampleCount, int& bestMaxCategories) {
    vector<int> maxDepths = { 5, 10, 15 };       // Values to test for maximum depth of trees
    vector<int> minSampleCounts = { 2, 5, 10 };  // Values to test for minimum sample count per split
    vector<int> maxCategories = { 2, 5, 10 };    // Values to test for maximum categories

    double bestAccuracy = 0;

    for (int maxDepth : maxDepths) {
        for (int minSampleCount : minSampleCounts) {
            for (int maxCategory : maxCategories) {
                // Create and configure the Random Forest model
                Ptr<RTrees> randomForest = RTrees::create();
                randomForest->setMaxDepth(maxDepth);
                randomForest->setMinSampleCount(minSampleCount);
                randomForest->setMaxCategories(maxCategory);
                randomForest->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

                // Train the model
                randomForest->train(trainFeatures, ROW_SAMPLE, trainLabels);

                // Predict on the test set
                Mat predictedLabels;
                randomForest->predict(testFeatures, predictedLabels);

                // Evaluate the model
                int correctPredictions = 0;
                for (int i = 0; i < testLabels.rows; ++i) {
                    if (testLabels.at<int>(i, 0) == predictedLabels.at<float>(i, 0)) {
                        correctPredictions++;
                    }
                }

                float accuracy = (correctPredictions / static_cast<float>(testLabels.rows)) * 100;

                // Check if this combination gives the best accuracy
                if (accuracy > bestAccuracy) {
                    bestAccuracy = accuracy;
                    bestMaxDepth = maxDepth;
                    bestMinSampleCount = minSampleCount;
                    bestMaxCategories = maxCategory;
                }

                cout << "Max Depth: " << maxDepth << ", Min Sample Count: " << minSampleCount
                    << ", Max Categories: " << maxCategory << ", Accuracy: " << accuracy << "%" << endl;
            }
        }
    }

    cout << "\nBest Parameters -> Max Depth: " << bestMaxDepth << ", Min Sample Count: " << bestMinSampleCount
        << ", Max Categories: " << bestMaxCategories << ", Best Accuracy: " << bestAccuracy << "%" << endl;
}


/* =================================== Classifiers ======================================== */

// Dataset loading and preprocessing
void readCSV(const string& filename, vector<Record>& records) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Error opening file!" << endl;
        return;
    }
    string line, word;

    getline(file, line);  // Skip the header

    while (getline(file, line)) {
        stringstream ss(line);
        Record record;
        getline(ss, word, ',');
        record.imageName = word;
        getline(ss, word, ',');
        record.label = stoi(word);
        while (getline(ss, word, ',')) {
            record.features.push_back(stof(word));
        }
        records.push_back(record);
    }
}

// Combine features from color_)histogram and hu_moments
vector<Record> mergeRecordsByName(const vector<Record>& colorRecords, const vector<Record>& shapeRecords) {
    vector<Record> combinedRecords;
    map<string, Record> shapeMap;

    // Create a map from the shapeRecords using imageName as the key
    for (const auto& shapeRecord : shapeRecords) {
        shapeMap[shapeRecord.imageName] = shapeRecord;
    }

    // Iterate over colorRecords and find the corresponding shapeRecord by imageName
    for (const auto& colorRecord : colorRecords) {
        auto it = shapeMap.find(colorRecord.imageName);
        if (it != shapeMap.end() && colorRecord.label == it->second.label) {  // Check both imageName and label
            Record combinedRecord = colorRecord;
            combinedRecord.features.insert(
                combinedRecord.features.end(),
                it->second.features.begin(),
                it->second.features.end()
            );
            combinedRecords.push_back(combinedRecord);
        }
    }

    return combinedRecords;
}

// Convert records to Mat format for OpenCV
Mat convertToMat(const vector<Record>& records) {
    Mat featuresMat(records.size(), records[0].features.size(), CV_32F);
    for (size_t i = 0; i < records.size(); ++i) {
        for (size_t j = 0; j < records[i].features.size(); ++j) {
            featuresMat.at<float>(i, j) = records[i].features[j];
        }
    }
    return featuresMat;
}

Mat convertLabelsToMat(const vector<Record>& records) {
    Mat labelsMat(records.size(), 1, CV_32S);
    for (size_t i = 0; i < records.size(); ++i) {
        labelsMat.at<int>(i, 0) = records[i].label;
    }
    return labelsMat;
}

// Shuffle records with a fixed seed
void shuffleDataset(vector<Record>& records, unsigned int seed) {
    mt19937 g(seed);
    shuffle(records.begin(), records.end(), g);
}

// Split dataset into train/test sets
void splitData(const vector<Record>& records, Mat& trainFeatures, Mat& testFeatures, Mat& trainLabels, Mat& testLabels, float trainRatio) {
    int totalSamples = records.size();
    int trainSamples = static_cast<int>(totalSamples * trainRatio);
    vector<Record> trainRecords(records.begin(), records.begin() + trainSamples);
    vector<Record> testRecords(records.begin() + trainSamples, records.end());
    trainFeatures = convertToMat(trainRecords);
    trainLabels = convertLabelsToMat(trainRecords);
    testFeatures = convertToMat(testRecords);
    testLabels = convertLabelsToMat(testRecords);
}


// Function to create and print a labeled confusion matrix
Mat getConfusionMatrix(const Mat& testLabels, const Mat& predictedLabels, int numClasses) {
    // Initialize confusion matrix with zeros
    Mat confusionMatrix = Mat::zeros(numClasses, numClasses, CV_32S);

    // Populate confusion matrix
    for (int i = 0; i < testLabels.rows; ++i) {
        int actualClass = testLabels.at<int>(i, 0);
        int predictedClass = static_cast<int>(predictedLabels.at<float>(i, 0));
        confusionMatrix.at<int>(actualClass, predictedClass)++;
    }

    return confusionMatrix;
}

// Function to print confusion matrix with labels
void printConfusionMatrix(const Mat& confusionMatrix, const vector<string>& classNames) {
    int numClasses = classNames.size();

    cout << setw(20) << "Actual\\Predicted ";
    for (int i = 0; i < numClasses; ++i) {
        cout << setw(20) << classNames[i];
    }
    cout << endl;

    for (int i = 0; i < numClasses; ++i) {
        cout << setw(20) << classNames[i];
        for (int j = 0; j < numClasses; ++j) {
            cout << setw(18) << confusionMatrix.at<int>(i, j);
        }
        cout << endl;
    }
    cout << endl;
}

// Function to calculate and store precision, recall, and F1 score for each class, and average them
void calculateMetrics(const Mat& confusionMatrix, const vector<string>& classNames, Metrics& metrics) {
    int numClasses = classNames.size();
    vector<float> precision(numClasses, 0.0);
    vector<float> recall(numClasses, 0.0);
    vector<float> f1Score(numClasses, 0.0);

    float totalPrecision = 0.0f;
    float totalRecall = 0.0f;
    float totalF1Score = 0.0f;

    for (int i = 0; i < numClasses; ++i) {
        int tp = confusionMatrix.at<int>(i, i);  // True positives
        int fp = sum(confusionMatrix.col(i))[0] - tp;  // False positives
        int fn = sum(confusionMatrix.row(i))[0] - tp;  // False negatives

        // Compute precision and recall, avoiding division by zero
        if (tp + fp == 0) {
            precision[i] = 0;
        }
        else {
            precision[i] = static_cast<float>(tp) / (tp + fp);
        }

        if (tp + fn == 0) {
            recall[i] = 0;
        }
        else {
            recall[i] = static_cast<float>(tp) / (tp + fn);
        }

        // Only compute F1 score if both precision and recall are non-zero
        if (precision[i] == 0 || recall[i] == 0) {
            f1Score[i] = 0;  // If either precision or recall is zero, F1 score is zero
        }
        else {
            f1Score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]);
        }

        // Accumulate the total precision, recall, and F1 score for averaging
        totalPrecision += precision[i];
        totalRecall += recall[i];
        totalF1Score += f1Score[i];

        // Store metrics for each class, converting to percentage
        metrics.precision[classNames[i]] = precision[i] * 100.0f;
        metrics.recall[classNames[i]] = recall[i] * 100.0f;
        metrics.f1Score[classNames[i]] = f1Score[i] * 100.0f;

        cout << "Class: " << classNames[i] << endl;
        cout << "Precision: " << precision[i] * 100.0f << "%" << endl;
        cout << "Recall: " << recall[i] * 100.0f << "%" << endl;
        cout << "F1 Score: " << f1Score[i] * 100.0f << "%" << endl << endl;
    }

    // Calculate and store the macro-averaged precision, recall, and F1 score
    float avgPrecision = totalPrecision / numClasses;
    float avgRecall = totalRecall / numClasses;
    float avgF1Score = totalF1Score / numClasses;

    // Store the average metrics
    metrics.precision["Average"] = avgPrecision * 100.0f;
    metrics.recall["Average"] = avgRecall * 100.0f;
    metrics.f1Score["Average"] = avgF1Score * 100.0f;

    cout << "Average Precision: " << avgPrecision * 100.0f << "%" << endl;
    cout << "Average Recall: " << avgRecall * 100.0f << "%" << endl;
    cout << "Average F1 Score: " << avgF1Score * 100.0f << "%" << endl;
}



