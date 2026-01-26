/*
  Author: Krushna Sanjay Sharma
  Date: January 24, 2026
  Purpose: Read and display an image from a file using OpenCV.
           Task 1: Basic image display with keyboard controls.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <ctime>

using namespace cv;
using namespace std;

/*
  Function: generateTimestampFilename
  Purpose: Generate a unique filename based on current timestamp
  Arguments:
    prefix - string prefix for the filename (e.g., "image")
    extension - file extension (e.g., ".jpg")
  Return value: string containing the generated filename
*/
string generateTimestampFilename(const string& prefix, const string& extension) {
    time_t now = time(0);
    tm* ltm = localtime(&now);
    
    char buffer[100];
    sprintf(buffer, "%s_%04d%02d%02d_%02d%02d%02d%s",
            prefix.c_str(),
            1900 + ltm->tm_year,
            1 + ltm->tm_mon,
            ltm->tm_mday,
            ltm->tm_hour,
            ltm->tm_min,
            ltm->tm_sec,
            extension.c_str());
    
    return string(buffer);
}

/*
  Function: printControls
  Purpose: Display available keyboard controls to the user
  Arguments: none
  Return value: none
*/
void printControls() {
    cout << "\n=== Image Display Controls ===" << endl;
    cout << "  q/ESC : Quit application" << endl;
    cout << "  s     : Save image" << endl;
    cout << "  i     : Display image information" << endl;
    cout << "  r     : Reset to original image" << endl;
    cout << "  g     : Convert to greyscale" << endl;
    cout << "  h     : Show this help" << endl;
    cout << "  +/=   : Increase brightness" << endl;
    cout << "  -/_   : Decrease brightness" << endl;
    cout << "==============================\n" << endl;
}

/*
  Function: displayImageInfo
  Purpose: Display detailed information about the image
  Arguments:
    image - the image to analyze
    filename - the source filename
  Return value: none
*/
void displayImageInfo(const Mat& image, const string& filename) {
    cout << "\n=== Image Information ===" << endl;
    cout << "Filename: " << filename << endl;
    cout << "Dimensions: " << image.cols << " x " << image.rows << " pixels" << endl;
    cout << "Channels: " << image.channels() << endl;
    cout << "Depth: " << image.depth() << " (";
    
    switch(image.depth()) {
        case CV_8U:  cout << "8-bit unsigned"; break;
        case CV_8S:  cout << "8-bit signed"; break;
        case CV_16U: cout << "16-bit unsigned"; break;
        case CV_16S: cout << "16-bit signed"; break;
        case CV_32S: cout << "32-bit signed"; break;
        case CV_32F: cout << "32-bit float"; break;
        case CV_64F: cout << "64-bit float"; break;
        default:     cout << "unknown"; break;
    }
    cout << ")" << endl;
    
    cout << "Type: " << image.type() << endl;
    cout << "Size in memory: " << (image.total() * image.elemSize() / 1024.0) << " KB" << endl;
    cout << "========================\n" << endl;
}

/*
  Function: adjustBrightness
  Purpose: Adjust the brightness of an image using manual pixel manipulation
  Arguments:
    src - source image (CV_8UC3)
    dst - destination image (CV_8UC3)
    brightness - brightness factor (-1.0 to 1.0, where 0 is no change)
  Return value: none
  
  Algorithm:
    offset = brightness * 255
    For each pixel:
      b = b + offset (clipped to [0, 255])
      g = g + offset (clipped to [0, 255])
      r = r + offset (clipped to [0, 255])
*/
void adjustBrightness(const Mat& src, Mat& dst, float brightness) {
    // Create destination image if needed
    dst.create(src.size(), src.type());
    
    // Calculate offset from brightness factor
    int offset = static_cast<int>(brightness * 255.0f);
    
    // Iterate through each row using row pointers
    for (int row = 0; row < src.rows; row++) {
        // Get pointers to the current row in source and destination
        const Vec3b* srcRow = src.ptr<Vec3b>(row);
        Vec3b* dstRow = dst.ptr<Vec3b>(row);
        
        // Iterate through each pixel in the row
        for (int col = 0; col < src.cols; col++) {
            // Get BGR values from source pixel
            int blue = srcRow[col][0];
            int green = srcRow[col][1];
            int red = srcRow[col][2];
            
            // Add offset to each channel
            blue += offset;
            green += offset;
            red += offset;
            
            // Clip values to valid range [0, 255]
            if (blue < 0) blue = 0;
            if (blue > 255) blue = 255;
            
            if (green < 0) green = 0;
            if (green > 255) green = 255;
            
            if (red < 0) red = 0;
            if (red > 255) red = 255;
            
            // Write clipped values to destination pixel
            dstRow[col][0] = static_cast<uchar>(blue);
            dstRow[col][1] = static_cast<uchar>(green);
            dstRow[col][2] = static_cast<uchar>(red);
        }
    }
}

/*
  Function: main
  Purpose: Read an image file, display it in a window, and handle user keypresses
  Arguments:
    argc - number of command line arguments
    argv - array of command line argument strings (expects image path as argv[1])
  Return value: 0 on success, -1 on error
*/
int main(int argc, char** argv) {
    cout << "=== Image Display Application ===" << endl;
    cout << "Task 1: Read and display image from file\n" << endl;
    
    // Check if image path was provided as command line argument
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        cout << "Example: " << argv[0] << " image.jpg" << endl;
        cout << "\nPress Enter to exit...";
        cin.get();
        return -1;
    }
    
    // Get the image path from command line arguments
    string imagePath = argv[1];
    
    // Read the image from file
    // IMREAD_COLOR ensures we get a color image (BGR format)
    Mat originalImage = imread(imagePath, IMREAD_COLOR);
    
    // Check if image was loaded successfully
    if (originalImage.empty()) {
        cout << "Error: Could not open or find the image at: " << imagePath << endl;
        cout << "\nPlease check that:" << endl;
        cout << "  1. The file path is correct" << endl;
        cout << "  2. The file exists" << endl;
        cout << "  3. The file format is supported (jpg, png, bmp, etc.)" << endl;
        cout << "\nPress Enter to exit...";
        cin.get();
        return -1;
    }
    
    // Image loaded successfully
    cout << "Image loaded successfully!" << endl;
    displayImageInfo(originalImage, imagePath);
    printControls();
    
    // Create a window to display the image
    string windowName = "Image Display - " + imagePath;
    namedWindow(windowName, WINDOW_AUTOSIZE);
    
    // Current image being displayed (can be modified)
    Mat currentImage = originalImage.clone();
    
    // Display the image in the window
    imshow(windowName, currentImage);
    
    // State variables
    int savedCount = 0;
    float brightnessLevel = 0.0f;  // Range: -1.0 to 1.0
    
    // Main event loop - wait for keypress
    cout << "Image displayed. Waiting for keyboard input..." << endl;
    
    while (true) {
        // Wait for a key press (1ms timeout to keep window responsive)
        // This allows the window to refresh and respond to system events
        int key = waitKey(1);
        
        // Check if no key was pressed (timeout)
        if (key == -1) {
            continue;
        }
        
        // Process the keypress
        char keyChar = (char)key;
        
        // Handle different keypresses
        if (key == 'q' || key == 'Q') {
            // Quit the application
            cout << "\n'q' pressed - Quitting application..." << endl;
            cout << "Total images saved: " << savedCount << endl;
            break;
        }
        else if (key == 27) {  // ESC key
            // Alternative quit option
            cout << "\nESC pressed - Quitting application..." << endl;
            cout << "Total images saved: " << savedCount << endl;
            break;
        }
        else if (key == 's' || key == 'S') {
            // Save the current image
            string filename = generateTimestampFilename("saved_image", ".jpg");
            
            bool success = imwrite(filename, currentImage);
            
            if (success) {
                savedCount++;
                cout << "Image saved as: " << filename << endl;
            }
            else {
                cout << "Error: Failed to save image" << endl;
            }
        }
        else if (key == 'i' || key == 'I') {
            // Display image information
            displayImageInfo(currentImage, imagePath);
        }
        else if (key == 'r' || key == 'R') {
            // Reset to original image
            currentImage = originalImage.clone();
            brightnessLevel = 0.0f;
            imshow(windowName, currentImage);
            cout << "Image reset to original" << endl;
        }
        else if (key == 'g' || key == 'G') {
            // Convert to greyscale
            Mat greyImage;
            cvtColor(originalImage, greyImage, COLOR_BGR2GRAY);
            
            // Convert back to 3-channel for display consistency
            cvtColor(greyImage, currentImage, COLOR_GRAY2BGR);
            
            brightnessLevel = 0.0f;
            imshow(windowName, currentImage);
            cout << "Converted to greyscale" << endl;
        }
        else if (key == 'h' || key == 'H') {
            // Show help
            printControls();
        }
        else if (key == '+' || key == '=') {
            // Increase brightness
            brightnessLevel += 0.1f;
            if (brightnessLevel > 1.0f) brightnessLevel = 1.0f;
            
            adjustBrightness(originalImage, currentImage, brightnessLevel);
            imshow(windowName, currentImage);
            
            int percentage = static_cast<int>(brightnessLevel * 100.0f);
            cout << "Brightness: " << (brightnessLevel >= 0 ? "+" : "") 
                 << percentage << "%" << endl;
        }
        else if (key == '-' || key == '_') {
            // Decrease brightness
            brightnessLevel -= 0.1f;
            if (brightnessLevel < -1.0f) brightnessLevel = -1.0f;
            
            adjustBrightness(originalImage, currentImage, brightnessLevel);
            imshow(windowName, currentImage);
            
            int percentage = static_cast<int>(brightnessLevel * 100.0f);
            cout << "Brightness: " << (brightnessLevel >= 0 ? "+" : "") 
                 << percentage << "%" << endl;
        }
        else {
            // Unknown key pressed
            cout << "Unknown key pressed (code: " << key << ")" << endl;
            cout << "Press 'h' for help" << endl;
        }
    }
    
    // Cleanup - destroy all windows
    destroyAllWindows();
    
    cout << "\nApplication closed successfully." << endl;
    return 0;
}
