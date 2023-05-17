# FaceDigitClassification
CS440 - Introduction to Artificial Intelligence (Spring 2023)  
Rutgers the State University of New Jersey

### Project Details
**Acknowledgement:** This project is based on the one created by Dan Klein and John DeNero that was given as part of the programming assignments of Berkeley’s CS188 course.  
In this project, you will design three classifiers: a naive Bayes classifier, a perceptron classifier and a classifier of your choice. You will test your classifiers on two image data sets: a set of scanned handwritten digit images and a set of face images in which edges have already been detected. Even with simple features, your classifiers will be able to do quite well on these tasks when given enough training data.  
Optical character recognition (OCR) is the task of extracting text from image sources. The first data set on which you will run your classifiers is a collection of handwritten numerical digits (0-9). This is a very commercially useful technology, similar to the technique used by the US post office to route mail by zip codes. There are systems that can perform with over 99% classification accuracy (see LeNet-5 for an example system in action).  
Face detection is the task of localizing faces within video or still images. The faces can be at any location and vary in size. There are many applications for face detection, including human computer interaction and surveillance. You will attempt a simplified face detection task in which your system is presented with an image that has been pre-processed by an edge detection algorithm. The task is to determine whether the edge image is a face or not.  
Please refer to http://inst.eecs.berkeley.edu/~cs188/sp11/projects/classification/classification.html for a brief description of the Perceptron and Naive Bayes classifiers.

### How to Run Code
**python** dataClassifier.py -c naiveBayes --autotune  
**python** dataClassifier.py -c perceptron  
**python** dataClassifier.py -c mira --autotune  
**python** dataClassifier.py -c perceptron -w  
**python** dataClassifier.py -d digits -c perceptron -f -a -t &lt;trainingfactor&gt; -i 30  

**TO RUN WITHOUT RECEIVING STANDARD DEVIATION**  
**python** dataClassifier.py -d digits -c perceptron -f -a -t &lt;trainingfactor&gt;  
**python** dataClassifier.py -d digits -c naiveBayes -f -a -t &lt;trainingfactor&gt;  
**python** dataClassifier.py -d digits -c mira -f -a -t &lt;trainingfactor&gt;  
**&lt;trainingfactor&gt;** is a float between 0 to 1  
  
**TO RUN WHILE RECEIVING STANDARD DEVIATION**  
**python** dataClassifier.py -d digits -c perceptron -f -a -t &lt;trainingfactor&gt; -r  
**python** dataClassifier.py -d digits -c naiveBayes -f -a -t &lt;trainingfactor&gt; -r  
**python** dataClassifier.py -d digits -c mira -f -a -t &lt;trainingfactor&gt; -r  
**&lt;trainingfactor&gt;** is a float between 0 to 1  