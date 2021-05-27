/*
Call a C function in python       : https://www.journaldev.com/31907/calling-c-functions-from-python 
                                    https://realpython.com/python-bindings-overview/
Pass a numpy array into C program : https://stackoverflow.com/questions/5862915/passing-numpy-arrays-to-a-c-function-for-input-and-output
iterate a numpy array in C program: https://stackoverflow.com/questions/22425921/pass-a-2d-numpy-array-to-c-using-ctypes
*/

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include <stdio.h>
#include <stdbool.h>
#include <iostream>
#include <vector>

#include <sys/time.h>

struct keypoint_xy
{
    int x;
    int y;
};

struct fingerkeypoint
{
    keypoint_xy firstpoint;
    keypoint_xy secondpoint;
    keypoint_xy thirdpoint;
    keypoint_xy fourthpoint;
};

struct handkeypoint
{
    fingerkeypoint firstfinger;
    fingerkeypoint secondfinger;
    fingerkeypoint thirdfinger;
    fingerkeypoint forthfinger;
    fingerkeypoint thumb;
    keypoint_xy mid_pt;
};

void fillintohandkeypoint(std::vector<double> handkeypoint_vector, handkeypoint &handkeypoint_)
{
    // 1-2us for this noob assignment
    handkeypoint_.mid_pt.x = handkeypoint_vector.at(0);
    handkeypoint_.mid_pt.y = handkeypoint_vector.at(1);

    // thumb ---> fourthfinger
    handkeypoint_.thumb.firstpoint.x = handkeypoint_vector.at(2);
    handkeypoint_.thumb.firstpoint.y = handkeypoint_vector.at(3);
    handkeypoint_.thumb.secondpoint.x = handkeypoint_vector.at(4);
    handkeypoint_.thumb.secondpoint.y = handkeypoint_vector.at(5);
    handkeypoint_.thumb.thirdpoint.x = handkeypoint_vector.at(6);
    handkeypoint_.thumb.thirdpoint.y = handkeypoint_vector.at(7);
    handkeypoint_.thumb.fourthpoint.x = handkeypoint_vector.at(8);
    handkeypoint_.thumb.fourthpoint.y = handkeypoint_vector.at(9);

    handkeypoint_.firstfinger.firstpoint.x = handkeypoint_vector.at(10);
    handkeypoint_.firstfinger.firstpoint.y = handkeypoint_vector.at(11);
    handkeypoint_.firstfinger.secondpoint.x = handkeypoint_vector.at(12);
    handkeypoint_.firstfinger.secondpoint.y = handkeypoint_vector.at(13);
    handkeypoint_.firstfinger.thirdpoint.x = handkeypoint_vector.at(14);
    handkeypoint_.firstfinger.thirdpoint.y = handkeypoint_vector.at(15);
    handkeypoint_.firstfinger.fourthpoint.x = handkeypoint_vector.at(16);
    handkeypoint_.firstfinger.fourthpoint.y = handkeypoint_vector.at(17);

    handkeypoint_.secondfinger.firstpoint.x = handkeypoint_vector.at(18);
    handkeypoint_.secondfinger.firstpoint.y = handkeypoint_vector.at(19);
    handkeypoint_.secondfinger.secondpoint.x = handkeypoint_vector.at(20);
    handkeypoint_.secondfinger.secondpoint.y = handkeypoint_vector.at(21);
    handkeypoint_.secondfinger.thirdpoint.x = handkeypoint_vector.at(22);
    handkeypoint_.secondfinger.thirdpoint.y = handkeypoint_vector.at(23);
    handkeypoint_.secondfinger.fourthpoint.x = handkeypoint_vector.at(24);
    handkeypoint_.secondfinger.fourthpoint.y = handkeypoint_vector.at(25);

    handkeypoint_.thirdfinger.firstpoint.x = handkeypoint_vector.at(26);
    handkeypoint_.thirdfinger.firstpoint.y = handkeypoint_vector.at(27);
    handkeypoint_.thirdfinger.secondpoint.x = handkeypoint_vector.at(28);
    handkeypoint_.thirdfinger.secondpoint.y = handkeypoint_vector.at(29);
    handkeypoint_.thirdfinger.thirdpoint.x = handkeypoint_vector.at(30);
    handkeypoint_.thirdfinger.thirdpoint.y = handkeypoint_vector.at(31);
    handkeypoint_.thirdfinger.fourthpoint.x = handkeypoint_vector.at(32);
    handkeypoint_.thirdfinger.fourthpoint.y = handkeypoint_vector.at(33);

    handkeypoint_.forthfinger.firstpoint.x = handkeypoint_vector.at(34);
    handkeypoint_.forthfinger.firstpoint.y = handkeypoint_vector.at(35);
    handkeypoint_.forthfinger.secondpoint.x = handkeypoint_vector.at(36);
    handkeypoint_.forthfinger.secondpoint.y = handkeypoint_vector.at(37);
    handkeypoint_.forthfinger.thirdpoint.x = handkeypoint_vector.at(38);
    handkeypoint_.forthfinger.thirdpoint.y = handkeypoint_vector.at(39);
    handkeypoint_.forthfinger.fourthpoint.x = handkeypoint_vector.at(40);
    handkeypoint_.forthfinger.fourthpoint.y = handkeypoint_vector.at(41);
}

float get_Euclidean_DistanceAB(float a_x, float a_y, float b_x, float b_y)
{
    float dist = std::pow(a_x - b_x, 2) + pow(a_y - b_y, 2);
    return std::sqrt(dist);
}

bool isThumbNearFirstFinger(keypoint_xy thumb_point1, keypoint_xy firstfinger_point2)
{
    float distance = get_Euclidean_DistanceAB(thumb_point1.x, thumb_point1.y, firstfinger_point2.x, firstfinger_point2.y);
    return distance < 0.2;
}

boost::python::object gesture_recognition(boost::python::numpy::ndarray &keypoint_hand)
{

    // struct timeval stop, start;
    // gettimeofday(&start, NULL);

    using boost::python::object;

    // // finger states
    bool thumbIsOpen = false;
    bool firstFingerIsOpen = false;
    bool secondFingerIsOpen = false;
    bool thirdFingerIsOpen = false;
    bool fourthFingerIsOpen = false;

    int input_size = keypoint_hand.shape(0);
    double *input_ptr = reinterpret_cast<double *>(keypoint_hand.get_data());
    std::vector<double> v(input_size);
    for (int i = 0; i < input_size; i++)
        v[i] = *(input_ptr + i);

    handkeypoint hand_keypoint;
    fillintohandkeypoint(v, hand_keypoint);

    float pseudoFixKeyPoint = hand_keypoint.thumb.secondpoint.x;
    if (hand_keypoint.firstfinger.firstpoint.x > hand_keypoint.forthfinger.firstpoint.x &&
        hand_keypoint.thumb.thirdpoint.x > pseudoFixKeyPoint && hand_keypoint.thumb.fourthpoint.x > pseudoFixKeyPoint)
    {
        // palm facing the camera and thumb is open
        // std::cout<<"Left backside or Right frontside"<<std::endl;
        // std::cout<<"ThumbIsOpen: first statement"<<std::endl;
        thumbIsOpen = true;
    }

    else if (hand_keypoint.firstfinger.firstpoint.x < hand_keypoint.forthfinger.firstpoint.x &&
             (hand_keypoint.thumb.thirdpoint.x < pseudoFixKeyPoint && hand_keypoint.thumb.fourthpoint.x < pseudoFixKeyPoint))
    {
        // thumb is on right of index finger. Either this is left hand or right hand with palm facing the user
        // std::cout<<"ThumbIsOpen: second statement"<<std::endl;
        thumbIsOpen = true;
    }

    pseudoFixKeyPoint = hand_keypoint.firstfinger.secondpoint.y;
    if (hand_keypoint.firstfinger.thirdpoint.y < pseudoFixKeyPoint && hand_keypoint.firstfinger.fourthpoint.y < pseudoFixKeyPoint)
    {
        firstFingerIsOpen = true;
    }

    pseudoFixKeyPoint = hand_keypoint.secondfinger.secondpoint.y;
    if (hand_keypoint.secondfinger.thirdpoint.y < pseudoFixKeyPoint && hand_keypoint.secondfinger.fourthpoint.y < pseudoFixKeyPoint)
    {
        secondFingerIsOpen = true;
    }

    pseudoFixKeyPoint = hand_keypoint.thirdfinger.secondpoint.y;
    if (hand_keypoint.thirdfinger.thirdpoint.y < pseudoFixKeyPoint && hand_keypoint.thirdfinger.fourthpoint.y < pseudoFixKeyPoint)
    {
        thirdFingerIsOpen = true;
    }

    pseudoFixKeyPoint = hand_keypoint.forthfinger.secondpoint.y;
    if (hand_keypoint.forthfinger.thirdpoint.y < pseudoFixKeyPoint && hand_keypoint.forthfinger.fourthpoint.y < pseudoFixKeyPoint)
    {
        fourthFingerIsOpen = true;
    }

    // std::cout << "Finger States: " << thumbIsOpen << firstFingerIsOpen << secondFingerIsOpen << thirdFingerIsOpen << fourthFingerIsOpen<<"___"<<std::endl;

    // Hand gesture recognition
    if (thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && thirdFingerIsOpen && fourthFingerIsOpen)
    {
        // std::cout << "FIVE!" << std::endl;
        return object("FIVE");
    }
    else if (!thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && thirdFingerIsOpen && fourthFingerIsOpen)
    {
        // std::cout << "FOUR!" << std::endl;
        return object("FOUR");
    }
    else if (thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        // std::cout << "THREE!" << std::endl;
        return object("THREE");
    }
    else if (thumbIsOpen && firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        // std::cout << "TWO!" << std::endl;
        return object("TWO");
    }
    else if (!thumbIsOpen && firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        // std::cout << "ONE!" << std::endl;
        return object("ONE");
    }
    else if (!thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        // std::cout << "YEAH!" << std::endl;
        return object("YEAH");
    }
    else if (!thumbIsOpen && firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && fourthFingerIsOpen)
    {
        // std::cout << "ROCK!" << std::endl;
        return object("ROCK");
    }
    else if (thumbIsOpen && firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && fourthFingerIsOpen)
    {
        // std::cout << "SPIDERMAN!" << std::endl;
        return object("SPIDERMAN");
    }
    // else if (!thumbIsOpen && !firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    // {
    //     // std::cout << "FIST!"<<std::endl;
    //     return object("FIST");
    // }
    // else if (!firstFingerIsOpen && secondFingerIsOpen && thirdFingerIsOpen && fourthFingerIsOpen && isThumbNearFirstFinger(hand_keypoint.thumb.fourthpoint,hand_keypoint.firstfinger.fourthpoint))
    // {
    //     std::cout << "OK!";
    // }
    else
    {
        // std::cout << "Finger States: " << thumbIsOpen << firstFingerIsOpen << secondFingerIsOpen << thirdFingerIsOpen << fourthFingerIsOpen;
        // std::cout << "___" << std::endl;
        return object("No gesture");
    }

    // gettimeofday(&stop, NULL);
    // printf("took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
}

BOOST_PYTHON_MODULE(gesture_recognition)
{
    using namespace boost::python;
    Py_Initialize();
    boost::python::numpy::initialize();
    def("gesture_recognition", &gesture_recognition);
}