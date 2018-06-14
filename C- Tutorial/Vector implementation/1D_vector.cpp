//TODO: Use this as a playground to practice with vectors


//TODO:
// Fill out your program's header. The header should contain any necessary
// include statements and also function declarations
#include <iostream>
#include <vector>
using namespace std;
vector <float> subtract(vector<float> vector1, vector<float> vector2);


//TODO:
// Write your main program. Remember that all C++ programs need
// a main function. The most important part of your program goes
// inside the main function. 

int main()
{
    // declare and initialize vectors
	vector<float> v1(3);
	vector<float> v2(3);
	
	v1[0] = 5.0;
	v1[1] = 10.0;
	v1[2] = 27.0;
	
	v2[0] = 2.0;
	v2[1] = 17.0;
	v2[2] = 12.0;
    vector<float> result(v1.size());
    result = subtract(v1, v2);
    for (int i = 0; i<result.size();i++)
    {
        cout<< "Result = " <<result[i]<<endl;
        
    }//end of for loop
    return 0;
    
}//end of main

vector <float> subtract(vector<float> vector1,vector<float> vector2)
{
    vector<float> result(vector1.size());
    for(int i = 0;i<vector1.size();i++)
    {
        result[i] = vector2[i] - vector1[i];

    }//end of for loop
    return result;
    
}//end of subtract method
