// Utility functions and classes for TreeSHAP

#ifndef __UTILS
#define __UTILS

#include <vector>
#include <stack>

using namespace std;

// Custom Types
template <typename T>
using Matrix = vector<vector<T>>;
template <typename T>
using Tensor = vector<vector<vector<T>>>;


template<typename T>
Matrix<T> createMatrix(int n, int m, T* data){
    Matrix<T> mat;
    for (int i(0); i < n; i++){
        mat.push_back(vector<T> ());
        for (int j(0); j < m; j++){
            mat[i].push_back((data[m * i + j]));
        }
    }
    return mat;
}

template<typename T>
Tensor<T> createTensor(int n, int m, int l, T* data){
    Tensor<T> mat;
    for (int i(0); i < n; i++){
        mat.push_back(Matrix<T> ());
        for (int j(0); j < m; j++){
            mat[i].push_back(vector<T> ());
            for (int k(0); k < l; k++){
                mat[i][j].push_back((data[l * m * i + l * j + k]));
            }
        }
    }
    return mat;
}

template<typename T>
void printMatrix(Matrix<T> mat){
    int n = mat.size();
    int m = mat[0].size();
    for (int i(0); i < n; i++){
        for (int j(0); j < m; j++){
            cout << mat[i][j] << " ";
        }
        cout << "\n";
    }
}

template<typename T>
void printTensor(Tensor<T> mat){
    int n = mat.size();
    int m = mat[0].size();
    int l = mat[0][0].size();
    for (int i(0); i < n; i++){
        for (int j(0); j < m; j++){
            for (int k(0); k < l; k++){
                cout << mat[i][j][k] << " ";
            }
            cout << "\n";
        }
        cout << "\n\n";
    }
}



void compute_W(Matrix<double> &W)
{
    int D = W.size();
    for (double j(0); j < D; j++){
        W[0][j] = 1 / (j + 1);
        W[j][j] = 1 / (j + 1);
    }
    for (double j(2); j < D; j++){
        for (double i(j-1); i > 0; i--){
            W[i][j] = (j - i) / (i + 1) * W[i+1][j];
        }
    }
}


#endif