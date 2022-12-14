#include <vector>
#include <iostream>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iomanip>
#include "progressbar.hpp"


using namespace std;
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



// Recursion function for treeSHAP
pair<double, double> recurse(int n,
                            vector<double> &x, vector<double> &z, 
                            int* categorical_to_features,
                            vector<int> &feature,
                            vector<int> &child_left,
                            vector<int> &child_right,
                            vector<double> &threshold,
                            vector<double> &value,
                            vector<vector<double>> &W,
                            int n_features,
                            vector<double> &phi,
                            vector<int> &in_SX,
                            vector<int> &in_SZ)
{
    int current_feature = feature[n];
    int x_child(0), z_child(0);
    // num_players := |S_{AB}|
    int num_players = 0;

    // Arriving at a Leaf
    if (child_left[n] < 0)
    {
        double pos(0.0), neg(0.0);
        num_players = in_SX[n_features] + in_SZ[n_features];
        if (in_SX[n_features] > 0)
        {
            pos = W[in_SX[n_features]-1][num_players-1] * value[n];
        }
        if (in_SZ[n_features] > 0)
        {
            neg = W[in_SX[n_features]][num_players-1] * value[n];
        }
        return make_pair(pos, neg);
    }
    
    // Find children of x and z
    if (x[current_feature] <= threshold[n]){
        x_child = child_left[n];
    } else {x_child = child_right[n];}
    if (z[current_feature] <= threshold[n]){
        z_child = child_left[n];
    } else {z_child = child_right[n];}

    // Scenario 1 : x and z go the same way so we avoid the type B edge
    if (x_child == z_child){
        return recurse(x_child, x, z, categorical_to_features, feature, child_left, child_right,
                            threshold, value, W, n_features, phi, in_SX, in_SZ);
    }

    // Senario 2: x and z go different ways and we have seen this feature i in S_X U S_Z.
    // Hence we go down the correct edge to ensure that S_X and S_Z are kept disjoint
    if (in_SX[ categorical_to_features[current_feature] ] || 
        in_SZ[ categorical_to_features[current_feature] ]){
        if (in_SX[current_feature]){
            return recurse(x_child, x, z, categorical_to_features, feature, child_left, child_right,
                            threshold, value, W, n_features, phi, in_SX, in_SZ);
        }
        else{
            return recurse(z_child, x, z, categorical_to_features, feature, child_left, child_right,
                            threshold, value, W, n_features, phi, in_SX, in_SZ);
        }
    }

    // Scenario 3 : x and z go different ways and we have not yet seen this feature
    else {
        // Go to x's child
        in_SX[categorical_to_features[current_feature]]++;
        in_SX[n_features]++;
        pair<double, double> pairf = recurse(x_child, x, z, categorical_to_features, feature, child_left, child_right,
                                            threshold, value, W, n_features, phi, in_SX, in_SZ);
        in_SX[categorical_to_features[current_feature]]--;
        in_SX[n_features]--;

        // Go to z's child
        in_SZ[categorical_to_features[current_feature]]++;
        in_SZ[n_features]++;
        pair<double, double> pairb = recurse(z_child, x, z, categorical_to_features, feature, child_left, child_right,
                                            threshold, value, W, n_features, phi, in_SX, in_SZ);
        in_SZ[categorical_to_features[current_feature]]--;
        in_SZ[n_features]--;

        // Add contribution to the feature
        phi[ categorical_to_features[current_feature] ] += pairf.first - pairb.second;

        return make_pair(pairf.first + pairb.first, pairf.second + pairb.second);
    }
}



// Main function for TreeSHAP
Tensor<double> treeSHAP(Matrix<double> &X_f, 
                        Matrix<double> &X_b,
                        int* categorical_to_features, 
                        Matrix<int> &feature,
                        Matrix<int> &left_child,
                        Matrix<int> &right_child,
                        Matrix<double> &threshold,
                        Matrix<double> &value,
                        Matrix<double> &W)
{
    // Setup
    int n_features = categorical_to_features[X_f[0].size()-1] + 1;
    int n_trees = feature.size();
    int size_background = X_b.size();
    int size_foreground = X_f.size();

    // Initialize the SHAP values to zero
    Tensor<double> phi_f_b(size_foreground, Matrix<double> (n_trees, vector<double> (n_features, 0)));
    progressbar bar(size_foreground);
    // Iterate over all foreground instances
    for (int i(0); i < size_foreground; i++){
        // Iterate over all background instances
        for (int j(0); j < size_background; j++){
            // Iterate over all trees in the ensemble
            for (int t(0); t < n_trees; t++){
                // Last index is the size of the set
                vector<int> in_SX(n_features+1, 0);
                vector<int> in_SZ(n_features+1, 0);
                vector<double> phi(n_features, 0);

                // Start the recursion
                recurse(0, X_f[i], X_b[j], categorical_to_features, feature[t], left_child[t], right_child[t],
                            threshold[t], value[t], W, n_features, phi, in_SX, in_SZ);

                // Add the contribution of the tree and background instance
                for (int f(0); f < n_features; f++){
                    phi_f_b[i][t][f] += phi[f];
                }
            }
        }
        // Rescale w.r.t the number of background instances
        for (int t(0); t < n_trees; t++){
            for (int f(0); f < n_features; f++){
                phi_f_b[i][t][f] /= size_background;
            }
        }
        bar.update();
    }
    return phi_f_b;
}



////// Wrapping the C++ functions with a C interface //////



extern "C"
int main_treeshap(int Nx, int Nz, int Nt, int d, int depth, double* foreground, double* background,
                  int* categorical_to_features, double* threshold_, double* value_, int* feature_, 
                  int* left_child_, int* right_child_, double* result) {
    
    // Load data instances
    Matrix<double> X_f = createMatrix<double>(Nx, d, foreground);
    Matrix<double> X_b = createMatrix<double>(Nz, d, background);

    // Load tree structure
    Matrix<double> threshold = createMatrix<double>(Nt, depth, threshold_);
    Matrix<double> value = createMatrix<double>(Nt, depth, value_);
    Matrix<int> feature = createMatrix<int>(Nt, depth, feature_);
    Matrix<int> left_child  = createMatrix<int>(Nt, depth, left_child_);
    Matrix<int> right_child = createMatrix<int>(Nt, depth, right_child_);

    // Precompute the SHAP weights
    int n_features = categorical_to_features[d-1] + 1;
    Matrix<double> W(n_features, vector<double> (n_features));
    compute_W(W);
    
    Tensor<double> phi = treeSHAP(X_f, X_b, categorical_to_features, 
                                    feature, left_child, right_child, 
                                    threshold, value, W);
    std::cout << std::endl;
    
    // Save the results
    int n_instances = phi.size();
    int n_trees = phi[0].size();
    for (int i(0); i < n_instances; i++){
        for (int j(0); j < n_trees; j++){
            for (int k(0); k < n_features; k++){
                result[i * n_features * n_trees + j * n_features + k] = phi[i][j][k];
            }
        }
    }
    return 0;
}
