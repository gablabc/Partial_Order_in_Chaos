#include <vector>
#include <iostream>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iomanip>
#include <stdexcept>
#include "progressbar.hpp"
#include "utils.hpp"



// Recursion function for treeSHAP
pair<double, double> recurse(int n,
                            vector<double> &x, vector<double> &z, 
                            int* I_map,
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
        return recurse(x_child, x, z, I_map, feature, child_left, child_right,
                            threshold, value, W, n_features, phi, in_SX, in_SZ);
    }

    // Senario 2: x and z go different ways and we have seen this feature i in I(S_X) U I(S_Z).
    // Hence we go down the correct edge to ensure that I(S_X) and I(S_Z) are kept disjoint
    if (in_SX[ I_map[current_feature] ] || in_SZ[ I_map[current_feature] ]){
        if (in_SX[ I_map[current_feature] ]){
            return recurse(x_child, x, z, I_map, feature, child_left, child_right,
                            threshold, value, W, n_features, phi, in_SX, in_SZ);
        }
        else{
            return recurse(z_child, x, z, I_map, feature, child_left, child_right,
                            threshold, value, W, n_features, phi, in_SX, in_SZ);
        }
    }

    // Scenario 3 : x and z go different ways and we have not yet seen this feature
    else {
        // Go to x's child
        in_SX[ I_map[current_feature] ]++;
        in_SX[n_features]++;
        pair<double, double> pairf = recurse(x_child, x, z, I_map, feature, child_left, child_right,
                                            threshold, value, W, n_features, phi, in_SX, in_SZ);
        in_SX[ I_map[current_feature] ]--;
        in_SX[n_features]--;

        // Go to z's child
        in_SZ[ I_map[current_feature] ]++;
        in_SZ[n_features]++;
        pair<double, double> pairb = recurse(z_child, x, z, I_map, feature, child_left, child_right,
                                            threshold, value, W, n_features, phi, in_SX, in_SZ);
        in_SZ[ I_map[current_feature] ]--;
        in_SZ[n_features]--;

        // Add contribution to the feature
        phi[ I_map[current_feature] ] += pairf.first - pairb.second;

        return make_pair(pairf.first + pairb.first, pairf.second + pairb.second);
    }
}



// Main function for Interventional TreeSHAP
Tensor<double> int_treeSHAP(Matrix<double> &X_f, 
                            Matrix<double> &X_b,
                            int* I_map, 
                            Matrix<int> &feature,
                            Matrix<int> &left_child,
                            Matrix<int> &right_child,
                            Matrix<double> &threshold,
                            Matrix<double> &value,
                            Matrix<double> &W)
    {
    // Setup
    int n_features = I_map[X_f[0].size()-1] + 1;
    int n_trees = feature.size();
    int Nz = X_b.size();
    int Nx = X_f.size();

    // Initialize the SHAP values to zero
    Tensor<double> phi_f_b(Nx, Matrix<double> (n_features, vector<double> (n_trees, 0)));
    progressbar bar(n_trees);
    // Iterate over all trees
    for (int t(0); t < n_trees; t++){
        // Iterate over all foreground instances
        for (int i(0); i < Nx; i++){
            // Iterate over all background instances
            for (int j(0); j < Nz; j++){
                // Last index is the size of the set
                vector<int> in_SX(n_features+1, 0);
                vector<int> in_SZ(n_features+1, 0);
                vector<double> phi(n_features, 0);

                // Start the recursion
                recurse(0, X_f[i], X_b[j], I_map, feature[t], left_child[t], right_child[t],
                            threshold[t], value[t], W, n_features, phi, in_SX, in_SZ);

                // Add the contribution of the tree and background instance
                for (int f(0); f < n_features; f++){
                    phi_f_b[i][f][t] += (phi[f] - phi_f_b[i][f][t]) / (j+1);
                }
            }
        }
        bar.update();
    }
    return phi_f_b;
}




////// Wrapping the C++ functions with a C interface //////



extern "C"
int main_int_treeshap(int Nx, int Nz, int Nt, int d, int depth, double* foreground, double* background,
                      int* I_map, double* threshold_, double* value_, int* feature_, 
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
    int n_features = I_map[d-1] + 1;
    Matrix<double> W(n_features, vector<double> (n_features));
    compute_W(W);
    
    Tensor<double> phi = int_treeSHAP(X_f, X_b, I_map, 
                                     feature, left_child, right_child, 
                                     threshold, value, W);
    std::cout << std::endl;
    
    // Save the results
    for (int i(0); i < Nx; i++){
        for (int j(0); j < n_features; j++){
            for (int k(0); k < Nt; k++){
                result[i*n_features*Nt + j*Nt + k] = phi[i][j][k];
            }
        }
    }
    return 0;
}
