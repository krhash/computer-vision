/**
 * @file    EmbeddingPlot.h
 * @brief   2D PCA projection and scatter plot of CNN embeddings.
 *
 *          Extension: Display the embeddings of objects in a 2D plot by
 *          computing the top 2 eigenvectors of the embedding covariance
 *          matrix and projecting all embeddings onto them.
 *
 *          Each label gets a unique color. Well-separated clusters indicate
 *          the CNN embedding space distinguishes objects effectively.
 *
 *          PCA is computed from scratch using cv::PCACompute.
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include "Embedding.h"

/**
 * @brief Compute 2D PCA projection of all embeddings and render scatter plot.
 *
 * @param db        Embedding DB containing labeled 512-dim vectors.
 * @param plotSize  Output image size in pixels (square).
 * @param dst       Output BGR scatter plot image.
 */
void renderEmbeddingPlot(const EmbeddingDB& db,
                          int plotSize,
                          cv::Mat& dst);