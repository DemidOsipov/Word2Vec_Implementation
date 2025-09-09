# Word2Vec Implementation from Scratch

A comprehensive implementation of the Word2Vec algorithm using PyTorch, built from first principles with all key optimizations from the original paper.

## Overview

This project implements the skip-gram Word2Vec model with negative sampling, recreating the seminal work of Mikolov et al. (2013). Rather than using pre-built libraries, this implementation demonstrates the underlying mathematics and algorithmic components that make Word2Vec effective for learning word embeddings.

## Key Features

**Algorithm Optimizations**
- Subsampling of frequent words with theoretical probability distributions
- Negative sampling with 3/4 power smoothing for improved rare word representations  
- Hierarchical softmax alternative through efficient sampling strategies

**Implementation Details**
- Custom embedding layers with Xavier initialization
- Batch generation with dynamic negative sampling
- Learning rate scheduling with plateau detection
- Memory-efficient training without GPU requirements

**Evaluation & Visualization**
- Semantic similarity testing between related word pairs
- Nearest neighbor analysis using cosine similarity
- UMAP-based dimensionality reduction for embedding visualization
- Interactive Bokeh plots for exploring the learned embedding space

## Theory

The implementation maximizes the skip-gram objective with negative sampling:

$$\mathcal{L} = \log \sigma({\mathbf{v}'_{w_O}}^\top \mathbf{v}_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma({-\mathbf{v}'_{w_i}}^\top \mathbf{v}_{w_I}) \right]$$

Where frequent word subsampling follows:
$$P_\text{drop}(w_i)=1 - \sqrt{\frac{t}{f(w_i)}}$$

And negative sampling uses the 3/4 power distribution:
$$P_n(w) = \frac{U(w)^{3/4}}{Z}$$

## Results

The trained embeddings successfully capture semantic relationships, demonstrated through:
- Higher similarity scores for related word pairs (e.g., "iPhone" vs "Apple" > "iPhone" vs "Dell")
- Meaningful nearest neighbors in the embedding space
- Clear clustering of semantically related terms in visualization

## Usage

The notebook provides a complete pipeline from raw text preprocessing through final embedding evaluation. Key functions include `subsample_frequent_words()` for frequency balancing and `get_negative_sampling_prob()` for creating the negative sampling distribution.
