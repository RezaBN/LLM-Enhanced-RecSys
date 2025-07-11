# LLM-Enhanced Few-Shot Recommender System

This repository presents an implementation and analysis of a few-shot recommender system, building upon the methodology described in the paper ["Empowering Few-Shot Recommender Systems With Large Language Models-Enhanced Representations"](https://ieeexplore.ieee.org/document/10440582/). Our work explores the use of Large Language Models (LLMs) to enhance user and item representations for improved performance in scenarios with limited interaction data.

## Project Overview

The core idea is to leverage the power of LLMs to generate rich, contextual representations of users and items based on their explicit feedback (movie reviews). These enhanced representations are then integrated into traditional recommender models to tackle the "few-shot" challenge, where conventional systems struggle due to a scarcity of data.

While the original paper utilized ChatGPT for representation generation, our implementation adapts this approach using a "Free BERT method" due to resource constraints. We then evaluate the performance of these BERT-enhanced representations across two key recommendation tasks: interaction prediction and direct recommendation.

## Key Features

* **Few-Shot Scenario Simulation**: Implements a "leave-two-out" strategy for data splitting to simulate few-shot scenarios, mirroring the paper's approach.
* **LLM-Enhanced Representations**: Generates user and item representations from textual reviews using a BERT-based LLM.
* **Trainable Projection Layers**: Incorporates trainable projection layers to adapt LLM-generated embeddings to the recommender models.
* **Baseline Models**: Integrates established recommender baselines, including BPR-MF and NCF (NCF-MLP, NCF-Linear).
* **Dynamic Experiment Management**: Automatically creates unique folders for each experimental run based on hyperparameters, ensuring organized results.
* **Checkpointing & Resuming**: Supports checkpointing to save model states and allows for seamless training resumption.
* **Separated Result Files**: Saves results for recommendation and interaction prediction tasks into distinct, clearly named files for easy analysis.
* **Shared Caching**: Caches datasets and embeddings in a parent directory to optimize execution time across multiple runs.

## Experimental Setup

### Dataset

* **Original Paper**: Douban Chinese Moviedata-10M - [Link](https://m.douban.com/doulist/901995/).
* [cite_start]**Our Implementation**: Amazon Movie and TV dataset - [Link](https://jmcauley.ucsd.edu/data/amazon/).
    * **Preprocessing**: Similar preprocessing steps (e.g., "leave-two-out" strategy, sampling 2000 users, max 10 reviews per user) were applied to simulate the few-shot scenario as described in the paper.

### LLM Approach for Representation Generation

* **Original Paper**: ChatGPT (gpt-3.5-turbo).
    * **Capabilities Highlighted**: Generative and logical reasoning, association, and inference.
* **Our Implementation**: "Free BERT method" (specifically `bert-base-uncased` for representation generation, and `sentence-transformers/all-MiniLM-L6-v2` for basic embeddings).
    * **Rationale for Change**: Resource limitations.
    * **Implication**: BERT focuses on contextual understanding rather than the generative and associative capabilities emphasized for ChatGPT ([More Information](https://huggingface.co/docs/transformers/en/model_doc/bert)).

### Embedding Model

* **Original Paper**: ChatGPT and MacBERT.
* **Our Implementation**: `sentence-transformers/all-MiniLM-L6-v2` for basic embeddings and `bert-base-uncased` for LLM-enhanced embeddings.

### Models and Metrics

We evaluated our models on two primary tasks:

1.  **Interaction Prediction**: Predicting whether a user will engage in an interaction with a specific item.
    * **Baselines**: MLP (as described in the paper).
    * **Metrics**: Accuracy, Precision, F1 Score.
2.  **Direct Recommendation**: Recommending items most likely to align with a user's preferences.
    * **Baselines**: BPR-MF, NCF-MLP.
    * **Metrics**: Top-K Hit Ratio (HR@K), Top-K Mean Reciprocal Rank (MRR@K).
        * **Note**: The paper primarily reports HR@100 and MRR@100 for BPR-MF, and HR@10 and MRR@10 for NCF models. Our `K_METRICS` are set to 10.


## Repository Contents

* `Advanced_LLM_Enhanced_Recommender_System.ipynb`: Jupyter Notebook containing the full experimental pipeline.
* `advanced_llm_enhanced_recommender_system.py`: Python script version of the experimental pipeline.
* `Report.pdf`: Detailed report of experimental findings and comparison with the referenced paper.
* `LLM_RecSys_Experiments_Parent/`: (Created upon first run) This directory stores all experiment outputs, including:
    * Run-specific folders (e.g., `run_llm-bert-base-uncased_pdim128_lr0.001_e200_k10_u1000_rev10_b256/`) based on hyperparameters for dynamic experiment management.
    * `checkpoints/`: Model and optimizer states for resuming training.
    * `history_recommendation.xlsx`: Training and validation history for recommendation tasks.
    * `history_interaction_prediction.xlsx`: Training and validation history for interaction prediction tasks.
    * `final_results_recommendation.xlsx`: Final test results for recommendation tasks.
    * `final_results_interaction_prediction.xlsx`: Final test results for interaction prediction tasks.
    * `learning_curves_recommendation.png`: Plots of learning curves for recommendation.
    * `learning_curves_interaction.png`: Plots of learning curves for interaction prediction.
    * `experiment_log.log`: Comprehensive logging of the experiment run.
    * `sampled_1000_user_dataset.csv`: Sampled dataset used for experiments (cached).
    * `bert_embeddings_bert-base-uncased_1000_users_cache.json`: BERT embeddings cache (cached) - **Was not included because of its large size**.

## Getting Started

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd llm-enhanced-recsys
    ```
2.  **Mount Google Drive (if using Colab)**: The script automatically attempts to mount `/content/drive` to save results and caches.
3.  **Install Dependencies**: The script will automatically install necessary libraries (e.g., `sentence-transformers`, `scikit-learn`, `transformers[torch]`, `matplotlib`) if not found.
4.  **Prepare Dataset**: Ensure the `Movies_and_TV_5.json` dataset is placed in `/content/drive/MyDrive/Datasets/` (or adjust `real_dataset` path in `get_experiment_paths()` if running locally). You can download this from [Amazon Customer Reviews Dataset](https://nijianmo.github.io/amazon/index.html).
5.  **Run the Experiment**:
    * **Jupyter Notebook**: Open `Advanced_LLM_Enhanced_Recommender_System.ipynb` in Google Colab or your local Jupyter environment and run all cells.
    * **Python Script**: Execute `python advanced_llm_enhanced_recommender_system.py` from your terminal.

The script will automatically set up directories, process data, generate embeddings, train models, and save all results and plots.

## Hyperparameters

You can modify the following hyperparameters in the script:

```python
SBERT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2' # For basic embeddings
LLM_MODEL_NAME = 'bert-base-uncased' # For LLM-enhanced embeddings
EPOCHS = 150
LEARNING_RATE = 0.001
BATCH_SIZE = 256
K_METRICS = 10 # Top-K for recommendation metrics
PROJECTION_DIM = 128
NUM_USERS_TO_SAMPLE = 2000
MAX_REVIEWS_PER_USER = 10
```

Feel free to experiment with different configurations to observe their impact on performance.

## License
This project is licensed under the Apache License, Version 2.0. See the LICENSE file for details.

## Reference paper

Wang, Z., 2024. Empowering few-shot recommender systems with large language models-enhanced representations. IEEE Access. (https://ieeexplore.ieee.org/document/10440582)
