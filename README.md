# VueICL: Entity-Aware Video Question Answering through In-Context Learning of Visual Annotations

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![GitHub stars](https://img.shields.io/github/stars/yourusername/VueICL?style=social)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Models](#models)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Introduction
VueICL is a framework designed for **Entity-Aware Video Question Answering** leveraging **In-Context Learning** of visual annotations. By integrating personalized visual markers with state-of-the-art Vision-Language Models (VLMs), VueICL enables efficient and accurate answering of entity-specific queries in long videos without the need for extensive model fine-tuning.

## Features
- **Two-Stage Pipeline**: Annotate video frames with visual markers and integrate these annotations into VLMs.
- **Entity-Aware Reasoning**: Enhanced capability to reference and differentiate multiple entities within a single conversation context.
- **Scalable and Efficient**: Avoids computational overhead associated with traditional fine-tuning methods.
- **Benchmark and Evaluation Scripts**: Provides a curated benchmark for entity-aware video question answering.

## Installation

### Prerequisites
- **Python 3.8+**
- **Conda** (for environment management)

### Steps

1. **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/VueICL.git
    cd VueICL
    ```

2. **Create a Conda Environment**
    ```bash
    conda create -n vueicl_env python=3.8
    conda activate vueicl_env
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

### Folder Structure

Ensure that your data directory follows the structure below:

```plaintext
raw_videos_characters/
├── video1/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── video1.mp4
│   └── ...
├── video2/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── 4.png
│   ├── video2.mp4
│   └── ...
└── ...
```

### Details
- **raw_videos_characters/**: Root directory containing all raw videos.
- **videoX/**: Subdirectories for each video, containing annotated frames.

### Data and Evaluation Scripts
We provide all necessary data and evaluation scripts in our [Google Drive](https://drive.google.com/drive/folders/1qUsLUFxwfwY2myfWlmWCqLwQw32ofSvz?usp=drive_link). Please download and place the `raw_videos_characters` folder in the root directory of the repository.

## Models

### LongVU
Download the LongVU model from the official repository:
- **GitHub Repository**: [LongVU GitHub](https://github.com/yourusername/LongVU)

### Qwen2-VL
Download the Qwen2-VL model from the official repository:
- **GitHub Repository**: [Qwen2-VL GitHub](https://github.com/yourusername/Qwen2-VL)

**Note**: Ensure that both models are correctly placed in the specified directories as per the instructions in their respective repositories.

## Usage

### Running the Framework
1. **Annotate Videos**
    Ensure that your videos are annotated with red bounding boxes and corresponding entity labels as per the folder structure.

2. **Execute the Inference Script**
    ```bash
    python inference.py --model LongVU --model_path path_to_longvu_model --data_path raw_videos_characters
    ```
    Replace `path_to_longvu_model` with the actual path to your LongVU model.

3. **Evaluate Performance**
    ```bash
    python evaluate.py --results results.json --benchmark_path path_to_benchmark
    ```
    Replace `path_to_benchmark` with the actual path to your benchmark dataset.

## Evaluation

Our evaluation is based on a curated benchmark specifically designed for entity-aware video question answering. The benchmark includes a set of closed-ended questions tailored to assess the model's ability to identify and reason about specific entities within video content.

### Metrics
- **Questions**: Number of questions answered correctly out of 100.
- **Videos**: Number of videos where all questions were answered correctly out of 22.

## Results

Refer to the [Results](#results) section in the paper for detailed performance metrics. Below is a summary of our findings:

| **Method**                  | **Questions** | **Videos** |
|-----------------------------|---------------|------------|
| LongVU Empty video          | 44/100        | 1/22       |
| LongVU No annotation        | 54/100        | 3/22       |
| Qwen2-VL                    | 49/100        | 4/22       |
| **VueICL long prompt**      | **67/100**    | **6/22**   |
| **VueICL short prompt**     | **68/100**    | **6/22**   |

*Figure 4*: Performance comparison graph of different methods on the entity-aware video understanding benchmark.

![Results Graph](path_to_results_graph_placeholder)  
*Figure 4*: Performance comparison graph of different methods on the entity-aware video understanding benchmark. VueICL methods significantly outperform existing baselines and state-of-the-art models in both question-solving accuracy and video-level comprehension.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the Repository**
2. **Create a Feature Branch**
    ```bash
    git checkout -b feature/YourFeature
    ```
3. **Commit Your Changes**
    ```bash
    git commit -m "Add Your Feature"
    ```
4. **Push to the Branch**
    ```bash
    git push origin feature/YourFeature
    ```
5. **Open a Pull Request**

Please ensure that your contributions adhere to the project's coding standards and include appropriate tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite it as follows:

```bibtex
@inproceedings{your2025vueicl,
  title={VueICL: Entity-Aware Video Question Answering through In-Context Learning of Visual Annotations},
  author={Levinson, Shahaf and Elgov, Ram and Benizri, Yonatan and Schwartz, Idan},
  booktitle={Proceedings of the 38th International Conference on Machine Learning (ICML)},
  year={2025},
}
```

Contact

For any questions or suggestions, please contact:
	•	Shahaf Levinson - shahafl@mail.tau.ac.il
	•	Ram Elgov - ramelgov@mail.tau.ac.il
	•	Yonatan Benizri - benizrilevi@mail.tau.ac.il
	•	Idan Schwartz - idan.schwartz@biu.ac.il

