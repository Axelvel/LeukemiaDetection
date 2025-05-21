# Leukemia Detection

## Description
A machine learning project to detect leukemia from blood cell images.

## Installation

### Environment Setup
To set up the environment, install the required packages using the `environment.txt` file:
```bash
pip install -r environment.txt
```

### Dataset Download
Create an API Token from [Kaggle](https://www.kaggle.com/<your_username>/account) and place the `kaggle.json` file in your `~/.kaggle/` folder (for Linux/macOS) or `C:\Users\<YourUserName>\.kaggle\` (for Windows).

Then, download and extract the dataset using the following commands:
```bash
# Replace <dataset-identifier> with the actual dataset identifier from Kaggle
kaggle datasets download -d <dataset-identifier>
# Replace <dataset-zip-file> with the name of the downloaded zip file
unzip <dataset-zip-file>.zip -d ./data
# Ensure the data is extracted to a directory named 'data' in the project root
```
*Note: You will need to replace `<dataset-identifier>` and `<dataset-zip-file>` with the actual values for the C-NMC dataset. The dataset should be extracted into a folder named `data` in the root of the project directory.*

## How to Run
To run the project, navigate to the project's root directory in your terminal and execute the following command:
```bash
python main.py
```
This will start the leukemia detection process, which may include training the model (if not already trained) and then performing predictions, depending on the implementation in `main.py`.

## Dataset
The project uses the C-NMC Leukemia dataset, which contains images of blood cells. The dataset is divided into training and testing sets, with images labeled as either leukemia positive or negative.

## Model
The model used in this project is a `LeukemiaDetector`.

- **Input:** The model takes blood cell images as input.
- **Output:** The model outputs a prediction indicating whether leukemia is detected in the image.

## Training
The model is trained using the C-NMC Leukemia dataset. The training process involves feeding the model with labeled images and adjusting its parameters to minimize the prediction error.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/your-feature-name`).
6. Open a Pull Request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.