# Face Mask Detector 😷

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

**Real-time Face Mask Detection using OpenCV and Machine Learning**

---

## Table of Contents
- [About the Project](#about-the-project)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## About the Project
This project detects whether a person is wearing a face mask or not in real-time using a webcam.  
It uses **Haarcascades** for face detection and a pre-trained model for mask classification.  

**Why this project:** Helps monitor mask compliance in public areas and workplaces.

---

## Features
- Real-time face detection  
- Mask / No Mask classification  
- Webcam integration  
- Lightweight and easy to use  


## Installation
1. Clone the repository:  
git clone https://github.com/hemaamurthy/face-mask-detector.git
cd face-mask-detector
2.(Recommended) Create a virtual environment:
python -m venv .venv
3.Activate the virtual environment:
Windows:
.venv\Scripts\activate
Mac/Linux:
source .venv/bin/activate
4.Install dependencies:
pip install -r requirements.txt
## Usage
Run the main script:
python app.py
*Your webcam will open and detect faces with or without masks in real-time.
*(If your main script is named differently, replace app.py with your script name.)
## Folder Structure
```
face-mask-detector/
│
├── .gitignore
├── README.md
├── requirements.txt
├── app.py <-- main script for detection
├── check_model.py <-- helper script to verify model
├── models/
│ └── mask_detector_model.h5
├── images/
│ ├── with_mask/
│ └── without_mask/
└── utils/
└── helpers.py
 ```
 (Keep or remove files/folders if your project is slightly different.)
## Technologies Used
* Python 3.x
* OpenCV
* TensorFlow / Keras
* NumPy
  (Add any other libraries you used in your project.)
## Future Enhancements
* Add real-time alerts for no-mask detection
* Deploy as a web application using Flask or Streamlit
* Replace Haarcascades with YOLO/SSD for faster detection
## Contributing 
 Contributions are welcome! Please create a pull request or open an issue.
## License
This project is licensed under the [MIT License](LICENSE).

