# Smart Disease & Health AI

![Streamlit App](https://raw.githubusercontent.com/ankit7221/smart-disease-ai/main/assets) ## Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Live Demo](#live-demo)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running Locally](#running-locally)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About the Project

This project, **Smart Disease & Health AI**, is a comprehensive web application built using Streamlit that leverages machine learning and generative AI to provide preliminary disease predictions based on symptoms and offer general health advice. It aims to be a personal health companion, helping users understand potential health conditions and receive AI-powered guidance.

**Key functionalities include:**
* **Symptom-based Disease Prediction:** Predicts potential diseases using machine learning models (Naive Bayes and Random Forest) based on user-selected symptoms, their severity, and duration.
* **AI Health Advisor Chat:** Provides general health insights and answers questions related to predicted diseases or general health using the Google Gemini API.
* **User Authentication:** Allows users to log in and manage their prediction history.
* **Prediction History Tracking:** Saves and displays past prediction results for logged-in users.
* **Multilingual Support:** (If applicable, mention specifically, e.g., "Currently supports English and Hindi for predictions and interface.")

## Features

* Intuitive user interface for symptom input.
* Predictions from multiple machine learning models.
* Confidence scores for predictions.
* Interactive AI chat for health advice.
* User registration and login system.
* Personalized prediction history.
* Responsive design.

## Live Demo

Experience the application live on Streamlit Community Cloud:
[https://smart-disease-ai-9rtue2ftzeyypmt8k65qvo.streamlit.app/] ## Technologies Used

* **Python:** Primary programming language.
* **Streamlit:** For building the interactive web application.
* **scikit-learn:** For machine learning models (Naive Bayes, Random Forest).
* **joblib:** For saving and loading machine learning models.
* **Google Generative AI (Gemini API):** For the AI Health Advisor Chat.
* **pandas:** For data manipulation.
* **numpy:** For numerical operations.
* **Git & GitHub:** For version control and hosting the codebase.
* **Streamlit Community Cloud:** For deployment.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)
* A Google Gemini API key (for the AI Health Advisor feature)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ankit7221/smart-disease-ai.git](https://github.com/ankit7221/smart-disease-ai.git)
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd smart-disease-ai
    ```
3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/Scripts/activate # On Windows
    # source venv/bin/activate    # On macOS/Linux
    ```
4.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Set up your Google Gemini API Key:**
    Create a file named `.streamlit/secrets.toml` in your project root.
    Add your API key to this file:
    ```toml
    GOOGLE_API_KEY = "your_gemini_api_key_here"
    ```
    **Note:** For local development, you might use a `.env` file or direct environment variables. For Streamlit Cloud deployment, use Streamlit secrets.

### Running Locally

1.  **Generate the machine learning models:**
    Run the training script to create the `.pkl` model files:
    ```bash
    python train_model.py
    ```
2.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    Your application will open in your default web browser at `http://localhost:8501`.

## Usage

1.  **Select Symptoms:** On the left sidebar, choose the symptoms you are currently experiencing from the dropdown.
2.  **Enter Details:** For each selected symptom, adjust its severity and duration using the sliders.
3.  **Predict Disease:** Click the "Predict Disease" button to get predictions from Naive Bayes and Random Forest models.
4.  **AI Health Advisor:** After a prediction, the AI Health Advisor chat will be enabled. You can ask questions related to the predicted disease or general health.
5.  **Login/Sign Up:** Use the login/sign-up options to access your personalized prediction history.

## Project Structure
smart-disease-ai/
├── .streamlit/             # Streamlit specific configurations (e.g., secrets.toml, config.toml)
│   └── secrets.toml        # For Gemini API key on Streamlit Cloud
├── data/                   # Contains dataset used for training
│   └── dataset.csv
├── model/                  # Trained machine learning models (generated by train_model.py)
│   ├── naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   └── symptom_binarizer.pkl
├── utils/                  # Utility functions (e.g., language processing, voice-to-text)
│   ├── language_utils.py
│   └── voice_to_text.py
├── app.py                  # Main Streamlit application file
├── requirements.txt        # Python dependencies
├── style.css               # Custom CSS for styling the app
└── train_model.py          # Script to train and save the ML models

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:
1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

Distributed under the [ MIT License]. See `LICENSE` for more information.

## Contact

[ankit7221] - [ankitchoudhary2451@gmail.com]
Project Link: [https://github.com/ankit7221/smart-disease-ai](https://github.com/ankit7221/smart-disease-ai)
