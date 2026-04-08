# Aquapor 💧

**Aquapor** is an interactive, Streamlit-based dashboard designed to forecast water levels based on rainfall and temperature trends. It empowers users to analyze water resources, observe historical trends, and prepare proactive steps using actionable alerts (e.g., Drought and Flood risks).

## 🌟 Key Features

*   **🔮 Forecasts & Alerts**: Uses historical data to predict water levels for the next 7 days, providing real-time alerts if predicted levels breach safe thresholds.
*   **📊 Data Analysis**: Visualize interactive scatter plots and correlation heatmaps to understand the relationships between rainfall, temperature, and water levels.
*   **💡 Insights Engine**: Auto-generates actionable decision recommendations based on recent trend lines and current water stress indices.
*   **🎛️ "What-If" Analysis**: Adjust hypothetical rainfall and temperature inputs using sliders to instantly predict the simulated water level.
*   **📂 Custom Data Upload**: Test and interact with your own data by seamlessly uploading local CSV files.

## 🛠️ Tech Stack

*   **Python**: Core programming language.
*   **[Streamlit](https://streamlit.io/)**: Provides the frontend web interface.
*   **[Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)**: Robust and efficient data manipulation.
*   **[Scikit-learn](https://scikit-learn.org/)**: Performs linear regression modeling and data imputation.
*   **[Altair](https://altair-viz.github.io/)**: Generates clean, fast, and interactive chart visualizations.

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed on your machine. You will need to install the project dependencies before running the application.

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <repository-directory>
   ```

2. **Install the requirements:**
   ```bash
   pip install -r requirements.txt
   ```
   *(If you don't have the explicit requirements, install `streamlit`, `pandas`, `numpy`, `scikit-learn`, and `altair`.)*

### Running the App

Start your local Streamlit server by running the following command from the root of the project:

```bash
streamlit run app.py
```

The application will automatically launch and open your default web browser at `http://localhost:8501`. 

## 📁 Project Structure

*   `app.py`: The main Streamlit dashboard file managing the entire UI layout and user interaction.
*   `model.py`: Implements machine learning workflows including splitting, training, evaluation, and forecasting logic.
*   `data.csv`: Sample historical data to demonstrate out-of-the-box functionality.
*   `requirements.txt`: Python package dependencies necessary for execution.
