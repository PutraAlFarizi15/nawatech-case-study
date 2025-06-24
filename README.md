# Nawatech - Machine Learning Engineer Case Study

This is the submission repository for the Nawatech Machine Learning Engineer case study.

## Repository Structure

* `/Case1_SentimentAnalysis`: Contains the notebook and data for the sentiment analysis task. 
* `/Case2_FAQ_Chatbot`: Contains the Streamlit chatbot application and its corresponding Dockerfile. 

---

### Main Documentation

**For all detailed explanations, analysis, visualizations, and model evaluation results, please refer to the presentation (PPT) file submitted as per the instructions.** 

---

### How to Run Case #2 (Chatbot)

First clone the repository

```bash
git clone https://github.com/PutraAlFarizi15/nawatech-case-study.git
```

You can run the chatbot application in two ways:

#### Option 1: Using Docker (Recommended as per Case Scenario)

[cite_start]This method uses containerization to run the application in an isolated environment, as requested in the case study task. 

1.  Navigate to the chatbot directory:
    ```bash
    cd Case2_FAQ_Chatbot
    ```

2.  Build the Docker image:
    ```bash
    docker build -t nawatech-chatbot .
    ```

3.  Run the Docker container: Don't forget to change the API key 

    ```bash
    docker run -p 8501:8501 -e OPENAI_API_KEY="your_api_key" nawatech-chatbot
    ```
4.  Open your browser and navigate to `http://localhost:8501` or `http://127.0.0.1:8501 `.

#### Option 2: Running Locally

This method runs the application directly on your machine using a Python virtual environment.

1.  Navigate to the chatbot directory:
    ```bash
    cd Case2_FAQ_Chatbot
    ```

2.  Create a virtual environment:
    ```bash
    python -m venv venv
    ```

3.  Activate the virtual environment:
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

5.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

6.  Open your browser and navigate to the local URL provided in your terminal.