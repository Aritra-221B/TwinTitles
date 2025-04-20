# Twin Titles: Research Title Similarity Checker

A web application for automatically verifying new research paper title submissions by checking for similarities with existing titles. Built for a one-day hackathon.

## Features

1. **Title Similarity Analysis**: Compares submitted research titles with a database of existing titles.
2. **Semantic Similarity Graph**: Visual representation of the semantic relationships between titles.
3. **Plagiarism Heatmap**: Highlights similar words between titles to identify potential plagiarism.
4. **Title Enhancement Suggestions**: Provides recommendations to improve titles.
5. **Abstract Generation**: Creates a short abstract based on the title.
6. **Uniqueness Score**: Visual pie chart showing the uniqueness percentage.
7. **PDF Report Generation**: Creates a downloadable PDF analysis report.

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript, Bootstrap, Chart.js
- **Backend**: Flask (Python)
- **NLP Model**: Sentence-Transformers
- **Data Visualization**: Matplotlib, NetworkX
- **PDF Generation**: ReportLab

## Project Structure

```
Twin_titles/
├── app.py                     # Main Flask application
├── model_folder/              # Model files and prediction logic
│   ├── predict.py             # Title prediction functionality
│   ├── titles.json            # Database of existing titles
│   ├── title_embeddings.npy   # Pre-computed embeddings for titles
│   └── ...                    # Other model files
├── static/                    # Static assets
│   ├── css/                   # CSS stylesheets
│   ├── js/                    # JavaScript files
│   └── img/                   # Images and generated graph visualizations
├── templates/                 # HTML templates
│   └── index.html             # Main application template
├── temp/                      # Temporary files for PDF generation
└── requirements.txt           # Python dependencies
```

## Setup and Installation

1. **Clone the Repository**
   ```
   git clone <repository-url>
   cd Twin_titles
   ```

2. **Create a Virtual Environment**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```
   python app.py
   ```

5. **Access the Application**
   Open your browser and navigate to http://127.0.0.1:5000/

## Usage

1. Enter your research title in the provided text area.
2. Click "Analyze Title" to process the title.
3. View the analysis results including similarity score, semantic graph, and enhancement suggestions.
4. Download a comprehensive PDF report by clicking the "Download Report" button.

## Future Enhancements

- Integration with academic databases for more extensive title comparison
- User authentication and history tracking
- API endpoints for integration with other systems
- Advanced language model for better abstract generation
- Multi-language support

## Credits

- NLP Model: Sentence-BERT / Sentence Transformers
- Frontend Framework: Bootstrap 5
- Visualization: Chart.js and matplotlib
- PDF Generation: ReportLab
