from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, session
import os
import json
import numpy as np
from model_folder.predict import find_top_similar_titles
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid thread issues
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import base64
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import networkx as nx
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
import textwrap
import uuid
import time
import re
import traceback
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
# Configure Flask JSON handling
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.json_encoder = NumpyEncoder
app.secret_key = 'twin_titles_secret_key' # Set a secret key for session management

# Database setup
DB_PATH = 'twin_titles.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        status TEXT NOT NULL,
        similarity_index REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database tables
init_db()

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# Get model path
model_path = os.path.abspath('model_folder')

# Load pre-trained model instead of using local folder directly
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load titles and embeddings
with open(os.path.join(model_path, "titles.json"), "r") as f:
    titles = json.load(f)
title_embeddings = np.load(os.path.join(model_path, "title_embeddings.npy"))

# Create temp directory if it doesn't exist
if not os.path.exists('temp'):
    os.makedirs('temp')
    print("Created temp directory for PDF reports")

# Create img directory if it doesn't exist
if not os.path.exists('static/img'):
    os.makedirs('static/img')
    print("Created image directory for charts and graphs")

@app.route('/')
def landing():
    # If user is already logged in, redirect to dashboard
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('landing.html')

@app.route('/dashboard')
def index():
    user = None
    if 'user_id' in session:
        # Fetch user details from database
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, email FROM users WHERE id = ?", (session['user_id'],))
        user = cursor.fetchone()
        conn.close()
        return render_template('index.html', user=user)
    else:
        # Redirect to landing page if user is not logged in
        return redirect(url_for('landing'))

# User Authentication Routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate form data
        if not username or not email or not password:
            flash('All fields are required', 'danger')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
        
        try:
            # Check if username or email already exists
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, email))
            existing_user = cursor.fetchone()
            
            if existing_user:
                conn.close()
                flash('Username or email already exists', 'danger')
                return render_template('register.html')
            
            # Create new user
            password_hash = generate_password_hash(password)
            cursor.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                          (username, email, password_hash))
            conn.commit()
            
            # Get the user ID for session
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            conn.close()
            
            if user:
                # Clear any existing session data
                session.clear()
                session['user_id'] = user[0]
                # For debugging
                print(f"Session data after registration: {session}")
                
                flash('Registration successful! You are now logged in.', 'success')
                return redirect(url_for('index'))
            else:
                flash('Registration failed. Please try again.', 'danger')
        except Exception as e:
            print(f"Registration error: {str(e)}")
            flash('An error occurred during registration. Please try again.', 'danger')
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Validate form data
        if not username or not password:
            flash('Both username and password are required', 'danger')
            return render_template('login.html')
        
        try:
            # Check if user exists
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            conn.close()
            
            if user and check_password_hash(user['password_hash'], password):
                # Set session data
                session.clear()  # Clear any existing session data
                session['user_id'] = user['id']
                # For debugging, print the session
                print(f"Session data after login: {session}")
                
                flash('Login successful!', 'success')
                
                next_page = request.args.get('next')
                if next_page:
                    return redirect(next_page)
                return redirect(url_for('index'))
            else:
                flash('Invalid username or password', 'danger')
        except Exception as e:
            print(f"Login error: {str(e)}")
            flash('An error occurred during login. Please try again.', 'danger')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

def format_db_row(row):
    """Convert a SQLite Row to a dictionary with properly formatted dates."""
    if not row:
        return None
        
    # Convert to dict
    row_dict = dict(row)
    
    # Format created_at if exists
    if 'created_at' in row_dict and row_dict['created_at']:
        try:
            # Handle SQLite date string format
            date_str = row_dict['created_at']
            # Simply display the date string as is, or format if needed
            if 'T' in date_str:  # ISO format
                date_parts = date_str.split('T')[0].split('-')
                if len(date_parts) == 3:
                    row_dict['created_at'] = f"{date_parts[1]}/{date_parts[2]}/{date_parts[0]}"
        except Exception as e:
            print(f"Error formatting date: {e}")
    
    return row_dict

@app.route('/profile')
@login_required
def profile():
    try:
        print(f"Profile route accessed. User ID in session: {session.get('user_id')}")
        conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get user info
        cursor.execute("SELECT * FROM users WHERE id = ?", (session['user_id'],))
        user_row = cursor.fetchone()
        
        if not user_row:
            # User not found in database despite being in session
            print(f"User ID {session.get('user_id')} not found in database")
            session.pop('user_id', None)
            flash('Session expired. Please log in again.', 'warning')
            return redirect(url_for('login'))
        
        # Format user data
        user = format_db_row(user_row)
        print(f"User found: {user['username']}")
        
        # Get user's search history
        cursor.execute("""
            SELECT id, title, status, similarity_index, created_at 
            FROM user_history 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT 10
        """, (session['user_id'],))
        history_rows = cursor.fetchall() or []
        
        # Format history data
        history = [format_db_row(row) for row in history_rows]
        print(f"User history retrieved: {len(history)} records")
        
        conn.close()
        
        # Check if profile.html exists
        template_path = os.path.join(os.path.dirname(__file__), 'templates', 'profile.html')
        if not os.path.exists(template_path):
            print(f"Profile template not found at: {template_path}")
            flash('Profile template not found.', 'danger')
            return redirect(url_for('index'))
            
        print("Rendering profile template")
        return render_template('profile.html', user=user, history=history)
    except Exception as e:
        # Log the error
        print(f"Error in profile route: {str(e)}")
        print(traceback.format_exc())
        flash('An error occurred while loading your profile. Please try again later.', 'danger')
        return redirect(url_for('index'))

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        title = request.form.get('title', '')
        
        if not title or len(title.strip()) == 0:
            return jsonify({"error": "No title provided"}), 400
        
        # Get prediction from model
        result = find_top_similar_titles(title)
        
        # Make sure result has the expected structure
        if not isinstance(result, dict) or 'label' not in result or 'top_matches' not in result:
            return jsonify({"error": "Invalid result format from model"}), 500
        
        # Convert any numpy float32 to regular float for JSON serialization
        top_matches = []
        for match in result['top_matches']:
            if len(match) == 2:
                top_matches.append((str(match[0]), float(match[1])))
        
        # Replace the original with converted values
        result['top_matches'] = top_matches
        
        # Generate a graph visualization
        graph_path = generate_similarity_graph(title, result['top_matches'])
        
        # Generate a heatmap for plagiarism detection
        heatmap_data = generate_plagiarism_heatmap(title, result['top_matches'][0][0])
        
        # Generate suggestions for title enhancement
        suggestions = generate_title_suggestions(title)
        
        # Generate abstract
        abstract = generate_abstract(title)
        
        # Generate uniqueness score data for pie chart
        uniqueness_data = {
            "unique": 100 - int(result['top_matches'][0][1] * 100),
            "similar": int(result['top_matches'][0][1] * 100)
        }
        
        # Find common words between the user's title and similar titles
        common_words_data = find_common_words(title, result['top_matches'])
        
        # Generate a temporary PDF report
        try:
            pdf_path = generate_pdf_report(title, result, graph_path, heatmap_data, suggestions, abstract, uniqueness_data, common_words_data)
            # If error report is returned, generate a fallback PDF
            if pdf_path == "error_report.pdf":
                pdf_path = generate_fallback_pdf(title, result['top_matches'])
        except Exception as pdf_error:
            print(f"Error in PDF generation: {str(pdf_error)}")
            pdf_path = generate_fallback_pdf(title, result['top_matches'])
        
        # If user is logged in, save the search to history
        if 'user_id' in session:
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO user_history (user_id, title, status, similarity_index) VALUES (?, ?, ?, ?)",
                    (session['user_id'], title, result['label'], float(result['top_matches'][0][1]))
                )
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Error saving user history: {str(e)}")
        
        # Prepare final JSON response manually to avoid serialization issues
        response_data = {
            "status": result['label'],
            "similarity_index": float(result['top_matches'][0][1]),
            "top_matches": result['top_matches'],
            "graph_path": graph_path,
            "heatmap_data": {
                'words1': heatmap_data['words1'],
                'words2': heatmap_data['words2'],
                'similar_pairs': [
                    {
                        'word1': p['word1'],
                        'word2': p['word2'],
                        'similarity': float(p['similarity']),
                        'index1': int(p['index1']),
                        'index2': int(p['index2'])
                    }
                    for p in heatmap_data['similar_pairs']
                ],
                'matrix': [[float(cell) for cell in row] for row in heatmap_data['matrix']]
            },
            "suggestions": suggestions,
            "abstract": abstract,
            "uniqueness_data": uniqueness_data,
            "pdf_path": pdf_path,
            "common_words_data": common_words_data
        }
        
        return app.json.dumps(response_data)
    
    except Exception as e:
        error_msg = f"Error during analysis: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({"error": error_msg}), 500

@app.route('/get_pdf/<filename>')
def get_pdf(filename):
    try:
        # Ensure filename is safe and doesn't contain path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({"error": "Invalid filename"}), 400
            
        file_path = os.path.join('temp', filename)
        
        # Debug output
        print(f"Attempting to serve PDF: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")
        
        if not os.path.exists(file_path):
            # Return a specific error for better debugging
            if filename == "error_report.pdf":
                return jsonify({"error": "PDF generation failed. Check server logs for details."}), 500
            return jsonify({"error": "PDF file not found"}), 404
            
        # Send the file with explicit MIME type
        return send_file(file_path, mimetype='application/pdf', as_attachment=True, 
                         download_name=f"TwinTitles_Analysis_{int(time.time())}.pdf")
    except Exception as e:
        print(f"Error serving PDF: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Error serving PDF: {str(e)}"}), 500

def find_common_words(user_title, similar_titles):
    # Prepare user title words
    user_words = re.findall(r'\b\w+\b', user_title.lower())
    
    # Find common words for each similar title
    result = []
    for title, similarity in similar_titles:
        title_words = re.findall(r'\b\w+\b', str(title).lower())
        common = list(set(user_words).intersection(set(title_words)))
        
        # Count occurrences of each common word in both titles
        common_details = []
        for word in common:
            if len(word) > 2:  # Ignore very short words
                common_details.append({
                    'word': word,
                    'user_count': user_words.count(word),
                    'match_count': title_words.count(word)
                })
        
        result.append({
            'title': title,
            'common_words': common_details
        })
    
    return result

def generate_similarity_graph(title, top_matches):
    try:
        # Create a graph
        G = nx.Graph()
        
        # Add nodes
        G.add_node(title, size=3000, color='red')
        for match, similarity in top_matches:
            # Size based on similarity
            node_size = 1500 + (similarity * 1500)
            # Node color based on similarity level
            if similarity > 0.95:
                node_color = 'orange'  # Duplicate - very high similarity
            elif similarity > 0.8:
                node_color = 'yellow'  # Similar
            else:
                node_color = 'lightblue'  # More distinct
                
            G.add_node(match, size=node_size, color=node_color)
            
            # Edge weight and length inversely proportional to similarity
            # Higher similarity = shorter edge, thicker connection
            edge_weight = similarity * 5  # Line thickness
            # Edge length inversely proportional to similarity
            # Higher similarity (closer to 1) means shorter edge (closer to 0.1)
            # Lower similarity (closer to 0) means longer edge (closer to 2)
            edge_length = 2 - (similarity * 1.9)
            
            G.add_edge(title, match, weight=edge_weight, length=edge_length)
        
        # Set positions using spring layout with custom edge lengths
        edge_lengths = {(u, v): G[u][v]['length'] for u, v in G.edges()}
        pos = nx.spring_layout(G, weight='length', iterations=100, k=0.5)
        
        # Create figure with white background
        plt.figure(figsize=(12, 10), facecolor='white')
        
        # Get node attributes
        node_sizes = [G.nodes[node]['size'] for node in G.nodes()]
        node_colors = [G.nodes[node]['color'] for node in G.nodes()]
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        
        # Draw the graph
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, edge_color='gray')
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
        
        # Add labels with custom positions
        label_pos = {k: (v[0], v[1] + 0.05) for k, v in pos.items()}
        
        # Create custom labels with line wrapping
        labels = {}
        for node in G.nodes():
            wrapped_text = textwrap.fill(str(node), 30)
            # Highlight the main title
            if node == title:
                labels[node] = wrapped_text
            else:
                # Add similarity percentage to similar titles
                for match_title, similarity in top_matches:
                    if node == match_title:
                        sim_percent = f"{similarity:.2f}"
                        labels[node] = f"{wrapped_text}\n({sim_percent})"
        
        nx.draw_networkx_labels(G, label_pos, labels=labels, font_size=9, font_weight='bold')
        
        # Add title and legend
        plt.title("Semantic Similarity Network", fontsize=16, fontweight='bold')
        
        # Create a legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='Your Title'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=15, label='High Similarity (>95%)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=15, label='Medium Similarity (>80%)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=15, label='Low Similarity')
        ]
        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        
        # Set tight layout and remove axis
        plt.tight_layout()
        plt.axis('off')
        
        # Save to a temporary file
        filename = f"graph_{int(time.time())}.png"
        filepath = f"static/img/{filename}"
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close()
        
        return filename
    except Exception as e:
        print(f"Error generating similarity graph: {str(e)}")
        print(traceback.format_exc())
        # Return a placeholder if there's an error
        return "placeholder.png"

def generate_plagiarism_heatmap(title1, title2):
    try:
        # Tokenize titles into words
        words1 = str(title1).lower().split()
        words2 = str(title2).lower().split()
        
        # Create embeddings for each word
        embeddings1 = [model.encode([word])[0] for word in words1]
        embeddings2 = [model.encode([word])[0] for word in words2]
        
        # Calculate similarity matrix
        similarity_matrix = []
        for emb1 in embeddings1:
            row = []
            for emb2 in embeddings2:
                sim = float(cosine_similarity([emb1], [emb2])[0][0])
                row.append(sim)
            similarity_matrix.append(row)
        
        # Find highly similar word pairs (similarity > 0.8)
        similar_pairs = []
        for i, word1 in enumerate(words1):
            for j, word2 in enumerate(words2):
                if similarity_matrix[i][j] > 0.8:
                    similar_pairs.append({
                        'word1': word1,
                        'word2': word2,
                        'similarity': float(similarity_matrix[i][j]),
                        'index1': i,
                        'index2': j
                    })
        
        return {
            'words1': words1,
            'words2': words2,
            'matrix': similarity_matrix,
            'similar_pairs': similar_pairs
        }
    except Exception as e:
        print(f"Error generating heatmap: {str(e)}")
        print(traceback.format_exc())
        # Return placeholder data if there's an error
        return {
            'words1': [],
            'words2': [],
            'matrix': [],
            'similar_pairs': []
        }

def generate_title_suggestions(title):
    # Simple suggestions based on title enhancement
    suggestions = [
        f"Consider adding specificity: \"{title} - A Comprehensive Review\"",
        f"For empirical studies: \"An Empirical Analysis of {title}\"",
        f"More concise alternative: \"{' '.join(title.split()[:5])}...\""
    ]
    return suggestions

def generate_abstract(title):
    # Generate a simple abstract based on the title
    abstract = f"This research explores {title.lower()}. The study examines various aspects of this topic, including methodologies, key findings, and implications for future research. Through extensive analysis, we provide insights into the underlying mechanisms and highlight potential applications in real-world scenarios."
    return abstract

def generate_pdf_report(title, result, graph_path, heatmap_data, suggestions, abstract, uniqueness_data, common_words_data):
    try:
        # Ensure temp directory exists
        if not os.path.exists('temp'):
            os.makedirs('temp')
            print("Created temp directory for PDF reports")
            
        # Create a unique filename
        filename = f"report_{uuid.uuid4().hex}.pdf"
        filepath = os.path.abspath(f"temp/{filename}")
        
        print(f"Generating PDF report at: {filepath}")
        
        try:
            # Create PDF with more professional settings
            doc = SimpleDocTemplate(
                filepath, 
                pagesize=letter,
                rightMargin=50,
                leftMargin=50,
                topMargin=50,
                bottomMargin=50
            )
            
            # Get standard styles and create custom ones
            styles = getSampleStyleSheet()
            
            # Define color scheme for a cohesive design
            primary_color = colors.navy
            secondary_color = colors.dodgerblue
            accent_color = colors.orange
            background_color = colors.whitesmoke
            border_color = colors.lightgrey
            highlight_color = colors.yellow
            
            # Enhanced styles with the color scheme
            title_style = ParagraphStyle(
                'Title',
                parent=styles['Title'],
                fontSize=24,
                fontName='Helvetica-Bold',
                alignment=1,  # Center alignment
                spaceAfter=5,
                textColor=primary_color,
                leading=28
            )
            
            subtitle_style = ParagraphStyle(
                'Subtitle',
                parent=styles['Heading1'],
                fontSize=14,
                fontName='Helvetica',
                alignment=1,  # Center alignment
                spaceAfter=20,
                textColor=colors.gray
            )
            
            heading_style = ParagraphStyle(
                'Heading',
                parent=styles['Heading2'],
                fontSize=16,
                fontName='Helvetica-Bold',
                spaceAfter=10,
                textColor=primary_color,
                borderPadding=[0, 0, 3, 8],  # top, right, bottom, left
                borderWidth=0,
                borderColor=secondary_color,
                borderRadius=0,
                leftIndent=0
            )
            
            subheading_style = ParagraphStyle(
                'SubHeading',
                parent=styles['Heading3'],
                fontSize=13,
                fontName='Helvetica-Bold',
                spaceAfter=6,
                textColor=secondary_color,
                leftIndent=0
            )
            
            normal_style = ParagraphStyle(
                'Normal',
                parent=styles['Normal'],
                fontSize=10,
                fontName='Helvetica',
                spaceAfter=8,
                leading=14,
                textColor=colors.darkslategray
            )
            
            note_style = ParagraphStyle(
                'Note',
                parent=styles['Italic'],
                fontSize=9,
                fontName='Helvetica-Oblique',
                textColor=colors.gray,
                spaceAfter=10
            )
            
            info_style = ParagraphStyle(
                'Info',
                parent=note_style,
                textColor=colors.darkblue,
                backColor=colors.lightcyan,
                borderPadding=10,
                leftIndent=10,
                rightIndent=10
            )
            
            # Create section divider
            def add_section_divider(story):
                # Add a subtle divider line between sections
                divider_data = [['']]
                divider = Table(divider_data, colWidths=[470])
                divider.setStyle(TableStyle([
                    ('LINEBELOW', (0, 0), (0, 0), 0.5, colors.lightgrey),
                    ('TOPPADDING', (0, 0), (0, 0), 0),
                    ('BOTTOMPADDING', (0, 0), (0, 0), 0),
                ]))
                story.append(divider)
                story.append(Spacer(1, 15))
            
            # Build the PDF content with improved layout
            story = []
            
            # Add logo if exists
            logo_path = "static/img/logo.png"
            if os.path.exists(logo_path):
                logo = Image(logo_path, width=150, height=50)
                logo.hAlign = 'CENTER'
                story.append(logo)
                story.append(Spacer(1, 10))
            
            # Report title with date in an elegant header
            current_time = time.strftime("%B %d, %Y %H:%M")
            
            # Create a header table
            header_data = [[Paragraph(f"TwinTitles Analysis Report", title_style)], 
                          [Paragraph(f"Generated on {current_time}", subtitle_style)]]
            
            header_table = Table(header_data, colWidths=[470])
            header_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (0, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LINEBELOW', (0, 1), (0, 1), 1, primary_color),
                ('BOTTOMPADDING', (0, 0), (0, 0), 0),
                ('TOPPADDING', (0, 1), (0, 1), 0),
            ]))
            
            story.append(header_table)
            story.append(Spacer(1, 30))
            
            # Title and metadata in a card-like format
            story.append(Paragraph("Analysis Summary", heading_style))
            story.append(Spacer(1, 10))
            
            # Create a shaded table for basic info with improved visual design
            basic_info = [
                ["Title", title],
                ["Status", result['label']],
                ["Similarity Index", f"{float(result['top_matches'][0][1]):.2f} ({float(result['top_matches'][0][1])*100:.1f}%)"],
                ["Uniqueness Score", f"{uniqueness_data['unique']}%"]
            ]
            
            # Add visual progress bar for similarity/uniqueness
            similarity = float(result['top_matches'][0][1])
            uniqueness = 1 - similarity
            
            # Create progress bar for similarity
            sim_color = colors.red if similarity > 0.8 else colors.orange if similarity > 0.5 else colors.green
            sim_bar_data = [[
                Paragraph(f"Similarity: {int(similarity*100)}%", normal_style),
                Paragraph(f'<para backColor="{sim_color}" borderColor="{sim_color}" borderWidth="1" borderRadius="5" borderPadding="2" width="{int(similarity*150)}px" height="12px"> </para>', normal_style)
            ]]
            
            sim_bar = Table(sim_bar_data, colWidths=[120, 180])
            sim_bar.setStyle(TableStyle([
                ('VALIGN', (0, 0), (1, 0), 'MIDDLE'),
            ]))
            
            # Create progress bar for uniqueness
            uniq_color = colors.green if uniqueness > 0.8 else colors.orange if uniqueness > 0.5 else colors.red
            uniq_bar_data = [[
                Paragraph(f"Uniqueness: {int(uniqueness*100)}%", normal_style),
                Paragraph(f'<para backColor="{uniq_color}" borderColor="{uniq_color}" borderWidth="1" borderRadius="5" borderPadding="2" width="{int(uniqueness*150)}px" height="12px"> </para>', normal_style)
            ]]
            
            uniq_bar = Table(uniq_bar_data, colWidths=[120, 180])
            uniq_bar.setStyle(TableStyle([
                ('VALIGN', (0, 0), (1, 0), 'MIDDLE'),
            ]))
            
            # Create a title card with background
            title_card_data = [
                [Paragraph(f"<b>Title:</b> {title}", normal_style)],
                [Paragraph(f"<b>Status:</b> <font color='{sim_color}'>{result['label']}</font>", normal_style)],
                [sim_bar],
                [uniq_bar]
            ]
            
            title_card = Table(title_card_data, colWidths=[470])
            title_card.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), background_color),
                ('VALIGN', (0, 0), (0, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (0, -1), 15),
                ('RIGHTPADDING', (0, 0), (0, -1), 15),
                ('BOTTOMPADDING', (0, 0), (0, -1), 10),
                ('TOPPADDING', (0, 0), (0, -1), 10),
                ('LINEBELOW', (0, 0), (0, 0), 0.5, border_color),
                ('BOX', (0, 0), (0, -1), 1, border_color),
                ('ROUNDEDCORNERS', [5, 5, 5, 5]),
            ]))
            
            story.append(title_card)
            story.append(Spacer(1, 20))
            
            # Executive Summary with improved styling
            story.append(Paragraph("Executive Summary", heading_style))
            
            # Create a simple status message based on the similarity
            if similarity > 0.9:
                status_msg = "Your title has a very high similarity with existing research titles. Consider a significant revision to differentiate your work."
                status_icon = "⚠️"
            elif similarity > 0.7:
                status_msg = "Your title shows substantial similarity with existing research. Some revision is recommended to better highlight the uniqueness of your work."
                status_icon = "⚠️"
            elif similarity > 0.5:
                status_msg = "Your title shares moderate similarity with existing research. Minor adjustments could help emphasize its distinctive aspects."
                status_icon = "ℹ️"
            else:
                status_msg = "Your title appears to be relatively unique compared to existing research. It effectively distinguishes your work."
                status_icon = "✓"
            
            # Create an executive summary card
            summary_data = [[Paragraph(f"{status_icon} {status_msg}", normal_style)]]
            summary_table = Table(summary_data, colWidths=[470])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, 0), colors.white),
                ('LEFTPADDING', (0, 0), (0, 0), 15),
                ('RIGHTPADDING', (0, 0), (0, 0), 15),
                ('BOTTOMPADDING', (0, 0), (0, 0), 15),
                ('TOPPADDING', (0, 0), (0, 0), 15),
                ('BOX', (0, 0), (0, 0), 1, border_color),
                ('ROUNDEDCORNERS', [3, 3, 3, 3]),
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 20))
            add_section_divider(story)
            
            # Top Similar Titles - Visual Format with improved cards
            story.append(Paragraph("Top Similar Titles", heading_style))
            story.append(Spacer(1, 10))
            
            # For each match, create a visual card with similarity badge
            for i, (match_title, similarity) in enumerate(result['top_matches'][:5]):
                # Get common words for this title
                common_words = []
                if i < len(common_words_data):
                    common_words = [item['word'] for item in common_words_data[i]['common_words']]
                
                # Create data for similarity badge - circular with color based on similarity
                similarity_percentage = int(float(similarity) * 100)
                badge_color = colors.red if similarity > 0.8 else colors.orange if similarity > 0.5 else colors.green
                
                # Create a table for the similarity badge (circular badge look)
                badge_data = [[f"{similarity_percentage}%"]]
                badge_table = Table(badge_data, colWidths=[50])
                badge_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, 0), badge_color),
                    ('TEXTCOLOR', (0, 0), (0, 0), colors.white),
                    ('ALIGNMENT', (0, 0), (0, 0), 'CENTER'),
                    ('VALIGN', (0, 0), (0, 0), 'MIDDLE'),
                    ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (0, 0), 12),
                    ('ROUNDEDCORNERS', [25, 25, 25, 25]),
                ]))
                
                # Create highlighted title text
                words = str(match_title).split()
                highlighted_title = []
                
                for word in words:
                    clean_word = re.sub(r'[^\w]', '', word.lower())
                    if clean_word in common_words:
                        highlighted_title.append(f'<span style="background-color:#FFF2CC; padding:2px; border-radius:2px;">{word}</span>')
                    else:
                        highlighted_title.append(word)
                
                # Create data for the entire card-like row with improved styling
                card_data = [[badge_table, Paragraph(" ".join(highlighted_title), normal_style)]]
                card = Table(card_data, colWidths=[70, 400])
                card.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (1, 0), 'MIDDLE'),
                    ('BOTTOMPADDING', (0, 0), (1, 0), 15),
                    ('TOPPADDING', (0, 0), (1, 0), 15),
                    ('BACKGROUND', (0, 0), (1, 0), colors.white),
                    ('BOX', (0, 0), (1, 0), 1, border_color),
                    ('ROUNDEDCORNERS', [3, 3, 3, 3]),
                ]))
                
                story.append(card)
                story.append(Spacer(1, 10))
            
            story.append(Spacer(1, 5))
            story.append(Paragraph("<i>Note: Words highlighted in yellow appear in both your title and the similar title.</i>", note_style))
            story.append(Spacer(1, 15))
            add_section_divider(story)
            
            # Generated Abstract Section with modern card design
            story.append(Paragraph("Generated Abstract", heading_style))
            story.append(Spacer(1, 10))
            
            # Create a box with a blue left border for the abstract
            abstract_style = ParagraphStyle(
                'Abstract',
                parent=normal_style,
                backColor=colors.white,
                borderPadding=15,
                borderWidth=1,
                borderColor=colors.lightgrey,
                borderRadius=5,
                leftIndent=10,
                rightIndent=10,
                spaceAfter=5
            )
            
            # Create a container with a left accent border
            abstract_data = [[Paragraph(abstract, abstract_style)]]
            abstract_table = Table(abstract_data, colWidths=[470])
            abstract_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, 0), colors.white),
                ('VALIGN', (0, 0), (0, 0), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (0, 0), 5),
                ('RIGHTPADDING', (0, 0), (0, 0), 5),
                ('BOTTOMPADDING', (0, 0), (0, 0), 5),
                ('TOPPADDING', (0, 0), (0, 0), 5),
                ('BOX', (0, 0), (0, 0), 1, border_color),
                ('LINEAFTER', (0, 0), (0, 0), 4, secondary_color),
            ]))
            
            story.append(abstract_table)
            
            # Add note about the abstract with icon
            story.append(Spacer(1, 10))
            story.append(Paragraph("<i>ℹ️ This abstract is automatically generated based on your title and can be used as a starting point for your research paper.</i>", info_style))
            story.append(Spacer(1, 20))
            add_section_divider(story)
            
            # Visual Title Comparison with enhanced design
            story.append(PageBreak())
            story.append(Paragraph("Visual Title Comparison", heading_style))
            story.append(Paragraph("This section provides a side-by-side comparison between your input title and the matched titles, highlighting common terms.", normal_style))
            story.append(Spacer(1, 15))
            
            # Add visual comparisons for top 3 matches with improved styling
            for i, (match_title, similarity) in enumerate(result['top_matches'][:3]):
                # Get common words for this title
                common_words = []
                if i < len(common_words_data):
                    common_words = [item['word'] for item in common_words_data[i]['common_words']]
                
                # Create highlighted versions of both titles
                input_title_words = title.split()
                match_title_words = str(match_title).split()
                
                highlighted_input = []
                for word in input_title_words:
                    clean_word = re.sub(r'[^\w]', '', word.lower())
                    if clean_word in common_words:
                        highlighted_input.append(f'<span style="background-color:#FFF2CC; padding:2px; border-radius:2px;">{word}</span>')
                    else:
                        highlighted_input.append(word)
                
                highlighted_match = []
                for word in match_title_words:
                    clean_word = re.sub(r'[^\w]', '', word.lower())
                    if clean_word in common_words:
                        highlighted_match.append(f'<span style="background-color:#FFF2CC; padding:2px; border-radius:2px;">{word}</span>')
                    else:
                        highlighted_match.append(word)
                
                # Create a match header with badge showing similarity percentage
                similarity_percentage = int(float(similarity) * 100)
                badge_color = colors.red if similarity > 0.8 else colors.orange if similarity > 0.5 else colors.green
                
                match_header_data = [[
                    Paragraph(f"Match #{i+1}", subheading_style),
                    Paragraph(f'<para backColor="{badge_color}" color="white" borderColor="{badge_color}" borderWidth="1" borderRadius="10" borderPadding="5">{similarity_percentage}% Similar</para>', normal_style)
                ]]
                
                match_header = Table(match_header_data, colWidths=[235, 235])
                match_header.setStyle(TableStyle([
                    ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                    ('VALIGN', (0, 0), (1, 0), 'MIDDLE'),
                    ('BOTTOMPADDING', (0, 0), (1, 0), 5),
                    ('TOPPADDING', (0, 0), (1, 0), 5),
                ]))
                
                story.append(match_header)
                story.append(Spacer(1, 5))
                
                # Create data for the header row with improved styling
                header_style = ParagraphStyle(
                    'HeaderCell',
                    parent=normal_style,
                    fontName='Helvetica-Bold',
                    fontSize=10,
                    textColor=colors.white,
                    alignment=1,  # Center alignment
                )
                
                # Create a table for side-by-side comparison with styled headers
                comparison_data = [
                    [Paragraph('<para color="white">Input Title</para>', header_style), 
                     Paragraph('<para color="white">Matched Title</para>', header_style)],
                    [Paragraph(" ".join(highlighted_input), normal_style), 
                     Paragraph(" ".join(highlighted_match), normal_style)]
                ]
                
                # Create comparison table with enhanced styling
                comparison_table = Table(comparison_data, colWidths=[235, 235])
                comparison_table.setStyle(TableStyle([
                    # Header row styling
                    ('BACKGROUND', (0, 0), (1, 0), primary_color),
                    ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
                    ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                    ('VALIGN', (0, 0), (1, 0), 'MIDDLE'),
                    ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (1, 0), 8),
                    ('TOPPADDING', (0, 0), (1, 0), 8),
                    
                    # Content cells
                    ('BACKGROUND', (0, 1), (1, 1), colors.white),
                    ('VALIGN', (0, 1), (1, 1), 'TOP'),
                    ('ALIGN', (0, 1), (1, 1), 'LEFT'),
                    ('LEFTPADDING', (0, 1), (1, 1), 10),
                    ('RIGHTPADDING', (0, 1), (1, 1), 10),
                    ('BOTTOMPADDING', (0, 1), (1, 1), 15),
                    ('TOPPADDING', (0, 1), (1, 1), 15),
                    
                    # Table border with rounded corners
                    ('BOX', (0, 0), (1, 1), 1, border_color),
                    ('INNERGRID', (0, 0), (1, 1), 1, border_color),
                    ('LINEBELOW', (0, 0), (1, 0), 1, border_color),
                    ('ROUNDEDCORNERS', [5, 5, 5, 5]),
                ]))
                
                story.append(comparison_table)
                story.append(Spacer(1, 20))
            
            story.append(Spacer(1, 5))
            story.append(Paragraph("<i>Note: Words highlighted in yellow appear in both titles, indicating potential similarity.</i>", note_style))
            add_section_divider(story)
            
            # Title Enhancement Suggestions with modern card design
            story.append(PageBreak())
            story.append(Paragraph("Title Enhancement Suggestions", heading_style))
            story.append(Spacer(1, 10))
            
            # Add icon to paragraph
            suggestion_intro = Paragraph("Based on our analysis, here are some suggestions to improve your title:", normal_style)
            story.append(suggestion_intro)
            story.append(Spacer(1, 10))
            
            # Create card-like boxes for each suggestion with improved styling
            for i, suggestion in enumerate(suggestions):
                suggestion_data = [[Paragraph(suggestion, normal_style)]]
                suggestion_table = Table(suggestion_data, colWidths=[470])
                
                # Alternate background color for better readability
                bg_color = colors.white if i % 2 == 0 else background_color
                
                suggestion_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, 0), bg_color),
                    ('VALIGN', (0, 0), (0, 0), 'MIDDLE'),
                    ('LEFTPADDING', (0, 0), (0, 0), 15),
                    ('RIGHTPADDING', (0, 0), (0, 0), 15),
                    ('BOTTOMPADDING', (0, 0), (0, 0), 15),
                    ('TOPPADDING', (0, 0), (0, 0), 15),
                    ('LINEAFTER', (0, 0), (0, 0), 4, primary_color),
                    ('BOX', (0, 0), (0, 0), 1, border_color),
                    ('ROUNDEDCORNERS', [5, 5, 5, 5]),
                ]))
                
                story.append(suggestion_table)
                story.append(Spacer(1, 10))
            
            story.append(Spacer(1, 20))
            add_section_divider(story)
            
            # Semantic Similarity Network and Plagiarism Heatmap with improved styling
            story.append(PageBreak())
            story.append(Paragraph("Semantic Similarity Network", heading_style))
            story.append(Spacer(1, 5))
            
            img_path = f"static/img/{graph_path}"
            if os.path.exists(img_path):
                img = Image(img_path, width=450, height=350)
                img.hAlign = 'CENTER'
                story.append(img)
                
                # Add legend with better styling
                legend_data = [
                    [Paragraph('<para backColor="red" textColor="white"><b>\u25CF</b></para>', normal_style), 
                     Paragraph("Your Title", normal_style)],
                    [Paragraph('<para backColor="dodgerblue" textColor="white"><b>\u25CF</b></para>', normal_style), 
                     Paragraph("Similar Titles", normal_style)],
                    [Paragraph('<b>\u2500\u2500</b>', normal_style), 
                     Paragraph("Similarity Strength", normal_style)]
                ]
                
                legend_table = Table(legend_data, colWidths=[20, 450])
                legend_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (0, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 0), (-1, -1), 5),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ]))
                
                story.append(Spacer(1, 10))
                story.append(legend_table)
            else:
                story.append(Paragraph("Graph could not be generated.", normal_style))
            
            story.append(Spacer(1, 20))
            add_section_divider(story)
            
            # Add Features Documentation in a cleaner format
            story.append(PageBreak())
            story.append(Paragraph("TwinTitles Features", heading_style))
            story.append(Spacer(1, 10))
            
            # Format features documentation with icons and better layout
            features_intro = Paragraph(
                "TwinTitles is a research title similarity analyzer that helps researchers, academics, and students evaluate " +
                "the uniqueness of their research titles, using advanced natural language processing and semantic analysis.", normal_style)
            story.append(features_intro)
            story.append(Spacer(1, 15))
            
            # List of features with icons in a table format
            features = [
                ["📊", "Similarity Analysis", "Comprehensive similarity analysis comparing your title with a database of existing research titles."],
                ["🔍", "Visual Title Comparison", "Side-by-side comparison highlighting common words between titles."],
                ["🕸️", "Semantic Similarity Network", "Visual representation of relationships between your title and similar titles."],
                ["🔥", "Plagiarism Heatmap", "Highlights similar words between titles to identify potential areas of concern."],
                ["💡", "Title Enhancement", "Actionable suggestions to improve your title's uniqueness while maintaining focus."],
                ["📝", "Abstract Generation", "Auto-generated abstract based on your title as a starting point for your research."],
                ["📄", "PDF Reports", "Comprehensive reports for sharing or reference (like this one)."],
                ["🔤", "Common Word Analysis", "Identification of shared terms to help make targeted adjustments."]
            ]
            
            # Create a table with all features
            feature_rows = []
            for icon, name, description in features:
                feature_rows.append([
                    Paragraph(f"{icon}", normal_style),
                    Paragraph(f"<b>{name}</b>", normal_style),
                    Paragraph(description, normal_style)
                ])
            
            features_table = Table(feature_rows, colWidths=[20, 120, 330])
            features_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ]))
            
            story.append(features_table)
            
            # Add footer with copyright
            def add_page_number(canvas, doc):
                canvas.saveState()
                # Draw a top header line
                canvas.setStrokeColor(primary_color)
                canvas.setLineWidth(1)
                canvas.line(50, letter[1]-40, letter[0]-50, letter[1]-40)
                
                # Draw page info
                canvas.setFont('Helvetica', 8)
                canvas.drawString(50, 30, f"TwinTitles Analysis Report - {time.strftime('%Y-%m-%d')}")
                canvas.drawRightString(letter[0] - 50, 30, f"Page {doc.page} of {doc.build.maxPages or '?'}")
                
                # Draw bottom line
                canvas.setStrokeColor(border_color)
                canvas.setLineWidth(0.5)
                canvas.line(50, 20, letter[0]-50, 20)
                
                # Add copyright
                canvas.setFont('Helvetica', 8)
                canvas.drawCentredString(letter[0]/2, 30, "© 2025 TwinTitles | Hack4Bengal Project")
                canvas.restoreState()
            
            # Build the document with page numbers
            doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
            
            return filename
                
        except Exception as inner_e:
            print(f"ERROR during PDF creation process: {str(inner_e)}")
            print(traceback.format_exc())
            return "error_report.pdf"
            
    except Exception as e:
        print(f"ERROR generating PDF report: {str(e)}")
        print(traceback.format_exc())
        # Return a placeholder filename - we'll handle missing files in the get_pdf route
        return "error_report.pdf"

def generate_fallback_pdf(title, top_matches):
    """Generate a simple fallback PDF when the main report generation fails"""
    try:
        # Ensure temp directory exists
        if not os.path.exists('temp'):
            os.makedirs('temp')
            
        # Create a unique filename
        filename = f"fallback_report_{uuid.uuid4().hex}.pdf"
        filepath = os.path.abspath(f"temp/{filename}")
        
        print(f"Generating fallback PDF report at: {filepath}")

        # Create a simple PDF
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas as reportlab_canvas
        
        c = reportlab_canvas.Canvas(filepath, pagesize=letter)
        width, height = letter
        
        # Add title
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(width/2, height-50, "TwinTitles Analysis Report")
        
        # Add timestamp
        c.setFont("Helvetica", 12)
        c.drawCentredString(width/2, height-70, f"Generated on {time.strftime('%B %d, %Y %H:%M')}")
        
        # Add analyzed title in a card
        y_position = height - 120
        
        # Draw a box for the title info
        c.setFillColor(colors.whitesmoke)
        c.rect(50, y_position-60, width-100, 60, fill=1, stroke=0)
        c.setStrokeColor(colors.grey)
        c.rect(50, y_position-60, width-100, 60, fill=0, stroke=1)
        
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(60, y_position-20, "Title:")
        c.setFont("Helvetica", 12)
        c.drawString(120, y_position-20, title)
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(60, y_position-40, "Status:")
        status = "Similar" if float(top_matches[0][1]) > 0.5 else "Unique"
        c.drawString(120, y_position-40, status)
        
        # Add top matches section title
        y_position = y_position - 90
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, "Top Similar Titles")
        y_position -= 30
        
        # Draw similar titles in a visual format with badges
        for i, (match_title, similarity) in enumerate(top_matches[:5]):
            # Draw similarity badge (circle)
            badge_x = 80
            badge_y = y_position - 15
            badge_radius = 25
            
            c.setFillColor(colors.dodgerblue)
            c.circle(badge_x, badge_y, badge_radius, fill=1)
            
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 12)
            percentage = int(float(similarity) * 100)
            c.drawCentredString(badge_x, badge_y-4, f"{percentage}%")
            
            # Draw title text
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 11)
            
            # Truncate long titles
            display_title = match_title
            if len(str(match_title)) > 60:
                display_title = str(match_title)[:57] + "..."
                
            c.drawString(120, y_position-20, display_title)
            
            # Draw divider line
            y_position -= 50
            if i < len(top_matches[:5]) - 1:  # No line after last item
                c.setStrokeColor(colors.lightgrey)
                c.line(50, y_position+10, width-50, y_position+10)
        
        # Check if we need a new page
        if y_position < 300:
            c.showPage()
            y_position = height - 50
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_position, "Analysis Continued")
            y_position -= 30
        
        # Abstract section
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, "Generated Abstract")
        y_position -= 30
        
        # Create a box for the abstract
        abstract = f"This research explores {title.lower()}. The study examines various aspects of this topic, including methodologies, key findings, and implications for future research."
        
        # Abstract text box
        c.setFillColor(colors.whitesmoke)
        c.rect(50, y_position-80, width-100, 70, fill=1, stroke=0)
        c.setStrokeColor(colors.grey)
        c.rect(50, y_position-80, width-100, 70, fill=0, stroke=1)
        
        # Abstract text with wrapping
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 10)
        
        # Wrap text
        wrapped_text = []
        words = abstract.split()
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if c.stringWidth(test_line, "Helvetica", 10) < width - 120:
                current_line = test_line
            else:
                wrapped_text.append(current_line)
                current_line = word
        if current_line:
            wrapped_text.append(current_line)
        
        # Draw wrapped text lines
        text_y = y_position - 20
        for line in wrapped_text:
            c.drawString(60, text_y, line)
            text_y -= 15
        
        # Add info about abstract
        y_position -= 100
        c.setFillColor(colors.lightcyan)
        c.rect(50, y_position-30, width-100, 20, fill=1, stroke=0)
        c.setStrokeColor(colors.lightblue)
        c.rect(50, y_position-30, width-100, 20, fill=0, stroke=1)
        
        c.setFillColor(colors.darkblue)
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(60, y_position-18, "This abstract is automatically generated based on your title and can be used as a starting point.")
        
        # Suggestions section
        y_position -= 50
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, "Title Enhancement Suggestions")
        y_position -= 30
        
        # Sample suggestions
        suggestions = [
            f"Consider adding specificity: \"{title} - A Comprehensive Review\"",
            f"For empirical studies: \"An Empirical Analysis of {title}\"",
            f"More concise alternative: \"{' '.join(title.split()[:3])}...\""
        ]
        
        # Draw suggestion boxes with blue left border
        for suggestion in suggestions:
            # Box for suggestion
            c.setFillColor(colors.white)
            c.rect(50, y_position-30, width-100, 25, fill=1, stroke=0)
            c.setStrokeColor(colors.lightgrey)
            c.rect(50, y_position-30, width-100, 25, fill=0, stroke=1)
            
            # Blue left border
            c.setFillColor(colors.blue)
            c.rect(50, y_position-30, 4, 25, fill=1, stroke=0)
            
            # Suggestion text
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 10)
            
            # Truncate if needed
            display_suggestion = suggestion
            if len(suggestion) > 80:
                display_suggestion = suggestion[:77] + "..."
                
            c.drawString(60, y_position-15, display_suggestion)
            
            y_position -= 35
        
        # Add a new page for visual title comparison
        c.showPage()
        y_position = height - 50
        
        # Visual Title Comparison header
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(width/2, y_position, "Visual Title Comparison")
        y_position -= 30
        
        c.setFont("Helvetica", 11)
        c.drawString(50, y_position, "This section provides a side-by-side comparison between your input title and the matched titles.")
        y_position -= 30
        
        # Draw visual comparisons for top 3 matches
        for i, (match_title, similarity) in enumerate(top_matches[:3]):
            # Draw match number and similarity
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y_position, f"Match #{i+1} (Similarity: {similarity:.4f})")
            y_position -= 25
            
            # Draw table headers with blue background
            c.setFillColor(colors.navy)
            c.rect(50, y_position-20, (width-100)/2, 20, fill=1, stroke=0)
            c.rect(50 + (width-100)/2, y_position-20, (width-100)/2, 20, fill=1, stroke=0)
            
            # Draw header text
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 10)
            c.drawCentredString(50 + (width-100)/4, y_position-10, "Input Title")
            c.drawCentredString(50 + (width-100)/2 + (width-100)/4, y_position-10, "Matched Title")
            
            # Draw content cells
            c.setFillColor(colors.white)
            cell_height = 60
            c.rect(50, y_position-20-cell_height, (width-100)/2, cell_height, fill=1, stroke=0)
            c.rect(50 + (width-100)/2, y_position-20-cell_height, (width-100)/2, cell_height, fill=1, stroke=0)
            
            # Draw cell borders
            c.setStrokeColor(colors.black)
            c.rect(50, y_position-20, (width-100)/2, 20, fill=0, stroke=1)  # Header 1
            c.rect(50 + (width-100)/2, y_position-20, (width-100)/2, 20, fill=0, stroke=1)  # Header 2
            c.rect(50, y_position-20-cell_height, (width-100)/2, cell_height, fill=0, stroke=1)  # Content 1
            c.rect(50 + (width-100)/2, y_position-20-cell_height, (width-100)/2, cell_height, fill=0, stroke=1)  # Content 2
            
            # Draw titles - handling common words
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 10)
            
            # Input title
            input_y = y_position - 40
            words = title.split()
            x_pos = 60
            
            for word in words:
                # Check if this is a common word (simplified check)
                is_common = False
                clean_word = re.sub(r'[^\w]', '', word.lower())
                
                # Check if it appears in matched title
                if clean_word in str(match_title).lower():
                    is_common = True
                    
                # Draw word
                word_width = c.stringWidth(word, "Helvetica", 10)
                
                # Draw highlight for common words
                if is_common:
                    c.setFillColor(colors.yellow)
                    c.rect(x_pos-2, input_y-10, word_width+4, 14, fill=1, stroke=0)
                
                c.setFillColor(colors.black)
                c.drawString(x_pos, input_y, word)
                
                # Move to next word position
                x_pos += word_width + 5
                
                # Wrap to next line if needed
                if x_pos > 50 + (width-100)/2 - 20:
                    x_pos = 60
                    input_y -= 15
            
            # Matched title
            match_y = y_position - 40
            words = str(match_title).split()
            x_pos = 50 + (width-100)/2 + 10
            
            for word in words:
                # Check if this is a common word
                is_common = False
                clean_word = re.sub(r'[^\w]', '', word.lower())
                
                # Check if it appears in input title
                if clean_word in title.lower():
                    is_common = True
                    
                # Draw word
                word_width = c.stringWidth(word, "Helvetica", 10)
                
                # Draw highlight for common words
                if is_common:
                    c.setFillColor(colors.yellow)
                    c.rect(x_pos-2, match_y-10, word_width+4, 14, fill=1, stroke=0)
                
                c.setFillColor(colors.black)
                c.drawString(x_pos, match_y, word)
                
                # Move to next word position
                x_pos += word_width + 5
                
                # Wrap to next line if needed
                if x_pos > 50 + (width-100) - 20:
                    x_pos = 50 + (width-100)/2 + 10
                    match_y -= 15
            
            # Move down for next comparison
            y_position -= cell_height + 50
            
            # Check if we need a new page
            if y_position < 100:
                c.showPage()
                y_position = height - 50
                c.setFont("Helvetica-Bold", 14)
                c.drawString(50, y_position, "Visual Title Comparison - Continued")
                y_position -= 30
        
        # Add note about highlighting
        if y_position < 80:
            c.showPage()
            y_position = height - 50
        
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColor(colors.grey)
        c.drawString(50, y_position-20, "Note: Words highlighted in yellow appear in both titles, indicating potential similarity.")
        
        # Continue with features documentation
        c.showPage()
        
        # Features Documentation header
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(width/2, height-50, "TwinTitles Features Documentation")
        
        y_position = height - 80
        
        # About TwinTitles
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, "About TwinTitles")
        y_position -= 25
        
        c.setFont("Helvetica", 10)
        about_text = "TwinTitles is a research title similarity analyzer that helps researchers, academics, and students evaluate the uniqueness of their research titles using advanced natural language processing."
        
        # Wrap text for proper display
        text_lines = []
        words = about_text.split()
        line = ""
        for word in words:
            test_line = line + " " + word if line else word
            if c.stringWidth(test_line, "Helvetica", 10) < width - 100:
                line = test_line
            else:
                text_lines.append(line)
                line = word
        if line:
            text_lines.append(line)
            
        for text_line in text_lines:
            c.drawString(50, y_position, text_line)
            y_position -= 15
            
        y_position -= 10
            
        # Key Features
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, "Key Features")
        y_position -= 25
        
        features = [
            "1. Similarity Analysis: Compares your title with a database of existing research titles.",
            "2. Visual Title Comparison: Side-by-side comparison with highlighted common words.",
            "3. Semantic Similarity Network: Visual representation of relationships between your title and similar titles.",
            "4. Plagiarism Heatmap: Highlights similar words between titles to identify potential areas of concern.",
            "5. Title Enhancement: Actionable suggestions to improve your title's uniqueness while maintaining focus.",
            "6. Abstract Generation: Auto-generated abstract based on your title as a starting point for your research.",
            "7. PDF Reports: Comprehensive reports for sharing or reference (like this one).",
            "8. Common Word Analysis: Identification of shared terms to help make targeted adjustments."
        ]
        
        c.setFont("Helvetica", 10)
        for feature in features:
            # Split feature into key and description
            parts = feature.split(": ", 1)
            key = parts[0]
            description = parts[1] if len(parts) > 1 else ""
            
            c.setFont("Helvetica-Bold", 10)
            c.drawString(50, y_position, key)
            c.setFont("Helvetica", 10)
            
            # Handle potential line wrapping for descriptions
            if description:
                desc_lines = []
                words = description.split()
                line = ""
                for word in words:
                    test_line = line + " " + word if line else word
                    if c.stringWidth(test_line, "Helvetica", 10) < width - 180:
                        line = test_line
                    else:
                        desc_lines.append(line)
                        line = word
                if line:
                    desc_lines.append(line)
                
                if desc_lines:
                    c.drawString(180, y_position, desc_lines[0])
                    y_position -= 15
                    
                    for i in range(1, len(desc_lines)):
                        c.drawString(180, y_position, desc_lines[i])
                        y_position -= 15
            else:
                y_position -= 15
                
            y_position -= 5
            
            # Check if we need a new page
            if y_position < 100:
                c.showPage()
                y_position = height - 50
                c.setFont("Helvetica", 10)
                c.drawString(50, height-30, "TwinTitles Features - Continued")
        
        # Save the PDF
        c.save()
        
        # Verify the file was created
        if os.path.exists(filepath):
            print(f"Fallback PDF created successfully: {filepath}")
            return filename
        else:
            print(f"Failed to create fallback PDF: {filepath}")
            return "error_report.pdf"
            
    except Exception as e:
        print(f"Error generating fallback PDF: {str(e)}")
        print(traceback.format_exc())
        return "error_report.pdf"

if __name__ == '__main__':
    app.run(debug=True) 