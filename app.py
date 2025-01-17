import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import fitz  # PyMuPDF
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///resume_data.db'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'docx'}
db = SQLAlchemy(app)

# Ensure the uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
CUSTOM_STOPWORDS = set(stopwords.words('english')).union({'experience', 'looking', 'for', 'supporting', 'responsible', 'required',
    'desired', 'excellent', 'ability', 'must', 'should', 'will', 'have', 'has',
    'and', 'or', 'with', 'to', 'the', 'ideally', 'like', 'cc', 'example', 'record',
    'worked', 'key', 'track', 'demonstrated', 'understanding', 'proven', 'ensure',
    'strong', 'knowledge', 'skills', 'including', 'ability', 'team', 'across',
    'preferred', 'extensive', 'based', 'solid', 'years', 'minimum', 'work', 'job',
    'role', 'responsibilities', 'applicants', 'hands-on', 'etc', 'using', 'applicant',
    'within', 'applicable', 'help', 'others', 'apply', 'support', 'train', 'over',
    'develop', 'meet', 'understand', 'maintain', 'good', 'solutions', 'assist',
    'day-to-day', 'need', 'opportunity', 'similar', 'projects', 'deliver', 'result',
    'performs', 'current', 'future', 'required', 'desirable', 'position', 'seek', 
    'perform', 'plan', 'process', 'highly', 'function', 'goal', 'achieve', 
    'various', 'relevant', 'preferred', 'multiple', 'manage', 'managing', 'plus', 
    'related', 'time', 'excellent', 'leading', 'details', 'accomplish', 'seeks', 
    'meetings', 'clear', 'ability', 'directly', 'tools', 'guidance', 'enhance',
    'address', 'helpful', 'ensure', 'efforts', 'plans', 'methods', 'reports',
    'reviews', 'design', 'assess', 'complete', 'adhere', 'timely', 'programs'
})

# Predefined skill set
SKILL_SET = { "embedded_engineer": {
        "c", "c++", "python", "verilog", "vhdl", "embedded c", "rtos",
        "arduino", "raspberry pi", "microcontrollers", "fpga", "spi", "i2c",
        "can", "uart", "arm cortex", "keil", "freeRTOS", "linux", "gpio",
        "peripheral programming", "pcb design", "altium", "eagle", "jtag",
        "matlab", "simulink", "systemverilog", "firmware development"
    },
    "ux_designer": {
        "adobe xd", "figma", "sketch", "photoshop", "illustrator", "zeplin",
        "ux research", "wireframing", "prototyping", "user flows",
        "user-centered design", "information architecture", "usability testing",
        "design thinking", "responsive design", "interaction design",
        "storyboarding", "html", "css", "javascript", "invision", "heuristic evaluation"
    },
    "software_developer": {
        "java", "python", "javascript", "typescript", "c#", "c++", "ruby", "go", "scala",
        "node.js", "react", "angular", "flask", "django", "spring boot",
        "sql", "nosql", "postgresql", "mongodb", "docker", "kubernetes",
        "aws", "azure", "git", "github", "unit testing", "integration testing",
        "microservices", "rest apis", "graphql", "ci/cd pipelines", "agile"
    },
    "firmware_engineer": {
        "c", "c++", "assembly", "embedded c", "rtos", "freeRTOS", "linux",
        "uart", "spi", "i2c", "usb", "ethernet", "arm cortex", "stm32",
        "nordic", "ti-msp430", "keil", "altium", "matlab", "simulink",
        "debugging tools", "jtag", "isp", "firmware updates", "bootloaders",
        "bare-metal programming", "real-time systems", "peripheral programming",
        "hardware-software co-design"
    }
}


# Database model
class Interaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_description = db.Column(db.Text, nullable=False)
    resume_text = db.Column(db.Text, nullable=False)
    matched_skills = db.Column(db.Text, nullable=True)
    missing_skills = db.Column(db.Text, nullable=True)

# Initialize the database
with app.app_context():
    db.create_all()

# Helper function: Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Helper function: Extract text from a PDF file
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        text = ""
    return text

# Helper function: Extract text from a DOCX file
def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        text = ""
    return text

# Helper function: Preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text.strip()

# Resume analysis function
def analyze_resume(resume_text, job_description, vectorizer=None):
    # Tokenize and normalize text
    resume_words = word_tokenize(resume_text.lower())
    job_words = word_tokenize(job_description.lower())

    # Apply lemmatization and filter stopwords
    lemmatized_resume = [lemmatizer.lemmatize(word) for word in resume_words if word.isalnum() and word not in CUSTOM_STOPWORDS]
    lemmatized_job = [lemmatizer.lemmatize(word) for word in job_words if word.isalnum() and word not in CUSTOM_STOPWORDS]

    # Convert to sets for comparison
    resume_set = set(lemmatized_resume)
    job_set = set(lemmatized_job)

    # Find matches and missing words
    matches = resume_set.intersection(job_set)
    missing_words = job_set - resume_set

    # Match percentage
    match_percentage = (len(matches) / len(job_set)) * 100 if job_set else 0

    # Weighted score based on frequency
    resume_frequency = {word: lemmatized_resume.count(word) for word in matches}
    weighted_score = sum(resume_frequency.values()) / len(lemmatized_job) * 100 if lemmatized_job else 0

    # Extract skills
    resume_skills = resume_set.intersection(SKILL_SET)
    job_skills = job_set.intersection(SKILL_SET)

    # Calculate similarity if vectorizer is provided
    similarity_score = 0
    if vectorizer:
        resume_vector = vectorizer.transform([resume_text])
        job_vector = vectorizer.transform([job_description])
        similarity_score = cosine_similarity(resume_vector, job_vector).flatten()[0]

    return {
        "match_percentage": match_percentage,
        "weighted_score": weighted_score,
        "similarity_score": similarity_score,
        "matches": matches,
        "missing_words": missing_words,
        "resume_skills": list(resume_skills),
        "job_skills": list(job_skills)
    }

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get file and job description
        file = request.files.get('file')
        job_description = request.form.get('job_description')

        # Validate inputs
        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file format. Please upload a .pdf, .txt, or .docx file."}), 400

        if not job_description.strip():
            return jsonify({"error": "Job description cannot be empty."}), 400

        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"Uploaded file saved at: {file_path}")

        # Extract text from the file
        if file.filename.endswith('.pdf'):
            resume_text = extract_text_from_pdf(file_path)
        elif file.filename.endswith('.docx'):
            resume_text = extract_text_from_docx(file_path)
        else:
            with open(file_path, 'r') as f:
                resume_text = f.read()

        # Preprocess the extracted text
        resume_text = preprocess_text(resume_text)
        job_description = preprocess_text(job_description)

        # Check for empty extracted text
        if not resume_text.strip():
            return jsonify({"error": "Failed to extract text from the resume. Please check the file and try again."}), 400

        print(f"Job Description: {job_description}")
        print(f"Resume Text (first 500 chars): {resume_text[:500]}")

        # Load all previous interactions
        interactions = Interaction.query.all()
        job_descriptions = [i.job_description for i in interactions]
        resumes = [i.resume_text for i in interactions]

        # Train TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        if job_descriptions and resumes:
            vectorizer.fit(job_descriptions + resumes)
        else:
            vectorizer.fit([job_description, resume_text])

        # Perform analysis
        result = analyze_resume(resume_text, job_description, vectorizer=vectorizer)

        # Save the interaction to the database
        interaction = Interaction(
            job_description=job_description,
            resume_text=resume_text,
            matched_skills=",".join(result["resume_skills"]),
            missing_skills=",".join(result["missing_words"]),
        )
        db.session.add(interaction)
        db.session.commit()

        return jsonify({
            "match_percentage": result["match_percentage"],
            "weighted_score": result["weighted_score"],
            "similarity_score": result["similarity_score"],
            "matches": list(result["matches"]),
            "missing_words": list(result["missing_words"]),
            "resume_skills": result["resume_skills"],
            "job_skills": result["job_skills"]
        })

    except Exception as e:
        # Log the error for debugging
        print(f"Error during analysis: {e}")
        return jsonify({"error": "Something went wrong while analyzing your resume. Please try again."}), 500

if __name__ == '__main__':
    app.run(debug=True)
