# AI-Resume-Parser-with-Job-Recommendation (Flask + Gemini API)

This project is an AI-powered resume parser built using **Python, Flask**, and **Gemini (Google Generative AI)**. It extracts structured information from PDF resumes (like name, email, skills, etc.) and recommends jobs based on the extracted skills, maintaining the skill order.

---

## 📌 Features

- Upload a PDF resume
- Extracts key data using Gemini (Google Generative AI):
  - Full name
  - Email
  - Phone
  - Experience
  - Career Objective
  - Skills (in order)
- Matches extracted skills with job postings
- Displays matching jobs in order of user's skills
- Clean, responsive UI with cards and grid layout

---

## 🛠️ Technologies & Libraries Used

### 🔙 Backend
- `Python`
- `Flask`
- `pdfminer.six` → Extract text from PDF resumes
- `spaCy` → (Optional) basic NLP processing
- `google-generativeai` → Gemini AI API for resume data extraction
- `json` → Job postings & structured output
- `re` → Regular expressions (for cleaning Gemini raw output)

### 🎨 Frontend
- HTML5 / CSS3 / Jinja2
- Responsive 3-column job layout with hover cards

---

## 📁 Folder Structure

├── app.py # Flask backend
├── templates/
│ └── index.html # Home page
│ └── results.html # Results UI
├── static/ # (optional) for CSS, images, etc.
├── resumes/ # Uploaded PDF files
├── job_postings.json # All available job listings
├── requirements.txt # Python libraries
└── README.md

---

## 🚀 How to Run This Project Locally

1️⃣ Clone the Repository

git clone https://github.com/your-username/resume-parser-gemini.git
cd resume-parser-gemini

2️⃣ Set up Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

Then download the spaCy model: python -m spacy download en_core_web_sm

4️⃣ Set Up Gemini API Key

genai.configure(api_key="YOUR_GEMINI_API_KEY")

5️⃣ Run the App

python app.py
