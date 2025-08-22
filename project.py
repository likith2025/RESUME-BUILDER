# Smart Resume Parser
# Complete implementation with PDF/DOCX support and Streamlit UI

import streamlit as st
import spacy
import re
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import io
import zipfile

# File processing imports
try:
    import fitz  # PyMuPDF
except ImportError:
    st.error("PyMuPDF not installed. Run: pip install PyMuPDF")

try:
    from docx import Document
except ImportError:
    st.error("python-docx not installed. Run: pip install python-docx")

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy English model not found. Run: python -m spacy download en_core_web_sm")
        return None

class ResumeParser:
    def _init_(self):
        self.nlp = load_spacy_model()
        
        # Common patterns for different sections
        self.section_patterns = {
            'education': r'(?i)\b(education|academic|degree|university|college|school|qualification)\b',
            'experience': r'(?i)\b(experience|employment|work|career|professional|job|position)\b',
            'skills': r'(?i)\b(skills|technical|competencies|abilities|expertise|proficiencies)\b',
            'projects': r'(?i)\b(projects|portfolio|work samples)\b',
            'certifications': r'(?i)\b(certification|certificate|license|credential)\b',
            'contact': r'(?i)\b(contact|phone|email|address|linkedin|github)\b'
        }
        
        # Skills keywords (expanded list)
        self.skills_keywords = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'express'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
            'data_science': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'tableau', 'power bi'],
            'tools': ['git', 'jenkins', 'jira', 'confluence', 'slack', 'trello']
        }
        
        # Date patterns
        self.date_patterns = [
            r'\b\d{4}\s*-\s*\d{4}\b',  # 2020-2023
            r'\b\d{4}\s*–\s*\d{4}\b',  # 2020–2023
            r'\b\d{1,2}/\d{4}\s*-\s*\d{1,2}/\d{4}\b',  # 01/2020-12/2023
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}\b',  # January 2020
        ]
    
    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            st.error(f"Error extracting PDF: {e}")
            return ""
    
    def extract_text_from_docx(self, file_bytes: bytes) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(io.BytesIO(file_bytes))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting DOCX: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s@.-]', ' ', text)
        return text.strip()
    
    def extract_email(self, text: str) -> List[str]:
        """Extract email addresses"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(email_pattern, text)
    
    def extract_phone(self, text: str) -> List[str]:
        """Extract phone numbers"""
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US format
            r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b',  # (123) 456-7890
            r'\b\+\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b'  # International
        ]
        phones = []
        for pattern in phone_patterns:
            phones.extend(re.findall(pattern, text))
        return phones
    
    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs (LinkedIn, GitHub, etc.)"""
        url_pattern = r'https?://[^\s]+'
        return re.findall(url_pattern, text)
    
    def extract_dates(self, text: str) -> List[str]:
        """Extract date ranges"""
        dates = []
        for pattern in self.date_patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))
        return dates
    
    def identify_sections(self, text: str) -> Dict[str, str]:
        """Identify and extract different sections"""
        sections = {}
        lines = text.split('\n')
        current_section = 'general'
        section_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header
            section_found = False
            for section_name, pattern in self.section_patterns.items():
                if re.search(pattern, line) and len(line.split()) <= 5:
                    # Save previous section
                    if section_content:
                        sections[current_section] = '\n'.join(section_content)
                    current_section = section_name
                    section_content = []
                    section_found = True
                    break
            
            if not section_found:
                section_content.append(line)
        
        # Save last section
        if section_content:
            sections[current_section] = '\n'.join(section_content)
        
        return sections
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills from text"""
        found_skills = {category: [] for category in self.skills_keywords}
        text_lower = text.lower()
        
        for category, skills_list in self.skills_keywords.items():
            for skill in skills_list:
                if skill.lower() in text_lower:
                    found_skills[category].append(skill)
        
        # Remove empty categories
        found_skills = {k: v for k, v in found_skills.items() if v}
        return found_skills
    
    def extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education information"""
        education_keywords = ['degree', 'bachelor', 'master', 'phd', 'diploma', 'certificate', 'university', 'college']
        education_entries = []
        
        sentences = text.split('\n')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in education_keywords):
                # Extract dates from the sentence
                dates = self.extract_dates(sentence)
                education_entries.append({
                    'description': sentence,
                    'dates': dates[0] if dates else 'N/A'
                })
        
        return education_entries
    
    def extract_experience(self, text: str) -> List[Dict[str, str]]:
        """Extract work experience"""
        experience_keywords = ['manager', 'developer', 'engineer', 'analyst', 'consultant', 'specialist', 'coordinator']
        experience_entries = []
        
        sentences = text.split('\n')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in experience_keywords) or self.extract_dates(sentence):
                dates = self.extract_dates(sentence)
                experience_entries.append({
                    'description': sentence,
                    'dates': dates[0] if dates else 'N/A'
                })
        
        return experience_entries
    
    def extract_name(self, text: str) -> str:
        """Extract candidate name (simple heuristic)"""
        if not self.nlp:
            return "N/A"
        
        doc = self.nlp(text[:500])  # Check first 500 chars
        names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        
        if names:
            # Return the first name found
            return names[0]
        else:
            # Fallback: first line that looks like a name
            lines = text.split('\n')[:5]  # Check first 5 lines
            for line in lines:
                line = line.strip()
                if line and len(line.split()) <= 3 and len(line) > 2:
                    return line
        
        return "N/A"
    
    def parse_resume(self, text: str) -> Dict:
        """Main parsing function"""
        if not text:
            return {}
        
        # Clean text
        clean_text = self.clean_text(text)
        
        # Identify sections
        sections = self.identify_sections(clean_text)
        
        # Extract structured information
        parsed_data = {
            'name': self.extract_name(text),
            'contact_info': {
                'emails': self.extract_email(text),
                'phones': self.extract_phone(text),
                'urls': self.extract_urls(text)
            },
            'skills': self.extract_skills(clean_text),
            'education': self.extract_education(sections.get('education', clean_text)),
            'experience': self.extract_experience(sections.get('experience', clean_text)),
            'sections': sections,
            'extracted_dates': self.extract_dates(clean_text),
            'parsing_timestamp': datetime.now().isoformat()
        }
        
        return parsed_data

# Streamlit App
def main():
    st.set_page_config(
        page_title="Smart Resume Parser",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("🔍 Smart Resume Parser")
    st.markdown("Extract structured information from PDF and DOCX resumes")
    
    # Initialize parser
    parser = ResumeParser()
    
    if parser.nlp is None:
        st.error("Cannot initialize spaCy model. Please install required dependencies.")
        st.stop()
    
    # Sidebar
    st.sidebar.title("Options")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Resume Files",
        type=['pdf', 'docx'],
        accept_multiple_files=True
    )
    
    # Export format
    export_format = st.sidebar.selectbox(
        "Export Format",
        ["JSON", "CSV", "Excel"]
    )
    
    # Processing section
    if uploaded_files:
        st.header("📊 Parsing Results")
        
        parsed_results = []
        
        # Progress bar
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            st.subheader(f"📄 {uploaded_file.name}")
            
            # Extract text based on file type
            file_bytes = uploaded_file.read()
            
            if uploaded_file.type == "application/pdf":
                text = parser.extract_text_from_pdf(file_bytes)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = parser.extract_text_from_docx(file_bytes)
            else:
                st.error(f"Unsupported file type: {uploaded_file.type}")
                continue
            
            if not text:
                st.error(f"Could not extract text from {uploaded_file.name}")
                continue
            
            # Parse resume
            with st.spinner(f"Parsing {uploaded_file.name}..."):
                parsed_data = parser.parse_resume(text)
                parsed_data['filename'] = uploaded_file.name
                parsed_results.append(parsed_data)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("*Basic Information:*")
                st.write(f"Name: {parsed_data.get('name', 'N/A')}")
                st.write(f"Emails: {', '.join(parsed_data['contact_info']['emails']) or 'N/A'}")
                st.write(f"Phones: {', '.join(parsed_data['contact_info']['phones']) or 'N/A'}")
                
                st.write("*Skills Found:*")
                skills = parsed_data.get('skills', {})
                if skills:
                    for category, skill_list in skills.items():
                        st.write(f"- {category.title()}: {', '.join(skill_list)}")
                else:
                    st.write("No skills detected")
            
            with col2:
                st.write("*Education:*")
                education = parsed_data.get('education', [])
                if education:
                    for edu in education[:3]:  # Show first 3
                        st.write(f"- {edu['description'][:100]}...")
                else:
                    st.write("No education information detected")
                
                st.write("*Experience:*")
                experience = parsed_data.get('experience', [])
                if experience:
                    for exp in experience[:3]:  # Show first 3
                        st.write(f"- {exp['description'][:100]}...")
                else:
                    st.write("No experience information detected")
            
            # Show extracted text preview
            with st.expander(f"View extracted text from {uploaded_file.name}"):
                st.text_area("Extracted Text", text[:1000] + "..." if len(text) > 1000 else text, height=200)
            
            # Update progress
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Export functionality
        if parsed_results:
            st.header("📥 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Download JSON"):
                    json_data = json.dumps(parsed_results, indent=2)
                    st.download_button(
                        label="📄 Download JSON",
                        data=json_data,
                        file_name=f"parsed_resumes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("Download CSV"):
                    # Flatten data for CSV
                    flattened_data = []
                    for result in parsed_results:
                        flat_row = {
                            'filename': result['filename'],
                            'name': result.get('name', ''),
                            'emails': '; '.join(result['contact_info']['emails']),
                            'phones': '; '.join(result['contact_info']['phones']),
                            'urls': '; '.join(result['contact_info']['urls']),
                            'skills_programming': '; '.join(result.get('skills', {}).get('programming', [])),
                            'skills_web': '; '.join(result.get('skills', {}).get('web', [])),
                            'skills_database': '; '.join(result.get('skills', {}).get('database', [])),
                            'education_count': len(result.get('education', [])),
                            'experience_count': len(result.get('experience', [])),
                            'parsing_timestamp': result.get('parsing_timestamp', '')
                        }
                        flattened_data.append(flat_row)
                    
                    df = pd.DataFrame(flattened_data)
                    csv_data = df.to_csv(index=False)
                    
                    st.download_button(
                        label="📊 Download CSV",
                        data=csv_data,
                        file_name=f"parsed_resumes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col3:
                if st.button("View Summary Table"):
                    # Create summary dataframe
                    summary_data = []
                    for result in parsed_results:
                        summary_data.append({
                            'Filename': result['filename'],
                            'Name': result.get('name', 'N/A'),
                            'Email Count': len(result['contact_info']['emails']),
                            'Skills Categories': len(result.get('skills', {})),
                            'Education Entries': len(result.get('education', [])),
                            'Experience Entries': len(result.get('experience', []))
                        })
                    
                    df_summary = pd.DataFrame(summary_data)
                    st.dataframe(df_summary, use_container_width=True)
    
    else:
        # Instructions
        st.header("📋 How to Use")
        st.markdown("""
        1. *Upload Resume Files*: Support for PDF and DOCX formats
        2. *Automatic Parsing*: Extracts key information including:
           - Contact information (name, email, phone, URLs)
           - Skills by category (programming, web, database, etc.)
           - Education history
           - Work experience
           - Section identification
        3. *Export Options*: Download results as JSON or CSV
        4. *Batch Processing*: Upload multiple resumes at once
        
        *Required Dependencies:*
        bash
        pip install streamlit spacy PyMuPDF python-docx pandas
        python -m spacy download en_core_web_sm
        
        """)
        
        # Sample test data
        st.header("🧪 Test with Sample Data")
        if st.button("Generate Sample Resume Data"):
            sample_resumes = [
                {
                    "filename": "john_doe.pdf",
                    "name": "John Doe",
                    "contact_info": {
                        "emails": ["john.doe@email.com"],
                        "phones": ["(555) 123-4567"],
                        "urls": ["https://linkedin.com/in/johndoe"]
                    },
                    "skills": {
                        "programming": ["python", "java", "javascript"],
                        "web": ["react", "node.js", "html", "css"],
                        "database": ["sql", "mongodb"]
                    },
                    "education": [
                        {"description": "Bachelor of Computer Science, University of Technology 2018-2022", "dates": "2018-2022"}
                    ],
                    "experience": [
                        {"description": "Software Developer at Tech Corp 2022-2024", "dates": "2022-2024"},
                        {"description": "Junior Developer at StartupXYZ 2021-2022", "dates": "2021-2022"}
                    ]
                }
            ]
            
            st.json(sample_resumes[0])
            
            # Download sample JSON
            json_sample = json.dumps(sample_resumes, indent=2)
            st.download_button(
                label="📄 Download Sample JSON",
                data=json_sample,
                file_name="sample_parsed_resume.json",
                mime="application/json"
            )

if _name_ == "_main_":
    main()