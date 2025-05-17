import os
import json
from ML.pdf_analyser import pdfAnalyser
from ML.pdf_extractor import pdfextractor
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Initialize the Flask app with the template folder pointing to 'public'
app = Flask(__name__)
CORS(app)

# Set the upload folder for PDF files
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Helper function to check if file is allowed (only PDFs)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/")
def home():
    return render_template("firstPage.html")  # Render the HTML page

# Handle PDF upload
from flask import request
from werkzeug.utils import secure_filename
import os
from flask import Response

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'profileImage' not in request.files:
        return '<script>alert("No file uploaded!"); window.location.href = "/";</script>'

    files = request.files.getlist('profileImage')

    if not files or files[0].filename == '':
        return '<script>alert("No file selected!"); window.location.href = "/";</script>'

    saved_any = False
    pdfNameList=[]
    for file in files:
        if file and allowed_file(file.filename):
            filename = file.filename
            pdfNameList.append(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            saved_any = True
    if saved_any:
        # pdf extraction 
        with open('./ML/extracted_pdf_data.json', 'w') as file:
            json.dump([], file)
        for pdf in pdfNameList:
            print(pdf)
            pdfextractor(pdf)

        return Response(status=204)
    else:
        return '<script>alert("Invalid file type! Only PDF files are allowed."); window.location.href = "/";</script>'

@app.route('/analyze', methods=['POST'])
def analyze():
    results = []
    try:
        with open("./ML/extracted_pdf_data.json", "r") as json_file:
            pdfData = json.load(json_file)
        with open('./ML/analysed_data.json', 'w') as json_file:
            json.dump([], json_file, indent=4)   
        results = [] 
        for data in pdfData:
            print("start")
            result = pdfAnalyser(data)
            results.append(result)
        return jsonify({"results":results})
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analysedjson', methods=['GET'])
def get_json():
    try:
        with open('./ML/analysed_data.json', 'r') as json_file:
            analysedData = json.load(json_file)
        return jsonify(analysedData)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/extractedjson', methods=['GET'])
def get_json2():
    try:
       with open("./ML/extracted_pdf_data.json", "r") as json_file:
            extracted = json.load(json_file)
            return jsonify(extracted)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
