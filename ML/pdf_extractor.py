import fitz  # PyMuPDF
import re
import json
import os

def pdfextractor(PDF):
    # Load PDF
    pdfloc="./uploads/"+PDF
    doc = fitz.open(pdfloc)

    #NAME
    # first_page = doc[0]  # get first page
    # text = first_page.get_text()

    # # Split into lines and get the first one
    # first_line = text.strip().split('\n')[0]

    # print(first_line)
    # doc.close()

    # Output the name
    # print(first_line)

    def extract_text_at_position(target_pos, tolerance=5.0): 
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:  # Only process text blocks
                    for line in block["lines"]:
                        for span in line["spans"]:
                            x, y = span["origin"]  # Bottom-left of the text
                            
                            # Check if the span's position is close to the target
                            if (abs(x - target_pos[0]) <= tolerance and 
                                abs(y - target_pos[1]) <= tolerance):
                                return span["text"].strip()
        
        return None  # No text found at the position

    # Output name
    target_position = (223.55999755859375, 65.5469970703125)
    name = extract_text_at_position(target_position)

    if name:
        print(name)
    else:
        print("No name found at the specified position.") 

    print("###################################################################################")

     #LINKEDIN URL
    url_text = ""
    headings = ["(LinkedIn)"]

    for page in doc:
        text = page.get_text()
        url_start = text.find("www.")
        url_chunk = text[url_start:]
            
        # Try stopping at any known next section
        end_index = -1
        for heading in headings:
            index = url_chunk.find(heading)
            if index != -1:
                # print("found before "+heading)
                end_index = index
                break
            
        if end_index != -1:
            url_chunk = url_chunk[:end_index]

        url_text = url_chunk.strip()
        break

    # Output the url
    if url_text == "":
        print("No url ")
    else:
        url_text=url_text.replace("\n","")
        print(url_text)

    print("###################################################################################")

    # Location
    locations =[]
    for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:  # Only process text blocks
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["color"]==11645361:
                                locations.append(span["text"])
    location=locations[0]

    print(location)

    print("###################################################################################")
    #HEADING
    summary_text = ""
    headings = ["Summary", "Experience", "Education", "Projects", "Skills", "Certifications", "Languages", "Contact"]

    for page in doc:
        text = page.get_text()
        heading_start = text.find(name)
        heading_chunk = text[heading_start+len(name):]
            
        # Try stopping at any known next section
        end_index = -1
        for heading in headings:
            index = heading_chunk.find(heading)
            if index != -1:
                # print("found before "+heading)
                end_index = index
                break
            
        if end_index != -1:
            heading_chunk = heading_chunk[:end_index]

        heading_text = heading_chunk.strip()
        break

    # Output the heading
    if heading_text == "":
        print("No heading ")
    else:
        heading_text=heading_text.replace(location,"")
        heading_text=heading_text.replace("\n"," ")
        print(heading_text)

    # for page in doc:
    #         blocks = page.get_text("dict")["blocks"]
            
    #         for block in blocks:
    #             if "lines" in block:  # Only process text blocks
    #                 for line in block["lines"]:
    #                     for span in line["spans"]:
    #                         if span["color"]==11645361:
    #                             print(span)    

    print("###################################################################################")

    #SUMMARY
    summary_text = ""
    headings = ["Experience", "Education", "Projects", "Skills", "Certifications", "Languages", "Contact"]

    for page in doc:
        text = page.get_text()
        if "Summary" in text:
            summary_start = text.find("Summary")
            summary_chunk = text[summary_start:]
            
            # Try stopping at any known next section
            end_index = -1
            for heading in headings:
                index = summary_chunk.find(heading)
                if index != -1:
                    # print("found before "+heading)
                    end_index = index
                    break
            
            if end_index != -1:
                summary_chunk = summary_chunk[:end_index]

            summary_text = summary_chunk.strip()
            break

            
    # Output the summary
    if summary_text == "":
        print("No Summary ")
    else:
        summary_text=summary_text.replace("\n"," ")
        print(summary_text)
    print("###################################################################################")

    #####
    #complete pdf text
    # def extract_structured_text():
    #     full_text = []
        
    #     for page in doc:
    #         blocks = page.get_text("blocks")  # Get text as blocks
    #         for block in blocks:
    #             text = block[4].strip()  # Block format: (x0,y0,x1,y1,text,block_no,block_type)
    #             if text:
    #                 full_text.append(text)
        
    #     return "\n".join(full_text)

    # # Usage
    # structured_text = extract_structured_text()
    # # print(structured_text)

    # # remove page number 
    # def remove_page_x_of_y(text):
    #     # Strict pattern for "Page X of Y" only
    #     pattern = re.compile(r'^Page\s+\d+\s+of\s+\d+\s*$', re.IGNORECASE)
        
    #     # Process each line
    #     lines = []
    #     for line in text.split('\n'):
    #         if not pattern.fullmatch(line.strip()):
    #             lines.append(line)
        
    #     return '\n'.join(lines)
    # #####

    # #general fn for target section 
    # def extract_section(structured_text,target_section):
    #     # Split text into sections using common headers
    #     sections = re.split(r'\n(Experience|Education|Skills|Projects|Summary)\n', structured_text, flags=re.IGNORECASE)

    #     if "Experience" in sections:
    #         if "Education" in sections:
    #             for page in doc:    
    #                 text = page.get_text()
    #                 start = text.find("Experience")
    #                 chunk = text[start:] 
    #                 end = chunk.find("Education")
    #                 chunk= chunk[:end]
    #         else:
    #             start = text.find("Experience")
    #             chunk = text[start:] 

        
    #     # Find the index of "target_section"
    #     try:
    #         exp_index = sections.index(target_section) if target_section in sections else -1
    #         if exp_index != -1 and exp_index + 1 <= len(sections):
    #             return sections[exp_index + 1].strip()
    #     except ValueError:
    #         pass
        
    #     return target_section+" section not found"

    # SKILLS
    top_skills = []
    capture = False

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                line_text = " ".join(span["text"] for span in line["spans"]).strip()

                if "Top Skills" in line_text:
                    capture = True
                    continue  # Skip the header itself

                if capture:
                    if len(top_skills) < 3:
                        top_skills.append(line_text)
                    if len(top_skills) == 3:
                        capture = False
                        break  # No need to continue after capturing 3 skills

    print(top_skills)

    print("###################################################################################")

    #EXPERIENCE
    capture = False
    experience_blocks = []

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue

            block_text = " ".join(
                span["text"] for line in block["lines"] for span in line["spans"]
            ).strip()

            if "Experience" in block_text:
                capture = True
                continue  # Skip the header block itself

            if capture:
                # Stop capturing when reaching the "Education" section
                if "Education" in block_text:
                    capture = False
                    break

                experience_blocks.append(block)

    # Now print the experience blocks
    Experience = []
    flag=False
    title=''
    description=''
    for block in experience_blocks:
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if not text or text == "\xa0":
                    continue  # Skip empty or non-breaking space
                if span["size"]==12 and span["font"]=="ArialUnicodeMS":
                    flag=True
                    title=span["text"]
                    continue
                if flag==True and span["size"]==11.5:
                    description=span["text"]
                    flag=False
                    job={
                        "title":title,
                        "description":description
                    }
                    title=''
                    description=''
                    Experience.append(job)
                    break
    print(Experience)

    print("###################################################################################")

     #Education
    capture = False
    education_blocks = []

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue

            block_text = " ".join(
                span["text"] for line in block["lines"] for span in line["spans"]
            ).strip()

            if "Education" in block_text:
                capture = True
                continue  # Skip the header block itself

            if capture:
                education_blocks.append(block)

    Education = []
    flag=False
    university=''
    degree=''
    for block in education_blocks:
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if not text or text == "\xa0":
                    continue  # Skip empty or non-breaking space
                if span["size"]==12 and span["font"]=="ArialUnicodeMS":
                    flag=True
                    university=span["text"]
                    continue
                if flag==True and span["size"]==10.5:
                    degree=span["text"]
                    degree=degree.split(",")
                    degree=degree[0]
                    flag=False
                    job={
                        "university":university,
                        "degree":degree
                    }
                    university=''
                    degree=''
                    Education.append(job)
                    break
    print(Education)

    print("###################################################################################")

    #OTHERS SECTION
    Certification=""
    HonorsAwards=""

    full_text = []
        
    for page in doc:
        blocks = page.get_text("blocks")  # Get text as blocks
        for block in blocks:
            text = block[4].strip()  # Block format: (x0,y0,x1,y1,text,block_no,block_type)
            if text:
                full_text.append(text)
        
    structured_text= "\n".join(full_text)
    # Split text into sections using common headers
    sections = re.split(r'\n(Experience|Education|Skills|Projects|Summary|Certifications|Honors-Awards)\n', structured_text, flags=re.IGNORECASE)

    if "Honors-Awards" in sections:
        HonorsAwards="yes"
    if "Certifications" in sections:
        Certification="yes"

    # Create a dictionary to hold all the extracted data
    extracted_data = {
        "name": name if name else "",
        "linkedin_url": url_text if 'url_text' in locals() else "",
        "location": location if location else "",
        "heading": heading_text if 'heading_text' in locals() else "",
        "summary": summary_text if 'summary_text' in locals() else "",
        "Certification": Certification if 'Certification' in locals() else "",
        "Honors-Awards": HonorsAwards if 'HonorsAwards' in locals() else "",
        "others_Score" : otherSum if 'otherSum' in locals() else "",
        "top_skills" : top_skills if 'top_skills' in locals() else "",
        "experience": Experience if 'Experience' in locals() else "",
        "education": Education if 'Education' in locals() else ""
    }
    
    # Save to JSON file
    # Check if file exists and load existing data, otherwise create a new list
    data=[]
    if os.path.exists('./ML/extracted_pdf_data.json'):
        with open('./ML/extracted_pdf_data.json', 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append the new profile data
    data.append(extracted_data)

    # Write updated data back to the file
    with open('./ML/extracted_pdf_data.json', 'w') as file:
        json.dump(data, file, indent=4)
    
    print("Data successfully saved to extracted_data.json")                    

    doc.close()
            

# pdfextractor()
