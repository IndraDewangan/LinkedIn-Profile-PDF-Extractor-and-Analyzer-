var Links=[];
var linkArrayIndex=0;
pdfNameList=[]
document.addEventListener('DOMContentLoaded',function(){
  document.getElementById('pdfFile').addEventListener('change', function(event) {
    const files = event.target.files;
    const previewContainer = document.getElementById('pdfPreview');
    previewContainer.innerHTML = '';  // clear previous previews
    k=0
    Array.from(files).forEach(file => {
      if (file.type === "application/pdf") {
        pdfNameList[k++]=file.name
        const previewBox = document.createElement('div');
        previewBox.style = "width:100px;height:130px;border:1px solid #ccc;border-radius:8px;padding:10px;margin:10px;background:#fff;box-shadow:0 0 8px rgba(0,0,0,0.1);text-align:center;";
        
        previewBox.innerHTML = `
          <i class="fa-solid fa-file-pdf" style="font-size: 36px; color: #e53935;"></i>
          <div style="font-size: 10px; margin-top: 8px; word-break: break-word; color: #333;">${file.name}</div>
        `;
  
        if (previewContainer.firstChild) {
          previewContainer.insertBefore(previewBox, previewContainer.firstChild);
        } else {
          previewContainer.appendChild(previewBox);
        }
      }
    });
    console.log(pdfNameList)
    previewContainer.style.display = "flex";
  });

    document.querySelector(".link").addEventListener("click", async function(){
        var unfilteredLinks=document.getElementById("ProfileLinks").value.split("\n");
        //
        Links=[];
        var count=0;
        unfilteredLinks.forEach(function(link){
            if(link.includes("www.linkedin.com")) {
                Links.push(link);
                count++;
            }      
        });    
        document.querySelector(".lab1111").textContent=`You Have Entered ${count} LinkedIn Profile's Link`;
        document.querySelector(".labelDiv1111").classList.remove("labelDiv2222");
        setTimeout(setTimer,5000);
        function setTimer(){
            document.querySelector(".labelDiv1111").classList.add("labelDiv2222");
        }

        //
        await unfilteredFilteredLengthCheck(unfilteredLinks,Links);
        await linkLengthCheck(Links);
        var Links2=Links[linkArrayIndex];

        await openTab(Links,Links2);
        linkArrayIndex++;        
    });
});
function uploadpdf(){
  document.querySelector('.analyze-container').style.display = 'flex';
  showSection2('analysis')
}

function uploadAndAnalyze() {
          document.querySelector('.analyze-button').style.display = 'none';
          extractedDatas=[]
          analysedDatas=[]
          document.getElementById('analyzing-dots').style.display = 'block';

          let dotCount = 1;
          const dotInterval = setInterval(() => {
          dotCount = (dotCount % 3) + 1;
          document.getElementById('dots').textContent = '.'.repeat(dotCount);
          }, 500);

          fetch('/analyze', {
              method: 'POST'
          })
          .then(response => response.json())
          .then(data => {
              console.log(data.results)
              analysedDatas=data.results
          })
          .catch(error => {
              console.error('Error during analysis:', error);
          });

          fetch('/extractedjson', {
            method: 'GET'
        })
        .then(response => response.json())
        .then(data => {
            extractedDatas=data
        })
        .catch(error => {
            console.error('Error fetching extracted file:', error);
        });
      // 
      const pollInterval = setInterval(() => {
                  fetch('/analysedjson',{method:'GET'}) // or your actual file path
                      .then(res => res.json())
                      .then(json => {
                          const data = json;  // if json is an array, use json.length
                          const listLength=data.length;
                          const extlength= extractedDatas.length
                          if (listLength == extlength) {
                              clearInterval(pollInterval);
                              clearInterval(dotInterval);
        
                              // Hide loading dots
                              document.querySelector('.analyze-container').style.display = 'none';
        
                              // Show results section
                              // document.getElementById('analysis-section').style.display = 'flex';
        
                              // Hide other elements
                              document.querySelector(".title").style.display = 'none';
                              document.querySelector(".score-circle").style.display = 'none';
                              
                              //cardHTML inserter
                              insertAnalysisCards(analysedDatas,extractedDatas)
                              //
        
                              showSection2('analysis');

                              
                          }
                      })
                      .catch(err => console.error("Polling error:", err));
        }, 1000); // Poll every 1 second
}


function showSection(sectionId) {
        document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
        document.getElementById(sectionId).classList.add('active');

        document.querySelectorAll('.nav button').forEach(btn => btn.classList.remove('active'));
        const clickedBtn = Array.from(document.querySelectorAll('.nav button')).find(btn => btn.textContent.toLowerCase() === sectionId);
        if (clickedBtn) clickedBtn.classList.add('active');
}
function showSection2(sectionId) {
        document.querySelectorAll('.section2').forEach(s => s.classList.remove('active'));
        document.getElementById(sectionId).classList.add('active');

        document.querySelectorAll('.nav-links button').forEach(btn => btn.classList.remove('active'));
        const clickedBtn = Array.from(document.querySelectorAll('.nav-links button')).find(btn => btn.textContent.toLowerCase() === sectionId);
        if (clickedBtn) clickedBtn.classList.add('active');
}

// link
async function unfilteredFilteredLengthCheck(unfilteredLinks,Links){
        if(unfilteredLinks.length>Links.length){
            var diff=unfilteredLinks.length-Links.length;
            document.querySelector(".lab1").textContent=`Your Entered ${diff} Link is/are not LinkedIn Profile Link \n Please enter only LinkedIn Profile Link...!!`;
            document.querySelector(".labelDiv1").classList.remove("labelDiv2");
            setTimeout(setTimer,5000);
            function setTimer(){
                document.querySelector(".labelDiv1").classList.add("labelDiv2");
            }
        }
}
async function linkLengthCheck(Links){
            document.querySelector(".lab11").textContent=`${Links.length} is/are LinkedIn Profile Link(s)`;
            document.querySelector(".labelDiv11").classList.remove("labelDiv22");
            setTimeout(setTimer,5000);
            function setTimer(){
                document.querySelector(".labelDiv11").classList.add("labelDiv22");
            }         
}
async function openTab(Links,Links2){
    if(Links.length>0){
            document.querySelector(".lab111").textContent="Opening wait a second ......";
            document.querySelector(".labelDiv111").classList.remove("labelDiv222");
            setTimeout(setTimer,5000);
            function setTimer(){
                document.querySelector(".labelDiv111").classList.add("labelDiv222");
                tab(Links,Links2)
            }
    }
}
async function tab(Links, Links2) {
    // Open the first link (Links2)
    const firstWindow = window.open(Links2, 'tab0', 'toolbars=0,width=415,height=400,left=200,top=200,scrollbars=0,resizable=1');
  
    // Wait until the first window closes
    const waitForClose = setInterval(() => {
      if (firstWindow.closed) {
        clearInterval(waitForClose);
        // Start opening links in the Links array one by one
        openSequentially(Links, 1);
      }
    }, 500);
  }
  
  // Recursive function to open each link one by one
  function openSequentially(links, index) {
    if (index >= links.length){
        showSection2('uploadpdf')
        return; 
    } // All links done
  
    const newWindow = window.open(links[index], `tab${index + 1}`, 'toolbars=0,width=415,height=400,left=200,top=200,scrollbars=0,resizable=1');
  
    const check = setInterval(() => {
      if (newWindow.closed) {
        clearInterval(check);
        openSequentially(links, index + 1); // Open next link
      }
    }, 500);
  }
  

  // main page score
  function getProfileFeedback(score) {
    if (score >= 85) {
        return "Excellent work! Your profile demonstrates a strong professional presence with great clarity, relevance, and structure. Just a few tweaks could make it perfect!";
    } else if (score >= 70) {
        return "Good job! Your profile is solid, but there’s room to enhance impact and specificity. Consider refining your headline or adding more measurable achievements.";
    } else if (score >= 50) {
        return "Decent foundation, but key improvements are needed. Strengthen your skills section, clarify your value proposition, and polish your experience descriptions for better results.";
    } else {
        return "Your profile needs significant improvement. Focus on clearer language, stronger keywords, and a more compelling summary to better highlight your potential.";
    }
}

function insertAnalysisCards(analysedDatas,extractedDatas) {
  const container = document.getElementById('analysis-section');
  const navBar = document.querySelector(".nav-links");
  
  analysedDatas.forEach((data, index) => {
    console.log(data["experience_scores"])
      const profileId = `profile-card-${index + 1}`;
      const comm=getProfileFeedback(data['overall_score'])
      const exdata = extractedDatas[index]
      // Create card HTML
      const cardHTML = `<div class="anscard" id=${profileId}>
      <div class="sidebar">
        <div class="logo">Linked Profile Extractor & Analyser</div>
        <div class="score-circle-wrapper">
          <div class="score-circle">${data['overall_score']}</div>
        </div>
        <div class="nav">
          <button onclick="showSection('(${index + 1})score')" class="active">(${index + 1})Score</button>
          <button onclick="showSection('(${index + 1})headline')">(${index + 1})Headline</button>
          <button onclick="showSection('(${index + 1})summary')">(${index + 1})Summary</button>
          <button onclick="showSection('(${index + 1})experience')">(${index + 1})Experience</button>
          <button onclick="showSection('(${index + 1})education')">(${index + 1})Education</button>
          <button onclick="showSection('(${index + 1})jobprediction')">(${index + 1})Job Prediction</button>
          <button onclick="showSection('(${index + 1})category')">(${index + 1})Category</button>
          <button onclick="showSection('(${index + 1})skills')">(${index + 1})Skills</button>
          <button onclick="showSection('(${index + 1})others')">(${index + 1})others</button>
        </div>
      </div>
    
      <div class="content">
        <div id="(${index + 1})score" class="section active">
          <div class="score-card">
            <div class="score-header">Overview</div>
            <div class="score-subtext">Welcome to your LinkedIn profile review, ${exdata['name']}!</div>
    
            <div class="score-details">
              <div class="score-breakdown">
                <div class="card-title">Score</div>
                <div class="score-circle-wrapper">
                  <div class="score-circle">${data['overall_score']}</div>
                </div>
                <div class="card-subtitle">${comm}</div>
              </div>
    
              <div class="score-steps">
                <ul>
                  <li><b>Breakdown</b>:<br>Your score is based on these five sections.</li>
                  
                </ul>
              </div>
            </div>
    
            <div class="score-bar">
              <div class="score-item">
                
                <div class="score-item-bar" style="width: ${data["headline_analysis"]["overall_score"]*10}%">Headline ${data["headline_analysis"]["overall_score"]}/10</div>
                
              </div>
              <div class="score-item">
                
                <div class="score-item-bar" style="width: ${data["summary_score"]["overall_score"]*10}%">Summary ${data["summary_score"]["overall_score"]}/10</div>
                
              </div>
              <div class="score-item">
                
                <div class="score-item-bar" style="width: ${data["summary_score"]["overall_score"]*10}%">Experience ${data["summary_score"]["overall_score"]}/10</div>
                
              </div>
              <div class="score-item">
                
                <div class="score-item-bar" style="width: ${data["education_scores"]["overall_score"]*10}%">Education ${data["education_scores"]["overall_score"]}/10</div>
                
              </div>
              <div class="score-item">
                
                <div class="score-item-bar" style="width: ${data["other"]["score"]*10}%">Other ${data["other"]["score"]}/10</div>
                
              </div>
            </div>
          </div>
        </div>
    
        <div id="(${index + 1})headline" class="section">
          <div class="score-card">                                   
            <div>
              <div class="card-title" style="font-size: 1.3rem;">Headline: Founder of The App Brewery | NHS Doctor | International Speaker Teaching 2 million people how to code.</div>
              <p style="margin-top: 15px; line-height: 1.6;">
                Your LinkedIn headline is the most seen component of your LinkedIn profile. It appears everywhere your name does and gives recruiters or your prospects a quick overview of who you are and why they should click your profile.
              </p>
              <hr style="margin: 30px 0; border: 0.5px solid #eee;">
              ${headlineScoreCard(data["headline_analysis"]["overall_score"],"headline","Your headline passes our checks on first look. It's important to check that you've included in the right keywords that you expect your prospects to be searching for.","Your headline might be missing important keywords or clarity. Consider refining it to better reflect your value and attract the right audience")}
            </div>
            
            <hr style="margin: 30px 0; border: 0.5px solid #eee;">
            
            <div>
              <div class="card-title" style="font-size: 1.3rem;">Core checks</div>
              <p style="margin-top: 10px; font-style: italic;">The following are common indicators of a strong LinkedIn headline. We evaluated your headline against each of these criteria. In addition to these criteria, you should also ensure you've included relevant keywords in your headline.</p>
              
              <div style="margin-top: 20px; padding: 15px; background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
              ${bulletPoints(data["headline_analysis"]["component_score"])}
              </div>
            </div>

            <hr style="margin: 30px 0; border: 0.5px solid #eee;">
            
            <div>
              <div class="card-title" style="font-size: 1.3rem;">Recommendation</div>
              <p style="margin-top: 10px; font-style: italic;"></p>
              
              <div style="margin-top: 20px; padding: 15px; background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                ${bulletPoints(data["headline_analysis"]["recommendations"])}
              </div>
            </div>
          </div>
        </div>
    
        <div id="(${index + 1})summary" class="section">
          <div class="score-card">                                   
            <div>
              <div class="card-title" style="font-size: 1.3rem;">Summary / 'About' section</div>
              <p style="margin-top: 15px; line-height: 1.6;">
                Your summary gives you the chance to make a great first impression to recruiters, potential clients, and other professionals who rely on LinkedIn, and is where you explain why you're the right fit.
              </p>
              <hr style="margin: 30px 0; border: 0.5px solid #eee;">
              ${headlineScoreCard(data["summary_score"]["overall_score"],"summary","Your summary effectively communicates your background and value proposition. It provides a clear narrative, uses relevant keywords, and highlights achievements that matter to your target audience.","Your summary may lack clarity or focus. Consider rewriting it to better highlight your strengths, include measurable results, and align more closely with the roles or opportunities you're targeting.")}
            </div>
            
            <hr style="margin: 30px 0; border: 0.5px solid #eee;">
            
            <div>
              <div class="card-title" style="font-size: 1.3rem;">Core checks</div>
              <p style="margin-top: 10px; font-style: italic;">Here's how we calculated your score - these are indicators of a strong LinkedIn summary.</p>
              
              <div style="margin-top: 20px; padding: 15px; background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
              ${bulletPoints(data["summary_score"]["component_score"])}
              </div>
            </div>

            <hr style="margin: 30px 0; border: 0.5px solid #eee;">
            
            <div>
              <div class="card-title" style="font-size: 1.3rem;">Recommendation</div>
              <p style="margin-top: 10px; font-style: italic;"></p>
              
              <div style="margin-top: 20px; padding: 15px; background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
              ${bulletPoints(data["summary_score"]["feedback"])}
              </div>
            </div>
          </div>
        </div>
  
      <div id="(${index + 1})experience" class="section">
        <div class="score-card">                                   
          <div>
            <div class="card-title" style="font-size: 1.3rem;">Experience</div>
            <p style="margin-top: 15px; line-height: 1.6;">
            People who've made it to your work experience have read your summary and are interested in your story so far. They now want to make sure you have the right experience to back up everything. Recruiters and your prospects will pay close attention to the job titles you've held and your achievements.
            </p>
            <hr style="margin: 30px 0; border: 0.5px solid #eee;">
            ${headlineScoreCard(data["experience_scores"]["overall_score"],"experience","Great job! Your experience section demonstrates strong alignment with your professional goals. It’s well-structured, achievement-focused, and clearly communicates your value to potential employers.","Your experience section seems to be lacking in detail or impact. Try adding measurable achievements, using action verbs, and emphasizing how your work made a difference to better capture recruiters’ attention.")}
          </div>
          
          <hr style="margin: 30px 0; border: 0.5px solid #eee;">
          
          <div>
            <div class="card-title" style="font-size: 1.3rem;">Core checks</div>
            <p style="margin-top: 10px; font-style: italic;">Each of your work experiences are checked against a set of key criteria that are considered indicators of a strong work experience. Often, recruiters or your prospects will only pay attention to your three most recent work experiences, so make sure all three fit the criteria..</p>

            <div style="margin-top: 20px; padding: 15px; background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            
  <div class="container-e">
    <div class="navigation-e">
      <button class="prevBtn-e" id="prev${data["overall-score"]}">PREVIOUS JOB</button>
      <button class="button-e" id="next${data["overall-score"]}">NEXT JOB</button>
    </div>
      
    <div id="experienceContainer">
    
      ${experienceCard(data["experience_scores"])}
    </div>
  </div>
            </div>
          </div>
        </div>
      </div>
  
      <div id="(${index + 1})education" class="section">
        <div class="score-card">
          <div class="score-header">UNICED IN REVIEW RESULTS</div>
          
          <div class="score-details" style="margin-top: 20px; align-items: center;">
            <div class="score-breakdown" style="text-align: center;">
              <div style="font-size: 2.5rem; font-weight: bold;">45</div>
              <div style="font-size: 1rem; margin-bottom: 20px;">Overall</div>
              
              <div class="card-title">Score</div>
              <ul style="list-style-type: none; margin-top: 10px;">
                <li>Headline 10/10</li>
                <li>Summary 1/10</li>
                <li>Experience 3/10</li>
                <li>Education 10/10</li>
                <li>Other 10/10</li>
              </ul>
            </div>
            
            <div class="score-steps">
              <div class="card-title">Keywords To Add</div>
              <ul style="list-style-type: none; margin-top: 10px;">
                <li>Keyword Analysis</li>
                <li>Networking</li>
              </ul>
            </div>
          </div>
          
          <hr style="margin: 30px 0; border: 0.5px solid #eee;">
          
          <div>
            <div class="card-title" style="font-size: 1.5rem;">Education</div>
            <p style="margin-top: 15px; line-height: 1.6;">
              Adding in your education gives you a more rounded profile and increases your network's overall reach. 
              Additionally, recruiters often pay particular attention to this section to check that you meet 
              qualifications they are looking for.
            </p>
          </div>
          
          <div style="margin-top: 30px;">
            <div class="card-title" style="font-size: 1.2rem; color: #0a66c2;">VIEW YOUR EDUCATION</div>
            <div style="margin-top: 20px; padding: 20px; background-color: #f8f9fa; border-radius: 8px;">
              <p>
                Your education section looks <strong>complete - well done!</strong> It contains the name of the 
                educational institutions you want to and the dates you went there.
              </p>
              <p style="margin-top: 15px;">
                Consider also adding in keywords which represent skills you learnt or any Activities or Societies 
                you took part in. This gives you a more rounded profile and ensures you rank higher when your 
                skills are searched for.
              </p>
              <p style="margin-top: 15px;">
                It's worth noting that adding in your education also ensures you are shown to alumni that attended 
                your school/university, and increases your network's overall reach.
              </p>
              <p style="margin-top: 15px;">
                Unlike a resume, you can also add in any additional courses (e.g. online courses) or certifications 
                that matter to your career or personal direction. If you graduated in the last 15 years, we'd 
                recommend also <strong>adding the dates</strong> you graduated from or attended this educational institution.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div id="(${index + 1})jobprediction" class="section">
        <div class="score-card">
          <div class="score-header">Job Title Prediction</div>
          
          <div style="margin-top: 20px;">
            <div class="card-title">Top Matching Roles</div>
            <div style="display: flex; gap: 20px; margin-top: 15px; flex-wrap: wrap;">
              <div style="padding: 15px; background: #f0f7ff; border-radius: 8px; flex: 1; min-width: 200px;">
                <div style="font-weight: bold;">1. Data Scientist</div>
                <div style="margin-top: 5px; font-size: 0.9rem;">87% match</div>
              </div>
              <div style="padding: 15px; background: #f0f7ff; border-radius: 8px; flex: 1; min-width: 200px;">
                <div style="font-weight: bold;">2. ML Engineer</div>
                <div style="margin-top: 5px; font-size: 0.9rem;">82% match</div>
              </div>
              <div style="padding: 15px; background: #f0f7ff; border-radius: 8px; flex: 1; min-width: 200px;">
                <div style="font-weight: bold;">3. AI Researcher</div>
                <div style="margin-top: 5px; font-size: 0.9rem;">78% match</div>
              </div>
            </div>
          </div>
    
          <div style="margin-top: 30px;">
            <div class="card-title">Improvement Suggestions</div>
            <ul style="margin-top: 10px; margin-left: 20px;">
              <li>Add more ML project details to increase match for AI roles</li>
              <li>Include specific frameworks (TensorFlow, PyTorch) you're proficient with</li>
              <li>Highlight publications or research experience</li>
            </ul>
          </div>
        </div>
      </div>

      <div id="(${index + 1})category" class="section">
        <div class="score-card">
          <div class="score-header">Industry Category Prediction</div>
          
          <div style="margin-top: 20px;">
            <div class="card-title">Your Primary Industry</div>
            <div style="margin-top: 10px; padding: 15px; background: #fff4e6; border-radius: 8px;">
              <div style="font-weight: bold; font-size: 1.2rem;">Artificial Intelligence & Machine Learning</div>
              <div style="margin-top: 5px;">92% confidence</div>
            </div>
          </div>
    
          <div style="margin-top: 25px;">
            <div class="card-title">Secondary Categories</div>
            <div style="display: flex; gap: 15px; margin-top: 10px; flex-wrap: wrap;">
              <div style="padding: 10px 15px; background: #f8f8f8; border-radius: 20px;">Data Science</div>
              <div style="padding: 10px 15px; background: #f8f8f8; border-radius: 20px;">Software Engineering</div>
              <div style="padding: 10px 15px; background: #f8f8f8; border-radius: 20px;">Cloud Computing</div>
            </div>
          </div>
        </div>
      </div>
  
      <div id="(${index + 1})skills" class="section">
        <div class="score-card">
          <div class="score-header">Skill Recommendations</div>
          
          <div style="margin-top: 20px;">
            <div class="card-title">Missing Key Skills</div>
            <div style="margin-top: 15px; display: flex; gap: 15px; flex-wrap: wrap;">
              <div style="padding: 10px 15px; background: #e6f7ff; border-radius: 20px; border: 1px solid #91d5ff;">
                <div style="font-weight: bold;">AWS/GCP</div>
                <div style="font-size: 0.8rem;">High demand</div>
              </div>
              <div style="padding: 10px 15px; background: #e6f7ff; border-radius: 20px; border: 1px solid #91d5ff;">
                <div style="font-weight: bold;">Docker</div>
                <div style="font-size: 0.8rem;">High demand</div>
              </div>
              <div style="padding: 10px 15px; background: #e6f7ff; border-radius: 20px; border: 1px solid #91d5ff;">
                <div style="font-weight: bold;">CI/CD</div>
                <div style="font-size: 0.8rem;">Growing importance</div>
              </div>
            </div>
          </div>
    
          <div style="margin-top: 30px;">
            <div class="card-title">Skill Optimization Tips</div>
            <ul style="margin-top: 10px; margin-left: 20px;">
              <li>Get 2-3 endorsements for your top skills</li>
              <li>Place most relevant skills in first 3 positions</li>
              <li>Add skills mentioned in your target job postings</li>
            </ul>
          </div>
        </div>
      </div>
  
      <div id="(${index + 1})others" class="section">
          <h2>Other Checks</h2>
          <p>You're almost done! Here are some final things you should do to make sure your profile is complete and memorable.</p>
          <div class="card-grid">
              <div class="card">
                  <h4 class="check">✔ You have a custom LinkedIn profile URL</h4>
                  <p>Great. Your LinkedIn profile URL is customized. This makes it easier to share and more memorable.</p>
              </div>
              <div class="card">
                  <h4 class="cross">✘ No Honors and Awards section found</h4>
                  <p>You haven’t included an Honors and Awards section. Consider adding awards you’ve received throughout your career.</p>
              </div>
              <div class="card">
                  <h4 class="check">✔ Location found on your public profile</h4>
                  <p>We found a location on your LinkedIn profile. This helps recruiters filter candidates by location.</p>
              </div>
              <div class="card">
                  <h4 class="check">✔ We found a Certifications section</h4>
                  <p>Great! You've included a Certifications section, helping you show up in search results.</p>
              </div>
              <div class="card">
                  <h4 class="check">✔ Professional profile photo</h4>
                  <p>You have a professional photo. This increases profile clicks and improves impressions.</p>
              </div>
              <div class="card">
                  <h4 class="check">✔ Found Skills section</h4>
                  <p>You have at least 3 skills listed on your LinkedIn profile. Great for appearing in skill-based searches!</p>
              </div>
          </div>
      </div>
    </div>
  </div>`;

      // Inject card into DOM
      const wrapper = document.createElement('div');
      wrapper.innerHTML = cardHTML;
      container.appendChild(wrapper);

      // Create nav button
      const navBtn = document.createElement('button');
      navBtn.textContent = `${index + 1}`;
      navBtn.classList.add('profile-tab');
      if (index === 0) navBtn.classList.add('active-tab');

      navBtn.addEventListener('click', () => {
          // Hide all cards
          document.querySelectorAll('.anscard').forEach(card => {
              card.style.display = 'none';
          });
          // Show selected card
          document.getElementById(profileId).style.display = 'flex';
          showSection(`(${index + 1})score`)
          // Update active tab styling
          document.querySelectorAll('.profile-tab').forEach(btn => btn.classList.remove('active-tab'));
          navBtn.classList.add('active-tab');
      });

      navBar.appendChild(navBtn);
      document.querySelectorAll('.anscard').forEach(card => {
        card.style.display = 'none';});
      document.getElementById('profile-card-1').style.display = 'flex';
      
      var experiences = document.querySelectorAll('.experience-box-e');
    let currentIndex = 0;

    document.getElementById("next"+data["overall-score"]).addEventListener('click', () => {
      if (currentIndex < experiences.length - 1) {
        experiences[currentIndex].classList.remove('active-e');
        currentIndex++;
        experiences[currentIndex].classList.add('active-e');
      }
    });

    document.getElementById("prev"+data["overall-score"]).addEventListener('click', () => {
      if (currentIndex > 0) {
        experiences[currentIndex].classList.remove('active-e');
        currentIndex--;
        experiences[currentIndex].classList.add('active-e');
      }
    });

  });
}
function headlineScoreCard(score,section,message1,message2) {
  const isGood = score >= 7;

  const icon = isGood ? '✔' : '✖';
  const colorClass = isGood ? 'green' : 'red';
  const message = isGood
    ? `Your ${section} looks good.`
    : `Your ${section} needs improvement.`;
  const description = isGood
    ? message1
    : message2;
  return `
    <div class="card-sm"> 
      <div class="score-circle-sm ${colorClass}"> 
        <div class="score-number-sm">${score}</div> 
        <div class="score-label-sm">${isGood ? "Perfect" : "Needs Work"}</div> 
      </div>
      <div class="content-sm"> 
        <h3>${icon} ${message}</h3>
        <p>${description}</p>
      </div>
    </div>
  `;
}
//for components and recommendation
function bulletPoints(subsection){
  item=''
  for(var bullet of subsection){
    item +=`<div style="display: flex; align-items: center; margin-bottom: 15px;">
    <div style="width: 30px; height: 30px; background-color: #ddd; border-radius: 50%; margin-right: 10px;"></div>
    <div>
      <div style="font-weight: bold;">${bullet}</div>
    </div>
  </div>`
  }
  return item

}

function experienceCard(section){
  temp=''
  for (const [expLabel, expData] of Object.entries(section)) {
    console.log(expData)
    if(expLabel=="overall_score"){
      continue
    }
    if(expLabel=="Experience_1"){
      temp+=`<div class="experience-box-e active-e">
          <div class="experience-title-e">${expLabel} of 5: Checks</div>
          <div class="score-indicator-e">${expData["overall_score"]}</div>
          <p>We scored each of your experiences on the following checks. Each experience was scored out of 10.</p>
          <div class="section-e">${expData["component_score"][0]}</div>
          <div class="section-e">${expData["component_score"][1]}</div>
          <div class="section-e">${expData["component_score"][2]}</div>
          <div class="section-e">${expData["component_score"][3]}</div>
          <div class="section-e">${expData["component_score"][4]}</div>
          <div class="section-e">${expData["component_score"][5]}</div>
          <div class="section-e">${expData["component_score"][6]}</div>
          <div class="section-e">${expData["component_score"][7]}</div>
          <div class="section-e">${expData["component_score"][8]}</div>
          <div class="section-e">${expData["component_score"][9]}</div>
        </div>`
    }else{
        temp+=`<div class="experience-box-e">
        <div class="experience-title-e">${expLabel} of 5: Checks</div>
        <div class="score-indicator-e"">${expData["overall_score"]}</div>
        <p>We scored each of your experiences on the following checks. Each experience was scored out of 10.</p>
        <div class="section-e">${expData["component_score"][0]}</div>
        <div class="section-e">${expData["component_score"][1]}</div>
        <div class="section-e">${expData["component_score"][2]}</div>
        <div class="section-e">${expData["component_score"][3]}</div>
        <div class="section-e">${expData["component_score"][4]}</div>
        <div class="section-e">${expData["component_score"][5]}</div>
        <div class="section-e">${expData["component_score"][6]}</div>
        <div class="section-e">${expData["component_score"][7]}</div>
        <div class="section-e">${expData["component_score"][8]}</div>
        <div class="section-e">${expData["component_score"][9]}</div>
      </div>`
    }
  }
  
    return temp
}