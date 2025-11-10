## Manjit Singh

## Data Science Portfolio



----------------------------------------------------------------------------------------------------------------------------------------------------------------------------



## Education

**Executive Post Graduate Program in Data Science & AI**  
*IIIT Bangalore*  
*Oct '23 – Jan '25*

**Bachelor of Commerce (B.Com)**  
*Gauhati University*  
*Jul '20 – Jun '23*



----------------------------------------------------------------------------------------------------------------------------------------------------------------------------



### **Professional Experience**

**Associate - Data Analyst**  
**Samhita | Remote | Nov 2024 – Present** 

At Samhita, I support data-driven decision-making for financial inclusion projects by managing datasets, uncovering insights, and automating processes across multiple programs. I work closely with program, tech, and MEL teams to ensure data flows efficiently from collection to actionable insight.

• Key Responsibilities:

🧹 Data Management & Reporting:
   - Maintain and update datasets across Access to Finance initiatives including Pre-Credit Score (PCS), Returnable Grants, and Formal Loans.
   - Consolidate repayment data from various sources and support data cleaning, validation, and standardization.
     

📊 Dashboarding & Visualization:
   - Assist in building and maintaining dashboards for internal monitoring and external reporting.
   - Support real-time updates and enhancements for evolving project needs.


🔍 Data Analysis & Insight Generation:
   - Conduct exploratory data analysis (EDA) to identify trends and anomalies.
   - Extract, transform, and analyze data from SQL databases to inform strategic decisions.


⚙️ Process Automation & GenAI Adoption:
   - Develop automation scripts using Python, SQL, and VBA to streamline recurring data workflows.
   - Contribute to identifying and piloting GenAI use cases to enhance team productivity.


🤝 Cross-Team Collaboration:
   - Coordinate with cross-functional teams to respond to data requests and support grant reports, donor updates, and tool development.
   - Assist in the design of data collection formats and tracking templates for improved monitoring.


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------



### Internships

**Business Analyst Intern**  
**Quest Global Technologies Ltd | Remote | Nov 2023 – May 2024**    

- Led the automation of reporting workflows using **Excel**, **SQL**, and visualization tools like **PowerBI**, enabling **real-time KPI tracking** and reducing manual effort by **10%**.  
- Designed and analyzed **A/B testing strategies** to evaluate marketing effectiveness, resulting in a **5% boost in conversion rates** and improved **ROI**.  
- Applied **data cleaning**, **segmentation**, and **exploratory analysis** to uncover business patterns and improve **customer targeting**.  
- Collaborated with **product** and **sales teams** to deliver **data-driven insights**, enhancing operational decision-making and **customer engagement**.  
- Created **reproducible reports**, dashboards, and **documentation** to support strategic planning and ensure **data transparency** across teams.




----------------------------------------------------------------------------------------------------------------------------------------------------------------------------



## Key Projects



### 🛍️ Sentiment-Based Product Recommendation System  
**Tech Stack:** Python, Flask, Scikit-learn, Numpy, Pandas, NLTK/SpaCy, HTML/CSS, Collaborative Filtering, Joblib, Git/GitHub

An intelligent **hybrid recommendation engine** that fuses **collaborative filtering** with **sentiment analysis** to deliver personalized and emotionally aware product recommendations. The system interprets user opinions from product reviews to refine recommendation accuracy beyond conventional approaches.

- **Problem Addressed:** Traditional recommendation systems rely solely on numerical ratings and overlook the emotional tone within user reviews — resulting in suggestions that often mismatch actual user sentiment.  
- **Solution Overview:**  
   - Processed review text using **NLTK** for tokenization, stopword removal, and lemmatization.
   - Transformed textual data with **TF-IDF vectorization** to capture key features.
   - Trained a **XGBoost** classifier to predict sentiment polarity (positive/negative).
   - Enhanced **collaborative filtering** by integrating sentiment scores to improve recommendation precision.
   - Deployed a **Flask web application** that interactively serves product recommendations based on both ratings and sentiment context.

✅ **Impact:** Delivered a **sentiment-aware recommender system** that significantly improves recommendation relevance and user satisfaction by aligning suggestions with customer opinions — creating a more personalized and engaging shopping experience.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/ManjitSingh2003/sentiment-based-recommendation-system)


### 📄 Helmate AI: Retrieval-Augmented Generation (RAG) System  
**Tech Stack:** Python, LangChain, GROQ AI, SBERT, ChromaDB, FAISS, Streamlit, PyPDF, dotenv, NumPy 

HelpMate AI is an advanced *Retrieval-Augmented Generation (RAG)* system built to simplify how users search, interpret, and compare complex **insurance policy documents**. The system enables natural language queries and returns **precise, context-aware answers** directly extracted from over **200 real policy PDFs**.

- **Problem Addressed:** Navigating lengthy insurance documents to find specific information is often tedious and time-consuming for customers and agents alike.
- **Solution Overview:**  
   - Implemented **semantic retrieval** using **SBERT embeddings** for accurate context matching.
   - Used **ChromaDB** as the vector store for fast and scalable document retrieval.
   - Integrated **LangChain** to orchestrate the retrieval–generation pipeline and **GROQ AI** for human-like response generation.
   - Optimized chunking, metadata handling, and text-splitting to enhance precision and maintain context.
   - Deployed a **Streamlit web app** to provide an intuitive interface for users to query documents in real time.
     
✅ **Impact:** Delivered a **scalable, open-source, and cost-efficient RAG** solution that improves information accessibility and interpretability—enabling users to obtain accurate, natural language answers from complex insurance policies **without paid APIs**.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/ManjitSingh2003/Helpmate-AI-RAG)


### 🧩 Customer Segmentation using Clustering  
**Tech Stack:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, Streamlit, Pytest, Docker, GitHub Actions

An end-to-end **unsupervised machine learning pipeline** designed to segment retail customers based on purchasing patterns. The system enables data-driven marketing strategies by uncovering behavioral clusters and delivering actionable insights through an interactive dashboard.

- **Objective:** Retail businesses often struggle to identify which customers are most valuable or at risk due to unstructured transactional data, leading to inefficient marketing and retention strategies.
- **Approach:**  
   - Engineered **RFM (Recency, Frequency, Monetary)** features from raw transactional data to quantify customer value.
   - Applied **K-Means** and **Hierarchical Clustering** to group customers with similar behaviors.
   - Validated cluster quality using **Elbow Method** and **Silhouette Scores** to ensure meaningful segmentation.
   - Built an interactive **Streamlit dashboard** to visualize segments, view metrics, and export labeled customer lists.
   - Containerized the solution using **Docker** and **automated testing/deployment** via **Pytest** and GitHub Actions.

✅ **Impact:** Delivered a production-ready segmentation system that enables targeted marketing, higher retention, and data-driven business decisions. The project demonstrates the integration of machine learning, visualization, and deployment workflows into a single, scalable analytics product.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/ManjitSingh2003/customer-segmentation-clustering-project)


### 🧾 Automatic Ticket Classification System  
**Tech Stack:** Python, Scikit-learn, NLTK, Pandas  

Built an intelligent **automated ticket classification** system to streamline customer support by accurately routing tickets to the relevant departments.

- **Problem Addressed:** Manual triaging of support tickets was time-consuming and error-prone, often delaying response times and affecting customer satisfaction.  
- **Solution Overview:**  
   - Applied **text preprocessing and NLP techniques** (tokenization, stopword removal, lemmatization) using NLTK.  
   - Performed **topic modeling** with Non-negative Matrix Factorization (NMF) to uncover hidden structures in unstructured ticket texts.  
   - Built and evaluated multiple classifiers using Scikit-learn for efficient ticket label prediction based on content.  
   - Developed a **reproducible Jupyter Notebook pipeline**, allowing for future scaling and integration.  

✅ **Impact:** Automated classification significantly reduced manual workload, improved response speed, and laid a foundation for intelligent support systems.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/ManjitSingh2003/automatic_ticket_classification)


### 🎬 RSVP Movies Analysis – SQL-Based Data Exploration  
**Tech Stack:** MySQL, SQL Joins, Aggregations, Subqueries  

Performed a comprehensive **SQL analysis on the RSVP Movies dataset** to derive actionable insights into movie performance, viewer behavior, and content trends.

- **Objective:** Analyze a real-world movie dataset to uncover trends related to movie views, genres, languages, platforms, and user engagement.  
- **Approach:**  
   - Wrote optimized **SQL queries** to explore viewer demographics, most-watched genres/languages, and regional popularity.  
   - Used **joins, window functions, subqueries**, and aggregations to extract multi-dimensional insights across different entity tables.  
   - Investigated **monthly trends** in user ratings and view counts to identify periods of peak engagement.  
   - Segmented data by **platform and genre** to assist content curation decisions for OTT platforms.  
   - Delivered conclusions in a structured manner for use in dashboards or further analysis.

✅ **Impact:** Enabled data-driven decisions for movie curation, marketing strategies, and platform-specific content optimization through insightful SQL reporting.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/ManjitSingh2003/RSVP_Movies_SQL_Analysis)


### 📉 Telecom Churn Prediction  
**Tech Stack:** Python, Scikit-learn, Pandas, Seaborn, Matplotlib  

Developed a **predictive machine learning model** to identify telecom customers likely to churn, enabling proactive retention strategies.

- **Objective:** Empowered telecom providers to **reduce customer churn** by predicting at-risk customers using historical usage and service data.  
- **Approach:**  
   - Conducted **extensive EDA** to uncover trends and patterns impacting churn (e.g., service issues, contract type, tenure).  
   - Handled missing values, encoded categorical features, and applied **feature selection techniques** to improve model performance.  
   - Trained and compared **multiple classification models** (Logistic Regression, Decision Trees, Random Forest, XGBoost), tuning hyperparameters for optimal results.  
   - Evaluated performance using **confusion matrix, ROC-AUC, precision, and recall** to ensure model reliability and interpretability.  
   - Visualized key insights with **bar plots, correlation heatmaps**, and model comparison charts.

✅ **Impact:** Equipped stakeholders with a reliable churn prediction system, enabling **targeted interventions** and improved customer retention.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/ManjitSingh2003/Telecom-Churn-Prediction)


### 🎯 Lead Scoring Model  
**Tech Stack:** Python, Scikit-learn, Pandas, Seaborn, Matplotlib  

Built a **machine learning model** to identify and prioritize marketing leads with the highest potential to convert, helping businesses optimize their sales funnel.

- **Objective:** Streamlined the lead qualification process by scoring leads based on behavioral and demographic attributes to **maximize conversion rates**.  
- **Approach:**  
   - Performed thorough **data cleaning, feature engineering**, and **outlier treatment** to ensure data quality.  
   - Conducted **EDA** to uncover key patterns, such as lead source, total time spent, and page activity influencing conversion likelihood.  
   - Applied **Logistic Regression** and **Random Forest** classifiers with **grid search-based hyperparameter tuning**.  
   - Evaluated model using **ROC-AUC, F1-score, and confusion matrix** to balance precision and recall for business decisions.  
   - Presented key takeaways through **visual dashboards and feature importance charts** for actionable stakeholder insights.

✅ **Impact:** Enabled targeted sales efforts by automating lead prioritization, thereby **reducing time-to-conversion and improving ROI**.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/ManjitSingh2003/Lead-Scoring)


### 🚗 Vehicle Dataset - Advanced EDA & Insights  
**Tech Stack:** Python, Pandas, Seaborn, Matplotlib, Plotly  

Conducted an **in-depth Exploratory Data Analysis (EDA)** on a vehicle dataset to uncover trends, patterns, and actionable insights in the automotive space.

- **Objective:** Explored and visualized key factors influencing vehicle pricing, fuel efficiency, and design to support data-driven decisions for buyers and sellers.  
- **Approach:**  
   - Handled **missing values, duplicates**, and formatted features for consistency.  
   - Applied **univariate, bivariate, and multivariate analysis** to understand the influence of attributes like engine size, brand, fuel type, and body style.  
   - Leveraged **correlation matrices, pair plots, violin plots, and box plots** to visually interpret data relationships.  
   - Used **interactive Plotly dashboards** for deeper insights and stakeholder-friendly visualizations.  
   - Summarized insights with **clear narratives and business implications** after each section to make the notebook self-explanatory.

✅ **Impact:** Revealed strong correlations between engine size, brand, and price; highlighted the impact of fuel type on efficiency; and created a foundation for future predictive modeling.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/ManjitSingh2003/Vehicle_EDA_Analysis)


### 🚴‍♂️ Bike Sharing Demand Prediction – Linear Regression Model  
**Tech Stack:** Python, Pandas, Seaborn, Matplotlib, Scikit-learn  

Built a linear regression model to **predict bike rental demand** based on historical usage patterns and environmental conditions.

- **Objective:** Analyze factors influencing daily bike rentals and develop a predictive model to estimate rental counts for a bike-sharing service.  
- **Approach:**  
   - Conducted **EDA** to understand seasonal trends, weather impact, and user behavior across weekdays vs. weekends.  
   - Applied **feature engineering** to extract relevant time-based and categorical features (e.g., working day, holiday, temperature bins).  
   - Built and evaluated a **multiple linear regression model**, interpreting coefficients to assess variable influence.  
   - Used **R² score, RMSE, and residual plots** to assess model accuracy and refine performance.  
   - Visualized relationships with scatter plots and heatmaps for better insight into dependencies.

✅ **Impact:** Provided an interpretable model that helps stakeholders anticipate rental demand and make informed operational decisions such as fleet planning and staff allocation.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/ManjitSingh2003/bike-sharing-linear-regression)


### 💳 Credit Data Analysis – Deep Dive into Borrower Behavior  
**Tech Stack:** Python, Pandas, Seaborn, Matplotlib, Plotly  

Performed a comprehensive **Exploratory Data Analysis (EDA)** on a credit dataset to understand patterns related to credit risk, borrower profiles, and loan characteristics.

- **Objective:** Investigated how factors like age, income, loan purpose, credit history, and employment status impact loan approval and creditworthiness.  
- **Approach:**  
   - Conducted **data cleaning, feature formatting, and null value handling** for analysis readiness.  
   - Performed **univariate and bivariate analysis** to explore distributions and identify key differentiators among defaulters and non-defaulters.  
   - Utilized **heatmaps, bar plots, histograms, and interactive visualizations** to reveal correlations and highlight risk indicators.  
   - Analyzed **demographic and financial features** in relation to loan status and provided clear, intuitive visual storytelling.  
   - Integrated **domain-relevant business interpretations** to derive real-world insights from patterns in the data.

✅ **Impact:** Enabled a clearer understanding of high-risk segments, borrower behavior, and key attributes associated with loan defaults—laying groundwork for future predictive modeling.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/ManjitSingh2003/Credit-EDA-Analysis)

