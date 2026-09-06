<h1 align="center">
🩺 TriNexus | VitaGuard AI
</h1>

<h3 align="center">
Explainable AI Based Healthcare Decision Support System for Early Diabetes Risk Screening
</h3>

<p align="center">

<img src="https://img.shields.io/badge/AI-Healthcare-blue">
<img src="https://img.shields.io/badge/ML-XGBoost-green">
<img src="https://img.shields.io/badge/XAI-SHAP-orange">
<img src="https://img.shields.io/badge/API-FastAPI-red">
<img src="https://img.shields.io/badge/Frontend-React.js-purple">

</p>


---

<h2>📌 Project Overview</h2>

<strong>VitaGuard AI</strong> is an <strong>Explainable Artificial Intelligence (XAI) based healthcare decision support system</strong> designed for early diabetes risk screening.

The system analyzes patient health information and predicts diabetes risk using Machine Learning models while providing transparent explanations using SHAP (SHapley Additive exPlanations).

The system assists healthcare professionals by identifying possible diabetes risk factors while keeping final medical decisions under qualified healthcare providers.


---

<h2>🎯 Project Objectives</h2>

| Objective | Description |
|---|---|
| Early Detection | Identify diabetes risk before severe complications develop |
| AI Assistance | Provide ML-based diabetes risk prediction |
| Explainability | Explain why a prediction was generated |
| Healthcare Support | Assist doctors and healthcare workers during decision making |


---

<h2>🚀 Key Features</h2>

| Feature | Description |
|---|---|
| 🧠 Risk Prediction | Predict diabetes risk using machine learning models |
| 📊 Risk Classification | Categorize patients into Low, Medium and High risk |
| 🔍 Explainable AI | Generate prediction explanations using SHAP |
| 📈 Feature Importance | Identify important health factors affecting prediction |
| 🏥 Healthcare Dashboard | Interactive patient analysis dashboard |
| 🤖 Decision Support | Provide AI-assisted clinical insights |


---

<h2>🏗️ AI Workflow</h2>

```text
Patient Health Information

          ↓

Data Preprocessing

          ↓

Machine Learning Model

          ↓

Diabetes Risk Prediction

          ↓

SHAP Explainable AI

          ↓

Risk Explanation

          ↓

Healthcare Dashboard

          ↓

Clinical Decision Support

<h2>🧬 System Architecture</h2>
                 User Input

                     ↓

        Patient Health Information

                     ↓

            Data Processing Layer

                     ↓

          Machine Learning Model

              ↙             ↘

 Risk Classification     SHAP Explanation

              ↘             ↙

          Healthcare Dashboard

                     ↓

        Medical Decision Support

<h2>🛠️ Technology Stack</h2> <h3>🤖 Machine Learning</h3>
| Technology   | Purpose                       |
| ------------ | ----------------------------- |
| Python       | Core programming language     |
| Scikit-learn | Machine learning development  |
| XGBoost      | Classification model          |
| Pandas       | Data processing               |
| NumPy        | Numerical computation         |
| SHAP         | Explainable AI implementation |
<h3>⚙️ Backend</h3>
| Technology | Purpose                 |
| ---------- | ----------------------- |
| FastAPI    | Backend API development |
| REST API   | AI model communication  |
<h3>🎨 Frontend</h3>
| Technology    | Purpose                    |
| ------------- | -------------------------- |
| React.js      | Interactive dashboard      |
| UI Components | Patient data visualization |
<h3>🔧 Development Tools</h3>
| Tool             | Usage                             |
| ---------------- | --------------------------------- |
| GitHub           | Version control and collaboration |
| Google Colab     | Model training                    |
| Jupyter Notebook | Data analysis and experiments     |
<h2>📊 Diabetes Risk Classification</h2>
| Risk Level     | Meaning                                  |
| -------------- | ---------------------------------------- |
| 🟢 Low Risk    | Few diabetes indicators detected         |
| 🟡 Medium Risk | Multiple risk factors require monitoring |
| 🔴 High Risk   | Further medical evaluation recommended   |
<h2>🔍 Explainable AI using SHAP</h2>

Traditional AI models often behave like black boxes.

VitaGuard AI uses SHAP to explain:
| Explanation Type     | Description                          |
| -------------------- | ------------------------------------ |
| Positive Factors     | Factors increasing diabetes risk     |
| Negative Factors     | Factors reducing diabetes risk       |
| Feature Contribution | Percentage influence of each feature |

Prediction:
High Diabetes Risk


Major Contributing Factors:

High Glucose Level       +35%

High BMI                 +20%

Age                      +15%

Low Physical Activity    +10%
<h2>📂 Project Structure</h2>

TriNexus-VitaGuard-AI

│
├── ML_Model
│   ├── preprocessing
│   ├── training
│   ├── evaluation
│   └── explainable_ai
│
├── backend
│   ├── FastAPI
│   ├── routes
│   └── services
│
├── frontend
│   ├── React Dashboard
│   └── UI Components
│
├── dataset
│
├── docs
│
└── README.md
<h2>⚡ Installation & Setup</h2> <h3>Clone Repository</h3>

git clone https://github.com/yourusername/TriNexus-VitaGuard-AI.git

cd TriNexus-VitaGuard-AI

<h3>Install Dependencies</h3>
pip install -r requirements.txt
<h3>Run Backend</h3>
uvicorn main:app --reload

<h3>Run Frontend</h3>
npm install

npm start

<h2>👥 Team Members & Responsibilities</h2>
| Member              | Branch          | Responsibility                              |
| ------------------- | --------------- | ------------------------------------------- |
| Tanvir Hosen Nishat | tanvir-ml       | AI Model Development, Dataset Analysis, XAI |
| Farhan              | farhan-backend  | Backend API and Integration                 |
| Siam                | siam-frontend   | Frontend Dashboard Development              |
| Monima              | monima-docs     | Documentation and Presentation              |
| Zayed               | Medical Review  | Healthcare Validation                       |
| Fardin              | Dataset Support | Dataset Organization and Analysis           |

<h2>🔄 GitHub Collaboration Workflow</h2>
| Step | Action                           |
| ---- | -------------------------------- |
| 1    | Pull latest changes from main    |
| 2    | Work on assigned branch          |
| 3    | Commit changes                   |
| 4    | Push own branch                  |
| 5    | Create Pull Request              |
| 6    | Code Review                      |
| 7    | Merge into protected main branch |

<h2>🚀 Future Scope</h2>
| Improvement                  | Description                       |
| ---------------------------- | --------------------------------- |
| Clinical Dataset Integration | Use large healthcare datasets     |
| Mobile Application           | Patient monitoring application    |
| IoT Integration              | Wearable health device connection |
| Real-Time Monitoring         | Continuous health tracking        |
| Advanced AI Models           | Improve prediction performance    |
| Healthcare Deployment        | Hospital-level implementation     |

<h2>⚠️ Disclaimer</h2>
VitaGuard AI is a healthcare decision support system and does not replace professional medical diagnosis.

AI predictions are intended only to assist healthcare professionals.

Final medical decisions must always be made by qualified healthcare providers.

<h2>🌟 Vision</h2> <blockquote>

Using Explainable Artificial Intelligence to make early diabetes screening smarter, transparent, and accessible.

</blockquote> ```