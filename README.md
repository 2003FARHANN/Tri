\# 🩺 TriNexus | VitaGuard AI



<p align="center">



<img src="https://img.shields.io/badge/AI-Healthcare-blue">

<img src="https://img.shields.io/badge/ML-XGBoost-green">

<img src="https://img.shields.io/badge/XAI-SHAP-orange">

<img src="https://img.shields.io/badge/API-FastAPI-red">

<img src="https://img.shields.io/badge/Frontend-React.js-purple">



</p>





\## 📌 Project Overview



\*\*VitaGuard AI\*\* is an \*\*Explainable Artificial Intelligence (XAI)-based healthcare decision support system\*\* designed for \*\*early diabetes risk screening\*\*.



The system analyzes patient health information including:



\- Age

\- BMI

\- Glucose Level

\- Blood Pressure

\- Lifestyle Factors

\- Health History



Using Machine Learning models, VitaGuard AI predicts diabetes risk levels and provides transparent explanations using \*\*SHAP (SHapley Additive exPlanations)\*\*.



The goal of this project is to assist healthcare professionals in identifying diabetes risk at an early stage while ensuring that final medical decisions remain with qualified healthcare providers.





\---



\# 🎯 Project Objectives



| Objective | Description |

|---|---|

| Early Detection | Identify individuals with potential diabetes risk before complications develop |

| AI Assistance | Provide machine learning-based risk prediction |

| Explainability | Explain why AI generated a specific prediction |

| Healthcare Support | Assist doctors and healthcare workers in decision-making |





\---



\# 🚀 Key Features





| Feature | Description |

|---|---|

| 🧠 Diabetes Risk Prediction | Predicts diabetes risk using Machine Learning models |

| 📊 Risk Classification | Categorizes patients into Low, Medium, and High risk groups |

| 🔍 Explainable AI | Uses SHAP to explain AI predictions |

| 📈 Feature Importance | Shows important health factors influencing prediction |

| 🏥 Healthcare Dashboard | Interactive dashboard for patient analysis |

| 🤖 Decision Support System | Helps healthcare professionals make informed decisions |





\---



\# 🏗️ AI Workflow





```

Patient Health Data



&#x20;       ↓



Data Preprocessing



&#x20;       ↓



Machine Learning Model



&#x20;       ↓



Diabetes Risk Prediction



&#x20;       ↓



SHAP Explainable AI



&#x20;       ↓



Risk Explanation



&#x20;       ↓



Healthcare Dashboard



&#x20;       ↓



Clinical Decision Support

```





\---



\# 🧬 System Architecture





```

&#x20;                   User Input



&#x20;                       ↓



&#x20;         Patient Health Information



&#x20;                       ↓



&#x20;             Data Processing Layer



&#x20;                       ↓



&#x20;           Machine Learning Model



&#x20;             ↙                  ↘



&#x20;Risk Classification        SHAP Explanation



&#x20;             ↘                  ↙



&#x20;         Healthcare Dashboard



&#x20;                       ↓



&#x20;         Medical Decision Support

```





\---



\# 🛠️ Technology Stack





\## 🤖 Machine Learning



| Technology | Purpose |

|---|---|

| Python | Core programming language |

| Scikit-learn | Machine Learning model development |

| XGBoost | Classification model |

| Pandas | Data processing and analysis |

| NumPy | Numerical computation |

| SHAP | Explainable AI implementation |





\---



\## ⚙️ Backend



| Technology | Purpose |

|---|---|

| FastAPI | Backend API development |

| REST API | Communication between AI model and application |





\---



\## 🎨 Frontend



| Technology | Purpose |

|---|---|

| React.js | Interactive dashboard development |

| UI Components | Patient data visualization |





\---



\## 🔧 Development Tools



| Tool | Usage |

|---|---|

| GitHub | Version control and collaboration |

| Google Colab | Model training and experimentation |

| Jupyter Notebook | Data analysis and testing |





\---



\# 📊 Diabetes Risk Classification





| Risk Level | Description |

|---|---|

| 🟢 Low Risk | Patient shows fewer diabetes risk indicators |

| 🟡 Medium Risk | Patient has several risk factors requiring monitoring |

| 🔴 High Risk | Patient requires further medical evaluation |





\---



\# 🔍 Explainable AI (SHAP)





Traditional AI models often work like a black box.



VitaGuard AI uses \*\*SHAP (SHapley Additive exPlanations)\*\* to explain:



\- Which health factors increased diabetes risk

\- Which factors reduced risk

\- Contribution percentage of each feature





Example:





```

Prediction: High Diabetes Risk





Major Contributing Factors:



↑ High Glucose Level        +35%



↑ High BMI                  +20%



↑ Age                       +15%



↓ Physical Activity         +10%

```





\---



\# 📂 Project Structure





```

VitaGuard-AI/



│

├── AI\_Model/

│   ├── preprocessing.py

│   ├── train\_model.py

│   ├── shap\_analysis.py

│

├── Backend/

│   ├── FastAPI/

│   ├── API Routes/

│

├── Frontend/

│   ├── React Dashboard/

│

├── Dataset/

│

├── Documentation/

│

└── README.md



```





\---



\# ⚡ Installation \& Setup





\## Clone Repository



```bash

git clone https://github.com/yourusername/VitaGuard-AI.git



cd VitaGuard-AI

```





\## Install Dependencies



```bash

pip install -r requirements.txt

```





\## Run Backend



```bash

uvicorn main:app --reload

```





\## Run Frontend



```bash

npm install



npm start

```





\---



\# 👥 Team Members





| Name | Responsibility |

|---|---|

| \*\*Tanvir Hosen Nishat\*\* | AI/ML Model Development, Dataset Analysis, Explainable AI |

| \*\*Zayed\*\* | Healthcare Research and Medical Validation |

| \*\*Farhan\*\* | Backend Development and API Integration |

| \*\*Siam\*\* | Frontend Development and Dashboard Design |

| \*\*Monima\*\* | Documentation, UX Design and Presentation |

| \*\*Fardin\*\* | Dataset Support and Data Analysis |





\---



\# 🚀 Future Scope





| Future Improvement | Description |

|---|---|

| 🏥 Real Healthcare Dataset | Integration with large-scale clinical datasets |

| 📱 Mobile Application | Patient-side health monitoring application |

| 🌐 IoT Integration | Connection with wearable health devices |

| 📡 Real-Time Monitoring | Continuous patient health tracking |

| 🧠 Advanced AI Models | Improve prediction accuracy |

| ☁️ Healthcare Deployment | Deployment in healthcare organizations |





\---



\# ⚠️ Disclaimer





VitaGuard AI is a \*\*healthcare decision support system\*\* and does not replace professional medical diagnosis.



The prediction generated by this system is intended only to assist healthcare professionals.



Final medical decisions must always be made by qualified healthcare providers.





\---



\# 🌟 Vision





> "Using Explainable Artificial Intelligence to make early diabetes screening smarter, transparent, and accessible."

