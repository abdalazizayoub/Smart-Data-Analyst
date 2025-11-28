# 🧠 Smart Data Analyst  
### *A Node-Based, Agentic Data Analysis Platform*

Smart Data Analyst is an **agentic, node-driven data analysis platform** that transforms raw datasets into automated insights.  
Using a workflow of connected analysis nodes, an **LLM agent** decides which statistical operations to run, interprets results, and orchestrates multi-step analysis, all without manual prompting.

---

## Key Features

### 🔗 Node-Based Workflow Engine  
- Create analysis pipelines using modular nodes  
- Each node represents a statistical or ML operation (correlation, regression, clustering, PCA, etc.)  
- Supports chained execution, branching logic, and agent-triggered node selection  
- Ideal for experimentation and building reusable analysis flows

### 🧠 LLM Agent (Gemini)  
- Reads dataset summaries and metadata  
- Chooses which analysis node should run next  
- Generates reasoning, recommendations, and follow-up actions  
- Enables fully autonomous, multi-step analysis workflows

### 📁 Robust Data Ingestion  
- Upload CSV datasets via API or UI  
- Files securely stored in **AWS S3**  
- Dataset metadata and analysis history stored in **SQLite**  
- Enables versioning and quick dataset retrieval

### 📊 Automated Analysis & Visualizations  
- Nodes communicate with FastAPI analysis endpoints  
- Generates plots, metrics, tables, and statistical summaries  
- Supports passing results into downstream nodes for deeper insights

### 🔊 Multimodal Output  
- Converts LLM-generated summaries into natural-sounding audio  
- Enables “listen to your dataset” functionality  
- Great for accessibility and data storytelling

### 🖥 User Interface  
- Flow-based UI (Streamlit or LangFlow-style) for building node pipelines  
- Interactive dataset manager  
- Run flows, visualize results, and inspect agent decisions


