from fastapi import FastAPI,UploadFile, HTTPException
import boto3
import os 
from dotenv import load_dotenv
from database import get_connection,create_db
from io import BytesIO
import pandas as pd
from google import genai
from google.genai import types


load_dotenv()
create_db()

google_client = genai.Client()

# Function to upload file to s3 and store metadata in sqlite
def upload_file_to_s3(bucket = "smart-da-bucket",file_content = None,filename=None):
    try:
        conn, cursor = get_connection()
        boto3_session = boto3.Session(
                aws_access_key_id=os.getenv("access_key_id"),  
            aws_secret_access_key=os.getenv("secret_access_key"),  
            region_name="eu-central-1")
        s3 = boto3_session.resource('s3')

        # if s3.Object(bucket, filename).load():
        #     return {"message":"File already exists in s3"}
        object= s3.Object('smart-da-bucket',f"{filename}.csv")
        result = object.put(Body=file_content)
        res = result.get('ResponseMetadata')
        if res.get('HTTPStatusCode') == 200:
            try:
                cursor.execute("INSERT INTO Datasets (dataset_name,dataset_file) VALUES (?,?) ",(filename,file_content))
                conn.commit()
                conn.close()    
                return {"message":"Dataset stored succesfully"}
            except Exception as e:
                return {"message":"Could not store metadata in database","error":str(e)}
           
    except Exception as e:
        return {"message":"Could not upload the file to s3","error":str(e)}
    
# Function to generate dataset description
def get_dataset_description(dataset_name ,data):
    try:
        df = pd.read_csv(BytesIO(data))
        columns = [str(col) for col in df.columns]
        
        # Separate numeric and non-numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Generate summary for numeric columns only
        numeric_summary = {}
        if numeric_columns:
            numeric_df = df[numeric_columns]
            summary_df = numeric_df.describe()
            
            for column in summary_df.columns:
                numeric_summary[str(column)] = {
                    str(stat): float(value) if pd.notna(value) else None
                    for stat, value in summary_df[column].items()
                }
        
        text_summary = {}
        for col in text_columns:
            text_summary[str(col)] = {
                "count": int(df[col].count()),
                "unique_values_list": df[col].dropna().unique().tolist(),
                "top_value": str(df[col].mode().iloc[0]) if not df[col].mode().empty else None,  
                                }
        
        return {
            "Dataset Name": dataset_name,
            "columns": columns,
            "numeric_columns": numeric_columns,
            "text_columns": text_columns,
            "numeric_summary": numeric_summary,
            "text_summary": text_summary,
            "dataset_shape": {
                "rows": int(len(df)),
                "columns": int(len(df.columns))
            }
        }
        
    except Exception as e:
        return {"error": f"Failed to generate description: {str(e)}"}
    
# LLM to generate non-technical description
def generate_description(summary):
    response = google_client.models.generate_content(
                            model="gemini-2.5-flash",
                            config=types.GenerateContentConfig(
                                        system_instruction="You are an expert Data Anaylst. Your name is Mo."),
                                        contents=f"generate a non-technical description given this dataset summary:{summary}"
                        )
    return response.text


def store_dataset_metadata(dataset_name: str, description: str = None):
    try:
        conn, cursor = get_connection()
        cursor.execute(
            "INSERT INTO Datasets (dataset_name, description) VALUES (?, ?)", 
            (dataset_name, description)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        raise Exception(f"Could not store metadata in database: {str(e)}")
    
app = FastAPI()
@app.get("/")
def health_check():
    return {"message":"HIII!"}

@app.post("/input")
def ingest_dataset(dataset_name,file:UploadFile):
    try:
        conn,cursor = get_connection()
    except Exception as e :
        raise HTTPException(status_code=503, detail="Could not connect to the database")
    
    try:
        contents = file.file.read()
        conn, cursor = get_connection()

        # Check if dataset name already exists
        cursor.execute("SELECT * FROM Datasets WHERE dataset_name = ?", (dataset_name,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Dataset name already exists")
        
        dataset_summary = get_dataset_description(dataset_name=dataset_name, data=contents)
        
        if "error" in dataset_summary:
            raise HTTPException(status_code=400, detail=dataset_summary["error"])
        
        # Upload to S3
        s3_success = upload_file_to_s3(
            file_content=contents, 
            filename=dataset_name
        )
        if not s3_success:
            raise HTTPException(status_code=500, detail="Failed to upload file to S3")
        
        dataset_description = generate_description(dataset_summary)
        
        store_dataset_metadata(
            dataset_name=dataset_name,
            description=dataset_description
        )
        
        return {
            "message": "Dataset stored successfully",
            "dataset_name": dataset_name,
            "description": dataset_description
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Could not save the dataset: {str(e)}")
    finally:
        if conn:
            conn.close()
       


@app.get("/datasets")
def list_datasets():
    try:
        conn, cursor = get_connection()
        cursor.execute("SELECT dataset_name, description FROM Datasets")
        datasets = cursor.fetchall()
        conn.close()
        
        dataset_list = [
            {"dataset_name": row[0], "description": row[1]} for row in datasets
        ]
        
        return {"datasets": dataset_list}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not retrieve datasets: {str(e)}")