from fastapi import FastAPI,UploadFile, HTTPException
import boto3
import os 
from dotenv import load_dotenv
from database import get_connection,create_db
from io import BytesIO
import pandas as pd
from google import genai
from google.genai import types 
from nodes import feature_importance, drop_na_rows, correlation_analysis, get_dataset_description
from llm_client import generate_description
load_dotenv()
create_db()

google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

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
        dataset_uri = f"s3://{bucket}/{filename}.csv"
        if res.get('HTTPStatusCode') == 200:
            try:
                cursor.execute("INSERT INTO Datasets (dataset_name,dataset_file) VALUES (?,?) ",(filename,dataset_uri))
                conn.commit()
                conn.close()    
                return {"message":"Dataset stored succesfully"}
            except Exception as e:
                return {"message":"Could not store metadata in database","error":str(e)}
           
    except Exception as e:
        return {"message":"Could not upload the file to s3","error":str(e)}
  

def store_dataset_metadata(dataset_name: str, description: str = None, summary:str = None):
    try:
        conn, cursor = get_connection()
        cursor.execute(
            "INSERT INTO Datasets (dataset_name, description,dataset_summary) VALUES (?, ?,?)", 
            (dataset_name, description,summary)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        raise Exception(f"Could not store metadata in database: {str(e)}")
#-------------------------------------backend Logic ---------------------------------------------------------------------------------
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    
@app.post("/drop-na-rows")
def drop_na_rows_endpoint(dataset_name: str):
    try:
        conn, cursor = get_connection()
        cursor.execute("SELECT dataset_file FROM Datasets WHERE dataset_name = ?", (dataset_name,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset_uri = result[0]
        
        # Download dataset from S3
        boto3_session = boto3.Session(
                aws_access_key_id=os.getenv("access_key_id"),  
            aws_secret_access_key=os.getenv("secret_access_key"),  
            region_name="eu-central-1")
        s3 = boto3_session.client('s3')
        
        bucket_name, key = dataset_uri.replace("s3://", "").split("/", 1)
        s3_object = s3.get_object(Bucket=bucket_name, Key=key)
        data = s3_object['Body'].read()
        
        df = pd.read_csv(BytesIO(data))
        
        cleaned_df = drop_na_rows(df)
        
        return {
            "original_shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
            "cleaned_shape": {"rows": int(len(cleaned_df)), "columns": int(len(cleaned_df.columns))}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing drop NA rows: {str(e)}")
    
    
@app.post("/feature-importance")
def feature_importance_endpoint(dataset_name: str, target_column: str, task_type: str):
    try:
        conn, cursor = get_connection()
        cursor.execute("SELECT dataset_file FROM Datasets WHERE dataset_name = ?", (dataset_name,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset_uri = result[0]
        
        # Download dataset from S3
        boto3_session = boto3.Session(
                aws_access_key_id=os.getenv("access_key_id"),  
            aws_secret_access_key=os.getenv("secret_access_key"),  
            region_name="eu-central-1")
        s3 = boto3_session.client('s3')
        
        bucket_name, key = dataset_uri.replace("s3://", "").split("/", 1)
        s3_object = s3.get_object(Bucket=bucket_name, Key=key)
        data = s3_object['Body'].read()
        
        df = pd.read_csv(BytesIO(data))
        
        results = feature_importance(
            df=df,
            class_label=target_column,
            test_size=0.2,
            task_type=task_type  
        )
        
        if 'error' in results:
            raise HTTPException(status_code=400, detail=results['error'])
        
        return {
            "task_type": results['task_type'],
            "test_score": results['test_score'],
            "importance": results['importance'].to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing feature importance: {str(e)}")
    
    
@app.post("/correlation-analysis")
def correlation_analysis_endpoint(dataset_name: str):
    try:
        conn, cursor = get_connection()
        cursor.execute("SELECT dataset_file FROM Datasets WHERE dataset_name = ?", (dataset_name,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset_uri = result[0]
        
        # Download dataset from S3
        boto3_session = boto3.Session(
                aws_access_key_id=os.getenv("access_key_id"),  
            aws_secret_access_key=os.getenv("secret_access_key"),  
            region_name="eu-central-1")
        s3 = boto3_session.client('s3')
        
        bucket_name, key = dataset_uri.replace("s3://", "").split("/", 1)
        s3_object = s3.get_object(Bucket=bucket_name, Key=key)
        data = s3_object['Body'].read()
        
        df = pd.read_csv(BytesIO(data))
        
        corr_matrix = correlation_analysis(df)
        
        return {
            "correlation_matrix": corr_matrix.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing correlation analysis: {str(e)}")



@app.get("dataset/{dataset_name}")
def get_dataset_details(dataset_name: str): 
    try:
        conn, cursor = get_connection()
        cursor.execute("SELECT dataset_name, description FROM Datasets WHERE dataset_name = ?", (dataset_name,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        return {
            "dataset_name": result[0],
            "description": result[1]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not retrieve dataset details: {str(e)}")
    
    
@app.get("/podcast/{dataset_name}")
def generate_podcate(dataset_name:str):
    try:    
        conn,cursor = get_connection()
        cursor.execute("Select dataset_name, description,summary FROM Datasets WHERE dataset_name = ?", (dataset_name,))
        result = cursor.fetchone()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not connect to the database: {str(e)}")
    
        
    