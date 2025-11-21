from fastapi import FastAPI,UploadFile, HTTPException
import boto3
import os 
import sys
from dotenv import load_dotenv
from database import get_connection,create_db
load_dotenv()

create_db()
conn, cursor = get_connection()


def upload_file_to_s3(bucket = "smart-da-bucket",file_conetnt = None,filename=None):
    try:

        boto3_session = boto3.Session(
                aws_access_key_id=os.getenv("access_key_id"),  
            aws_secret_access_key=os.getenv("secret_access_key"),  
            region_name="eu-central-1")
        s3 = boto3_session.resource('s3')

        # if s3.Object(bucket, filename).load():
        #     return {"message":"File already exists in s3"}
        object= s3.Object('smart-da-bucket',f"{filename}.csv")
        result = object.put(Body=file_conetnt)
        res = result.get('ResponseMetadata')
        if res.get('HTTPStatusCode') == 200:
            s3_uri = f"s3://{bucket}/{filename}.csv"
            cursor.executemany(
                "INSERT INTO Datasets (dataset_name, dataset_file) VALUES (?, ?)",(filename,s3_uri)
                )
            conn.commit()
            

            return {"message":"Dataset stored succesfully"}
    except Exception as e:
        return {"message":"Could not upload the file to s3","error":str(e)}
    

    

app = FastAPI()

@app.get("/Health")
def health_check():
    return {"message":"HIII!"}

@app.post("/input")
def save_dataset_in_s3(dataset_name,file:UploadFile):
    try:
        contents = file.file.read()
        result = upload_file_to_s3(file_conetnt=contents,filename=dataset_name)
        return result
       
        
        file_path_in_s3 = s3.object("smart-dat-bucket" ,f"{dataset_name}.csv")

    except Exception as e:
        raise HTTPException(status_code=500,detail="couldnot save the dataset")
        
