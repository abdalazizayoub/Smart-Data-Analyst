from fastapi import FastAPI,UploadFile, HTTPException
import Boto3
import os 
from dotenv import load_dotenv

load_dotenv()

def upload_file_to_s3(bucket = "smart-da-bucket",file_conetnt = None,filename:str ):
    boto3_session = boto3.Session(
        aws_access_key_id=f"{os.getenv("access_key_id")}",
        aws_secret_access_key=f"{os.getenv("secret_access_key")}",
        region = "Eu-central-1"
        )
    s3 = boto3_session.resource('s3')
    object= s3.object('smart-da-bucket',f"{filename}.csv")
    result = object.put(Body=file_conetnt)
    res = result.get('ResponseMetadata')
    if res.get('HTTPStatusCode') == 200:
        return {"message":"Dataset stored succesfully"}
    else:
        return {"message":"Dataset was not stored succesfully"}

    

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
        
