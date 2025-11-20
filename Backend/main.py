from fastapi import FastAPI,UploadFile
import Boto3
import os 
from dotenv import load_dotenv

load_dotenv()

boto3_session = boto3.Session(
    aws_access_key_id=f"{os.getenv("access_key_id")}",
    aws_secret_access_key=f"{os.getenv("secret_access_key")}",
    )
s3 = boto3_session.resource('s3')


app = FastAPI()

@app.get("/")
def health_check():
    return {"message":"HIII!"}

@app.post("/input")
def save_dataset_in_s3(dataset_name,file:UploadFile):
    try:
        contents = file.file.read()
        object=s3.object('smart-da-bucket',dataset_name.csv)
        object.put(Body=contents)