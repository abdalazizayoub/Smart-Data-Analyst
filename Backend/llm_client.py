from google import genai
from google.genai import types 

google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# LLM to generate non-technical description
def generate_description(summary):
    response = google_client.models.generate_content(
                            model="gemini-2.5-flash",
                            config=types.GenerateContentConfig(
                                        system_instruction="You are an expert Data Anaylst. Your name is Mo."),
                                        contents=f"generate a non-technical description given this dataset summary:{summary}"
                        )
    return response.text

#LLM to generate audio script for dataset 
def generate_audio_script(summary,description):
    response = google_client.models.generate_content(
        model="gemini-2.5-flash",
        config =types.GenerateContentConfig(
                                        system_instruction="You are a podcast script writer"),
                                        contents=f"""generate a podcast script with two hosts to explain and describe this dataset for a non techincal audience ,
                                        host one is called Abdalaziz and the other host is called Ayoub
                                        use the this summary{summary} and this description:{description} to generate the script """
                                        
                        )
    return response.text
        
    