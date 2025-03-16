INSTRUCTIONS FOR RUNNING THE CODE
Running the Code
1. Install the required dependencies:
   pip install fastapi uvicorn torch torchvision pillow
2. Download the dataset and organize it into folders.
3. Run the training script (Optional, as we already have the trained model file):
   python train.py
4. Save the trained model as ‘weather_modelF.pth’.
There is no need for performing the above steps as it has already been done and the trained model file has been shared. You can continue with the below steps for testing the model.

RUNNING THE API
1. Start the FastAPI server:
   python app.py
2. Access the API at ‘http://localhost:8000/’.
3. Use the ‘/predict’ endpoint to classify images:
   cURL:
    
     curl -X POST -F "file=@path_to_image.jpg" http://localhost:8000/predict
    
   Postman:
     Set the request type to POST.
     Enter the URL: ‘http://localhost:8000/predict’.
     Go to the ‘Body’ tab, select ‘form-data’, & upload an image file under the key ‘file’.
   HTML Form (optional):
     Open `index.html` in your browser.
     Upload an image and click `Predict`.
