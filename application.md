# Receipt Validator in Action

## To Start
Be sure to follow the set up steps to install Docker, set up the necessary API keys. Once that is complete, open the application in the Docker dev-container.

![image](https://github.com/user-attachments/assets/283615d4-9cb4-4e8e-a4df-98d9b87902be)


You can also run `pip install -r requirements.txt` in the terminal and run the application.

### Run main.py
To start the application, run main.py in VSCode or `python3 main.py` in the terminal. 
What's cool about this set up is you are also able to run the application in `DEBUG` mode if you are interested in stepping through the code line by line.

### ArVee UI
<img width="1443" height="1075" alt="image" src="https://github.com/user-attachments/assets/b059bdc5-0694-41b9-a640-7ee36de296e4" />

## Main Features
##### The demo shown will demonstrate 3 main features of RV:
### 1. Transaction-receipt matching & validation.
![Screenshot 2025-05-12 at 11 49 27 AM](https://github.com/user-attachments/assets/bbfab5c1-7077-4cab-8939-538dbba8676b)
### 2. Discrepancy Flagging
![Screenshot 2025-05-12 at 11 58 02 AM](https://github.com/user-attachments/assets/62180a06-ca45-496c-a56e-eec7c7b9d722)
### 3. Recommendation for Unmatched Transaction
![Screenshot 2025-05-12 at 11 53 55 AM](https://github.com/user-attachments/assets/5a6f1f6a-1c38-4448-a589-e5d8ff70bca7)


## Application Run

### Files Upload
<img width="1422" height="414" alt="image" src="https://github.com/user-attachments/assets/9a53335e-23b3-4e28-a4dc-1d1bd6dffaae" />

### Validate
After uploading the files, clicking `Validate` will start the validation process. The application starts reading in the uploaded transactions & receipt images.

1. The receipt images are decoded into base64 bytes payloads & get sent to OpenAI API for data extraction.
2. Statements in PDF forms are loaded via PyPDF Loader from LangChain into text & the text is also sent to OpenAI API for data extraction.
3. The image payload creation is done in multi-thread fashion to mitigate run time, as you can imagine there could be potentially a large amount of receipts to process.
API calls are also done multi-threadedly to improve performance.

#### ✅ Why Multithreading Works for API Calls
API calls are I/O-bound operations — the program spends most of its time waiting for network responses, not doing computation. Multithreading is well-suited for I/O-bound tasks because while one thread is waiting, another can proceed with its request.

### Validation Complete & Recommendations Provided
![Screenshot 2025-05-12 at 12 18 29 PM](https://github.com/user-attachments/assets/74d83dcf-95c0-4286-a0c1-b6163531494e)

### Final Results
Download the validated records using `Download Records` button.
![Screenshot 2025-05-12 at 11 29 30 AM](https://github.com/user-attachments/assets/5a2a1608-96a7-48e3-8242-cf2778c1f5e6)

