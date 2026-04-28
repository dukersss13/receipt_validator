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
<img width="1411" height="1042" alt="image" src="https://github.com/user-attachments/assets/61e2ae83-9d88-4df3-b761-fb6ac52a2cc9" />


### Validate
After uploading the files, clicking `Validate` will start the validation process. The application starts reading in the uploaded transactions & receipt images.

1. The receipt images are decoded into base64 bytes payloads and sent to Gemini API for data extraction.
2. Statements in PDF forms are loaded via PyPDF Loader from LangChain into text, and the text is also sent to Gemini API for data extraction.
3. The image payload creation is done asynchronously to reduce runtime, especially when processing a large number of receipts.
API calls are also done asynchronously to improve performance.
You can swap Gemini with any VLM/LLM of your choice based on your use case and provider preferences.

#### ✅ Why Async Works for API Calls
API calls are I/O-bound operations — the program spends most of its time waiting for network responses, not doing computation. Async I/O is well-suited for I/O-bound tasks because while one request is waiting, the event loop can run other pending requests.

### Cost Estimate (Gemini 2.5 Flash-Lite)
Estimates below use Gemini Developer API paid-tier pricing for Gemini 2.5 Flash-Lite:
- Input (text/image/video): $0.10 per 1,000,000 tokens
- Output: $0.40 per 1,000,000 tokens

#### Cost per 1,000 tokens
- Input tokens: $0.0001 per 1,000 tokens
- Output tokens: $0.0004 per 1,000 tokens

#### Estimated cost per 1,000 images
Assumption for this estimate:
- Each image is counted as 258 input tokens (common case for images <= 384 px on both dimensions)
- Estimated output is excluded unless noted, since output length depends on prompt/task

Input-only estimate:
- 1,000 images x 258 tokens = 258,000 input tokens
- 258,000 / 1,000,000 x $0.10 = $0.0258

Input + output example (if you average 120 output tokens per image):
- Output tokens: 120,000
- 120,000 / 1,000,000 x $0.40 = $0.0480
- Total estimated cost for 1,000 images = $0.0738

Note: Real cost can vary with image resolution (larger images can be tiled into more tokens), prompt size, and output length.

### Validation Complete & Recommendations Provided
<img width="1294" height="1136" alt="image" src="https://github.com/user-attachments/assets/af39c9f2-8e34-416f-a9fa-4798703e81f6" />

### Final Results
Download the validated records using `Download Records` button.
<img width="1405" height="1038" alt="image" src="https://github.com/user-attachments/assets/2873b4b5-d93a-42b2-8e12-adf78cb94f17" />


