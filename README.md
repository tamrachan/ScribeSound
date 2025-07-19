# ScribeSound
A personal project which makes playing music fully portable!   
It combines AI and music - allowing you to handwrite digits, lowercase and uppercase letters, map them to sounds, and play them with your fingers.

## How it works
This project that uses YOLO object detection to recognise alphanumerical data, trained by my own handwritten data set (about 1000 images so not super accurate). Once the AI has detected your handwritten digits/letters, you can map those labels to sounds in the 'notes' folder, which consists of wav files. Then, using your computer's webcam, you can tap on your handwritten digits/letters and the corresponding sounds will play.

## How to set up
**Create a virtual envionment**
`python -m venv venv`
**Activate the virtual environment**
`./venv/Scripts/activate`
**Install the requirements**
`pip install -r requirements.txt`

## Errors to note
If there are duplicate letters, it is random which sound will be produced.