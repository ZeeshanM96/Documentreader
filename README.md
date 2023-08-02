# Documentreader
DocumentReader application readers the text from an  image using opencv and google-cloud-vision 

This project aims to extract text from various types of documents and images using Optical Character Recognition (OCR).

**Background**

Initially, we were using OpenCV and PyTesseract for OCR, where OpenCV was used for image processing and PyTesseract was used to convert the processed image into text. However, this approach had a few limitations:

The quality of the text extraction was highly dependent on the quality and complexity of the input images.
It worked best with images containing clear, block text against a plain background, and performance decreased with more complex images.
Due to these limitations, we explored other OCR tools and finally settled on Google Cloud Vision API, which offers more powerful and flexible OCR capabilities.

**Google Cloud Vision API**

The Google Cloud Vision API is a machine learning model that works well on complex and multi-dimensional images. It can identify text in different shapes and in close proximity to various objects in an image. Compared to the combination of OpenCV and PyTesseract, the Google Cloud Vision API has proven to be more effective at extracting text from complex images.

**Setup**

To run this project, you will need to create a Google Cloud account and set up a Google Cloud Vision API project. Here is a quick guide on how to do it:

Create a Google Cloud account: Visit cloud.google.com and create a new account or sign in to your existing account.
Create a Google Cloud project: Visit the Google Cloud Console, click on the project drop-down and select or create the project that you wish to use for the Vision API.
Enable the Vision API for your project: Search for "Vision API" in the library and enable it for your project.
Create an API key: In the "Credentials" section, create an API key for your project.
Set up authentication: Download a private key file for your service account and point to it with the GOOGLE_APPLICATION_CREDENTIALS environment variable.

**Running the Project**

`python reader.py`
