
import cv2
import pickle
import face_recognition
import numpy as np
import os
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import uvicorn
from pydantic import BaseModel

# Create the FastAPI app
app = FastAPI(title="Face Recognition Backend API", description="Face Recognition API for MemoryVault")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories for storing images
MEMBER_IMAGES_DIR = "MembersData"
UNKNOWN_IMAGES_DIR = "UnknownImages"
IMAGES_DIR = "images"

# Ensure directories exist
os.makedirs(MEMBER_IMAGES_DIR, exist_ok=True)
os.makedirs(UNKNOWN_IMAGES_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Global variables for face encoding
encodeListKnown = []
studentIds = []

# Load the encoding file
def load_encodings():
    global encodeListKnown, studentIds
    print("Loading Encoded File ...")
    try:
        with open("EncodeFile.p", "rb") as file:
            encodeListKnownWithIds = pickle.load(file)
        
        # Ensure encodings are numpy arrays
        encodeListKnown = [np.array(encoding) for encoding in encodeListKnownWithIds[0]]
        studentIds = encodeListKnownWithIds[1]
        print("Encode File Loaded Successfully")
    except Exception as e:
        print(f"Error loading encoding file: {e}")
        encodeListKnown = []
        studentIds = []

# Function to generate encodings from images
def find_encodings(img_list):
    encode_list = []
    for img in img_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            # Get all face encodings
            face_encodings = face_recognition.face_encodings(img)
            if face_encodings:
                # Take the first face encoding
                encode = face_encodings[0]
                encode_list.append(encode)
            else:
                print(f"No face found in an image")
        except Exception as e:
            print(f"Error processing image: {e}")
    return encode_list

# Function to generate encodings and save to file
def generate_encodings():
    global encodeListKnown, studentIds
    
    folder_path = IMAGES_DIR
    path_list = os.listdir(folder_path)
    img_list = []
    student_ids = []

    for path in path_list:
        if path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_list.append(cv2.imread(os.path.join(folder_path, path)))
            student_ids.append(os.path.splitext(path)[0])

    print("Encoding Started...")
    encode_list_known = find_encodings(img_list)
    encode_list_known_with_ids = [encode_list_known, student_ids]
    print("Encoding Complete")

    file = open("EncodeFile.p", "wb")
    pickle.dump(encode_list_known_with_ids, file)
    file.close()
    print("File Saved")
    
    # Update global variables
    encodeListKnown = encode_list_known
    studentIds = student_ids
    
    return {"status": "success", "encoded_faces": len(encode_list_known)}

# Load encodings on startup
load_encodings()

# Function to check if image is a duplicate
def is_duplicate_image(new_img_path):
    # This is a basic implementation to avoid exact duplicates
    # You might want to enhance this with image similarity comparison if needed
    if not os.path.exists(UNKNOWN_IMAGES_DIR):
        return False
        
    # Get list of existing unknown images
    unknown_images = [f for f in os.listdir(UNKNOWN_IMAGES_DIR) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # If no unknown images exist, it's not a duplicate
    if not unknown_images:
        return False
    
    # Read the new image
    new_img = cv2.imread(new_img_path)
    
    # Simple file size check to quickly filter out obvious non-duplicates
    new_size = os.path.getsize(new_img_path)
    
    for img_name in unknown_images:
        img_path = os.path.join(UNKNOWN_IMAGES_DIR, img_name)
        
        # Quick size check first
        if abs(os.path.getsize(img_path) - new_size) > 1024:  # If size differs by more than 1KB
            continue
            
        # Only compare images that are similar in size
        existing_img = cv2.imread(img_path)
        
        # Simple hash-based comparison
        if existing_img.shape == new_img.shape:
            difference = cv2.norm(existing_img, new_img, cv2.NORM_L2)
            if difference < 100:  # Threshold for considering images as duplicates
                return True
                
    return False

# Face Recognition API Endpoints
@app.post("/detect-face/")
async def detect_face(file: UploadFile = File(...)):
    # Check if any encodings are available
    if not encodeListKnown:
        return {"recognized_person": "No encodings available"}

    # Read image bytes
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize & Convert
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Recognize faces
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    result = "Unknown"
    for encodeFace in encodeCurFrame:
        # Use a try-except block to handle potential errors
        try:
            # Compare faces with a lower tolerance to reduce false negatives
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.6)
            
            # If any match is found
            if True in matches:
                matchIndex = matches.index(True)
                result = studentIds[matchIndex]
                break
        except Exception as e:
            print(f"Error in face comparison: {e}")
            continue
    
    # If face is not recognized, store it in unknown faces (with duplicate check)
    if result == "Unknown" and faceCurFrame:
        try:
            # Generate unique filename
            filename = f"unknown_{uuid.uuid4()}.jpg"
            filepath = os.path.join(UNKNOWN_IMAGES_DIR, filename)
            
            # Save file temporarily
            cv2.imwrite(filepath, img)
            
            # Check if this is a duplicate of an existing unknown image
            if is_duplicate_image(filepath):
                # If it's a duplicate, delete it
                os.remove(filepath)
                print("Duplicate unknown face detected - not storing")
            else:
                print(f"Stored unknown face: {filename}")
        except Exception as e:
            print(f"Error handling unknown face: {e}")
            # Clean up if there was an error
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass

    return {"recognized_person": result}

@app.post("/store-member-image/")
async def store_member_image(
    file: UploadFile = File(...), 
    memberId: str = Form(...)
):
    # Generate unique filename
    filename = f"{memberId}_{uuid.uuid4()}.jpg"
    filepath = os.path.join(MEMBER_IMAGES_DIR, str(memberId), filename)
    
    # Ensure member directory exists
    os.makedirs(os.path.join(MEMBER_IMAGES_DIR, str(memberId)), exist_ok=True)
    
    # Save file
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"filename": filename, "status": "success"}

@app.post("/store-unknown-image/")
async def store_unknown_image(file: UploadFile = File(...)):
    # Generate unique filename
    filename = f"unknown_{uuid.uuid4()}.jpg"
    filepath = os.path.join(UNKNOWN_IMAGES_DIR, filename)
    
    # Save file
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"filename": filename, "status": "success"}

@app.get("/get-member-images/{member_id}")
async def get_member_images(member_id: str):
    member_dir = os.path.join(MEMBER_IMAGES_DIR, member_id)
    if not os.path.exists(member_dir):
        return {"images": []}
    
    images = [
        f for f in os.listdir(member_dir) 
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ]
    return {"images": images}

@app.delete("/delete-image/{member_id}/{filename}")
async def delete_image(member_id: str, filename: str):
    filepath = os.path.join(MEMBER_IMAGES_DIR, member_id, filename)
    
    try:
        os.remove(filepath)
        return {"status": "success", "message": "Image deleted"}
    except FileNotFoundError:
        return {"status": "error", "message": "File not found"}

@app.get("/get-unknown-images/")
async def get_unknown_images():
    """Get list of all unknown images"""
    try:
        if not os.path.exists(UNKNOWN_IMAGES_DIR):
            return {"images": []}
        
        images = [
            f for f in os.listdir(UNKNOWN_IMAGES_DIR) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        return {"images": images}
    except Exception as e:
        print(f"Error getting unknown images: {e}")
        return {"images": [], "error": str(e)}

@app.get("/get-unknown-image/{filename}")
async def get_unknown_image(filename: str):
    """Get a specific unknown image file"""
    file_path = os.path.join(UNKNOWN_IMAGES_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Return the image file
    return FileResponse(file_path)

@app.delete("/delete-unknown-image/{filename}")
async def delete_unknown_image(filename: str):
    """Delete an unknown image"""
    file_path = os.path.join(UNKNOWN_IMAGES_DIR, filename)
    
    try:
        os.remove(file_path)
        return {"status": "success", "message": "Unknown image deleted"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")

# Endpoint to regenerate encodings
@app.post("/generate-encodings/")
async def api_generate_encodings():
    return generate_encodings()

@app.get("/health/")
async def health_check():
    return {"status": "Service is running"}

# Main entry point
if __name__ == '__main__':
    # Generate encodings if the encode file doesn't exist
    if not os.path.exists("EncodeFile.p"):
        generate_encodings()
    
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)