
import React, { useRef, useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Webcam from "react-webcam";
import axios from "axios";
import { FaCamera, FaUpload, FaUser, FaCircleNotch } from "react-icons/fa6";
import { HiOutlineLightBulb } from "react-icons/hi";

const FaceRecognition = () => {
    const webcamRef = useRef(null);
    const fileInputRef = useRef(null);
    const [recognizedPerson, setRecognizedPerson] = useState("No one detected");
    const [isProcessing, setIsProcessing] = useState(false);
    const [cameraError, setCameraError] = useState(false);
    const [showAnimation, setShowAnimation] = useState(false);
    const navigate = useNavigate();

    useEffect(() => {
        // Check if camera is available
        const checkCamera = async () => {
            try {
                await navigator.mediaDevices.getUserMedia({ video: true });
                setCameraError(false);
            } catch (err) {
                console.error("Camera access error:", err);
                setCameraError(true);
            }
        };
        
        checkCamera();
    }, []);

    const processImage = async (imageFile) => {
        setIsProcessing(true);
        setShowAnimation(true);
        const formData = new FormData();
        formData.append("file", imageFile, "image.jpg");

        try {
            const response = await axios.post("http://127.0.0.1:8000/detect-face/", formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            
            const recognizedId = response.data.recognized_person;
            setRecognizedPerson(recognizedId);

            // Store image based on recognition
            if (recognizedId !== "Unknown") {
                await storeRecognizedImage(imageFile, recognizedId);
                setTimeout(() => {
                    navigate(`/${recognizedId}`);
                }, 1500); // Delay navigation to show the recognition result
            } else {
                await storeUnknownImage(imageFile);
            }
        } catch (error) {
            console.error("Error detecting face:", error);
            setRecognizedPerson("Error detecting face");
        } finally {
            setIsProcessing(false);
            setTimeout(() => {
                setShowAnimation(false);
            }, 2000);
        }
    };

    const storeRecognizedImage = async (imageFile, memberId) => {
        const formData = new FormData();
        formData.append("file", imageFile);
        formData.append("memberId", memberId);

        try {
            await axios.post("http://127.0.0.1:8000/store-member-image/", formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
        } catch (error) {
            console.error("Error storing member image:", error);
        }
    };

    const storeUnknownImage = async (imageFile) => {
        const formData = new FormData();
        formData.append("file", imageFile);

        try {
            await axios.post("http://127.0.0.1:8000/store-unknown-image/", formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
        } catch (error) {
            console.error("Error storing unknown image:", error);
        }
    };

    const capture = async () => {
        if (webcamRef.current) {
            const imageSrc = webcamRef.current.getScreenshot();
            if (imageSrc) {
                const blob = await fetch(imageSrc).then((res) => res.blob());
                await processImage(blob);
            }
        }
    };

    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (file) {
            await processImage(file);
        }
    };

    return (
        <div className="face-recognition-container">
            <div className="content-wrapper">
                <header className="recognition-header">
                    <h1>Smart Face Recognition</h1>
                    <p className="subtitle">
                        <HiOutlineLightBulb className="tip-icon" />
                        Position your face in the frame or upload an image
                    </p>
                </header>
                
                <div className="camera-container">
                    {cameraError ? (
                        <div className="camera-error">
                            <FaUser className="camera-error-icon" />
                            <p>Camera access denied or unavailable</p>
                        </div>
                    ) : (
                        <Webcam 
                            ref={webcamRef} 
                            screenshotFormat="image/jpeg"
                            className="webcam-preview"
                            mirrored={true}
                            audio={false}
                        />
                    )}
                    
                    {showAnimation && (
                        <div className={`recognition-overlay ${recognizedPerson !== "Unknown" && recognizedPerson !== "No one detected" ? "success" : recognizedPerson === "Unknown" ? "unknown" : ""}`}>
                            {isProcessing ? (
                                <div className="scanning-animation">
                                    <div className="scanning-line"></div>
                                </div>
                            ) : (
                                <div className="recognition-result">
                                    {recognizedPerson !== "Unknown" && recognizedPerson !== "No one detected" ? (
                                        <>
                                            <div className="recognition-icon success">✓</div>
                                            <p>Welcome, {recognizedPerson}!</p>
                                        </>
                                    ) : recognizedPerson === "Unknown" ? (
                                        <>
                                            <div className="recognition-icon unknown">?</div>
                                            <p>Face not recognized</p>
                                        </>
                                    ) : (
                                        <>
                                            <div className="recognition-icon error">!</div>
                                            <p>Error detecting face</p>
                                        </>
                                    )}
                                </div>
                            )}
                        </div>
                    )}
                </div>
                
                <div className="action-buttons">
                    <button
                        onClick={capture}
                        className="action-button capture-button"
                        disabled={isProcessing || cameraError}
                    >
                        {isProcessing ? <FaCircleNotch className="spinning" /> : <FaCamera />}
                        <span>Capture</span>
                    </button>
                    
                    <input 
                        type="file" 
                        ref={fileInputRef} 
                        className="file-input" 
                        accept="image/*"
                        onChange={handleFileUpload}
                    />
                    
                    <button
                        onClick={() => fileInputRef.current.click()}
                        className="action-button upload-button"
                        disabled={isProcessing}
                    >
                        {isProcessing ? <FaCircleNotch className="spinning" /> : <FaUpload />}
                        <span>Upload Image</span>
                    </button>
                </div>
                
                <div className={`recognition-status ${recognizedPerson !== "No one detected" ? "active" : ""}`}>
                    <div className="status-indicator"></div>
                    <p>{recognizedPerson !== "No one detected" ? 
                        `Recognized: ${recognizedPerson}` : 
                        "Ready to recognize"}
                    </p>
                </div>
            </div>
            
            <style jsx>{`
                .face-recognition-container {
                    max-width: 100%;
                    min-height: 100vh;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    background: linear-gradient(135deg, #e0f7fa 0%, #f5f5f5 100%);
                    padding: 20px;
                    font-family: 'Poppins', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                }
                
                .content-wrapper {
                    width: 100%;
                    max-width: 800px; /* Increased from 600px */
                    background: rgba(255, 255, 255, 0.9);
                    border-radius: 24px;
                    box-shadow: 0 10px 40px rgba(0, 150, 255, 0.08), 
                                0 0 20px rgba(0, 150, 255, 0.05);
                    padding: 40px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    position: relative;
                    overflow: hidden;
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.7);
                }
                
                .recognition-header {
                    text-align: center;
                    margin-bottom: 24px;
                    width: 100%;
                }
                
                .recognition-header h1 {
                    font-size: 32px;
                    font-weight: 700;
                    margin: 0 0 12px 0;
                    background: linear-gradient(90deg, #4fc3f7, #81d4fa);
                    -webkit-background-clip: text;
                    background-clip: text;
                    -webkit-text-fill-color: transparent;
                    letter-spacing: -0.5px;
                }
                
                .subtitle {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #78909c;
                    font-size: 16px;
                    margin: 0;
                }
                
                .tip-icon {
                    margin-right: 8px;
                    color: #4fc3f7;
                    font-size: 20px;
                }
                
                .camera-container {
                    width: 100%;
                    height: 450px; /* Increased from 300px */
                    margin-bottom: 32px;
                    border-radius: 20px;
                    overflow: hidden;
                    position: relative;
                    background-color: #e3f2fd;
                    box-shadow: 0 8px 20px rgba(0, 150, 255, 0.1);
                    border: 3px solid #f5f5f5;
                }
                
                .webcam-preview {
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                    border-radius: 16px;
                    transform: scale(1.01); /* Slight scale to avoid white borders */
                }
                
                .camera-error {
                    width: 100%;
                    height: 100%;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    color: #78909c;
                    background-color: #e3f2fd;
                    border-radius: 16px;
                }
                
                .camera-error-icon {
                    font-size: 60px;
                    margin-bottom: 20px;
                    color: #b3e5fc;
                }
                
                .recognition-overlay {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background-color: rgba(0, 0, 0, 0.3);
                    border-radius: 16px;
                    z-index: 2;
                    backdrop-filter: blur(3px);
                    transition: all 0.3s ease;
                }
                
                .recognition-overlay.success {
                    background-color: rgba(77, 208, 225, 0.4);
                }
                
                .recognition-overlay.unknown {
                    background-color: rgba(255, 193, 7, 0.4);
                }
                
                .scanning-animation {
                    width: 100%;
                    height: 100%;
                    position: relative;
                    overflow: hidden;
                }
                
                .scanning-line {
                    position: absolute;
                    width: 100%;
                    height: 4px;
                    background: linear-gradient(90deg, 
                        rgba(255,255,255,0) 0%, 
                        rgba(129,212,250,1) 50%, 
                        rgba(255,255,255,0) 100%);
                    animation: scan 1.8s ease-in-out infinite;
                    box-shadow: 0 0 15px rgba(77, 208, 225, 0.8);
                }
                
                @keyframes scan {
                    0% { top: 0; }
                    50% { top: 100%; }
                    100% { top: 0; }
                }
                
                .recognition-result {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 24px;
                    font-weight: 600;
                    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                }
                
                .recognition-icon {
                    width: 70px;
                    height: 70px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 30px;
                    margin-bottom: 20px;
                    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
                }
                
                .recognition-icon.success {
                    background-color: #4dd0e1;
                }
                
                .recognition-icon.unknown {
                    background-color: #ffc107;
                }
                
                .recognition-icon.error {
                    background-color: #ff5252;
                }
                
                .action-buttons {
                    display: flex;
                    gap: 20px;
                    margin-bottom: 30px;
                    width: 100%;
                }
                
                .action-button {
                    flex: 1;
                    padding: 16px 24px;
                    border-radius: 16px;
                    border: none;
                    font-size: 17px;
                    font-weight: 600;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 12px;
                    transition: all 0.3s ease;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
                }
                
                .action-button:disabled {
                    opacity: 0.7;
                    cursor: not-allowed;
                }
                
                .capture-button {
                    background: linear-gradient(45deg, #4fc3f7, #29b6f6);
                    color: white;
                }
                
                .capture-button:hover:not(:disabled) {
                    background: linear-gradient(45deg, #29b6f6, #03a9f4);
                    transform: translateY(-3px);
                    box-shadow: 0 6px 15px rgba(41, 182, 246, 0.4);
                }
                
                .upload-button {
                    background: linear-gradient(45deg, #b3e5fc, #81d4fa);
                    color: #0277bd;
                }
                
                .upload-button:hover:not(:disabled) {
                    background: linear-gradient(45deg, #81d4fa, #4fc3f7);
                    transform: translateY(-3px);
                    box-shadow: 0 6px 15px rgba(79, 195, 247, 0.4);
                    color: #01579b;
                }
                
                .file-input {
                    display: none;
                }
                
                .recognition-status {
                    display: flex;
                    align-items: center;
                    padding: 14px 20px;
                    border-radius: 16px;
                    background-color: #f5f5f5;
                    width: 100%;
                    transition: all 0.3s ease;
                }
                
                .recognition-status.active {
                    background-color: #e1f5fe;
                }
                
                .status-indicator {
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    background-color: #b0bec5;
                    margin-right: 14px;
                    transition: all 0.3s ease;
                }
                
                .recognition-status.active .status-indicator {
                    background-color: #29b6f6;
                    box-shadow: 0 0 0 4px rgba(41, 182, 246, 0.2);
                }
                
                .recognition-status p {
                    margin: 0;
                    color: #78909c;
                    font-size: 15px;
                    font-weight: 500;
                    transition: all 0.3s ease;
                }
                
                .recognition-status.active p {
                    color: #0277bd;
                    font-weight: 600;
                }
                
                .spinning {
                    animation: spin 1s linear infinite;
                }
                
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                
                @media (max-width: 768px) {
                    .content-wrapper {
                        padding: 30px 20px;
                        border-radius: 20px;
                    }
                    
                    .camera-container {
                        height: 350px;
                    }
                }
                
                @media (max-width: 500px) {
                    .content-wrapper {
                        padding: 20px 15px;
                    }
                    
                    .camera-container {
                        height: 300px;
                    }
                    
                    .action-buttons {
                        flex-direction: column;
                    }
                    
                    .action-button {
                        width: 100%;
                    }
                    
                    .recognition-header h1 {
                        font-size: 28px;
                    }
                }
            `}</style>
        </div>
    );
};

export default FaceRecognition;